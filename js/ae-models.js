/* ae-models.js — autoencoders over the 128-sample window, written from scratch.
 *
 * An autoencoder squeezes the window through a bottleneck of k numbers and
 * rebuilds it from them. Trained on what is normal, it rebuilds normal windows
 * well and everything else badly — so the reconstruction error is an anomaly
 * score that needs no labels at all.
 *
 * All models share one interface:
 *   forward(x, keep, training) → reconstruction (Float32Array, WIN samples)
 *   backward(dLoss/dOutput)    → extra loss of the bottleneck (the VAE's KL term)
 *   trainBatch(inputs, targets, lr, l2), reconstruct(x), score(x), code (the bottleneck)
 * and a list of stages the diagram draws.
 */

/* ------------------------------------------------------------- Upsample1D */
/** Repeats every sample twice: the decoder's way back up to the full length. */
class Upsample1D {
  constructor() { this.type = 'up'; }
  forward(x, L) {
    const C = x.length / L;
    this.C = C; this.L = L;
    const out = new Float32Array(C * 2 * L);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < L; t++) {
        const v = x[c * L + t];
        out[c * 2 * L + 2 * t] = v;
        out[c * 2 * L + 2 * t + 1] = v;
      }
    }
    return out;
  }
  backward(d) {
    const { C, L } = this;
    const dx = new Float32Array(C * L);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < L; t++) dx[c * L + t] = d[c * 2 * L + 2 * t] + d[c * 2 * L + 2 * t + 1];
    }
    return dx;
  }
}

/* ------------------------------------------------------------- Bottleneck */
/**
 * The k numbers in the middle. A plain autoencoder maps straight to them. A
 * variational one (VAE) predicts a mean μ and a log-variance for each and, in
 * training, samples z = μ + σ·ε; a KL term pulls every code towards N(0, 1), so
 * the space between the codes means something and can be sampled from.
 */
class Bottleneck {
  constructor(nin, k, vae, beta) {
    this.k = k; this.vae = !!vae; this.beta = beta || 0;
    this.mu = new Dense(nin, k);
    this.lv = this.vae ? new Dense(nin, k) : null;
    if (this.lv) { this.lv.W.forEach((_, i) => { this.lv.W[i] *= 0.1; }); this.lv.b.fill(-2); }
    this.params = this.vae ? [this.mu, this.lv] : [this.mu];
  }
  forward(h, training) {
    const mu = this.mu.forward(h);
    this.m = mu;
    if (!this.vae) { this.z = mu; return mu; }
    const lv = this.lv.forward(h);
    for (let i = 0; i < lv.length; i++) lv[i] = Math.max(-8, Math.min(8, lv[i]));
    this.l = lv;
    const z = new Float32Array(this.k), eps = new Float32Array(this.k);
    for (let i = 0; i < this.k; i++) {
      eps[i] = training ? randn() : 0;
      z[i] = mu[i] + Math.exp(0.5 * lv[i]) * eps[i];
    }
    this.eps = eps; this.z = z;
    return z;
  }
  /** β · KL( N(μ, σ²) ‖ N(0, 1) ) of the last forward pass. */
  kl() {
    if (!this.vae) return 0;
    let s = 0;
    for (let i = 0; i < this.k; i++) s += -0.5 * (1 + this.l[i] - this.m[i] * this.m[i] - Math.exp(this.l[i]));
    return this.beta * s;
  }
  backward(dz) {
    if (!this.vae) return this.mu.backward(dz);
    const dmu = new Float32Array(this.k), dlv = new Float32Array(this.k);
    for (let i = 0; i < this.k; i++) {
      const sd = Math.exp(0.5 * this.l[i]);
      dmu[i] = dz[i] + this.beta * this.m[i];
      dlv[i] = dz[i] * this.eps[i] * 0.5 * sd + this.beta * 0.5 * (Math.exp(this.l[i]) - 1);
    }
    const dh = this.mu.backward(dmu), dh2 = this.lv.backward(dlv);
    for (let i = 0; i < dh.length; i++) dh[i] += dh2[i];
    return dh;
  }
}

/* --------------------------------------------------------------- the base */
class AEBase {
  setup(cfg) {
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.t = 0;            // Adam step counter
    this.clip = 1.0;       // global gradient-norm clipping
  }
  zeroGrads() { for (const p of this.params) { p.gW.fill(0); p.gb.fill(0); } }
  step(lr, scale, l2) { RNNNet.prototype.step.call(this, lr, scale, l2); }
  paramCount() { let s = 0; for (const p of this.params) s += p.W.length + p.b.length; return s; }

  /** One mini-batch: the mean squared error between output and target, plus the bottleneck's term. */
  trainBatch(inputs, targets, lr, l2) {
    this.zeroGrads();
    let loss = 0;
    const n = inputs.length;
    for (let i = 0; i < n; i++) {
      const y = this.forward(inputs[i], false, true);
      const t = targets[i], g = new Float32Array(y.length);
      let l = 0;
      for (let j = 0; j < y.length; j++) {
        const e = y[j] - t[j];
        l += e * e;
        g[j] = 2 * e / y.length;
      }
      loss += l / y.length + this.backward(g);
    }
    this.step(lr, 1 / n, l2);
    return loss / n;
  }

  reconstruct(x) { return this.forward(x, false, false); }

  /** Anomaly score: the mean squared reconstruction error of the window. */
  score(x) {
    const y = this.reconstruct(x);
    let s = 0;
    for (let i = 0; i < y.length; i++) { const e = y[i] - x[i]; s += e * e; }
    return s / y.length;
  }
}

/* --------------------------------------------------------------- MLP AE */
class MLPAE extends AEBase {
  /** @param {{hidden:number, k:number, vae:boolean, beta:number}} cfg */
  constructor(cfg) {
    super();
    this.kind = 'mlp';
    this.setup(cfg);
    const h = cfg.hidden, k = cfg.k;
    this.e1 = new Dense(WIN, h);
    this.a1 = new Activation('tanh');
    this.bn = new Bottleneck(h, k, cfg.vae, cfg.beta);
    this.d1 = new Dense(k, h);
    this.a2 = new Activation('tanh');
    this.d2 = new Dense(h, WIN);
    this.params = [this.e1, ...this.bn.params, this.d1, this.d2];
    this.stages = [
      { kind: 'units', part: 'enc', label: 'ENCODER · ' + h + ' units', C: h, dense: this.e1, act: 'tanh' },
      { kind: 'code', part: 'code', label: 'CODE · k = ' + k + (cfg.vae ? ' · VAE' : ''), C: k, dense: this.bn.mu },
      { kind: 'units', part: 'dec', label: 'DECODER · ' + h + ' units', C: h, dense: this.d1, act: 'tanh' },
    ];
    this.outDense = this.d2;
  }

  forward(x, keep, training) {
    const h1 = this.a1.forward(this.e1.forward(x), 1);
    const z = this.bn.forward(h1, training);
    const h2 = this.a2.forward(this.d1.forward(z), 1);
    const y = this.d2.forward(h2);
    this.code = this.bn.m;
    if (keep) {
      this.input = x;
      const s = this.stages;
      s[0].snapshot = h1.slice(); s[0].input = x;
      s[1].snapshot = this.bn.m.slice(); s[1].input = h1.slice(); s[1].z = z.slice();
      s[2].snapshot = h2.slice(); s[2].input = z.slice();
      this.outInput = h2.slice();
      this.out = y.slice();
    }
    return y;
  }

  backward(g) {
    let d = this.d2.backward(g);
    d = this.d1.backward(this.a2.backward(d));
    d = this.bn.backward(d);
    this.e1.backward(this.a1.backward(d));
    return this.bn.kl();
  }

  /** The decoder alone: a window from a code. */
  decode(z) { return this.d2.forward(this.a2.forward(this.d1.forward(z), 1)); }
}

/* --------------------------------------------------------------- Conv AE */
class ConvAE extends AEBase {
  /** @param {{filters:number, k:number, vae:boolean, beta:number}} cfg */
  constructor(cfg) {
    super();
    this.kind = 'conv';
    this.setup(cfg);
    const f = cfg.filters, k = cfg.k;
    // encoder: three conv → ReLU → max-pool steps, 128 → 64 → 32 → 16
    this.enc = [7, 5, 5].map((K, i) => ({
      conv: new Conv1D(i === 0 ? 1 : f, f, K, 1, false), act: new Activation('relu'), pool: new MaxPool1D(2),
    }));
    this.bn = new Bottleneck(f * 16, k, cfg.vae, cfg.beta);
    // decoder: back to 16 positions, then three upsample → conv steps, 16 → 32 → 64 → 128
    this.dd = new Dense(k, f * 16);
    this.da = new Activation('relu');
    this.dec = [5, 5, 7].map((K, i) => ({
      up: new Upsample1D(), conv: new Conv1D(f, i === 2 ? 1 : f, K, 1, false), act: i === 2 ? null : new Activation('relu'),
    }));
    this.params = [...this.enc.map((s) => s.conv), ...this.bn.params, this.dd, ...this.dec.map((s) => s.conv)];
    this.stages = [
      { kind: 'map', part: 'enc', label: 'ENC 1 · K=7 · pool', C: f, L: 64, step: this.enc[0], Lin: WIN },
      { kind: 'map', part: 'enc', label: 'ENC 2 · K=5 · pool', C: f, L: 32, step: this.enc[1], Lin: 64 },
      { kind: 'map', part: 'enc', label: 'ENC 3 · K=5 · pool', C: f, L: 16, step: this.enc[2], Lin: 32 },
      { kind: 'code', part: 'code', label: 'CODE · k = ' + k + (cfg.vae ? ' · VAE' : ''), C: k, dense: this.bn.mu },
      { kind: 'map', part: 'dec', label: 'DEC 1 · dense', C: f, L: 16, dense: this.dd },
      { kind: 'map', part: 'dec', label: 'DEC 2 · up · K=5', C: f, L: 32, step: this.dec[0], Lin: 32 },
      { kind: 'map', part: 'dec', label: 'DEC 3 · up · K=5', C: f, L: 64, step: this.dec[1], Lin: 64 },
    ];
    this.outStep = this.dec[2];
  }

  forward(x, keep, training) {
    const st = this.stages;
    let a = x, L = WIN;
    this.enc.forEach((s, i) => {
      const inp = a;
      a = s.pool.forward(s.act.forward(s.conv.forward(a, L), L), L);
      L = L / 2;
      if (keep) { st[i].input = inp; st[i].snapshot = a.slice(); }
    });
    const z = this.bn.forward(a, training);
    this.code = this.bn.m;
    if (keep) { st[3].input = a.slice(); st[3].snapshot = this.bn.m.slice(); st[3].z = z.slice(); }
    a = this.da.forward(this.dd.forward(z), 1);
    if (keep) { st[4].input = z.slice(); st[4].snapshot = a.slice(); }
    L = 16;
    this.dec.forEach((s, i) => {
      const u = s.up.forward(a, L);
      L *= 2;
      const c = s.conv.forward(u, L);
      a = s.act ? s.act.forward(c, L) : c;
      if (keep && i < 2) { st[5 + i].input = u; st[5 + i].snapshot = a.slice(); }
      if (keep && i === 2) { this.outInput = u; }
    });
    if (keep) { this.input = x; this.out = a.slice(); }
    return a;
  }

  backward(g) {
    let d = g;
    for (let i = this.dec.length - 1; i >= 0; i--) {
      const s = this.dec[i];
      if (s.act) d = s.act.backward(d);
      d = s.up.backward(s.conv.backward(d));
    }
    d = this.dd.backward(this.da.backward(d));
    d = this.bn.backward(d);
    for (let i = this.enc.length - 1; i >= 0; i--) {
      const s = this.enc[i];
      d = s.conv.backward(s.act.backward(s.pool.backward(d)));
    }
    return this.bn.kl();
  }

  decode(z) {
    let a = this.da.forward(this.dd.forward(z), 1), L = 16;
    this.dec.forEach((s) => {
      const u = s.up.forward(a, L);
      L *= 2;
      const c = s.conv.forward(u, L);
      a = s.act ? s.act.forward(c, L) : c;
    });
    return a;
  }
}

/* --------------------------------------------------------------- LSTM AE */
/**
 * Sequence to sequence: an encoder LSTM reads the 128 samples one by one and its
 * last state becomes the code; a decoder LSTM gets the code at every step and a
 * linear readout turns its state into the sample (Srivastava et al. 2015).
 */
class LSTMAE extends AEBase {
  /** @param {{units:number, k:number, cell:string, vae:boolean, beta:number}} cfg */
  constructor(cfg) {
    super();
    this.kind = 'lstm';
    this.setup(cfg);
    const H = cfg.units, k = cfg.k, cell = cfg.cell || 'lstm';
    this.H = H; this.k = k;
    this.enc = new RNNDir(cell, 1, H);
    this.bn = new Bottleneck(H, k, cfg.vae, cfg.beta);
    this.dec = new RNNDir(cell, k, H);
    this.head = new Conv1D(H, 1, 1, 1, false);      // the same weights at every step
    this.params = [...this.enc.params, ...this.bn.params, ...this.dec.params, this.head];
    const name = cell.toUpperCase();
    this.stages = [
      { kind: 'seq', part: 'enc', label: 'ENCODER · ' + name + ' · ' + H + ' units', short: 'ENC', C: H, L: WIN, rnn: this.enc },
      { kind: 'code', part: 'code', label: 'CODE · k = ' + k + (cfg.vae ? ' · VAE' : ''), C: k, dense: this.bn.mu },
      { kind: 'seq', part: 'dec', label: 'DECODER · ' + name + ' · ' + H + ' units', short: 'DEC', C: H, L: WIN, rnn: this.dec },
    ];
    this.outStep = { conv: this.head };
  }

  forward(x, keep, training) {
    const L = WIN, H = this.H, k = this.k;
    const hs = this.enc.forward(x, L, false);
    const last = new Float32Array(H);
    for (let u = 0; u < H; u++) last[u] = hs[u * L + L - 1];
    const z = this.bn.forward(last, training);
    this.code = this.bn.m;
    const zin = new Float32Array(k * L);             // the code, repeated at every step
    for (let i = 0; i < k; i++) zin.fill(z[i], i * L, (i + 1) * L);
    const hd = this.dec.forward(zin, L, false);
    const y = this.head.forward(hd, L);
    if (keep) {
      const s = this.stages;
      s[0].input = x; s[0].snapshot = hs;
      s[1].input = last; s[1].snapshot = this.bn.m.slice(); s[1].z = z.slice();
      s[2].input = zin; s[2].snapshot = hd;
      this.input = x; this.outInput = hd; this.out = y.slice();
    }
    return y;
  }

  backward(g) {
    const L = WIN, H = this.H, k = this.k;
    const dzin = this.dec.backward(this.head.backward(g));
    const dz = new Float32Array(k);
    for (let i = 0; i < k; i++) { let s = 0; for (let t = 0; t < L; t++) s += dzin[i * L + t]; dz[i] = s; }
    const dlast = this.bn.backward(dz);
    const dhs = new Float32Array(H * L);
    for (let u = 0; u < H; u++) dhs[u * L + L - 1] = dlast[u];
    this.enc.backward(dhs);
    return this.bn.kl();
  }

  decode(z) {
    const L = WIN, zin = new Float32Array(this.k * L);
    for (let i = 0; i < this.k; i++) zin.fill(z[i], i * L, (i + 1) * L);
    return this.head.forward(this.dec.forward(zin, L, false), L);
  }
}

/* ---------------------------------------------------- Transformer MAE */
/**
 * Masked autoencoder (He et al. 2021): the window is cut into 16 patches of 8
 * samples, some patches are replaced by a learned [MASK] token, and the
 * Transformer has to fill them in from the ones it can see. There is no
 * bottleneck — what makes it an autoencoder is that the hidden patches have to
 * be predicted from the context. To score a whole window it is run four times,
 * each time hiding every fourth patch, so every patch gets predicted once.
 */
const MAE_GROUPS = 4;

class MAEAE extends AEBase {
  /** @param {{d:number, heads:number, layers:number, maskRatio:number}} cfg */
  constructor(cfg) {
    super();
    this.kind = 'mae';
    this.setup(cfg);
    const d = cfg.d, T = WIN / TF_PATCH;
    this.d = d; this.T = T;
    this.embed = new TokenLinear(TF_PATCH, d);
    this.pos = makeParam(T, d, 0.1, 0);                // a learned vector per position
    this.mtok = makeParam(1, d, 0.1, 0);               // the [MASK] token
    this.layers = Array.from({ length: cfg.layers }, () => new EncoderLayer(d, cfg.heads, false));
    this.lnF = new TokenLayerNorm(d);
    this.head = new TokenLinear(d, TF_PATCH);
    this.params = [this.embed, this.pos, this.mtok, ...this.layers.flatMap((l) => l.params), this.lnF, this.head];
    this.stages = [{ kind: 'seq', part: 'enc', label: 'EMBED · ' + d + ' dims · ' + T + ' tokens', short: 'EMBED', C: d, L: T, tfembed: true }];
    this.layers.forEach((layer, i) => this.stages.push({
      kind: 'seq', part: 'enc', label: 'ENCODER ' + (i + 1) + ' · attention', short: 'ENC ' + (i + 1), C: d, L: T, tflayer: true, layer,
    }));
    this.viewGroup = 0;
  }

  /** Token-major T×d → channel-major d×T, the layout the diagram draws. */
  toMaps(X) { return TransformerNet.prototype.toMaps.call(this, X); }

  /** A random mask: round(ratio·16) of the patches hidden, at least one, at most 15. */
  randomMask() {
    const T = this.T, n = Math.max(1, Math.min(T - 1, Math.round(this.cfg.maskRatio * T)));
    const idx = Array.from({ length: T }, (_, i) => i);
    for (let i = T - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const m = new Uint8Array(T);
    for (let i = 0; i < n; i++) m[idx[i]] = 1;
    return m;
  }
  groupMask(g) { const m = new Uint8Array(this.T); for (let p = g; p < this.T; p += MAE_GROUPS) m[p] = 1; return m; }

  /** One pass with a given mask; returns the 128 predicted samples (only the masked patches matter). */
  pass(x, mask, keep) {
    const { T, d } = this;
    const E = this.embed.forward(x, T);              // 128 samples already are 16 patches of 8 in a row
    const X0 = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < d; i++) X0[t * d + i] = (mask[t] ? this.mtok.W[i] : E[t * d + i]) + this.pos.W[t * d + i];
    }
    this.mask = mask;
    let X = X0;
    const outs = [];
    this.layers.forEach((layer) => { X = layer.forward(X, T, keep); if (keep) outs.push(X); });
    const N = this.lnF.forward(X, T);
    const Y = this.head.forward(N, T);
    let n = 0;
    const pooled = new Float32Array(d);               // the mean of the visible tokens: a summary for the latent plot
    for (let t = 0; t < T; t++) if (!mask[t]) { n++; for (let i = 0; i < d; i++) pooled[i] += N[t * d + i]; }
    for (let i = 0; i < d; i++) pooled[i] /= Math.max(1, n);
    if (keep) {
      this.E = E; this.X0 = X0; this.N = N; this.Y = Y;
      this.stages[0].snapshot = this.toMaps(X0); this.stages[0].masked = mask; this.stages[0].input = x;
      outs.forEach((o, i) => { this.stages[i + 1].snapshot = this.toMaps(o); this.stages[i + 1].masked = mask; });
    }
    return { Y, pooled };
  }

  forward(x, keep, training) {
    if (training) { const r = this.pass(x, this.randomMask(), false); this.code = r.pooled; return r.Y; }
    // four passes, every fourth patch hidden in each; the pass on view runs last so its trace stays
    const y = new Float32Array(WIN), code = new Float32Array(this.d);
    const order = [];
    for (let g = 0; g < MAE_GROUPS; g++) if (g !== this.viewGroup) order.push(g);
    order.push(this.viewGroup);
    order.forEach((g, n) => {
      const r = this.pass(x, this.groupMask(g), keep && n === order.length - 1);
      for (let p = g; p < this.T; p += MAE_GROUPS) for (let j = 0; j < TF_PATCH; j++) y[p * TF_PATCH + j] = r.Y[p * TF_PATCH + j];
      for (let i = 0; i < this.d; i++) code[i] += r.pooled[i] / MAE_GROUPS;
    });
    this.code = code;
    if (keep) { this.input = x; this.out = y.slice(); }
    return y;
  }

  /** Only the hidden patches count: predicting what it can see would be trivial. */
  trainBatch(inputs, targets, lr, l2) {
    this.zeroGrads();
    let loss = 0;
    const n = inputs.length;
    for (let b = 0; b < n; b++) {
      const y = this.forward(inputs[b], false, true), t = targets[b], mask = this.mask;
      let nm = 0;
      for (let p = 0; p < this.T; p++) nm += mask[p];
      const cnt = nm * TF_PATCH, g = new Float32Array(WIN);
      let l = 0;
      for (let p = 0; p < this.T; p++) {
        if (!mask[p]) continue;
        for (let j = 0; j < TF_PATCH; j++) {
          const i = p * TF_PATCH + j, e = y[i] - t[i];
          l += e * e; g[i] = 2 * e / cnt;
        }
      }
      loss += l / cnt;
      this.backward(g);
    }
    this.step(lr, 1 / n, l2);
    return loss / n;
  }

  backward(g) {
    const { T, d } = this;
    let dX = this.lnF.backward(this.head.backward(g));
    for (let l = this.layers.length - 1; l >= 0; l--) dX = this.layers[l].backward(dX);
    const dE = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < d; i++) {
        const v = dX[t * d + i];
        this.pos.gW[t * d + i] += v;
        if (this.mask[t]) this.mtok.gW[i] += v; else dE[t * d + i] = v;
      }
    }
    this.embed.backward(dE);
    return 0;
  }
}
