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
}
