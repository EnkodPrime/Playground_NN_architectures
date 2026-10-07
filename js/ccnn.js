/* ccnn.js — a continuous convolutional network (CKConv / FlexConv), from scratch.
 *
 * An ordinary convolution stores its kernel tap by tap: K numbers, tied to the
 * sample rate. Here the kernel is a continuous function of time, produced by a
 * small network (Romero et al., CKConv and FlexConv; Knigge et al., CCNN):
 *
 *     w(Δ) = KernelNet(u) · exp(−½ (u/σ)²),   u = Δt / span ∈ [−1, 1]
 *     KernelNet(u) = W₃ · sin(W₂ · sin(ω₀ (W₁ u + b₁)) + b₂) + b₃
 *
 * KernelNet gives one value per (output, input) channel pair at every u; the
 * Gaussian mask, with a learned width σ, decides how far the kernel reaches.
 * Because w is a function of time and not of the tap index, the same trained
 * layer can be sampled at another rate — the kernel is just read on a finer
 * grid and its sum scaled to match.
 */

const CC_HIDDEN = 16;        // width of the kernel network
// The first sine layer must be able to oscillate as fast as the signals it should catch: its
// frequency grows with the span, so the shortest period stays a few taps whatever the span.
const CC = { omegaPerTap: 1.5, gain: 0.3, zRms: 0.7 };

/** Uniform random in [−a, a]. */
function ccUniform(a) { return (Math.random() * 2 - 1) * a; }

class CKConvLayer {
  /**
   * @param cin, cout   channels
   * @param R           half span in samples at the training rate (kernel covers −R … R)
   * @param flex        learn the Gaussian mask (FlexConv); off: the kernel fills the whole span
   */
  constructor(cin, cout, R, flex, act) {
    this.type = 'ckconv';
    this.cin = cin; this.cout = cout; this.R = R; this.flex = !!flex;
    const H = CC_HIDDEN, P = cout * cin;
    this.k1 = makeParam(H, 1, 0, H);
    this.k2 = makeParam(H, H, 0, H);
    this.k3 = makeParam(P, H, 0, P);
    // SIREN initialisation: the first layer spans a few periods over u ∈ [−1, 1]
    for (let i = 0; i < H; i++) { this.k1.W[i] = ccUniform(1); this.k1.b[i] = ccUniform(1); }
    const a2 = Math.sqrt(6 / H);
    for (let i = 0; i < this.k2.W.length; i++) this.k2.W[i] = ccUniform(a2);
    // output scale: a kernel of about 2σR effective taps over cin channels gets He-sized weights
    const keff = Math.max(4, 2 * 0.35 * R);
    const a3 = CC.gain * Math.sqrt(2 / (cin * keff)) * Math.sqrt(3) / Math.sqrt(H * 0.5);
    for (let i = 0; i < this.k3.W.length; i++) this.k3.W[i] = ccUniform(a3);
    this.bias = makeParam(cout, 0, 0, cout);
    this.sig = makeParam(1, 1, 0, 0);
    this.sig.W[0] = Math.log(0.35);                     // σ: mask width as a share of the span
    this.act = new Activation(act || 'relu');
    this.params = [this.k1, this.k2, this.k3, this.bias];
    if (this.flex) this.params.push(this.sig);
    this.rate = 1;
    this.omega = CC.omegaPerTap * R;
  }

  get sigma() { return Math.exp(this.sig.W[0]); }

  /**
   * Samples the kernel on the grid of the current rate: taps −R·rate … R·rate,
   * u = Δ / (R·rate). Keeps what the backward pass needs.
   */
  makeKernel() {
    const H = CC_HIDDEN, P = this.cout * this.cin;
    const Rr = Math.round(this.R * this.rate), K = 2 * Rr + 1;
    const s = this.sigma;
    const W = new Float32Array(P * K), net = new Float32Array(P * K), mask = new Float32Array(K);
    const z1 = new Float32Array(K * H), h1 = new Float32Array(K * H), z2 = new Float32Array(K * H), h2 = new Float32Array(K * H);
    const scale = 1 / this.rate;                         // a finer grid has more taps: keep the sum the same
    for (let q = 0; q < K; q++) {
      const u = (q - Rr) / Rr;
      for (let i = 0; i < H; i++) {
        const z = this.omega * (this.k1.W[i] * u + this.k1.b[i]);
        z1[q * H + i] = z; h1[q * H + i] = Math.sin(z);
      }
      for (let i = 0; i < H; i++) {
        let z = this.k2.b[i];
        for (let j = 0; j < H; j++) z += this.k2.W[i * H + j] * h1[q * H + j];
        z2[q * H + i] = z; h2[q * H + i] = Math.sin(z);
      }
      mask[q] = this.flex ? Math.exp(-0.5 * (u / s) * (u / s)) : 1;
      for (let o = 0; o < P; o++) {
        let v = this.k3.b[o];
        for (let j = 0; j < H; j++) v += this.k3.W[o * H + j] * h2[q * H + j];
        net[o * K + q] = v;
        W[o * K + q] = v * mask[q] * scale;
      }
    }
    // taps where the mask is below 0.1 % add nothing: skip them in the convolution
    let qlo = 0, qhi = K - 1;
    if (this.flex) { while (qlo < Rr && mask[qlo] < 1e-3) qlo++; while (qhi > Rr && mask[qhi] < 1e-3) qhi--; }
    Object.assign(this, { K, Rr, kernel: W, net, mask, z1, h1, z2, h2, kscale: scale, qlo, qhi });
    if (!this.gK || this.gK.length !== W.length) this.gK = new Float32Array(W.length);
    return W;
  }

  /** Offset of tap q from the output position. */
  tapOffset(q) { return q - this.Rr; }

  forward(x, L, keep) {
    const { cin, cout } = this;
    const W = this.makeKernel(), K = this.K, Rr = this.Rr;
    this.x = x; this.L = L;
    const z = new Float32Array(cout * L);
    for (let co = 0; co < cout; co++) {
      for (let t = 0; t < L; t++) z[co * L + t] = this.bias.b[co];
      for (let ci = 0; ci < cin; ci++) {
        const wo = (co * cin + ci) * K, xo = ci * L;
        for (let q = this.qlo; q <= this.qhi; q++) {
          const w = W[wo + q], sh = q - Rr;
          const t0 = Math.max(0, -sh), t1 = Math.min(L, L - sh);
          for (let t = t0; t < t1; t++) z[co * L + t] += w * x[xo + t + sh];
        }
      }
    }
    if (keep) this.z = z;
    return this.act.forward(z, L);
  }

  backward(dout) {
    const { cin, cout, K, Rr, kernel: W, x, L } = this;
    const dz = this.act.backward(dout);
    const dx = new Float32Array(cin * L);
    for (let co = 0; co < cout; co++) {
      let sb = 0;
      for (let t = 0; t < L; t++) sb += dz[co * L + t];
      this.bias.gb[co] += sb;
      for (let ci = 0; ci < cin; ci++) {
        const wo = (co * cin + ci) * K, xo = ci * L;
        for (let q = this.qlo; q <= this.qhi; q++) {
          const w = W[wo + q], sh = q - Rr;
          const t0 = Math.max(0, -sh), t1 = Math.min(L, L - sh);
          let acc = 0;
          for (let t = t0; t < t1; t++) {
            const d = dz[co * L + t];
            acc += d * x[xo + t + sh];
            dx[xo + t + sh] += d * w;
          }
          this.gK[wo + q] += acc;            // the kernel net sees this once per batch, in flush()
        }
      }
    }
    return dx;
  }

  /** Hands the kernel gradient gathered over the batch to the kernel network and the mask. */
  flush() {
    if (!this.gK) return;
    const H = CC_HIDDEN, P = this.cout * this.cin, K = this.K, Rr = this.Rr, s = this.sigma;
    const { net, mask, h1, z1, h2, z2, kscale: sc } = this;
    const dh2 = new Float64Array(H), dz2 = new Float64Array(H), dh1 = new Float64Array(H);
    let dlogs = 0;
    for (let q = 0; q < K; q++) {
      const u = (q - Rr) / Rr;
      let dm = 0;
      dh2.fill(0);
      for (let o = 0; o < P; o++) {
        const g = this.gK[o * K + q] * sc;
        if (g === 0) continue;
        dm += g * net[o * K + q];
        const dn = g * mask[q];
        this.k3.gb[o] += dn;
        for (let j = 0; j < H; j++) {
          this.k3.gW[o * H + j] += dn * h2[q * H + j];
          dh2[j] += dn * this.k3.W[o * H + j];
        }
      }
      if (this.flex) dlogs += dm * mask[q] * (u / s) * (u / s);   // d mask / d log σ
      for (let i = 0; i < H; i++) dz2[i] = dh2[i] * Math.cos(z2[q * H + i]);
      dh1.fill(0);
      for (let i = 0; i < H; i++) {
        const g = dz2[i];
        this.k2.gb[i] += g;
        for (let j = 0; j < H; j++) {
          this.k2.gW[i * H + j] += g * h1[q * H + j];
          dh1[j] += g * this.k2.W[i * H + j];
        }
      }
      for (let i = 0; i < H; i++) {
        const g = dh1[i] * Math.cos(z1[q * H + i]) * this.omega;
        this.k1.gW[i] += g * u;
        this.k1.gb[i] += g;
      }
    }
    if (this.flex) this.sig.gW[0] += dlogs;
    this.gK.fill(0);
  }

  /** Taps where the mask still counts (above 10 %): the kernel size FlexConv has learned. */
  effectiveTaps() {
    if (!this.flex) return 2 * this.R + 1;
    const half = Math.min(1, this.sigma * Math.sqrt(2 * Math.log(10)));
    return 2 * Math.round(half * this.R) + 1;
  }
}

/* --------------------------------------------------------------- model */
class CCNNNet {
  /**
   * @param {{layers:{channels:number}[], span:number, flex:boolean, activation:string,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'ccnn';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
    this.rate = 1;
  }

  build() {
    const cfg = this.cfg, L = cfg.inputLen;
    this.layers = []; this.stages = []; this.params = [];
    let cin = 1;
    cfg.layers.forEach((ls, i) => {
      const C = ls.units || ls.channels;
      const layer = new CKConvLayer(cin, C, cfg.span, cfg.flex, cfg.activation);
      this.layers.push(layer);
      this.params.push(...layer.params);
      this.stages.push({ index: i, cc: layer, C, L, pooled: false, snapshot: null, ccnn: true });
      cin = C;
    });
    this.pool = new GlobalAvgPool();
    this.dense = new Dense(cin, cfg.nClasses);
    this.params.push(this.dense);
    this.finalC = cin; this.finalL = L; this.headKind = 'gap';
  }

  /**
   * Starting scale from the data (LSUV, Mishkin & Matas 2016): layer by layer, the kernel
   * network's output is rescaled until the layer's pre-activation has the same size on a
   * few training windows, whatever the span. Without it a long span starts too quiet or
   * too loud, and the sine layers of the kernel network then barely learn.
   */
  calibrate(xs) {
    const n = Math.min(32, xs.length);
    this.layers.forEach((layer, li) => {
      for (let pass = 0; pass < 2; pass++) {
        let s2 = 0, cnt = 0;
        for (let i = 0; i < n; i++) {
          let a = xs[i];
          for (let k = 0; k <= li; k++) a = this.layers[k].forward(a, a.length / (k === 0 ? 1 : this.layers[k - 1].cout), true);
          for (const v of layer.z) { s2 += v * v; cnt++; }
        }
        const f = CC.zRms / Math.max(1e-6, Math.sqrt(s2 / cnt));
        for (let i = 0; i < layer.k3.W.length; i++) layer.k3.W[i] *= f;
        for (let i = 0; i < layer.k3.b.length; i++) layer.k3.b[i] *= f;
      }
    });
  }

  /** Runs on windows sampled `rate` times faster than in training (the kernels follow). */
  setRate(rate) { this.rate = rate; this.layers.forEach((l) => { l.rate = rate; }); }

  forward(x, keepActs) {
    const L = x.length;
    let a = x;
    this.layers.forEach((layer, i) => {
      a = layer.forward(a, L, keepActs);
      if (keepActs && this.rate === 1) this.stages[i].snapshot = a.slice();
    });
    a = this.pool.forward(a, L);
    this.embedding = a;
    this.logits = this.dense.forward(a);
    return this.softmax(this.logits);
  }

  backward(probs, target) {
    const g = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) g[i] = probs[i];
    g[target] -= 1;
    let d = this.pool.backward(this.dense.backward(g));
    for (let l = this.layers.length - 1; l >= 0; l--) d = this.layers[l].backward(d);
    return -Math.log(Math.max(1e-9, probs[target]));
  }

  zeroGrads() {
    ConvNet1D.prototype.zeroGrads.call(this);
    this.layers.forEach((l) => { if (l.gK) l.gK.fill(0); });
  }

  step(lr, scale, l2) {
    // the kernel gradient of the whole batch goes through the kernel networks once
    this.layers.forEach((l) => l.flush());
    RNNNet.prototype.step.call(this, lr, scale, l2);
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}

/** A window resampled to `rate` times the sample rate by linear interpolation (same 40 ms). */
function ccResample(x, rate) {
  const n = x.length, m = n * rate, out = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    const p = i / rate, j = Math.min(n - 2, Math.floor(p)), f = p - j;
    out[i] = x[j] * (1 - f) + x[j + 1] * f;
  }
  return out;
}
