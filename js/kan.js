/* kan.js — a Kolmogorov–Arnold network written from scratch.
 *
 * An MLP puts a fixed nonlinearity on the nodes and a single number on every
 * edge. A KAN does the opposite: the node only adds up what arrives, and every
 * edge carries its own learned function
 *
 *     φ(x) = w_b · silu(x) + Σ_m c_m · B_m(x)
 *
 * where B_m are cubic B-splines on a uniform grid. The silu term keeps a slope
 * outside the grid; the spline gives the edge its shape. Every φ can be drawn,
 * which is what makes the network readable.
 *
 * A node adds up 128 edges, so its value can land far outside the grid of the
 * next layer, where every spline is zero. Each deeper layer therefore reads its
 * inputs standardised, u = (x − μ)/σ, with μ and σ tracked during training —
 * the simplest form of the grid update in the KAN paper: the grid follows the
 * values that actually arrive.
 *
 * The last step to the classes is an ordinary linear layer, so the softmax head,
 * the novelty score, quantisation and the watermark work exactly as for the
 * other playgrounds.
 */

const KAN_LO = -2, KAN_HI = 2;          // spline grid range
const KAN_G = 5;                        // grid intervals
const KAN_NB = KAN_G + 3;               // cubic B-spline bases on that grid
const KAN_STEP = (KAN_HI - KAN_LO) / KAN_G;

function sigmoidK(x) { return 1 / (1 + Math.exp(-x)); }
function siluK(x) { return x * sigmoidK(x); }
function siluGradK(x) { const s = sigmoidK(x); return s * (1 + x * (1 - s)); }

/**
 * The four cubic B-spline bases that can be non-zero at x, on the uniform grid.
 * Bases m0 … m0+3 take the values b[0..3]; db holds their slopes d/dx. Indices
 * outside 0 … KAN_NB−1 belong to no basis and are skipped by the caller.
 */
function kanBasis(x, b, db) {
  const s = (x - KAN_LO) / KAN_STEP;
  const m0 = Math.floor(s);
  const u = s - m0, u2 = u * u, u3 = u2 * u, v = 1 - u;
  b[0] = v * v * v / 6;
  b[1] = (3 * u3 - 6 * u2 + 4) / 6;
  b[2] = (-3 * u3 + 3 * u2 + 3 * u + 1) / 6;
  b[3] = u3 / 6;
  if (db) {
    db[0] = -v * v / 2 / KAN_STEP;
    db[1] = (3 * u2 - 4 * u) / 2 / KAN_STEP;
    db[2] = (-3 * u2 + 2 * u + 1) / 2 / KAN_STEP;
    db[3] = u2 / 2 / KAN_STEP;
  }
  return m0;
}

/* --------------------------------------------------------------- layer */
class KANLayer {
  constructor(nin, nout, norm) {
    this.type = 'kan';
    this.nin = nin; this.nout = nout;
    // running input statistics; the first layer reads raw samples, already inside the grid
    this.norm = !!norm;
    this.mu = new Float64Array(nin);
    this.ms = new Float64Array(nin).fill(1);
    this.sd = new Float64Array(nin).fill(1);
    this.training = false;
    // spline coefficients: one row of KAN_NB per edge (j, i)
    this.pc = makeParam(nout * nin, KAN_NB, 0.3 / Math.sqrt(nin), 0);
    // base weight on silu, one per edge
    this.pw = makeParam(nout, nin, 1 / Math.sqrt(nin), 0);
    this.params = [this.pc, this.pw];
  }

  /** Input value → grid coordinate, and back. */
  toGrid(i, x) { return this.norm ? (x - this.mu[i]) / this.sd[i] : x; }
  fromGrid(i, u) { return this.norm ? this.mu[i] + u * this.sd[i] : u; }

  /** φ_{j,i}(x) and its two parts, for drawing and for the arithmetic panel. */
  edge(j, i, x) {
    const u = this.toGrid(i, x);
    const b = new Float64Array(4);
    const m0 = kanBasis(u, b, null);
    const base = this.pw.W[j * this.nin + i] * siluK(u);
    const row = (j * this.nin + i) * KAN_NB;
    let spline = 0;
    const active = [];
    for (let r = 0; r < 4; r++) {
      const m = m0 + r;
      if (m < 0 || m >= KAN_NB) continue;
      spline += this.pc.W[row + m] * b[r];
      active.push({ m, B: b[r], c: this.pc.W[row + m] });
    }
    return { phi: base + spline, base, spline, active, wb: this.pw.W[j * this.nin + i], u };
  }

  forward(x, keep) {
    const { nin, nout } = this;
    const C = this.pc.W, Wb = this.pw.W;
    this.x = x;
    if (this.norm && this.training) {
      const a = 0.01;                                  // slow: the splines must keep up with it
      for (let i = 0; i < nin; i++) {
        this.mu[i] += a * (x[i] - this.mu[i]);
        this.ms[i] += a * (x[i] * x[i] - this.ms[i]);
        this.sd[i] = Math.sqrt(Math.max(1e-4, this.ms[i] - this.mu[i] * this.mu[i]));
      }
    }
    const inv = new Float64Array(nin);
    for (let i = 0; i < nin; i++) inv[i] = this.norm ? 1 / this.sd[i] : 1;
    this.inv = inv;
    // the basis depends only on the input value, so it is shared by every edge leaving it
    const m0 = new Int32Array(nin), B = new Float64Array(nin * 4), dB = new Float64Array(nin * 4);
    const sl = new Float64Array(nin), dsl = new Float64Array(nin);
    const b = new Float64Array(4), db = new Float64Array(4);
    for (let i = 0; i < nin; i++) {
      const u = this.norm ? (x[i] - this.mu[i]) * inv[i] : x[i];
      m0[i] = kanBasis(u, b, db);
      for (let r = 0; r < 4; r++) { B[i * 4 + r] = b[r]; dB[i * 4 + r] = db[r]; }
      sl[i] = siluK(u);
      dsl[i] = siluGradK(u);
    }
    this.m0 = m0; this.B = B; this.dB = dB; this.sl = sl; this.dsl = dsl;

    const out = new Float32Array(nout);
    const phi = keep ? new Float32Array(nout * nin) : null;
    for (let j = 0; j < nout; j++) {
      let s = 0;
      for (let i = 0; i < nin; i++) {
        const row = (j * nin + i) * KAN_NB;
        let e = Wb[j * nin + i] * sl[i];
        for (let r = 0; r < 4; r++) {
          const m = m0[i] + r;
          if (m >= 0 && m < KAN_NB) e += C[row + m] * B[i * 4 + r];
        }
        if (phi) phi[j * nin + i] = e;
        s += e;
      }
      out[j] = s;
    }
    this.phi = phi;
    return out;
  }

  backward(dout) {
    const { nin, nout, m0, B, dB, sl, dsl, inv } = this;
    const C = this.pc.W, gC = this.pc.gW, Wb = this.pw.W, gWb = this.pw.gW;
    const dx = new Float32Array(nin);
    for (let j = 0; j < nout; j++) {
      const g = dout[j];
      if (g === 0) continue;
      for (let i = 0; i < nin; i++) {
        const row = (j * nin + i) * KAN_NB;
        const k = j * nin + i;
        gWb[k] += g * sl[i];
        let slope = Wb[k] * dsl[i];
        for (let r = 0; r < 4; r++) {
          const m = m0[i] + r;
          if (m < 0 || m >= KAN_NB) continue;
          gC[row + m] += g * B[i * 4 + r];
          slope += C[row + m] * dB[i * 4 + r];
        }
        dx[i] += g * slope * inv[i];             // μ and σ are held fixed in the gradient
      }
    }
    return dx;
  }
}

/* --------------------------------------------------------------- model */
class KANNet {
  /**
   * @param {{layers:{units:number}[], nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'kan';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.stages = [];
    this.params = [];
    let nin = cfg.inputLen;
    cfg.layers.forEach((ls, i) => {
      const layer = new KANLayer(nin, ls.units, i > 0);
      this.seq.push(layer);
      this.params.push(...layer.params);
      this.stages.push({
        index: i, layer, C: ls.units, L: 1, pooled: false, snapshot: null, phi: null, input: null,
        units: ls.units, kan: true,
      });
      nin = ls.units;
    });
    this.dense = new Dense(nin, cfg.nClasses);
    this.seq.push(this.dense);
    this.params.push(this.dense);
    this.finalC = nin;
    this.finalL = 1;
    this.headKind = 'none';
  }

  forward(x, keepActs) {
    let a = x, si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'kan') {
        const st = this.stages[si++];
        if (keepActs) st.input = a;
        a = layer.forward(a, keepActs);
        if (keepActs) { st.snapshot = a.slice(); st.phi = layer.phi; }
      } else {
        this.embedding = a;
        a = layer.forward(a);
      }
    }
    this.logits = a;
    return this.softmax(a);
  }

  /**
   * A sampled curve of edge (j, i) of layer li, for drawing: evenly spaced in grid
   * coordinates from a little before the grid to a little after, with x in the
   * units the edge actually receives.
   */
  edgeCurve(li, j, i, n) {
    const layer = this.stages[li].layer;
    const lo = KAN_LO - 0.8, hi = KAN_HI + 0.8;
    const pts = [];
    for (let k = 0; k < n; k++) {
      const x = layer.fromGrid(i, lo + (hi - lo) * k / (n - 1));
      pts.push([x, layer.edge(j, i, x).phi]);
    }
    pts.gridLo = layer.fromGrid(i, KAN_LO);
    pts.gridHi = layer.fromGrid(i, KAN_HI);
    return pts;
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  backward(probs, target) { return ConvNet1D.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  // the spline coefficients see large, rare gradients; clip like the recurrent nets
  step(lr, scale, l2) { RNNNet.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) {
    // the input statistics of the deeper layers follow the training data only
    this.stages.forEach((st) => { st.layer.training = true; });
    try {
      return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2);
    } finally {
      this.stages.forEach((st) => { st.layer.training = false; });
    }
  }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}
