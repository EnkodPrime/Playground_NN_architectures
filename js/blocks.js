/* blocks.js — ResNet-1D and InceptionTime, the two standard deep baselines for
 * time series classification (Wang et al. 2017; Ismail Fawaz et al. 2020),
 * scaled down to run in the browser.
 *
 * The originals put batch norm after every convolution. Batch norm needs the
 * statistics of the whole mini-batch, and its gradient couples the examples of
 * the batch; this engine runs one example at a time and has neither. A version
 * that normalised with running averages instead was tried and made training
 * worse (38–66% where no normalisation reached 69–91%). These blocks use layer
 * normalisation — GroupNorm with one group, the usual stand-in when there is no
 * batch: every example is normalised over its own channels and positions, and
 * the gradient is exact.
 */

/* ------------------------------------------------------------ GroupNorm1D */
/**
 * Normalises one example over groups of its channels — every group over all its
 * channels and positions — then scales and shifts each channel. One group is
 * layer norm. It needs no batch, so its gradient is exact here, including the
 * part that flows through μ and σ.
 */
class GroupNorm1D {
  constructor(C, G) {
    this.type = 'gn';
    this.C = C;
    this.G = G || 1;
    this.cg = C / this.G;                       // channels per group
    this.p = makeParam(C, 1, 0, C);            // W = γ (scale), b = β (shift)
    this.p.W.fill(1);
  }

  forward(x, L) {
    const { C, G, cg } = this, N = cg * L;
    this.L = L;
    const m = new Float64Array(G), inv = new Float64Array(G);
    const out = new Float32Array(C * L), xh = new Float32Array(C * L);
    for (let g = 0; g < G; g++) {
      const o = g * N;
      let s = 0;
      for (let i = 0; i < N; i++) s += x[o + i];
      s /= N;
      let v = 0;
      for (let i = 0; i < N; i++) { const d = x[o + i] - s; v += d * d; }
      m[g] = s;
      inv[g] = 1 / Math.sqrt(v / N + 1e-5);
    }
    for (let c = 0; c < C; c++) {
      const g = Math.floor(c / cg), gm = this.p.W[c], b = this.p.b[c];
      for (let t = 0; t < L; t++) {
        const h = (x[c * L + t] - m[g]) * inv[g];
        xh[c * L + t] = h;
        out[c * L + t] = gm * h + b;
      }
    }
    this.xh = xh; this.inv = inv; this.m = m;
    return out;
  }

  backward(dout) {
    const { C, G, cg, L, xh, inv } = this, N = cg * L;
    const dh = new Float32Array(C * L);
    const s1 = new Float64Array(G), s2 = new Float64Array(G);
    for (let c = 0; c < C; c++) {
      const g = Math.floor(c / cg), gm = this.p.W[c];
      let sg = 0, sb = 0;
      for (let t = 0; t < L; t++) {
        const i = c * L + t, d = dout[i];
        sg += d * xh[i];
        sb += d;
        dh[i] = d * gm;
        s1[g] += dh[i];
        s2[g] += dh[i] * xh[i];
      }
      this.p.gW[c] += sg;
      this.p.gb[c] += sb;
    }
    const dx = new Float32Array(C * L);
    for (let c = 0; c < C; c++) {
      const g = Math.floor(c / cg), a = s1[g] / N, b = s2[g] / N;
      for (let t = 0; t < L; t++) {
        const i = c * L + t;
        dx[i] = inv[g] * (dh[i] - a - xh[i] * b);
      }
    }
    return dx;
  }

  /** The statistics of the last forward pass, kept with a trace so they can be explained later. */
  stat() { return { m: Float64Array.from(this.m), inv: Float64Array.from(this.inv) }; }

  /** One value through the normalisation, with the statistics of the example it came from. */
  explain(c, v, st) {
    const g = Math.floor(c / this.cg);
    const m = (st || this).m[g], inv = (st || this).inv[g];
    const xhat = (v - m) * inv;
    return { mu: m, sd: 1 / inv, gamma: this.p.W[c], beta: this.p.b[c], xhat, out: this.p.W[c] * xhat + this.p.b[c], group: g, groups: this.G };
  }
}

/** The normalisation of the deep networks — layer norm, or G groups — or none. */
function makeNorm(on, C, G) { return on ? new GroupNorm1D(C, G || 1) : null; }

/* ------------------------------------------------- max pool, stride 1 */
/** Max over a window of 3 around every position — the pooling branch of InceptionTime. */
class MaxPoolSame1D {
  constructor() { this.type = 'poolsame'; }
  forward(x, L) {
    const C = x.length / L;
    this.L = L; this.C = C;
    const out = new Float32Array(C * L), arg = new Int32Array(C * L);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < L; t++) {
        let best = -Infinity, bi = t;
        for (let j = Math.max(0, t - 1); j <= Math.min(L - 1, t + 1); j++) {
          const v = x[c * L + j];
          if (v > best) { best = v; bi = j; }
        }
        out[c * L + t] = best;
        arg[c * L + t] = c * L + bi;
      }
    }
    this.arg = arg;
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    for (let i = 0; i < dout.length; i++) dx[this.arg[i]] += dout[i];
    return dx;
  }
}

/* --------------------------------------------------------- ResNet block */
/** "8-5-3" → [8, 5, 3] */
function parseKernels(s) { return String(s).split('-').map(Number); }

/**
 * One residual block of ResNet-1D: conv → norm → ReLU for every kernel, except
 * that the last one stops after the norm, gets the skip added, and only then goes
 * through ReLU. The skip is the identity, or a 1×1 convolution (+ norm) when the
 * channel count changes. With skip off the same block is a plain deep CNN.
 */
class ResBlock1D {
  constructor(cin, F, kernels, opt) {
    this.type = 'resblock';
    this.cin = cin; this.F = F; this.kernels = kernels;
    this.useBn = !!opt.bn; this.skip = !!opt.skip;
    this.convs = []; this.bns = []; this.acts = [];
    let c = cin;
    kernels.forEach((k) => {
      this.convs.push(new Conv1D(c, F, k, 1, false));
      this.bns.push(this.useBn ? makeNorm(opt.bn, F) : null);
      this.acts.push(new Activation('relu'));
      c = F;
    });
    this.proj = this.skip && cin !== F ? new Conv1D(cin, F, 1, 1, false) : null;
    this.projBn = this.proj && this.useBn ? makeNorm(opt.bn, F) : null;
    this.normLayers = [...this.bns, this.projBn].filter(Boolean);
    this.params = [...this.convs, ...this.normLayers.map((b) => b.p)];
    if (this.proj) this.params.push(this.proj);
  }

  forward(x, L, keep) {
    this.L = L;
    const n = this.convs.length;
    const trace = keep ? [] : null;
    let a = x;
    for (let i = 0; i < n; i++) {
      const z = this.convs[i].forward(a, L);
      const nz = this.bns[i] ? this.bns[i].forward(z, L) : z;
      const stat = trace && this.bns[i] && this.bns[i].stat ? this.bns[i].stat() : null;
      if (i < n - 1) {
        a = this.acts[i].forward(nz, L);
        if (trace) trace.push({ z, n: nz, h: a, stat });
      } else {
        let s = null;
        if (this.skip) {
          s = x;
          if (this.proj) {
            s = this.proj.forward(x, L);
            if (this.projBn) s = this.projBn.forward(s, L);
          }
        }
        const sum = s ? new Float32Array(nz.length) : nz;
        if (s) for (let j = 0; j < nz.length; j++) sum[j] = nz[j] + s[j];
        a = this.acts[i].forward(sum, L);
        if (trace) trace.push({ z, n: nz, s, sum, h: a, stat });
      }
    }
    if (trace) { this.trace = trace; this.input = x; }
    return a;
  }

  backward(dout) {
    const n = this.convs.length;
    const dsum = this.acts[n - 1].backward(dout);
    let g = this.bns[n - 1] ? this.bns[n - 1].backward(dsum) : dsum;
    g = this.convs[n - 1].backward(g);
    for (let i = n - 2; i >= 0; i--) {
      g = this.acts[i].backward(g);
      if (this.bns[i]) g = this.bns[i].backward(g);
      g = this.convs[i].backward(g);
    }
    if (this.skip) {
      // the skip hands its gradient straight to the block input
      let ds = dsum;
      if (this.proj) {
        if (this.projBn) ds = this.projBn.backward(ds);
        ds = this.proj.backward(ds);
      }
      for (let j = 0; j < g.length; j++) g[j] += ds[j];
    }
    return g;
  }
}

/* ------------------------------------------------------ Inception module */
/**
 * InceptionTime module: a 1×1 bottleneck (when there is more than one input
 * channel), then three convolutions of different length side by side, plus a
 * max-pool branch with its own 1×1 convolution. The four results are stacked
 * into one map, normalised and passed through ReLU. Short kernels catch
 * impulses, long ones a whole stretch of the cycle.
 */
class InceptionModule {
  constructor(cin, f, kernels, opt) {
    this.type = 'inception';
    this.cin = cin; this.f = f; this.kernels = kernels;
    this.B = cin > 1 ? 4 : 0;                               // bottleneck width
    this.bott = this.B ? new Conv1D(cin, this.B, 1, 1, false) : null;
    const bin = this.B || cin;
    this.branches = kernels.map((k) => new Conv1D(bin, f, k, 1, false));
    this.mp = new MaxPoolSame1D();
    this.poolConv = new Conv1D(cin, f, 1, 1, false);
    this.nb = kernels.length + 1;
    this.C = f * this.nb;
    // one normalisation over the whole module: normalising each branch on its own was
    // tried and trained worse (10 epochs, four classes: 25–60% against 44–88%)
    this.bn = opt.bn ? makeNorm(opt.bn, this.C) : null;
    this.act = new Activation('relu');
    this.normLayers = this.bn ? [this.bn] : [];
    this.params = [...this.branches, this.poolConv, ...this.normLayers.map((b) => b.p)];
    if (this.bott) this.params.push(this.bott);
  }

  /** Which branch output channel ch belongs to, and its index inside the branch. */
  branchOf(ch) { return { bi: Math.floor(ch / this.f), j: ch % this.f }; }
  branchName(bi) { return bi < this.kernels.length ? 'K=' + this.kernels[bi] : 'pool'; }

  forward(x, L, keep) {
    this.L = L;
    const b = this.bott ? this.bott.forward(x, L) : x;
    const outs = this.branches.map((cv) => cv.forward(b, L));
    const pooled = this.mp.forward(x, L);
    outs.push(this.poolConv.forward(pooled, L));
    const cat = new Float32Array(this.C * L);
    outs.forEach((o, bi) => cat.set(o, bi * this.f * L));
    const n = this.bn ? this.bn.forward(cat, L) : cat;
    const stat = keep && this.bn && this.bn.stat ? this.bn.stat() : null;
    const out = this.act.forward(n, L);
    if (keep) { this.trace = { x, b, outs, pooled, cat, n, out, stat }; }
    return out;
  }

  backward(dout) {
    const { f, L } = this;
    let d = this.act.backward(dout);
    if (this.bn) d = this.bn.backward(d);
    const bin = this.B || this.cin;
    const db = new Float32Array(bin * L);
    this.branches.forEach((cv, bi) => {
      const g = cv.backward(d.subarray(bi * f * L, (bi + 1) * f * L));
      for (let i = 0; i < g.length; i++) db[i] += g[i];
    });
    const gp = this.mp.backward(this.poolConv.backward(d.subarray(this.branches.length * f * L)));
    const dx = this.bott ? this.bott.backward(db) : db;
    for (let i = 0; i < gp.length; i++) dx[i] += gp[i];
    return dx;
  }
}

/**
 * Three modules with a shortcut around them, as in the original (one shortcut
 * every three modules): y = ReLU(modules(x) + norm(1×1(x))).
 */
class InceptionGroup {
  constructor(modules, cin, opt) {
    this.type = 'incgroup';
    this.modules = modules;
    const C = modules[modules.length - 1].C;
    this.proj = new Conv1D(cin, C, 1, 1, false);
    this.projBn = opt.bn ? makeNorm(opt.bn, C) : null;
    this.act = new Activation('relu');
    this.normLayers = this.projBn ? [this.projBn] : [];
    this.params = [this.proj, ...this.normLayers.map((b) => b.p)];
  }
  forward(x, L, keep, onModule) {
    let a = x;
    this.modules.forEach((m, i) => {
      a = m.forward(a, L, keep);
      if (i < this.modules.length - 1 && onModule) onModule(i, a);
    });
    let s = this.proj.forward(x, L);
    if (this.projBn) s = this.projBn.forward(s, L);
    const sum = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) sum[i] = a[i] + s[i];
    const out = this.act.forward(sum, L);
    if (keep) this.trace = { x, s, sum, out };
    if (onModule) onModule(this.modules.length - 1, out);
    return out;
  }
  backward(dout) {
    const dsum = this.act.backward(dout);
    let g = dsum;
    for (let i = this.modules.length - 1; i >= 0; i--) g = this.modules[i].backward(g);
    let ds = this.projBn ? this.projBn.backward(dsum) : dsum;
    ds = this.proj.backward(ds);
    for (let i = 0; i < g.length; i++) g[i] += ds[i];
    return g;
  }
}

/* --------------------------------------------------------------- models */

class ResNet1DNet {
  /**
   * @param {{layers:{filters:number,kernels:string}[], bn:boolean, skip:boolean,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'resnet';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg, L = cfg.inputLen;
    this.seq = []; this.stages = []; this.params = []; this.normLayers = [];
    let cin = 1;
    cfg.layers.forEach((ls, i) => {
      const block = new ResBlock1D(cin, ls.filters, parseKernels(ls.kernels), cfg);
      this.seq.push(block);
      this.params.push(...block.params);
      this.normLayers.push(...block.normLayers);
      this.stages.push({ index: i, block, C: ls.filters, L, pooled: false, snapshot: null, resblock: true });
      cin = ls.filters;
    });
    this.pool = new GlobalAvgPool();
    this.dense = new Dense(cin, cfg.nClasses);
    this.seq.push(this.pool, this.dense);
    this.params.push(this.dense);
    this.finalC = cin; this.finalL = L; this.headKind = 'gap';
  }

  forward(x, keepActs) {
    const L = this.cfg.inputLen;
    let a = x, si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'resblock') {
        a = layer.forward(a, L, keepActs);
        const st = this.stages[si++];
        if (keepActs) st.snapshot = a.slice();
      } else if (layer.type === 'gap') {
        a = layer.forward(a, L);
        this.embedding = a;
      } else {
        a = layer.forward(a);
      }
    }
    this.logits = a;
    return this.softmax(a);
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  backward(probs, target) { return ConvNet1D.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { ConvNet1D.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}

class InceptionTimeNet {
  /**
   * @param {{layers:{filters:number}[], kernels:number[], bn:boolean, skip:boolean,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'inception';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg, L = cfg.inputLen;
    this.seq = []; this.stages = []; this.params = []; this.normLayers = [];
    this.group = null;
    const modules = [];
    let cin = 1;
    cfg.layers.forEach((ls, i) => {
      const m = new InceptionModule(cin, ls.filters, cfg.kernels, cfg);
      modules.push(m);
      this.params.push(...m.params);
      this.normLayers.push(...m.normLayers);
      this.stages.push({ index: i, module: m, C: m.C, L, pooled: false, snapshot: null, inception: true });
      cin = m.C;
    });
    // the original adds a shortcut around every three modules
    if (cfg.skip && modules.length === 3) {
      this.group = new InceptionGroup(modules, 1, cfg);
      this.seq.push(this.group);
      this.params.push(...this.group.params);
      this.normLayers.push(...this.group.normLayers);
      this.stages[2].shortcut = true;
    } else {
      this.seq.push(...modules);
    }
    this.pool = new GlobalAvgPool();
    this.dense = new Dense(cin, cfg.nClasses);
    this.seq.push(this.pool, this.dense);
    this.params.push(this.dense);
    this.finalC = cin; this.finalL = L; this.headKind = 'gap';
  }

  forward(x, keepActs) {
    const L = this.cfg.inputLen;
    let a = x, si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'inception') {
        a = layer.forward(a, L, keepActs);
        const st = this.stages[si++];
        if (keepActs) st.snapshot = a.slice();
      } else if (layer.type === 'incgroup') {
        a = layer.forward(a, L, keepActs, (i, out) => { if (keepActs) this.stages[i].snapshot = out.slice(); });
        si += layer.modules.length;
      } else if (layer.type === 'gap') {
        a = layer.forward(a, L);
        this.embedding = a;
      } else {
        a = layer.forward(a);
      }
    }
    this.logits = a;
    return this.softmax(a);
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  backward(probs, target) { return ConvNet1D.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { ConvNet1D.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}
