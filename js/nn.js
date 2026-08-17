/* nn.js — a 1D convolutional network written from scratch: forward, backprop, Adam.
 * No libraries. Tensors are flat Float32Arrays indexed as [channel * length + t].
 */

function heInit(arr, fanIn) {
  const std = Math.sqrt(2 / Math.max(1, fanIn));
  for (let i = 0; i < arr.length; i++) arr[i] = randn() * std;
}

/* ------------------------------------------------------------------ Conv1D */
class Conv1D {
  constructor(cin, cout, k, dil, causal) {
    this.type = 'conv';
    this.cin = cin; this.cout = cout; this.k = k;
    this.dil = dil || 1;                     // dilation: taps spaced this far apart
    this.causal = !!causal;                  // pad only on the left, no look-ahead
    this.pad = ((k - 1) * this.dil) >> 1;    // 'same' zero padding
    this.W = new Float32Array(cout * cin * k);
    this.b = new Float32Array(cout);
    this.rows = cout; this.cols = cin * k;     // grouping for per-channel quantisation
    heInit(this.W, cin * k);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(cout);
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(cout);
    this.vb = new Float32Array(cout);
  }
  /** Offset of tap j relative to the output position. */
  tapOffset(j) {
    return this.causal ? (j - (this.k - 1)) * this.dil : j * this.dil - this.pad;
  }
  forward(x, L) {
    this.x = x; this.L = L;
    const { cin, cout, k, W, b } = this;
    const out = new Float32Array(cout * L);
    for (let co = 0; co < cout; co++) {
      const ob = co * L;
      const bias = b[co];
      for (let t = 0; t < L; t++) out[ob + t] = bias;
      for (let ci = 0; ci < cin; ci++) {
        const wb = (co * cin + ci) * k;
        const xb = ci * L;
        for (let j = 0; j < k; j++) {
          const w = W[wb + j];
          if (w === 0) continue;
          const shift = this.tapOffset(j);
          const tStart = Math.max(0, -shift);
          const tEnd = Math.min(L, L - shift);
          for (let t = tStart; t < tEnd; t++) out[ob + t] += w * x[xb + t + shift];
        }
      }
    }
    return out;
  }
  backward(dout) {
    const { cin, cout, k, W, gW, gb, x, L } = this;
    const dx = new Float32Array(cin * L);
    for (let co = 0; co < cout; co++) {
      const ob = co * L;
      let sb = 0;
      for (let t = 0; t < L; t++) sb += dout[ob + t];
      gb[co] += sb;
      for (let ci = 0; ci < cin; ci++) {
        const wb = (co * cin + ci) * k;
        const xb = ci * L;
        for (let j = 0; j < k; j++) {
          const shift = this.tapOffset(j);
          const tStart = Math.max(0, -shift);
          const tEnd = Math.min(L, L - shift);
          let acc = 0;
          const w = W[wb + j];
          for (let t = tStart; t < tEnd; t++) {
            const d = dout[ob + t];
            acc += d * x[xb + t + shift];
            dx[xb + t + shift] += d * w;
          }
          gW[wb + j] += acc;
        }
      }
    }
    return dx;
  }
}

/* -------------------------------------------------------------- Activations */
class Activation {
  constructor(kind) { this.type = 'act'; this.kind = kind; }
  forward(x, L) {
    this.L = L;
    const out = new Float32Array(x.length);
    if (this.kind === 'relu') {
      for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    } else if (this.kind === 'tanh') {
      for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
    } else if (this.kind === 'leaky') {
      for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0.1 * x[i];
    } else { // abs — useful as an envelope detector
      for (let i = 0; i < x.length; i++) out[i] = Math.abs(x[i]);
      this.sign = x;
    }
    this.out = out;
    return out;
  }
  backward(dout) {
    const o = this.out;
    const dx = new Float32Array(o.length);
    if (this.kind === 'relu') {
      for (let i = 0; i < o.length; i++) dx[i] = o[i] > 0 ? dout[i] : 0;
    } else if (this.kind === 'tanh') {
      for (let i = 0; i < o.length; i++) dx[i] = dout[i] * (1 - o[i] * o[i]);
    } else if (this.kind === 'leaky') {
      for (let i = 0; i < o.length; i++) dx[i] = o[i] > 0 ? dout[i] : 0.1 * dout[i];
    } else {
      const s = this.sign;
      for (let i = 0; i < o.length; i++) dx[i] = dout[i] * (s[i] >= 0 ? 1 : -1);
    }
    return dx;
  }
}

/* -------------------------------------------------------------- MaxPool1D */
class MaxPool1D {
  constructor(size) { this.type = 'pool'; this.size = size || 2; }
  forward(x, L) {
    const s = this.size;
    const outL = Math.floor(L / s);
    const C = x.length / L;
    this.L = L; this.C = C; this.outL = outL;
    const out = new Float32Array(C * outL);
    const arg = new Int32Array(C * outL);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < outL; t++) {
        let best = -Infinity, bi = 0;
        for (let j = 0; j < s; j++) {
          const v = x[c * L + t * s + j];
          if (v > best) { best = v; bi = t * s + j; }
        }
        out[c * outL + t] = best;
        arg[c * outL + t] = bi;
      }
    }
    this.arg = arg;
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    for (let c = 0; c < this.C; c++) {
      for (let t = 0; t < this.outL; t++) {
        dx[c * this.L + this.arg[c * this.outL + t]] += dout[c * this.outL + t];
      }
    }
    return dx;
  }
}

/* --------------------------------------------------------- GlobalAvgPool */
class GlobalAvgPool {
  constructor() { this.type = 'gap'; }
  forward(x, L) {
    const C = x.length / L;
    this.C = C; this.L = L;
    const out = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let t = 0; t < L; t++) s += x[c * L + t];
      out[c] = s / L;
    }
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    for (let c = 0; c < this.C; c++) {
      const g = dout[c] / this.L;
      for (let t = 0; t < this.L; t++) dx[c * this.L + t] = g;
    }
    return dx;
  }
}

/* --------------------------------------------------------- GlobalMaxPool */
class GlobalMaxPool {
  constructor() { this.type = 'gmp'; }
  forward(x, L) {
    const C = x.length / L;
    this.C = C; this.L = L;
    const out = new Float32Array(C);
    const arg = new Int32Array(C);
    for (let c = 0; c < C; c++) {
      let best = -Infinity, bi = 0;
      for (let t = 0; t < L; t++) { const v = x[c * L + t]; if (v > best) { best = v; bi = t; } }
      out[c] = best; arg[c] = bi;
    }
    this.arg = arg;
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    for (let c = 0; c < this.C; c++) dx[c * this.L + this.arg[c]] += dout[c];
    return dx;
  }
}

/* ------------------------------------------------------------------ Dense */
class Dense {
  constructor(nin, nout) {
    this.type = 'dense';
    this.nin = nin; this.nout = nout;
    this.W = new Float32Array(nout * nin);
    this.b = new Float32Array(nout);
    this.rows = nout; this.cols = nin;
    heInit(this.W, nin);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(nout);
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(nout);
    this.vb = new Float32Array(nout);
  }
  forward(x) {
    this.x = x;
    const out = new Float32Array(this.nout);
    for (let j = 0; j < this.nout; j++) {
      let s = this.b[j];
      const wb = j * this.nin;
      for (let i = 0; i < this.nin; i++) s += this.W[wb + i] * x[i];
      out[j] = s;
    }
    // optional low-rank adapter: the effective weight is W + B·A
    if (this.lora) {
      const { pA, pB, r } = this.lora;
      const t = new Float32Array(r);
      for (let q = 0; q < r; q++) {
        let s = 0;
        for (let i = 0; i < this.nin; i++) s += pA.W[q * this.nin + i] * x[i];
        t[q] = s;
      }
      this.lora.t = t;
      for (let j = 0; j < this.nout; j++) {
        let s = 0;
        for (let q = 0; q < r; q++) s += pB.W[j * r + q] * t[q];
        out[j] += s;
      }
    }
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.nin);
    for (let j = 0; j < this.nout; j++) {
      const d = dout[j];
      if (d === 0) continue;
      const wb = j * this.nin;
      this.gb[j] += d;
      for (let i = 0; i < this.nin; i++) {
        this.gW[wb + i] += d * this.x[i];
        dx[i] += d * this.W[wb + i];
      }
    }
    if (this.lora) {
      const { pA, pB, r, t } = this.lora;
      const dt = new Float32Array(r);
      for (let j = 0; j < this.nout; j++) {
        const d = dout[j];
        for (let q = 0; q < r; q++) {
          pB.gW[j * r + q] += d * t[q];
          dt[q] += pB.W[j * r + q] * d;
        }
      }
      for (let q = 0; q < r; q++) {
        const g = dt[q];
        if (g === 0) continue;
        for (let i = 0; i < this.nin; i++) {
          pA.gW[q * this.nin + i] += g * this.x[i];
          dx[i] += g * pA.W[q * this.nin + i];
        }
      }
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class ConvNet1D {
  /**
   * @param {{layers:{filters:number,kernel:number,pool:boolean}[],
   *          activation:string, head:string, nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'cnn';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;                 // Adam step counter
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.convs = [];
    this.stages = [];           // metadata used by the visualisation
    let cin = 1, L = cfg.inputLen;

    cfg.layers.forEach((ls, i) => {
      const conv = new Conv1D(cin, ls.filters, ls.kernel, ls.dilation || 1, cfg.causal);
      this.seq.push(conv);
      this.convs.push(conv);
      const act = new Activation(cfg.activation);
      this.seq.push(act);
      let pooled = false;
      if (ls.pool && Math.floor(L / 2) >= 4) {
        this.seq.push(new MaxPool1D(2));
        L = Math.floor(L / 2);
        pooled = true;
      }
      this.stages.push({ index: i, conv, C: ls.filters, L, pooled, snapshot: null });
      cin = ls.filters;
    });

    this.headKind = cfg.head;
    if (cfg.head === 'gap') {
      this.pool = new GlobalAvgPool();
      this.seq.push(this.pool);
      this.dense = new Dense(cin, cfg.nClasses);
    } else if (cfg.head === 'gmp') {
      this.pool = new GlobalMaxPool();
      this.seq.push(this.pool);
      this.dense = new Dense(cin, cfg.nClasses);
    } else { // flatten
      this.pool = null;
      this.dense = new Dense(cin * L, cfg.nClasses);
    }
    this.finalC = cin;
    this.finalL = L;
    this.seq.push(this.dense);
    this.params = [...this.convs, this.dense];
  }

  /** Forward pass. With keepActs=true the activation maps are stored for drawing. */
  forward(x, keepActs) {
    let a = x, L = this.cfg.inputLen, si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'conv') { a = layer.forward(a, L); }
      else if (layer.type === 'act') {
        a = layer.forward(a, L);
        // with no pooling after it, this activation is what the stage outputs
        const st = this.stages[si];
        if (st && !st.pooled) { if (keepActs) st.snapshot = a.slice(); si++; }
      }
      else if (layer.type === 'pool') {
        a = layer.forward(a, L);
        L = layer.outL;
        const st = this.stages[si];
        if (st) { if (keepActs) st.snapshot = a.slice(); si++; }
      }
      else if (layer.type === 'gap' || layer.type === 'gmp') { a = layer.forward(a, L); this.embedding = a; }
      else if (layer.type === 'dense') { a = layer.forward(a); }
    }
    this.logits = a;
    return this.softmax(a);
  }

  softmax(z) {
    let m = -Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    const p = new Float32Array(z.length);
    let s = 0;
    for (let i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
    for (let i = 0; i < z.length; i++) p[i] /= s;
    this.probs = p;
    return p;
  }

  /** Backward pass from cross-entropy. Returns the loss for this example. */
  backward(probs, target) {
    const d = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) d[i] = probs[i];
    d[target] -= 1;
    let g = d;
    for (let i = this.seq.length - 1; i >= 0; i--) g = this.seq[i].backward(g);
    return -Math.log(Math.max(1e-9, probs[target]));
  }

  zeroGrads() {
    for (const p of this.params) { p.gW.fill(0); p.gb.fill(0); }
  }

  /** Adam update. scale = 1/batchSize */
  step(lr, scale, l2) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (const p of this.params) {
      upd(p.W, p.gW, p.mW, p.vW);
      upd(p.b, p.gb, p.mb, p.vb, true);
    }
    function upd(W, g, m, v, isBias) {
      for (let i = 0; i < W.length; i++) {
        let gi = g[i] * scale;
        if (l2 > 0 && !isBias) gi += l2 * W[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        W[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }

  /** One mini-batch update. Returns the mean loss. */
  trainBatch(xs, ys, idx, lr, l2) {
    this.zeroGrads();
    let loss = 0;
    for (const i of idx) {
      const p = this.forward(xs[i], false);
      loss += this.backward(p, ys[i]);
    }
    this.step(lr, 1 / idx.length, l2);
    return loss / idx.length;
  }

  /**
   * Evaluates a dataset. The confusion matrix is nClasses × (nClasses+1); the last
   * column counts examples rejected as "unknown".
   * @param {function():boolean} rejectFn optional, called after each forward pass
   */
  evaluate(ds, nClasses, limit, rejectFn) {
    const n = Math.min(ds.n, limit || ds.n);
    const stride = nClasses + 1;
    let loss = 0, correct = 0, rejected = 0;
    const conf = new Int32Array(nClasses * stride);
    for (let i = 0; i < n; i++) {
      const p = this.forward(ds.xs[i], false);
      const y = ds.ys[i];
      loss += -Math.log(Math.max(1e-9, p[y]));
      let arg = 0;
      for (let c = 1; c < p.length; c++) if (p[c] > p[arg]) arg = c;
      if (arg === y) correct++;
      const rej = rejectFn ? rejectFn() : false;
      if (rej) rejected++;
      conf[y * stride + (rej ? nClasses : arg)]++;
    }
    return { loss: loss / n, acc: correct / n, rejected: rejected / n, conf, n };
  }

  /** Number of trainable parameters. */
  paramCount() {
    let s = 0;
    for (const p of this.params) s += p.W.length + p.b.length;
    return s;
  }
}
