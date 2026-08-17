/* rnn.js — recurrent networks written from scratch: forward, BPTT, Adam.
 *
 * Three cells share one scaffolding: simple tanh RNN, GRU and LSTM. Sequences are
 * channel-major Float32Arrays indexed as [channel * length + t], the same layout the
 * convolutional side uses, so the diagram, the inspector and the metrics are shared.
 */

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

/** One trainable block: a weight matrix plus an optional bias, with Adam state. */
function makeParam(rows, cols, scale, nbias) {
  const W = new Float32Array(rows * cols);
  for (let i = 0; i < W.length; i++) W[i] = randn() * scale;
  const b = new Float32Array(nbias || 0);
  return {
    W, b,
    gW: new Float32Array(W.length), gb: new Float32Array(b.length),
    mW: new Float32Array(W.length), vW: new Float32Array(W.length),
    mb: new Float32Array(b.length), vb: new Float32Array(b.length),
  };
}

/* ---------------------------------------------------------- one direction */
/**
 * A single recurrent pass over a sequence in one direction.
 * Gate rows are stacked: LSTM = [i, f, o, g], GRU = [z, r, n], RNN = [h].
 */
class RNNDir {
  constructor(kind, D, H) {
    this.kind = kind; this.D = D; this.H = H;
    this.G = kind === 'lstm' ? 4 : kind === 'gru' ? 3 : 1;
    const R = this.G * H;
    this.px = makeParam(R, D, 1 / Math.sqrt(D), R);          // input weights + bias
    this.ph = makeParam(R, H, 1 / Math.sqrt(H), 0);          // recurrent weights
    if (kind === 'lstm') {                                   // forget-gate bias = 1
      for (let u = 0; u < H; u++) this.px.b[H + u] = 1;
    }
    this.params = [this.px, this.ph];
  }

  /**
   * @param {Float32Array} x channel-major input [D*L]
   * @param {number} L sequence length
   * @param {boolean} rev process the sequence backwards
   * @returns {Float32Array} hidden states, channel-major [H*L]
   */
  forward(x, L, rev) {
    const { D, H, G, kind } = this;
    const R = G * H;
    const Wx = this.px.W, b = this.px.b, Wh = this.ph.W;

    const hs = new Float32Array((L + 1) * H);      // hs[(s+1)*H+u] after step s
    const gs = new Float32Array(L * R);            // post-activation gates per step
    const cs = kind === 'lstm' ? new Float32Array((L + 1) * H) : null;
    const tc = kind === 'lstm' ? new Float32Array(L * H) : null;   // tanh(c)
    const qs = kind === 'gru' ? new Float32Array(L * H) : null;    // Whn · h_prev
    const z = new Float32Array(R);

    for (let s = 0; s < L; s++) {
      const t = rev ? L - 1 - s : s;
      const hp = s * H;                            // offset of h_{s-1}

      for (let r = 0; r < R; r++) {
        let acc = b[r];
        for (let d = 0; d < D; d++) acc += Wx[r * D + d] * x[d * L + t];
        z[r] = acc;
      }
      if (kind === 'gru') {
        // the reset gate multiplies only the recurrent part of the candidate
        for (let r = 0; r < 2 * H; r++) {
          let acc = 0;
          for (let v = 0; v < H; v++) acc += Wh[r * H + v] * hs[hp + v];
          z[r] += acc;
        }
        const zg = gs, off = s * R;
        for (let u = 0; u < H; u++) zg[off + u] = sigmoid(z[u]);
        for (let u = 0; u < H; u++) zg[off + H + u] = sigmoid(z[H + u]);
        for (let u = 0; u < H; u++) {
          let q = 0;
          for (let v = 0; v < H; v++) q += Wh[(2 * H + u) * H + v] * hs[hp + v];
          qs[s * H + u] = q;
          const npre = z[2 * H + u] + zg[off + H + u] * q;
          zg[off + 2 * H + u] = Math.tanh(npre);
        }
        for (let u = 0; u < H; u++) {
          const g = gs[off + u], n = gs[off + 2 * H + u];
          hs[(s + 1) * H + u] = (1 - g) * n + g * hs[hp + u];
        }
      } else {
        for (let r = 0; r < R; r++) {
          let acc = 0;
          for (let v = 0; v < H; v++) acc += Wh[r * H + v] * hs[hp + v];
          z[r] += acc;
        }
        const off = s * R;
        if (kind === 'lstm') {
          for (let u = 0; u < H; u++) {
            const i = sigmoid(z[u]), f = sigmoid(z[H + u]);
            const o = sigmoid(z[2 * H + u]), g = Math.tanh(z[3 * H + u]);
            gs[off + u] = i; gs[off + H + u] = f;
            gs[off + 2 * H + u] = o; gs[off + 3 * H + u] = g;
            const c = f * cs[hp + u] + i * g;
            cs[(s + 1) * H + u] = c;
            const th = Math.tanh(c);
            tc[s * H + u] = th;
            hs[(s + 1) * H + u] = o * th;
          }
        } else {
          for (let u = 0; u < H; u++) {
            const h = Math.tanh(z[u]);
            gs[off + u] = h;
            hs[(s + 1) * H + u] = h;
          }
        }
      }
    }

    this.x = x; this.L = L; this.rev = rev;
    this.hs = hs; this.gs = gs; this.cs = cs; this.tc = tc; this.qs = qs;

    const out = new Float32Array(H * L);
    for (let s = 0; s < L; s++) {
      const t = rev ? L - 1 - s : s;
      for (let u = 0; u < H; u++) out[u * L + t] = hs[(s + 1) * H + u];
    }
    return out;
  }

  /** BPTT. dout is channel-major [H*L]; returns dx channel-major [D*L]. */
  backward(dout) {
    const { D, H, G, kind, L, rev, hs, gs, cs, tc, qs, x } = this;
    const R = G * H;
    const Wx = this.px.W, Wh = this.ph.W;
    const gWx = this.px.gW, gb = this.px.gb, gWh = this.ph.gW;
    const dx = new Float32Array(D * L);
    const dh = new Float32Array(H);      // gradient flowing into h_s
    const dc = new Float32Array(H);      // gradient flowing into c_s (LSTM)
    const dz = new Float32Array(R);

    for (let s = L - 1; s >= 0; s--) {
      const t = rev ? L - 1 - s : s;
      const hp = s * H, off = s * R;
      for (let u = 0; u < H; u++) dh[u] += dout[u * L + t];
      dz.fill(0);
      const dhprev = new Float32Array(H);

      if (kind === 'lstm') {
        for (let u = 0; u < H; u++) {
          const i = gs[off + u], f = gs[off + H + u];
          const o = gs[off + 2 * H + u], g = gs[off + 3 * H + u];
          const th = tc[s * H + u];
          const dO = dh[u] * th;
          dc[u] += dh[u] * o * (1 - th * th);
          const dI = dc[u] * g, dG = dc[u] * i, dF = dc[u] * cs[hp + u];
          dz[u] = dI * i * (1 - i);
          dz[H + u] = dF * f * (1 - f);
          dz[2 * H + u] = dO * o * (1 - o);
          dz[3 * H + u] = dG * (1 - g * g);
          dc[u] = dc[u] * f;                       // carried to step s-1
        }
      } else if (kind === 'gru') {
        for (let u = 0; u < H; u++) {
          const zg = gs[off + u], r = gs[off + H + u], n = gs[off + 2 * H + u];
          const hprev = hs[hp + u];
          const dN = dh[u] * (1 - zg);
          const dZ = dh[u] * (hprev - n);
          dhprev[u] += dh[u] * zg;
          const dNpre = dN * (1 - n * n);
          dz[2 * H + u] = dNpre;                   // input part of the candidate
          const dR = dNpre * qs[s * H + u];
          dz[u] = dZ * zg * (1 - zg);
          dz[H + u] = dR * r * (1 - r);
        }
        // recurrent part of the candidate is gated by r
        for (let u = 0; u < H; u++) {
          const r = gs[off + H + u];
          const dq = dz[2 * H + u] * r;
          const row = (2 * H + u) * H;
          for (let v = 0; v < H; v++) {
            gWh[row + v] += dq * hs[hp + v];
            dhprev[v] += dq * Wh[row + v];
          }
        }
      } else {
        for (let u = 0; u < H; u++) {
          const h = gs[off + u];
          dz[u] = dh[u] * (1 - h * h);
        }
      }

      // input weights, bias and dx for every gate row
      for (let r = 0; r < R; r++) {
        const g = dz[r];
        if (g === 0) continue;
        gb[r] += g;
        const rw = r * D;
        for (let d = 0; d < D; d++) {
          gWx[rw + d] += g * x[d * L + t];
          dx[d * L + t] += g * Wx[rw + d];
        }
      }
      // recurrent weights: GRU handles its candidate row separately above
      const rEnd = kind === 'gru' ? 2 * H : R;
      for (let r = 0; r < rEnd; r++) {
        const g = dz[r];
        if (g === 0) continue;
        const row = r * H;
        for (let v = 0; v < H; v++) {
          gWh[row + v] += g * hs[hp + v];
          dhprev[v] += g * Wh[row + v];
        }
      }
      dh.set(dhprev);
    }
    return dx;
  }
}

/* ------------------------------------------------------------- RNN layer */
class RNNLayer {
  constructor(kind, D, H, bidir) {
    this.type = 'rnn';
    this.kind = kind; this.D = D; this.H = H; this.bidir = !!bidir;
    this.fwd = new RNNDir(kind, D, H);
    this.params = this.fwd.params.slice();
    if (this.bidir) {
      this.bwd = new RNNDir(kind, D, H);
      this.params = this.params.concat(this.bwd.params);
    }
    this.outC = this.bidir ? 2 * H : H;
  }
  forward(x, L) {
    this.L = L;
    const a = this.fwd.forward(x, L, false);
    if (!this.bidir) return a;
    const b = this.bwd.forward(x, L, true);
    const out = new Float32Array(this.outC * L);
    out.set(a, 0);
    out.set(b, this.H * L);
    return out;
  }
  backward(dout) {
    const L = this.L;
    if (!this.bidir) return this.fwd.backward(dout);
    const da = dout.subarray(0, this.H * L);
    const db = dout.subarray(this.H * L);
    const dxa = this.fwd.backward(da);
    const dxb = this.bwd.backward(db);
    for (let i = 0; i < dxa.length; i++) dxa[i] += dxb[i];
    return dxa;
  }
}

/* ---------------------------------------------------------------- readout */
/** Reduces a sequence to one vector: the last state, the mean or the max over time. */
class Readout {
  constructor(kind) { this.type = 'readout'; this.kind = kind; }
  forward(x, L) {
    const C = x.length / L;
    this.C = C; this.L = L;
    const out = new Float32Array(C);
    if (this.kind === 'mean') {
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let t = 0; t < L; t++) s += x[c * L + t];
        out[c] = s / L;
      }
    } else if (this.kind === 'max') {
      this.arg = new Int32Array(C);
      for (let c = 0; c < C; c++) {
        let best = -Infinity, bi = 0;
        for (let t = 0; t < L; t++) { const v = x[c * L + t]; if (v > best) { best = v; bi = t; } }
        out[c] = best; this.arg[c] = bi;
      }
    } else {                                    // last state of each direction
      for (let c = 0; c < C; c++) out[c] = x[c * L + (L - 1)];
    }
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    if (this.kind === 'mean') {
      for (let c = 0; c < this.C; c++) {
        const g = dout[c] / this.L;
        for (let t = 0; t < this.L; t++) dx[c * this.L + t] = g;
      }
    } else if (this.kind === 'max') {
      for (let c = 0; c < this.C; c++) dx[c * this.L + this.arg[c]] += dout[c];
    } else {
      for (let c = 0; c < this.C; c++) dx[c * this.L + (this.L - 1)] += dout[c];
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class RNNNet {
  /**
   * @param {{cell:string, layers:{units:number,bidir:boolean}[], readout:string,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'rnn';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  /**
   * Everything needed to write out one recurrence step: the gate values, the
   * weights that produced them, the previous state and the resulting state.
   * Valid after a forward pass with keepActs.
   */
  stepDetail(li, unit, t) {
    const st = this.stages[li], layer = st.layer;
    const back = layer.bidir && unit >= layer.H;
    const dir = back ? layer.bwd : layer.fwd;
    const u = back ? unit - layer.H : unit;
    const H = layer.H, G = dir.G, D = dir.D, L = dir.L;
    const s = back ? L - 1 - t : t;              // step index in this direction
    const hprev = new Float32Array(H);
    for (let v = 0; v < H; v++) hprev[v] = dir.hs[s * H + v];
    const gates = [], wx = [], wh = [], bias = [];
    for (let g = 0; g < G; g++) {
      gates.push(dir.gs[s * G * H + g * H + u]);
      bias.push(dir.px.b[g * H + u]);
      const rx = new Float32Array(D);
      for (let d = 0; d < D; d++) rx[d] = dir.px.W[(g * H + u) * D + d];
      const rh = new Float32Array(H);
      for (let v = 0; v < H; v++) rh[v] = dir.ph.W[(g * H + u) * H + v];
      wx.push(rx); wh.push(rh);
    }
    const xv = new Float32Array(D);
    for (let d = 0; d < D; d++) xv[d] = dir.x[d * L + t];
    return {
      dir, u, back, s, t, H, G, D, L, kind: dir.kind,
      hprev, gates, wx, wh, bias, xv,
      h: dir.hs[(s + 1) * H + u],
      c: dir.cs ? dir.cs[(s + 1) * H + u] : null,
      cprev: dir.cs ? dir.cs[s * H + u] : null,
      q: dir.qs ? dir.qs[s * H + u] : null,
    };
  }

  /** Input weights of one unit, as one row per gate: [gate][inputChannel]. */
  unitInputWeights(li, unit) {
    const st = this.stages[li], layer = st.layer;
    const back = layer.bidir && unit >= layer.H;
    const dir = back ? layer.bwd : layer.fwd;
    const u = back ? unit - layer.H : unit;
    const rows = [];
    for (let g = 0; g < dir.G; g++) {
      const row = new Float32Array(dir.D);
      for (let d = 0; d < dir.D; d++) row[d] = dir.px.W[(g * layer.H + u) * dir.D + d];
      rows.push(row);
    }
    return { rows, dir, u, back, bias: dir.px.b };
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.stages = [];
    this.params = [];
    let D = 1;
    const L = cfg.inputLen;

    cfg.layers.forEach((ls, i) => {
      const layer = new RNNLayer(cfg.cell, D, ls.units, ls.bidir);
      this.seq.push(layer);
      this.params = this.params.concat(layer.params);
      this.stages.push({
        index: i, layer, C: layer.outC, L, pooled: false, snapshot: null,
        units: ls.units, bidir: layer.bidir, kind: cfg.cell,
      });
      D = layer.outC;
    });

    this.readout = new Readout(cfg.readout);
    this.seq.push(this.readout);
    this.dense = new Dense(D, cfg.nClasses);
    this.seq.push(this.dense);
    this.params.push(this.dense);
    this.finalC = D;
    this.finalL = L;
    this.headKind = cfg.readout;
  }

  forward(x, keepActs) {
    let a = x;
    const L = this.cfg.inputLen;
    let si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'rnn') {
        a = layer.forward(a, L);
        const st = this.stages[si++];
        if (keepActs) st.snapshot = a.slice();
      } else if (layer.type === 'readout') {
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

  backward(probs, target) {
    const d = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) d[i] = probs[i];
    d[target] -= 1;
    let g = d;
    for (let i = this.seq.length - 1; i >= 0; i--) g = this.seq[i].backward(g);
    return -Math.log(Math.max(1e-9, probs[target]));
  }

  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }

  /**
   * Adam with global gradient-norm clipping. Backpropagating through 128 steps
   * produces occasional huge gradients; without clipping training oscillates
   * instead of converging.
   */
  step(lr, scale, l2) {
    const clip = this.clip || 1.0;
    let sum = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.gW.length; i++) { const g = p.gW[i] * scale; sum += g * g; }
      for (let i = 0; i < p.gb.length; i++) { const g = p.gb[i] * scale; sum += g * g; }
    }
    const norm = Math.sqrt(sum);
    const k = (norm > clip ? clip / norm : 1) * scale;
    for (const p of this.params) {
      for (let i = 0; i < p.gW.length; i++) p.gW[i] *= k;
      for (let i = 0; i < p.gb.length; i++) p.gb[i] *= k;
    }
    this.lastGradNorm = norm;
    ConvNet1D.prototype.step.call(this, lr, 1, l2);
  }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}
