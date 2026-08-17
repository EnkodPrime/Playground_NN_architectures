/* gnn.js — a graph neural network over the visibility graph of the window.
 *
 * The other architectures are handed a sequence. A graph network needs a graph, and
 * a single voltage window does not come with one, so the signal builds its own: in a
 * horizontal visibility graph two samples i < j are connected when every sample
 * between them is lower than both. The construction is standard for time series
 * (Luque et al. 2009) and it turns shape into topology — a lone impulse becomes a hub
 * that sees the whole window, while a clean sine produces an almost regular graph.
 *
 * Nodes keep their time order, so the node axis is the time axis and every panel that
 * already draws a channel over time keeps working unchanged.
 */

/**
 * Horizontal visibility graph in CSR form.
 * @returns {{off:Int32Array, nb:Int32Array, deg:Int32Array}}
 */
function visibilityGraph(x, L) {
  const lists = [];
  for (let i = 0; i < L; i++) lists.push([]);
  for (let i = 0; i < L; i++) {
    let m = -Infinity;
    for (let j = i + 1; j < L; j++) {
      if (x[j] > m) { lists[i].push(j); lists[j].push(i); m = x[j]; }
      if (x[j] >= x[i]) break;
    }
  }
  const off = new Int32Array(L + 1), deg = new Int32Array(L);
  let total = 0;
  for (let i = 0; i < L; i++) { deg[i] = lists[i].length; off[i] = total; total += deg[i]; }
  off[L] = total;
  const nb = new Int32Array(total);
  for (let i = 0; i < L; i++) {
    for (let k = 0; k < deg[i]; k++) nb[off[i] + k] = lists[i][k];
  }
  return { off, nb, deg };
}

/** Node features taken from the raw window: value, step difference, magnitude. */
function graphNodeFeatures(x, L) {
  const F = 3;
  const f = new Float32Array(F * L);
  for (let t = 0; t < L; t++) {
    f[0 * L + t] = x[t];
    f[1 * L + t] = t > 0 ? x[t] - x[t - 1] : 0;
    f[2 * L + t] = Math.abs(x[t]);
  }
  return f;
}

/* -------------------------------------------------------------- GNN layer */
class GNNLayer {
  /**
   * h'_i = act( W_self·h_i + W_neigh·aggregate_{j∈N(i)}(h_j) + b )
   * @param {number} D input channels per node
   * @param {number} H output channels per node
   * @param {string} agg 'mean' | 'max' | 'sum'
   */
  constructor(D, H, agg) {
    this.type = 'gnn';
    this.D = D; this.H = H; this.agg = agg;
    this.pself = makeParam(H, D, 1 / Math.sqrt(D), H);
    this.pneigh = makeParam(H, D, 1 / Math.sqrt(D), 0);
    this.params = [this.pself, this.pneigh];
    this.outC = H;
  }

  forward(x, L, graph) {
    const { D, H, agg } = this;
    this.x = x; this.L = L; this.graph = graph;
    const { off, nb } = graph;

    // aggregate the neighbourhood of every node
    const m = new Float32Array(D * L);
    const argmax = agg === 'max' ? new Int32Array(D * L).fill(-1) : null;
    for (let i = 0; i < L; i++) {
      const s = off[i], e = off[i + 1], n = e - s;
      for (let d = 0; d < D; d++) {
        if (n === 0) { m[d * L + i] = 0; continue; }
        if (agg === 'max') {
          let best = -Infinity, bi = -1;
          for (let k = s; k < e; k++) {
            const v = x[d * L + nb[k]];
            if (v > best) { best = v; bi = nb[k]; }
          }
          m[d * L + i] = best; argmax[d * L + i] = bi;
        } else {
          let acc = 0;
          for (let k = s; k < e; k++) acc += x[d * L + nb[k]];
          m[d * L + i] = agg === 'mean' ? acc / n : acc;
        }
      }
    }
    this.m = m; this.argmax = argmax;

    const pre = new Float32Array(H * L);
    for (let h = 0; h < H; h++) {
      const b = this.pself.b[h];
      for (let i = 0; i < L; i++) {
        let acc = b;
        for (let d = 0; d < D; d++) {
          acc += this.pself.W[h * D + d] * x[d * L + i] + this.pneigh.W[h * D + d] * m[d * L + i];
        }
        pre[h * L + i] = acc;
      }
    }
    this.pre = pre;
    const out = new Float32Array(H * L);
    for (let i = 0; i < out.length; i++) out[i] = pre[i] > 0 ? pre[i] : 0;   // ReLU
    return out;
  }

  backward(dout) {
    const { D, H, L, agg, x, m, graph } = this;
    const { off, nb } = graph;
    const dx = new Float32Array(D * L);
    const dm = new Float32Array(D * L);

    for (let h = 0; h < H; h++) {
      for (let i = 0; i < L; i++) {
        const g = this.pre[h * L + i] > 0 ? dout[h * L + i] : 0;
        if (g === 0) continue;
        this.pself.gb[h] += g;
        for (let d = 0; d < D; d++) {
          this.pself.gW[h * D + d] += g * x[d * L + i];
          this.pneigh.gW[h * D + d] += g * m[d * L + i];
          dx[d * L + i] += g * this.pself.W[h * D + d];
          dm[d * L + i] += g * this.pneigh.W[h * D + d];
        }
      }
    }
    // scatter the aggregation gradient back onto the neighbours
    for (let i = 0; i < L; i++) {
      const s = off[i], e = off[i + 1], n = e - s;
      if (n === 0) continue;
      for (let d = 0; d < D; d++) {
        const g = dm[d * L + i];
        if (g === 0) continue;
        if (agg === 'max') {
          dx[d * L + this.argmax[d * L + i]] += g;
        } else {
          const share = agg === 'mean' ? g / n : g;
          for (let k = s; k < e; k++) dx[d * L + nb[k]] += share;
        }
      }
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class GNNNet {
  /**
   * @param {{layers:{units:number}[], agg:string, readout:string,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'gnn';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.stages = [];
    this.params = [];
    let D = 3;                       // value, difference, magnitude
    const L = cfg.inputLen;

    cfg.layers.forEach((ls, i) => {
      const layer = new GNNLayer(D, ls.units, cfg.agg);
      this.seq.push(layer);
      this.params = this.params.concat(layer.params);
      this.stages.push({
        index: i, layer, C: layer.outC, L, pooled: false, snapshot: null,
        units: ls.units, bidir: false, kind: 'gnn', gnn: true,
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
    const L = this.cfg.inputLen;
    this.graph = visibilityGraph(x, L);
    let a = graphNodeFeatures(x, L);
    this.feat = a;
    let si = 0;
    for (const layer of this.seq) {
      if (layer.type === 'gnn') {
        a = layer.forward(a, L, this.graph);
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
  backward(probs, target) { return RNNNet.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { RNNNet.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }

  /** Message passing detail for one node, for the arithmetic panel. */
  stepDetail(li, ch, node) {
    const st = this.stages[li], layer = st.layer;
    const { off, nb } = this.graph;
    const src = li === 0 ? this.feat : this.stages[li - 1].snapshot;
    const D = layer.D, L = layer.L;
    const neigh = [];
    for (let k = off[node]; k < off[node + 1]; k++) neigh.push(nb[k]);
    const self = new Float32Array(D), agg = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      self[d] = src[d * L + node];
      agg[d] = layer.m[d * L + node];
    }
    const ws = new Float32Array(D), wn = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      ws[d] = layer.pself.W[ch * D + d];
      wn[d] = layer.pneigh.W[ch * D + d];
    }
    return {
      node, neigh, self, agg, ws, wn, D,
      bias: layer.pself.b[ch], agg_kind: layer.agg,
      pre: layer.pre[ch * L + node],
      out: st.snapshot ? st.snapshot[ch * L + node] : 0,
      degree: neigh.length,
    };
  }
}

/* ------------------------------------------------------------- drawing */
/** Arc diagram: nodes along the time axis, visibility edges drawn as arcs above. */
function drawGraphArcs(ctx, x, y, w, h, graph, probe, L, highlight) {
  ctx.clearRect(x, y, w, h);
  const px = (i) => x + (i * (w - 1)) / (L - 1);
  const base = y + h - 12;

  ctx.strokeStyle = 'rgba(43,108,176,0.30)';
  ctx.lineWidth = 0.7;
  const { off, nb } = graph;
  for (let i = 0; i < L; i++) {
    for (let k = off[i]; k < off[i + 1]; k++) {
      const j = nb[k];
      if (j <= i) continue;
      const hot = highlight != null && (i === highlight || j === highlight);
      const r = (px(j) - px(i)) / 2;
      ctx.beginPath();
      ctx.strokeStyle = hot ? 'rgba(224,52,43,0.85)' : 'rgba(43,108,176,0.28)';
      ctx.lineWidth = hot ? 1.4 : 0.7;
      ctx.arc((px(i) + px(j)) / 2, base, Math.min(r, h - 16), Math.PI, 0);
      ctx.stroke();
    }
  }
  // the signal itself along the bottom, so shape and topology line up
  let mx = 1e-6;
  for (let i = 0; i < L; i++) mx = Math.max(mx, Math.abs(probe[i]));
  ctx.beginPath();
  for (let i = 0; i < L; i++) {
    const Y = base + 10 - (probe[i] / mx) * 8;
    if (i === 0) ctx.moveTo(px(i), Y); else ctx.lineTo(px(i), Y);
  }
  ctx.strokeStyle = '#31404e'; ctx.lineWidth = 1;
  ctx.stroke();
  if (highlight != null) {
    ctx.fillStyle = '#e0342b';
    ctx.beginPath(); ctx.arc(px(highlight), base, 2.5, 0, 6.284); ctx.fill();
  }
}
