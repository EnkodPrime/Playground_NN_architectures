/* transformer.js — a small Transformer encoder over the window, from scratch.
 *
 * The 128 samples are cut into 16 patches of 8 (2.5 ms each; one 50 Hz cycle
 * is 8 patches) and every patch becomes a token of d numbers, plus a learned
 * position vector. Two tokenizers:
 *
 *   conv    d filters of length 7 slide over the samples, ReLU, and each patch
 *           keeps the largest response of every filter — the convolutional
 *           tokenizer of the Compact Convolutional Transformer (Hassani et al.
 *           2021). The default: on 480 windows it is what lets attention learn.
 *   linear  the same linear map turns the 8 raw samples of a patch into d numbers
 *           (ViT). Left in for comparison: it has to learn from far fewer cues.
 *
 * Each encoder layer then lets every token look at every other one:
 *
 *     x ← x + Attention(LN(x))        A = softmax(Q·Kᵀ / √d_h),  out = A·V·W_o
 *     x ← x + FFN(LN(x))              FFN = W₂ · ReLU(W₁ · ·)
 *
 * the pre-LN arrangement of ViT and most current models, which trains more
 * steadily than the post-LN of the original paper. The tokens are averaged and
 * a linear layer gives the classes. With "causal" on, token i may only attend
 * to tokens up to i — the decoder (GPT) form, which is what a Transformer uses
 * on a stream.
 */

const TF_PATCH = 8;                          // samples per token
const TF_KERNEL = 7;                         // filter length of the convolutional tokenizer

/** The same linear map applied to every token. Tokens are stored token-major: X[t·d + i]. */
class TokenLinear {
  constructor(din, dout, scale) {
    Object.assign(this, makeParam(dout, din, scale != null ? scale : 1 / Math.sqrt(din), dout));
    this.type = 'tlin';
    this.din = din; this.dout = dout;
  }
  forward(X, T) {
    const { din, dout, W, b } = this;
    this.X = X; this.T = T;
    const Y = new Float32Array(T * dout);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < dout; o++) {
        let s = b[o];
        for (let i = 0; i < din; i++) s += W[o * din + i] * X[t * din + i];
        Y[t * dout + o] = s;
      }
    }
    return Y;
  }
  backward(dY) {
    const { din, dout, W, gW, gb, X, T } = this;
    const dX = new Float32Array(T * din);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < dout; o++) {
        const g = dY[t * dout + o];
        if (g === 0) continue;
        gb[o] += g;
        for (let i = 0; i < din; i++) {
          gW[o * din + i] += g * X[t * din + i];
          dX[t * din + i] += g * W[o * din + i];
        }
      }
    }
    return dX;
  }
}

/** Layer norm over the d numbers of every token, with a learned scale γ and shift β. */
class TokenLayerNorm {
  constructor(d) {
    Object.assign(this, makeParam(d, 1, 0, d));    // W = γ, b = β
    this.W.fill(1);
    this.type = 'tln';
    this.d = d;
  }
  forward(X, T) {
    const d = this.d;
    this.T = T;
    const Y = new Float32Array(T * d), xh = new Float32Array(T * d);
    const mu = new Float64Array(T), inv = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      let m = 0;
      for (let i = 0; i < d; i++) m += X[t * d + i];
      m /= d;
      let v = 0;
      for (let i = 0; i < d; i++) { const q = X[t * d + i] - m; v += q * q; }
      mu[t] = m;
      inv[t] = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) {
        const h = (X[t * d + i] - m) * inv[t];
        xh[t * d + i] = h;
        Y[t * d + i] = this.W[i] * h + this.b[i];
      }
    }
    this.xh = xh; this.mu = mu; this.inv = inv;
    return Y;
  }
  backward(dY) {
    const { d, T, xh, inv } = this;
    const dX = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let s1 = 0, s2 = 0;
      for (let i = 0; i < d; i++) {
        const g = dY[t * d + i];
        this.gW[i] += g * xh[t * d + i];
        this.gb[i] += g;
        const dh = g * this.W[i];
        s1 += dh; s2 += dh * xh[t * d + i];
      }
      for (let i = 0; i < d; i++) {
        const dh = dY[t * d + i] * this.W[i];
        dX[t * d + i] = inv[t] * (dh - s1 / d - xh[t * d + i] * s2 / d);
      }
    }
    return dX;
  }
}

/** Multi-head self-attention over the tokens. */
class SelfAttention {
  constructor(d, heads, causal) {
    this.d = d; this.H = heads; this.dh = d / heads; this.causal = !!causal;
    this.q = new TokenLinear(d, d);
    this.k = new TokenLinear(d, d);
    this.v = new TokenLinear(d, d);
    this.o = new TokenLinear(d, d);
    this.params = [this.q, this.k, this.v, this.o];
  }

  forward(X, T) {
    const { d, H, dh } = this;
    const sc = 1 / Math.sqrt(dh);
    const Q = this.q.forward(X, T), K = this.k.forward(X, T), V = this.v.forward(X, T);
    const A = new Float32Array(H * T * T), S = new Float32Array(H * T * T);
    const O = new Float32Array(T * d);
    for (let h = 0; h < H; h++) {
      const ho = h * dh;
      for (let i = 0; i < T; i++) {
        const last = this.causal ? i : T - 1;          // a causal token sees only itself and the past
        let mx = -Infinity;
        for (let j = 0; j <= last; j++) {
          let s = 0;
          for (let k = 0; k < dh; k++) s += Q[i * d + ho + k] * K[j * d + ho + k];
          s *= sc;
          S[(h * T + i) * T + j] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let j = 0; j <= last; j++) {
          const e = Math.exp(S[(h * T + i) * T + j] - mx);
          A[(h * T + i) * T + j] = e;
          z += e;
        }
        for (let j = 0; j <= last; j++) A[(h * T + i) * T + j] /= z;
        for (let k = 0; k < dh; k++) {
          let s = 0;
          for (let j = 0; j <= last; j++) s += A[(h * T + i) * T + j] * V[j * d + ho + k];
          O[i * d + ho + k] = s;
        }
      }
    }
    this.Q = Q; this.K = K; this.V = V; this.A = A; this.S = S; this.O = O; this.T = T;
    return this.o.forward(O, T);
  }

  backward(dY) {
    const { d, H, dh, T, Q, K, V, A } = this;
    const sc = 1 / Math.sqrt(dh);
    const dO = this.o.backward(dY);
    const dQ = new Float32Array(T * d), dK = new Float32Array(T * d), dV = new Float32Array(T * d);
    const dA = new Float64Array(T);
    for (let h = 0; h < H; h++) {
      const ho = h * dh;
      for (let i = 0; i < T; i++) {
        const last = this.causal ? i : T - 1;
        let rowdot = 0;
        for (let j = 0; j <= last; j++) {
          const a = A[(h * T + i) * T + j];
          let s = 0;
          for (let k = 0; k < dh; k++) {
            s += dO[i * d + ho + k] * V[j * d + ho + k];
            dV[j * d + ho + k] += a * dO[i * d + ho + k];
          }
          dA[j] = s;
          rowdot += a * s;
        }
        for (let j = 0; j <= last; j++) {
          const dS = A[(h * T + i) * T + j] * (dA[j] - rowdot) * sc;   // through the softmax
          if (dS === 0) continue;
          for (let k = 0; k < dh; k++) {
            dQ[i * d + ho + k] += dS * K[j * d + ho + k];
            dK[j * d + ho + k] += dS * Q[i * d + ho + k];
          }
        }
      }
    }
    const dX = this.q.backward(dQ);
    const dXk = this.k.backward(dK), dXv = this.v.backward(dV);
    for (let i = 0; i < dX.length; i++) dX[i] += dXk[i] + dXv[i];
    return dX;
  }
}

/** One pre-LN encoder layer: x + Attention(LN(x)), then + FFN(LN(x)). */
class EncoderLayer {
  constructor(d, heads, causal) {
    this.type = 'encoder';
    this.d = d;
    this.ln1 = new TokenLayerNorm(d);
    this.attn = new SelfAttention(d, heads, causal);
    this.ln2 = new TokenLayerNorm(d);
    this.f1 = new TokenLinear(d, 2 * d, Math.sqrt(2 / d));
    this.relu = new Activation('relu');
    this.f2 = new TokenLinear(2 * d, d);
    this.params = [this.ln1, ...this.attn.params, this.ln2, this.f1, this.f2];
  }

  forward(X, T, keep) {
    const n1 = this.ln1.forward(X, T);
    const a = this.attn.forward(n1, T);
    const h = new Float32Array(X.length);
    for (let i = 0; i < h.length; i++) h[i] = X[i] + a[i];
    const n2 = this.ln2.forward(h, T);
    const z1 = this.f1.forward(n2, T);
    const r = this.relu.forward(z1, T);
    const z2 = this.f2.forward(r, T);
    const out = new Float32Array(h.length);
    for (let i = 0; i < out.length; i++) out[i] = h[i] + z2[i];
    if (keep) {
      const at = this.attn;
      this.trace = { X, n1, a, h, n2, z1, r, z2, out, Q: at.Q, K: at.K, V: at.V, A: at.A, S: at.S, O: at.O,
        ln1: { mu: this.ln1.mu, inv: this.ln1.inv }, ln2: { mu: this.ln2.mu, inv: this.ln2.inv } };
    }
    return out;
  }

  backward(dOut) {
    // out = h + FFN(LN2(h)),  h = x + Attn(LN1(x))
    const dn2 = this.f1.backward(this.relu.backward(this.f2.backward(dOut)));
    const dh = this.ln2.backward(dn2);
    for (let i = 0; i < dh.length; i++) dh[i] += dOut[i];
    const dX = this.ln1.backward(this.attn.backward(dh));
    for (let i = 0; i < dX.length; i++) dX[i] += dh[i];
    return dX;
  }
}

/* --------------------------------------------------------------- model */
class TransformerNet {
  /**
   * @param {{layers:object[], d:number, heads:number, causal:boolean, nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'transformer';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg, d = cfg.d;
    const T = Math.floor(cfg.inputLen / TF_PATCH);
    this.T = T; this.d = d;
    this.tokenizer = cfg.tokenizer === 'linear' ? 'linear' : 'conv';
    if (this.tokenizer === 'conv') {
      this.tconv = new Conv1D(1, d, TF_KERNEL, 1, false);
      this.tact = new Activation('relu');
      this.tpool = new MaxPool1D(TF_PATCH);
      this.embed = null;
    } else {
      this.embed = new TokenLinear(TF_PATCH, d);
    }
    this.pos = makeParam(T, d, 0.1, 0);              // a learned vector per position
    this.layers = cfg.layers.map(() => new EncoderLayer(d, cfg.heads, cfg.causal));
    this.lnF = new TokenLayerNorm(d);
    this.dense = new Dense(d, cfg.nClasses);
    this.stages = [{ index: 0, tfembed: true, C: d, L: T, pooled: false, snapshot: null }];
    this.layers.forEach((layer, i) => this.stages.push({
      index: i + 1, tflayer: true, layer, C: d, L: T, pooled: false, snapshot: null, causal: !!cfg.causal,
    }));
    this.params = [this.tconv || this.embed, this.pos, ...this.layers.flatMap((l) => l.params), this.lnF, this.dense];
    this.finalC = d; this.finalL = T; this.headKind = 'mean';
    this.clip = 1.0;
  }

  /** Token-major T×d → channel-major d×T, the layout the diagram draws. */
  toMaps(X) {
    const { T, d } = this, out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) out[i * T + t] = X[t * d + i];
    return out;
  }

  forward(x, keepActs) {
    const { T, d } = this, L = this.cfg.inputLen;
    let E;
    if (this.tokenizer === 'conv') {
      const z = this.tconv.forward(x, L);
      const a = this.tact.forward(z, L);
      const p = this.tpool.forward(a, L);              // d × T, the largest response in every patch
      E = new Float32Array(T * d);
      for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) E[t * d + i] = p[i * T + t];
      if (keepActs) { this.convZ = z; this.convA = a; }
    } else {
      E = this.embed.forward(x, T);                     // 128 samples already are 16 patches of 8 in a row
    }
    const X0 = new Float32Array(T * d);
    for (let i = 0; i < X0.length; i++) X0[i] = E[i] + this.pos.W[i];
    if (keepActs) { this.input = x; this.E = E; this.X0 = X0; this.stages[0].snapshot = this.toMaps(X0); }
    let X = X0;
    this.layers.forEach((layer, i) => {
      X = layer.forward(X, T, keepActs);
      if (keepActs) this.stages[i + 1].snapshot = this.toMaps(X);
    });
    const N = this.lnF.forward(X, T);
    const pooled = new Float32Array(d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) pooled[i] += N[t * d + i] / T;
    this.embedding = pooled;
    this.logits = this.dense.forward(pooled);
    return this.softmax(this.logits);
  }

  backward(probs, target) {
    const { T, d } = this;
    const g = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) g[i] = probs[i];
    g[target] -= 1;
    const dp = this.dense.backward(g);
    const dN = new Float32Array(T * d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) dN[t * d + i] = dp[i] / T;
    let dX = this.lnF.backward(dN);
    for (let l = this.layers.length - 1; l >= 0; l--) dX = this.layers[l].backward(dX);
    for (let i = 0; i < dX.length; i++) this.pos.gW[i] += dX[i];
    if (this.tokenizer === 'conv') {
      const dp2 = new Float32Array(d * T);
      for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) dp2[i * T + t] = dX[t * d + i];
      this.tconv.backward(this.tact.backward(this.tpool.backward(dp2)));
    } else {
      this.embed.backward(dX);
    }
    return -Math.log(Math.max(1e-9, probs[target]));
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { RNNNet.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}
