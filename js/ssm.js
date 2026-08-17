/* ssm.js — state space models: S4D (diagonal, time-invariant) and a Mamba-style
 * selective SSM. Both written from scratch with hand-derived gradients.
 *
 * S4D  : x_k = Ā ⊙ x_{k−1} + B̄ u_k ,  y_k = 2·Re(Σ C_n x_k,n) + D u_k
 *        A is complex diagonal, Ā = exp(ΔA), B̄ = (Ā−1)/A · B  (zero-order hold).
 *        Nothing depends on the input, so the whole layer is one fixed convolution
 *        kernel — that kernel is drawn in the inspector.
 * Mamba: A is real diagonal, while Δ, B and C are produced from the input at every
 *        step. The model is no longer time-invariant, so no single kernel exists;
 *        what can be shown instead is Δ_t, the selectivity, over the window.
 *
 * The complex state is carried as two real numbers, so every gradient below is
 * ordinary real-valued calculus.
 */

function softplus(z) { return z > 20 ? z : Math.log1p(Math.exp(z)); }
function sigmoidS(z) { return 1 / (1 + Math.exp(-z)); }

/* --------------------------------------------------------------- helpers */
/** Complex multiply. */
function cmul(ar, ai, br, bi) { return [ar * br - ai * bi, ar * bi + ai * br]; }
/** Complex divide. */
function cdiv(ar, ai, br, bi) {
  const d = br * br + bi * bi || 1e-30;
  return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
}
/** Complex exp. */
function cexp(ar, ai) { const e = Math.exp(ar); return [e * Math.cos(ai), e * Math.sin(ai)]; }
/**
 * Real gradient of L with respect to a parameter w, given dL/df = (gr, gi) for a
 * holomorphic f and its derivative f' = (pr, pi):  dL/dw = g · conj(f').
 */
function cgrad(gr, gi, pr, pi) { return [gr * pr + gi * pi, gi * pr - gr * pi]; }

/* ------------------------------------------------------------- SSM layer */
class SSMLayer {
  /**
   * @param {string} mode 's4' | 'mamba'
   * @param {number} D input channels
   * @param {number} H state-space channels (one independent SSM each)
   * @param {number} N state dimension per channel
   */
  constructor(mode, D, H, N) {
    this.type = 'ssm';
    this.mode = mode; this.D = D; this.H = H; this.N = N;

    // input projection D -> H
    this.pin = makeParam(H, D, 1 / Math.sqrt(D), H);
    this.params = [this.pin];

    // log of the real part magnitude: A_re = -exp(aLog)  (always stable)
    this.pa = makeParam(H, N, 0, 0);
    for (let h = 0; h < H; h++) {
      for (let n = 0; n < N; n++) this.pa.W[h * N + n] = Math.log(0.5);
    }
    this.params.push(this.pa);

    // timescale, one per channel: Δ = exp(dtLog), spread over decades
    this.pdt = makeParam(H, 1, 0, 0);
    for (let h = 0; h < H; h++) {
      this.pdt.W[h] = Math.log(0.001 * Math.pow(0.1 / 0.001, H > 1 ? h / (H - 1) : 0));
    }
    this.params.push(this.pdt);

    // feed-through term
    this.pd = makeParam(H, 1, 0, 0);
    for (let h = 0; h < H; h++) this.pd.W[h] = 1;
    this.params.push(this.pd);

    if (mode === 's4') {
      // S4D-Lin initialisation: imaginary parts on a line, A_n = −½ + iπn
      this.pb = makeParam(H, N, 0, 0);                       // imaginary part of A
      for (let h = 0; h < H; h++) {
        for (let n = 0; n < N; n++) this.pb.W[h * N + n] = Math.PI * n;
      }
      this.pcr = makeParam(H, N, 1 / Math.sqrt(N), 0);
      this.pci = makeParam(H, N, 1 / Math.sqrt(N), 0);
      this.params.push(this.pb, this.pcr, this.pci);
    } else {
      // selective: Δ, B and C are produced from the input at every step
      this.pdtp = makeParam(H, 1, 0.5, 0);                   // Δ_t depends on the channel value
      this.pbp = makeParam(N, D, 1 / Math.sqrt(D), N);       // B_t = W_B x_t + b
      this.pcp = makeParam(N, D, 1 / Math.sqrt(D), N);       // C_t = W_C x_t + b
      this.pg = makeParam(H, D, 1 / Math.sqrt(D), H);        // gate branch, as in Mamba
      for (let h = 0; h < H; h++) this.pg.b[h] = 1;          // start with the gate open
      // short causal depthwise convolution before the recurrence, as in the Mamba
      // block: A is real here, so the state alone is only a running average and
      // cannot resolve a frequency. This gives the block its local structure.
      this.KC = 4;
      this.pconv = makeParam(H, this.KC, 0.5, H);
      this.params.push(this.pdtp, this.pbp, this.pcp, this.pg, this.pconv);
    }
    this.outC = H;
  }

  /** Discretised Ā and B̄ for the time-invariant case (shared by every step). */
  discretise() {
    const { H, N } = this;
    const abar = new Float64Array(H * N * 2), bbar = new Float64Array(H * N * 2);
    for (let h = 0; h < H; h++) {
      const dt = Math.exp(this.pdt.W[h]);
      for (let n = 0; n < N; n++) {
        const i = h * N + n;
        const ar = -Math.exp(this.pa.W[i]), ai = this.pb.W[i];
        const [er, ei] = cexp(dt * ar, dt * ai);
        abar[i * 2] = er; abar[i * 2 + 1] = ei;
        const [br, bi] = cdiv(er - 1, ei, ar, ai);           // (Ā − 1)/A, with B = 1
        bbar[i * 2] = br; bbar[i * 2 + 1] = bi;
      }
    }
    return { abar, bbar };
  }

  /** Impulse response of one channel — the equivalent FIR kernel (S4 only). */
  kernel(h, L) {
    const { N } = this;
    const { abar, bbar } = this.discretise();
    const k = new Float64Array(L);
    const xr = new Float64Array(N), xi = new Float64Array(N);
    for (let t = 0; t < L; t++) {
      let acc = 0;
      for (let n = 0; n < N; n++) {
        const i = h * this.N + n;
        const ar = abar[i * 2], ai = abar[i * 2 + 1];
        const nr = ar * xr[n] - ai * xi[n] + (t === 0 ? bbar[i * 2] : 0);
        const ni = ar * xi[n] + ai * xr[n] + (t === 0 ? bbar[i * 2 + 1] : 0);
        xr[n] = nr; xi[n] = ni;
        acc += 2 * (this.pcr.W[i] * xr[n] - this.pci.W[i] * xi[n]);
      }
      k[t] = acc + (t === 0 ? this.pd.W[h] : 0);
    }
    return k;
  }

  forward(x, L) {
    const { D, H, N, mode } = this;
    this.x = x; this.L = L;

    // input projection, per timestep
    const u = new Float32Array(H * L);
    for (let h = 0; h < H; h++) {
      const bias = this.pin.b[h];
      for (let t = 0; t < L; t++) {
        let s = bias;
        for (let d = 0; d < D; d++) s += this.pin.W[h * D + d] * x[d * L + t];
        u[h * L + t] = s;
      }
    }
    this.uproj = u;

    // Mamba: depthwise causal conv + SiLU feed the recurrence; S4 feeds it directly
    let ua = u;
    if (mode !== 's4') {
      const KC = this.KC;
      const uc = new Float32Array(H * L);
      ua = new Float32Array(H * L);
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          let s = this.pconv.b[h];
          for (let j = 0; j < KC; j++) {
            const tt = t - (KC - 1) + j;
            if (tt >= 0) s += this.pconv.W[h * KC + j] * u[h * L + tt];
          }
          uc[h * L + t] = s;
          ua[h * L + t] = s * sigmoidS(s);
        }
      }
      this.uc = uc;
    }
    this.u = ua;
    const u2 = ua;

    const out = new Float32Array(H * L);
    const xs = new Float64Array(H * N * L * 2);     // every state, for the backward pass
    this.xs = xs;

    if (mode === 's4') {
      const { abar, bbar } = this.discretise();
      this.abar = abar; this.bbar = bbar;
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          let acc = this.pd.W[h] * u2[h * L + t];
          for (let n = 0; n < N; n++) {
            const i = h * N + n;
            const base = (i * L + t) * 2;
            const pr = t > 0 ? xs[(i * L + t - 1) * 2] : 0;
            const pi = t > 0 ? xs[(i * L + t - 1) * 2 + 1] : 0;
            const ar = abar[i * 2], ai = abar[i * 2 + 1];
            const nr = ar * pr - ai * pi + bbar[i * 2] * u2[h * L + t];
            const ni = ar * pi + ai * pr + bbar[i * 2 + 1] * u2[h * L + t];
            xs[base] = nr; xs[base + 1] = ni;
            acc += 2 * (this.pcr.W[i] * nr - this.pci.W[i] * ni);
          }
          out[h * L + t] = acc;
        }
      }
    } else {
      // selective scan: Δ_t, B_t and C_t change with the input
      const dts = new Float64Array(H * L), Bt = new Float64Array(N * L), Ct = new Float64Array(N * L);
      const dtPre = new Float64Array(H * L);
      for (let n = 0; n < N; n++) {
        for (let t = 0; t < L; t++) {
          let sb = this.pbp.b[n], sc = this.pcp.b[n];
          for (let d = 0; d < D; d++) {
            sb += this.pbp.W[n * D + d] * x[d * L + t];
            sc += this.pcp.W[n * D + d] * x[d * L + t];
          }
          Bt[n * L + t] = sb; Ct[n * L + t] = sc;
        }
      }
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          const z = this.pdt.W[h] + this.pdtp.W[h] * u2[h * L + t];
          dtPre[h * L + t] = z;
          dts[h * L + t] = softplus(z);
        }
      }
      this.dts = dts; this.dtPre = dtPre; this.Bt = Bt; this.Ct = Ct;
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          const dt = dts[h * L + t], uu = u2[h * L + t];
          let acc = this.pd.W[h] * uu;
          for (let n = 0; n < N; n++) {
            const i = h * N + n;
            const base = (i * L + t) * 2;
            const prev = t > 0 ? xs[(i * L + t - 1) * 2] : 0;
            const a = -Math.exp(this.pa.W[i]);
            const ab = Math.exp(dt * a);
            const nr = ab * prev + dt * Bt[n * L + t] * uu;
            xs[base] = nr; xs[base + 1] = ab;      // slot 1 caches Ā_t for the backward pass
            acc += Ct[n * L + t] * nr;
          }
          out[h * L + t] = acc;
        }
      }
    }

    /* The recurrence itself is linear, so without this the whole stack would be one
     * big linear map. S4 blocks put an activation here; Mamba multiplies by a gate
     * branch driven by the input. Both are what make the model expressive. */
    this.pre = Float32Array.from(out);
    if (mode === 's4') {
      for (let i = 0; i < out.length; i++) out[i] = out[i] * sigmoidS(out[i]);   // SiLU
    } else {
      const g = new Float32Array(H * L);
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          let s = this.pg.b[h];
          for (let d = 0; d < D; d++) s += this.pg.W[h * D + d] * x[d * L + t];
          g[h * L + t] = s;
        }
      }
      this.g = g;
      for (let i = 0; i < out.length; i++) out[i] = out[i] * (g[i] * sigmoidS(g[i]));
    }
    return out;
  }

  backward(doutRaw) {
    const { D, H, N, L, mode, u, xs, x } = this;
    const du = new Float32Array(H * L);
    const dx = new Float32Array(D * L);

    // undo the activation / gate first
    const dout = new Float32Array(doutRaw.length);
    if (mode === 's4') {
      for (let i = 0; i < dout.length; i++) {
        const a = this.pre[i], s = sigmoidS(a);
        dout[i] = doutRaw[i] * (s + a * s * (1 - s));         // d/da [a·σ(a)]
      }
    } else {
      const g = this.g;
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          const i = h * L + t;
          const gv = g[i], s = sigmoidS(gv), sg = gv * s;
          dout[i] = doutRaw[i] * sg;                          // through the SSM branch
          const dg = doutRaw[i] * this.pre[i] * (s + gv * s * (1 - s));
          this.pg.gb[h] += dg;
          for (let d = 0; d < D; d++) {
            this.pg.gW[h * D + d] += dg * x[d * L + t];
            dx[d * L + t] += dg * this.pg.W[h * D + d];
          }
        }
      }
    }

    if (mode === 's4') {
      const { abar, bbar } = this;
      const dAbar = new Float64Array(H * N * 2), dBbar = new Float64Array(H * N * 2);
      for (let h = 0; h < H; h++) {
        for (let n = 0; n < N; n++) {
          const i = h * N + n;
          let dxr = 0, dxi = 0;
          const ar = abar[i * 2], ai = abar[i * 2 + 1];
          for (let t = L - 1; t >= 0; t--) {
            const g = dout[h * L + t];
            dxr += 2 * this.pcr.W[i] * g;
            dxi += -2 * this.pci.W[i] * g;
            const cr = xs[(i * L + t) * 2], ci = xs[(i * L + t) * 2 + 1];
            this.pcr.gW[i] += 2 * g * cr;
            this.pci.gW[i] += -2 * g * ci;
            const pr = t > 0 ? xs[(i * L + t - 1) * 2] : 0;
            const pi = t > 0 ? xs[(i * L + t - 1) * 2 + 1] : 0;
            dAbar[i * 2] += dxr * pr + dxi * pi;
            dAbar[i * 2 + 1] += -dxr * pi + dxi * pr;
            const uu = u[h * L + t];
            dBbar[i * 2] += dxr * uu;
            dBbar[i * 2 + 1] += dxi * uu;
            du[h * L + t] += dxr * bbar[i * 2] + dxi * bbar[i * 2 + 1];
            const ndxr = dxr * ar + dxi * ai;
            const ndxi = -dxr * ai + dxi * ar;
            dxr = ndxr; dxi = ndxi;
          }
        }
        for (let t = 0; t < L; t++) {
          const g = dout[h * L + t];
          this.pd.gW[h] += g * u[h * L + t];
          du[h * L + t] += g * this.pd.W[h];
        }
      }
      // chain Ā and B̄ back to Δ and A
      for (let h = 0; h < H; h++) {
        const dt = Math.exp(this.pdt.W[h]);
        let gdt = 0;
        for (let n = 0; n < N; n++) {
          const i = h * N + n;
          const ar = -Math.exp(this.pa.W[i]), ai = this.pb.W[i];
          const Ar = abar[i * 2], Ai = abar[i * 2 + 1];
          const gAr = dAbar[i * 2], gAi = dAbar[i * 2 + 1];
          const gBr = dBbar[i * 2], gBi = dBbar[i * 2 + 1];
          // ∂Ā/∂Δ = A·Ā ; ∂Ā/∂A = Δ·Ā
          const [dA_dtr, dA_dti] = cmul(ar, ai, Ar, Ai);
          const dA_dAr = dt * Ar, dA_dAi = dt * Ai;
          // ∂B̄/∂Δ = Ā ; ∂B̄/∂A = (Δ·Ā·A − (Ā−1))/A²
          const [numr, numi] = cmul(dt * Ar, dt * Ai, ar, ai);
          const [a2r, a2i] = cmul(ar, ai, ar, ai);
          const [dB_dAr, dB_dAi] = cdiv(numr - (Ar - 1), numi - Ai, a2r, a2i);
          // real parameter Δ
          gdt += cgrad(gAr, gAi, dA_dtr, dA_dti)[0] + cgrad(gBr, gBi, Ar, Ai)[0];
          // complex parameter A -> (a_log, b)
          const ga = cgrad(gAr, gAi, dA_dAr, dA_dAi);
          const gb = cgrad(gBr, gBi, dB_dAr, dB_dAi);
          const gAre = ga[0] + gb[0], gAim = ga[1] + gb[1];
          this.pa.gW[i] += gAre * ar;                 // ∂A_re/∂a_log = −exp(a_log) = A_re
          this.pb.gW[i] += gAim;
        }
        this.pdt.gW[h] += gdt * dt;                   // ∂Δ/∂dtLog = Δ
      }
    } else {
      const { dts, dtPre, Bt, Ct } = this;
      const dBt = new Float64Array(N * L), dCt = new Float64Array(N * L);
      const ddt = new Float64Array(H * L);
      for (let h = 0; h < H; h++) {
        for (let n = 0; n < N; n++) {
          const i = h * N + n;
          const a = -Math.exp(this.pa.W[i]);
          let dstate = 0, ga = 0;
          for (let t = L - 1; t >= 0; t--) {
            const g = dout[h * L + t];
            dstate += Ct[n * L + t] * g;
            const cur = xs[(i * L + t) * 2];
            dCt[n * L + t] += g * cur;
            const ab = xs[(i * L + t) * 2 + 1];
            const prev = t > 0 ? xs[(i * L + t - 1) * 2] : 0;
            const dt = dts[h * L + t], uu = u[h * L + t];
            // x = exp(Δa)·prev + Δ·B·u
            ddt[h * L + t] += dstate * (a * ab * prev + Bt[n * L + t] * uu);
            ga += dstate * dt * ab * prev;
            dBt[n * L + t] += dstate * dt * uu;
            du[h * L + t] += dstate * dt * Bt[n * L + t];
            dstate = dstate * ab;
          }
          this.pa.gW[i] += ga * a;                    // ∂a/∂a_log = a
        }
        for (let t = 0; t < L; t++) {
          const g = dout[h * L + t];
          this.pd.gW[h] += g * u[h * L + t];
          du[h * L + t] += g * this.pd.W[h];
          // Δ_t = softplus(dtLog + w·u)
          const s = sigmoidS(dtPre[h * L + t]);
          const gz = ddt[h * L + t] * s;
          this.pdt.gW[h] += gz;
          this.pdtp.gW[h] += gz * u[h * L + t];
          du[h * L + t] += gz * this.pdtp.W[h];
        }
      }
      // B_t and C_t projections
      for (let n = 0; n < N; n++) {
        for (let t = 0; t < L; t++) {
          const gb = dBt[n * L + t], gc = dCt[n * L + t];
          this.pbp.gb[n] += gb; this.pcp.gb[n] += gc;
          for (let d = 0; d < D; d++) {
            const xv = x[d * L + t];
            this.pbp.gW[n * D + d] += gb * xv;
            this.pcp.gW[n * D + d] += gc * xv;
            dx[d * L + t] += gb * this.pbp.W[n * D + d] + gc * this.pcp.W[n * D + d];
          }
        }
      }
    }

    // Mamba: back through SiLU and the depthwise causal convolution
    if (mode !== 's4') {
      const KC = this.KC;
      const duProj = new Float32Array(H * L);
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < L; t++) {
          const c = this.uc[h * L + t], s = sigmoidS(c);
          const gc = du[h * L + t] * (s + c * s * (1 - s));
          if (gc === 0) continue;
          this.pconv.gb[h] += gc;
          for (let j = 0; j < KC; j++) {
            const tt = t - (KC - 1) + j;
            if (tt < 0) continue;
            this.pconv.gW[h * KC + j] += gc * this.uproj[h * L + tt];
            duProj[h * L + tt] += gc * this.pconv.W[h * KC + j];
          }
        }
      }
      du.set(duProj);
    }

    // back through the input projection
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < L; t++) {
        const g = du[h * L + t];
        if (g === 0) continue;
        this.pin.gb[h] += g;
        for (let d = 0; d < D; d++) {
          this.pin.gW[h * D + d] += g * x[d * L + t];
          dx[d * L + t] += g * this.pin.W[h * D + d];
        }
      }
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class SSMNet {
  /**
   * @param {{mode:string, layers:{units:number}[], stateDim:number, readout:string,
   *          nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'ssm';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.stages = [];
    this.params = [];
    let D = 1;
    const L = cfg.inputLen;

    cfg.layers.forEach((ls, i) => {
      const layer = new SSMLayer(cfg.mode, D, ls.units, cfg.stateDim);
      this.seq.push(layer);
      this.params = this.params.concat(layer.params);
      this.stages.push({
        index: i, layer, C: layer.outC, L, pooled: false, snapshot: null,
        units: ls.units, bidir: false, kind: cfg.mode, ssm: true,
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
      if (layer.type === 'ssm') {
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
  backward(probs, target) { return RNNNet.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { RNNNet.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }

  /** Per-step detail for the arithmetic panel. */
  stepDetail(li, unit, t) {
    const st = this.stages[li], layer = st.layer, N = layer.N, L = layer.L;
    const out = { mode: layer.mode, N, D: layer.D, u: layer.u[unit * L + t], t,
                  Dskip: layer.pd.W[unit], modes: [] };
    if (layer.mode === 's4') {
      const { abar, bbar } = layer.discretise();
      out.dt = Math.exp(layer.pdt.W[unit]);
      for (let n = 0; n < N; n++) {
        const i = unit * N + n;
        out.modes.push({
          n, aRe: -Math.exp(layer.pa.W[i]), aIm: layer.pb.W[i],
          abarRe: abar[i * 2], abarIm: abar[i * 2 + 1],
          bbarRe: bbar[i * 2], bbarIm: bbar[i * 2 + 1],
          cRe: layer.pcr.W[i], cIm: layer.pci.W[i],
          xRe: layer.xs[(i * L + t) * 2], xIm: layer.xs[(i * L + t) * 2 + 1],
          prevRe: t > 0 ? layer.xs[(i * L + t - 1) * 2] : 0,
          prevIm: t > 0 ? layer.xs[(i * L + t - 1) * 2 + 1] : 0,
        });
      }
    } else {
      out.dt = layer.dts[unit * L + t];
      out.dtBase = Math.exp(layer.pdt.W[unit]);
      out.dtW = layer.pdtp.W[unit];
      for (let n = 0; n < N; n++) {
        const i = unit * N + n;
        out.modes.push({
          n, aRe: -Math.exp(layer.pa.W[i]),
          abarRe: layer.xs[(i * L + t) * 2 + 1],
          B: layer.Bt[n * L + t], C: layer.Ct[n * L + t],
          xRe: layer.xs[(i * L + t) * 2],
          prevRe: t > 0 ? layer.xs[(i * L + t - 1) * 2] : 0,
        });
      }
    }
    return out;
  }
}
