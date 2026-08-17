/* quant.js — post-training quantisation of the weights, as a live experiment.
 *
 * Training always stays in float32. When the panel is on, a copy of the model is
 * quantised and evaluated next to the float one, so the cost of every bit width is
 * visible while the network learns instead of only at the end.
 *
 * Weights only. Activations stay in float — which is exactly what weight-only
 * quantisation means on a microcontroller, and it keeps the comparison honest.
 */

const quant = {
  on: false,
  bits: 8,
  perChannel: true,
  symmetric: true,
  fp: null,            // float master copy while a quantised model is installed
  metrics: null,       // {fpAcc, qAcc, fpLoss, qLoss, sqnr:[]}
  sweep: null,
  frozen: false,       // weights left quantised in place, so the diagram shows them
};

/** Quantises one array in place; returns the signal-to-quantisation-noise ratio in dB. */
function quantiseArray(W, rows, cols, bits, perChannel, symmetric) {
  const groups = perChannel && rows > 1 && rows * cols === W.length ? rows : 1;
  const gsize = W.length / groups;
  let sig = 0, err = 0;
  for (let g = 0; g < groups; g++) {
    const off = g * gsize;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < gsize; i++) { const v = W[off + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (symmetric) {
      const a = Math.max(Math.abs(mn), Math.abs(mx));
      const qmax = Math.pow(2, bits - 1) - 1;
      const scale = a / qmax || 1e-12;
      for (let i = 0; i < gsize; i++) {
        const v = W[off + i];
        const q = Math.max(-qmax - 1, Math.min(qmax, Math.round(v / scale)));
        const d = q * scale;
        sig += v * v; err += (v - d) * (v - d);
        W[off + i] = d;
      }
    } else {
      const qmax = Math.pow(2, bits) - 1;
      const scale = (mx - mn) / qmax || 1e-12;
      const zp = Math.round(-mn / scale);
      for (let i = 0; i < gsize; i++) {
        const v = W[off + i];
        const q = Math.max(0, Math.min(qmax, Math.round(v / scale) + zp));
        const d = (q - zp) * scale;
        sig += v * v; err += (v - d) * (v - d);
        W[off + i] = d;
      }
    }
  }
  return err > 0 ? 10 * Math.log10(sig / err) : Infinity;
}

function quantSaveFp() {
  return model.params.map((p) => ({ W: Float32Array.from(p.W), b: Float32Array.from(p.b) }));
}
function quantLoadFp(snap) {
  model.params.forEach((p, i) => { p.W.set(snap[i].W); p.b.set(snap[i].b); });
}

/** Quantises every weight tensor in place. Biases are left alone, as in practice. */
function quantiseModel(bits) {
  const sqnr = [];
  model.params.forEach((p) => {
    const rows = p.rows || (p.b && p.b.length && p.W.length % p.b.length === 0 ? p.b.length : 1);
    const cols = p.cols || p.W.length / rows;
    sqnr.push(quantiseArray(p.W, rows, cols, bits, quant.perChannel, quant.symmetric));
  });
  return sqnr;
}

/**
 * Evaluates the quantised model next to the float one without disturbing training.
 * Called from the metrics update, so the two curves move together.
 */
function quantEvaluate() {
  if (!model || !test) return null;
  const k = activeClasses().length;
  const snap = quantSaveFp();
  const sqnr = quantiseModel(quant.bits);
  const q = model.evaluate(test, k, 300);
  if (!quant.frozen) quantLoadFp(snap);
  const fp = quant.frozen ? q : model.evaluate(test, k, 300);
  quant.metrics = {
    bits: quant.bits, qAcc: q.acc, qLoss: q.loss,
    fpAcc: fp.acc, fpLoss: fp.loss, sqnr,
    conf: q.conf,
  };
  return quant.metrics;
}

/** Accuracy against bit width, from 2 bits up. */
function quantSweep() {
  const k = activeClasses().length;
  const snap = quantSaveFp();
  const fp = model.evaluate(test, k, 300).acc;
  const pts = [];
  for (const b of [2, 3, 4, 5, 6, 8, 10, 12, 16]) {
    quantLoadFp(snap);
    quantiseModel(b);
    pts.push({ bits: b, acc: model.evaluate(test, k, 300).acc });
  }
  quantLoadFp(snap);
  quant.sweep = { pts, fp, epoch: Math.floor(state.epoch) };
  return quant.sweep;
}

/* ------------------------------------------------------------- drawing */
function drawQuantSweep(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px system-ui,sans-serif';
  if (!quant.sweep) {
    ctx.fillStyle = '#98a2ad';
    ctx.fillText('Press "Bit sweep" to see where accuracy falls off a cliff.', 8, h / 2);
    return;
  }
  const padL = 30, padB = 18, padT = 10;
  const H = h - padB - padT, W = w - padL - 8;
  ctx.strokeStyle = GRID; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + i * H / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
    ctx.fillStyle = '#98a2ad';
    ctx.fillText((100 - i * 25) + '%', 4, y + 3);
  }
  const pts = quant.sweep.pts;
  const px = (i) => padL + (i * W) / (pts.length - 1);
  const py = (v) => padT + H * (1 - v);
  // float reference
  ctx.strokeStyle = '#c9d1d9'; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(padL, py(quant.sweep.fp)); ctx.lineTo(padL + W, py(quant.sweep.fp)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#98a2ad';
  ctx.fillText('float32', padL + W - 34, py(quant.sweep.fp) - 3);

  ctx.beginPath();
  pts.forEach((p, i) => { const X = px(i), Y = py(p.acc); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
  ctx.strokeStyle = '#2b6cb0'; ctx.lineWidth = 1.8; ctx.stroke();
  pts.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(px(i), py(p.acc), 2.6, 0, 6.284);
    ctx.fillStyle = p.bits === quant.bits ? '#e0342b' : '#2b6cb0';
    ctx.fill();
  });
  ctx.fillStyle = '#98a2ad';
  pts.forEach((p, i) => ctx.fillText(p.bits, px(i) - 4, h - 5));
}

/** Weight histogram of one tensor with the quantisation levels drawn over it. */
function drawQuantHist(ctx, w, h, param) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px system-ui,sans-serif';
  if (!param) return;
  const W = param.W;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < W.length; i++) { if (W[i] < mn) mn = W[i]; if (W[i] > mx) mx = W[i]; }
  const B = 48, bins = new Array(B).fill(0);
  for (let i = 0; i < W.length; i++) {
    bins[Math.max(0, Math.min(B - 1, Math.floor((W[i] - mn) / (mx - mn || 1) * B)))]++;
  }
  let bmax = 1;
  for (const b of bins) bmax = Math.max(bmax, b);
  const padB = 14, Hh = h - padB;
  const bw = w / B;
  for (let i = 0; i < B; i++) {
    const hh = (bins[i] / bmax) * (Hh - 4);
    ctx.fillStyle = 'rgba(43,108,176,0.45)';
    ctx.fillRect(i * bw, Hh - hh, bw - 0.6, hh);
  }
  // quantisation grid
  const a = Math.max(Math.abs(mn), Math.abs(mx));
  const qmax = Math.pow(2, quant.bits - 1) - 1;
  const scale = quant.symmetric ? a / qmax : (mx - mn) / (Math.pow(2, quant.bits) - 1);
  const nLevels = Math.round((mx - mn) / scale);
  if (nLevels <= 64) {
    ctx.strokeStyle = 'rgba(224,52,43,0.55)';
    ctx.lineWidth = 0.7;
    for (let q = 0; q <= nLevels; q++) {
      const v = mn + q * scale;
      const x = (v - mn) / (mx - mn || 1) * w;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, Hh); ctx.stroke();
    }
  }
  ctx.fillStyle = '#98a2ad';
  ctx.fillText(mn.toFixed(2), 2, h - 3);
  ctx.fillText(nLevels <= 64 ? nLevels + ' levels' : 'grid too fine to draw', w / 2 - 26, h - 3);
  ctx.fillText(mx.toFixed(2), w - 26, h - 3);
}
