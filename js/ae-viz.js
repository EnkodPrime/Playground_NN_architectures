/* ae-viz.js — draws an autoencoder as columns:
 * input → encoder stages → code → decoder stages → output (reconstruction over the input).
 * Uses the drawing helpers of viz.js (dpiSetup, drawWave, drawNodeBox, drawLink, …).
 */

const AE_IN_W = 128, AE_IN_H = 74, AE_OUT_W = 150, AE_OUT_H = 96;

/** Node sizes per stage kind. */
function aeStageNodes(st) {
  if (st.kind === 'units') return [{ w: 94, h: Math.max(90, Math.min(230, st.C * 5 + 12)) }];
  if (st.kind === 'seq') return [{ w: 132, h: Math.max(90, Math.min(230, st.C * 6 + 12)) }];
  if (st.kind === 'code') {
    const h = st.C > 8 ? 18 : 26;
    return Array.from({ length: st.C }, (_, c) => ({ w: 74, h, ch: c }));
  }
  return Array.from({ length: st.C }, (_, c) => ({ w: 94, h: 42, ch: c }));
}

function aeLayout(model, cssW) {
  const cols = [{ kind: 'input', nodes: [{ w: AE_IN_W, h: AE_IN_H }] }];
  model.stages.forEach((st, si) => cols.push({ kind: st.kind, stage: si, nodes: aeStageNodes(st) }));
  cols.push({ kind: 'output', nodes: [{ w: AE_OUT_W, h: AE_OUT_H }] });

  // tighten the gaps, then narrow the boxes, before scrolling sideways
  const n = cols.length, gapV = 8;
  let gap = 58;
  const widths = () => cols.reduce((s, c) => s + c.nodes[0].w, 0);
  if (widths() + gap * (n - 1) + 24 > cssW) {
    gap = Math.max(26, (cssW - 24 - widths()) / (n - 1));
    if (widths() + gap * (n - 1) + 24 > cssW) {
      const fixed = AE_IN_W + AE_OUT_W, mid = n - 2;
      const w = Math.max(54, (cssW - 24 - fixed - gap * (n - 1)) / mid);
      for (let c = 1; c < n - 1; c++) cols[c].nodes.forEach((nd) => { nd.w = Math.min(nd.w, w); });
    }
  }
  let total = widths() + gap * (n - 1);
  let x = Math.max(12, (cssW - total) / 2);
  const topPad = model.kind === 'mae' ? 30 + ATTN_PX + 24 : 30;
  let maxH = 0;
  cols.forEach((col) => {
    col.h = col.nodes.reduce((s, nd) => s + nd.h, 0) + (col.nodes.length - 1) * gapV;
    maxH = Math.max(maxH, col.h);
    col.x = x;
    x += col.nodes[0].w + gap;
  });
  cols.forEach((col) => {
    let y = topPad + (maxH - col.h) / 2;
    col.nodes.forEach((nd) => { nd.x = col.x; nd.y = y; y += nd.h + gapV; });
  });
  return { cols, width: Math.max(cssW, total + 24), height: topPad + maxH + 34 };
}

/** Strength of the link from node ci of one column to node co of the next. */
function aeLinkStrength(model, si, co, ci) {
  const st = model.stages[si];
  const conv = st.step && st.step.conv;
  if (conv && conv.cin > 1) return linkStrength(conv, co, ci);
  if (conv) return linkStrength(conv, co, 0);
  if (st.rnn) {
    // the input weights of every gate that read channel ci
    const dir = st.rnn, H = dir.H, D = dir.D;
    let sum = 0, signed = 0;
    for (let g = 0; g < dir.G; g++) { const w = dir.px.W[(g * H + co) * D + Math.min(ci, D - 1)]; sum += Math.abs(w); signed += w; }
    return { mag: sum / dir.G, sign: signed >= 0 ? 1 : -1 };
  }
  return { mag: 0.5, sign: 1 };
}

function aeDraw(ctx, o) {
  const { model, layout, probe, recon, hover, sel, mode } = o;
  const cols = layout.cols;
  ctx.clearRect(0, 0, layout.width, layout.height);

  // links
  for (let c = 1; c < cols.length; c++) {
    const prev = cols[c - 1], cur = cols[c];
    const si = cur.stage;
    let maxMag = 1e-6;
    const ls = [];
    cur.nodes.forEach((to, co) => prev.nodes.forEach((from, ci) => {
      const k = si != null ? aeLinkStrength(model, si, co, ci) : { mag: 0.5, sign: 1 };
      maxMag = Math.max(maxMag, k.mag);
      ls.push([from, to, k]);
    }));
    ls.forEach(([from, to, k]) => drawLink(ctx, from, to, k.mag / maxMag * 0.8, k.sign, false, null));
  }

  // input
  const inNd = cols[0].nodes[0];
  drawNodeBox(ctx, inNd, hover && hover.type === 'input', sel && sel.type === 'input');
  if (probe) {
    if (mode === 'freq') { const m = magSpectrum(probe, 0, WIN); drawSpectrum(ctx, inNd.x + 5, inNd.y + 5, inNd.w - 10, inNd.h - 10, m, maxOf(m)); }
    else drawWave(ctx, inNd.x + 5, inNd.y + 5, inNd.w - 10, inNd.h - 10, probe, 0, WIN, maxAbs(probe, 0, WIN));
  }
  label(ctx, inNd.x, inNd.y - 8, 'INPUT · ' + WIN + ' samples');

  // stages
  model.stages.forEach((st, si) => {
    const col = cols[si + 1];
    const narrow = col.nodes[0].w < 80;
    label(ctx, col.x, col.nodes[0].y - 8, narrow ? st.short || st.label.split(' ·')[0] : st.label);
    const snap = st.snapshot;
    const oneBox = st.kind === 'units' || st.kind === 'seq';
    col.nodes.forEach((nd, c) => {
      const isHot = hover && hover.type === 'node' && hover.stage === si && (oneBox || hover.ch === c);
      const isSel = sel && sel.type === 'node' && sel.stage === si && (oneBox || sel.ch === c);
      drawNodeBox(ctx, nd, isHot, isSel);
      if (!snap) return;
      if (st.kind === 'units') aeDrawUnits(ctx, nd, snap, sel && sel.type === 'node' && sel.stage === si ? sel.ch : -1);
      else if (st.kind === 'seq') aeDrawSeq(ctx, nd, snap, st.C, st.L, isSel ? sel.ch : -1, st.masked);
      else if (st.kind === 'code') aeDrawCode(ctx, nd, snap[c], maxAbs(snap, 0, snap.length), st.part === 'code' ? 'z' + (c + 1) : '');
      else {
        const L = st.L;
        if (mode === 'freq') { const m = magSpectrum(snap, c * L, L); drawSpectrum(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, m, maxOf(m)); }
        else drawWave(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, snap, c * L, L, maxAbs(snap, 0, snap.length));
        if (st.masked) aeShadeMasked(ctx, nd, st.masked, L);
      }
      if (isSel && st.L) {                  // the position being computed
        const t = Math.min(o.tPos, st.L - 1);
        const px = st.kind === 'seq' ? nd.x + 4 + (t + 0.5) * (nd.w - 8) / st.L
          : nd.x + 4 + (st.L <= 1 ? 0 : t * (nd.w - 9) / (st.L - 1));
        ctx.fillStyle = 'rgba(29,78,216,0.55)';
        ctx.fillRect(px, nd.y + 2, 1.5, nd.h - 4);
      }
    });
  });

  // MAE: attention above every encoder column, averaged over the heads
  // (row = token that looks, column = token it looks at; masked tokens are marked on the left)
  model.stages.forEach((st, si) => {
    if (!st.tflayer || !st.layer.trace) return;
    const col = cols[si + 1], A = st.layer.trace.A, T = st.L, H = st.layer.attn.H;
    const size = Math.min(ATTN_PX, col.nodes[0].w - 6);
    const x0 = col.x + col.nodes[0].w / 2 - size / 2, y0 = col.nodes[0].y - 22 - size, cs = size / T;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x0, y0, size, size);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        let a = 0;
        for (let h = 0; h < H; h++) a += A[(h * T + i) * T + j];
        a /= H;
        if (a <= 0) continue;
        ctx.fillStyle = 'rgba(29,78,216,' + Math.min(1, Math.pow(a, 0.6)).toFixed(3) + ')';
        ctx.fillRect(x0 + j * cs, y0 + i * cs, cs + 0.3, cs + 0.3);
      }
      if (st.masked && st.masked[i]) { ctx.fillStyle = '#e0342b'; ctx.fillRect(x0 - 4, y0 + i * cs, 2.5, cs); }
    }
    ctx.strokeStyle = '#cfd6de'; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, size, size);
    if (sel && sel.type === 'node' && sel.stage === si) {
      ctx.strokeStyle = '#e0342b'; ctx.lineWidth = 1.4;
      ctx.strokeRect(x0, y0 + Math.min(o.tPos, T - 1) * cs, size, cs);
    }
    ctx.fillStyle = '#98a2ad';
    ctx.font = '9px system-ui, sans-serif';
    ctx.fillText(H > 1 ? 'attention · mean of ' + H : 'attention', x0, y0 - 3);
  });

  // output: the reconstruction over the input, and the error underneath
  const outNd = cols[cols.length - 1].nodes[0];
  drawNodeBox(ctx, outNd, hover && hover.type === 'output', sel && sel.type === 'output');
  label(ctx, outNd.x, outNd.y - 8, 'OUTPUT · reconstruction');
  if (probe && recon) aeDrawRecon(ctx, outNd.x + 5, outNd.y + 5, outNd.w - 10, outNd.h - 10, probe, recon, o.mask, o.masked);
  if (sel && (sel.type === 'output' || sel.type === 'input')) {
    const nd = sel.type === 'output' ? outNd : inNd;
    const px = nd.x + 5 + o.tPos * (nd.w - 11) / (WIN - 1);
    ctx.fillStyle = 'rgba(29,78,216,0.55)';
    ctx.fillRect(px, nd.y + 2, 1.5, nd.h - 4);
  }
}

/** One box holding a whole dense layer: a centred bar per unit. */
function aeDrawUnits(ctx, nd, vals, hi) {
  const n = vals.length, m = maxAbs(vals, 0, n);
  const top = nd.y + 6, rowH = (nd.h - 12) / n, cx = nd.x + nd.w / 2, half = nd.w / 2 - 8;
  ctx.strokeStyle = '#e3e9f1'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, top + n * rowH); ctx.stroke();
  for (let i = 0; i < n; i++) {
    const v = vals[i], len = half * Math.max(-1, Math.min(1, v / m));
    ctx.fillStyle = v >= 0 ? POS : NEG;
    ctx.fillRect(len >= 0 ? cx : cx + len, top + i * rowH + rowH * 0.15, Math.abs(len), Math.max(1, rowH * 0.7));
    if (i === hi) {
      ctx.strokeStyle = '#e0342b'; ctx.lineWidth = 1.2;
      ctx.strokeRect(nd.x + 3, top + i * rowH, nd.w - 6, rowH);
    }
  }
}

/**
 * A whole sequence layer in one box: a row per unit (or token dimension), a column per
 * step, coloured by the value — orange positive, blue negative. Masked tokens are hatched.
 */
function aeDrawSeq(ctx, nd, snap, C, L, hi, masked) {
  const x0 = nd.x + 4, y0 = nd.y + 6, w = nd.w - 8, rowH = (nd.h - 12) / C, cw = w / L;
  const m = maxAbs(snap, 0, snap.length) || 1;
  for (let c = 0; c < C; c++) {
    for (let t = 0; t < L; t++) {
      const v = snap[c * L + t] / m, a = Math.min(1, Math.abs(v));
      if (a < 0.02) continue;
      ctx.fillStyle = (v >= 0 ? 'rgba(194,118,15,' : 'rgba(8,119,189,') + Math.pow(a, 0.7).toFixed(3) + ')';
      ctx.fillRect(x0 + t * cw, y0 + c * rowH, cw + 0.3, rowH + 0.3);
    }
  }
  if (masked) {
    ctx.strokeStyle = 'rgba(29,43,56,0.35)'; ctx.lineWidth = 0.8;
    masked.forEach((on, t) => {
      if (!on) return;
      ctx.beginPath();
      for (let yy = 0; yy < nd.h - 12; yy += 6) { ctx.moveTo(x0 + t * cw, y0 + yy + 6); ctx.lineTo(x0 + (t + 1) * cw, y0 + yy); }
      ctx.stroke();
    });
  }
  if (hi >= 0) {
    ctx.strokeStyle = '#e0342b'; ctx.lineWidth = 1.2;
    ctx.strokeRect(nd.x + 2, y0 + hi * rowH, nd.w - 4, rowH);
  }
}

/** One latent number: a bar around the centre and its value. */
function aeDrawCode(ctx, nd, v, m, name) {
  const cx = nd.x + nd.w / 2, half = nd.w / 2 - 6, len = half * Math.max(-1, Math.min(1, v / m));
  ctx.fillStyle = '#eef1f4'; ctx.fillRect(nd.x + 6, nd.y + nd.h - 8, nd.w - 12, 4);
  ctx.fillStyle = v >= 0 ? POS : NEG;
  ctx.fillRect(len >= 0 ? cx : cx + len, nd.y + nd.h - 8, Math.abs(len), 4);
  ctx.fillStyle = '#31404e';
  ctx.font = '600 10px ui-monospace, Consolas, monospace';
  const s = (name ? name + ' ' : '') + (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
  ctx.fillText(s, nd.x + 6, nd.y + 11);
}

/** Input (grey) and reconstruction (blue), with the squared error as a red band at the bottom. */
function aeDrawRecon(ctx, x, y, w, h, a, b, mask, masked) {
  const n = a.length, top = h * 0.72;
  let m = 1e-6;
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i]), Math.abs(b[i]));
  const X = (i) => x + i * (w - 1) / (n - 1), Y = (v) => y + top / 2 - v / m * (top / 2 - 2);
  if (mask) {
    ctx.fillStyle = 'rgba(224,52,43,0.09)';
    for (let i = 0; i < n; i++) if (mask[i]) ctx.fillRect(X(i) - w / n / 2, y, w / n + 0.5, top);
  }
  if (masked) {            // MAE: the patches the encoder did not see
    ctx.fillStyle = 'rgba(29,78,216,0.07)';
    const P = n / masked.length;
    masked.forEach((on, p) => { if (on) ctx.fillRect(X(p * P), y, w * P / n, top); });
  }
  ctx.strokeStyle = '#e3e9f1'; ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.moveTo(x, y + top / 2); ctx.lineTo(x + w, y + top / 2); ctx.stroke();
  const line = (arr, color, lw) => {
    ctx.beginPath();
    for (let i = 0; i < n; i++) { if (i) ctx.lineTo(X(i), Y(arr[i])); else ctx.moveTo(X(i), Y(arr[i])); }
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  };
  line(a, '#b6c0ca', 1.6);
  line(b, '#2b6cb0', 1.2);
  let em = 1e-9;
  const e = new Float32Array(n);
  for (let i = 0; i < n; i++) { e[i] = (a[i] - b[i]) * (a[i] - b[i]); em = Math.max(em, e[i]); }
  const by = y + top + 2, bh = h - top - 2;
  ctx.fillStyle = 'rgba(224,52,43,0.75)';
  for (let i = 0; i < n; i++) {
    const hh = e[i] / em * bh;
    ctx.fillRect(X(i) - w / n / 2, by + bh - hh, w / n + 0.3, hh);
  }
}

/** Light shading over masked tokens in a token map. */
function aeShadeMasked(ctx, nd, masked, L) {
  ctx.fillStyle = 'rgba(29,78,216,0.10)';
  const w = (nd.w - 8) / L;
  masked.forEach((on, t) => { if (on) ctx.fillRect(nd.x + 4 + t * w, nd.y + 3, w, nd.h - 6); });
}

function aeHit(layout, mx, my) {
  const cols = layout.cols;
  for (let ci = 0; ci < cols.length; ci++) {
    const col = cols[ci];
    for (let i = 0; i < col.nodes.length; i++) {
      const nd = col.nodes[i];
      if (mx >= nd.x && mx <= nd.x + nd.w && my >= nd.y && my <= nd.y + nd.h) {
        if (col.kind === 'input') return { type: 'input', nd };
        if (col.kind === 'output') return { type: 'output', nd };
        // a whole dense layer is one box: the row under the cursor picks the unit
        if (col.kind === 'units' || col.kind === 'seq') return { type: 'node', stage: col.stage, ch: -1, nd, my };
        return { type: 'node', stage: col.stage, ch: i, nd };
      }
    }
  }
  return null;
}
