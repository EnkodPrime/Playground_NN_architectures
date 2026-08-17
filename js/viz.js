/* viz.js — draws the network, signals, spectra and metrics onto canvases. */

const POS = '#f0921f';   // positive values (orange, as in TF Playground)
const NEG = '#0877bd';   // negative values (blue)
const GRID = '#e3e6ea';
const AXIS = '#b9c0c8';

function dpiSetup(canvas, cssW, cssH) {
  const r = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(cssW * r));
  const H = Math.max(1, Math.round(cssH * r));
  const ctx = canvas.getContext('2d');
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W; canvas.height = H;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
  }
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return ctx;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Filled curve around zero: orange above, blue below. */
function drawWave(ctx, x, y, w, h, data, off, len, scale) {
  const mid = y + h / 2;
  const s = scale > 1e-6 ? scale : 1e-6;
  const px = (i) => x + (len <= 1 ? 0 : (i * (w - 1)) / (len - 1));
  const py = (v) => mid - Math.max(-1.15, Math.min(1.15, v / s)) * (h / 2 - 1);

  // zero line
  ctx.strokeStyle = AXIS;
  ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x + w, mid); ctx.stroke();

  // fill
  ctx.beginPath();
  ctx.moveTo(px(0), mid);
  for (let i = 0; i < len; i++) ctx.lineTo(px(i), py(data[off + i]));
  ctx.lineTo(px(len - 1), mid);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, y, 0, y + h);
  grad.addColorStop(0, POS + '55');
  grad.addColorStop(0.5, POS + '18');
  grad.addColorStop(0.5, NEG + '18');
  grad.addColorStop(1, NEG + '55');
  ctx.fillStyle = grad;
  ctx.fill();

  // stroke
  ctx.beginPath();
  for (let i = 0; i < len; i++) {
    const X = px(i), Y = py(data[off + i]);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.strokeStyle = '#31404e';
  ctx.lineWidth = 1.1;
  ctx.stroke();
}

/** Magnitude spectrum as a filled curve (0..Nyquist). */
function drawSpectrum(ctx, x, y, w, h, mags, scale) {
  const n = mags.length;
  const s = scale > 1e-9 ? scale : 1e-9;
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  for (let i = 0; i < n; i++) {
    const X = x + (i * (w - 1)) / Math.max(1, n - 1);
    const Y = y + h - Math.min(1, mags[i] / s) * (h - 2);
    ctx.lineTo(X, Y);
  }
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
  ctx.fillStyle = NEG + '33';
  ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const X = x + (i * (w - 1)) / Math.max(1, n - 1);
    const Y = y + h - Math.min(1, mags[i] / s) * (h - 2);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.strokeStyle = '#0f5f8f';
  ctx.lineWidth = 1.1;
  ctx.stroke();
}

/** The filter kernel as a stem plot. */
function drawKernel(ctx, x, y, w, h, W, off, k) {
  let m = 1e-6;
  for (let i = 0; i < k; i++) m = Math.max(m, Math.abs(W[off + i]));
  const mid = y + h / 2;
  const step = w / k;
  ctx.strokeStyle = AXIS; ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.moveTo(x, mid); ctx.lineTo(x + w, mid); ctx.stroke();
  for (let i = 0; i < k; i++) {
    const v = W[off + i];
    const bh = (Math.abs(v) / m) * (h / 2 - 2);
    const bx = x + i * step + step * 0.22;
    const bw = step * 0.56;
    ctx.fillStyle = v >= 0 ? POS : NEG;
    if (v >= 0) ctx.fillRect(bx, mid - bh, bw, bh);
    else ctx.fillRect(bx, mid, bw, bh);
  }
}

/* ------------------------------------------------------------ Layout */
const NODE_W = 94, NODE_H = 46, NODE_VGAP = 10, COL_GAP = 74;
const IN_W = 128, IN_H = 74;

function layoutNetwork(model, classes, cssW, oodOn) {
  const cols = [];
  const nLayers = model.stages.length;

  cols.push({ kind: 'input', nodes: [{ w: IN_W, h: IN_H }] });
  for (let i = 0; i < nLayers; i++) {
    const st = model.stages[i];
    const nodes = [];
    for (let c = 0; c < st.C; c++) nodes.push({ w: NODE_W, h: NODE_H, layer: i, ch: c });
    cols.push({ kind: 'conv', layer: i, nodes });
  }
  const outH = Math.max(56, classes.length * 20 + 10) + (oodOn ? 22 : 0);
  cols.push({ kind: 'output', nodes: [{ w: 132, h: outH }] });

  // horizontal placement
  let totalW = 0;
  cols.forEach((c) => { totalW += c.nodes[0].w; });
  totalW += COL_GAP * (cols.length - 1);
  let x = Math.max(12, (cssW - totalW) / 2);
  const topPad = 30;
  let maxH = 0;
  cols.forEach((col) => {
    const n = col.nodes.length;
    const h = n * col.nodes[0].h + (n - 1) * NODE_VGAP;
    maxH = Math.max(maxH, h);
    col.x = x;
    col.h = h;
    x += col.nodes[0].w + COL_GAP;
  });
  cols.forEach((col) => {
    let y = topPad + (maxH - col.h) / 2;
    col.nodes.forEach((nd) => { nd.x = col.x; nd.y = y; y += nd.h + NODE_VGAP; });
  });
  return { cols, width: Math.max(cssW, totalW + 24), height: topPad + maxH + 34 };
}

/** Strength of the link between input channel ci and filter co. */
function linkStrength(conv, co, ci) {
  const k = conv.k;
  const base = (co * conv.cin + ci) * k;
  let sum = 0, signed = 0;
  for (let j = 0; j < k; j++) { sum += Math.abs(conv.W[base + j]); signed += conv.W[base + j]; }
  return { mag: sum / k, sign: signed >= 0 ? 1 : -1 };
}

/** Same idea for a recurrent unit: all gate weights that read input channel ci. */
function rnnLinkStrength(stage, unit, ci) {
  const layer = stage.layer;
  const back = layer.bidir && unit >= layer.H;
  const dir = back ? layer.bwd : layer.fwd;
  const u = back ? unit - layer.H : unit;
  let sum = 0, signed = 0;
  for (let g = 0; g < dir.G; g++) {
    const w = dir.px.W[(g * layer.H + u) * dir.D + ci];
    sum += Math.abs(w); signed += w;
  }
  return { mag: sum / dir.G, sign: signed >= 0 ? 1 : -1 };
}

/** Uniform accessor so the diagram does not care which model it is drawing. */
function stageLink(model, li, co, ci) {
  const st = model.stages[li];
  return st.conv ? linkStrength(st.conv, co, ci) : rnnLinkStrength(st, co, ci);
}

function stageInputCount(model, li) {
  const st = model.stages[li];
  return st.conv ? st.conv.cin : st.layer.fwd.D;
}

function outLinkStrength(model, ci) {
  const d = model.dense;
  let sum = 0, signed = 0;
  const per = d.nin / model.finalC;   // 1 for GAP/GMP, finalL for flatten
  for (let j = 0; j < d.nout; j++) {
    for (let q = 0; q < per; q++) {
      const w = d.W[j * d.nin + ci * per + q];
      sum += Math.abs(w); signed += w;
    }
  }
  return { mag: sum / (d.nout * per), sign: signed >= 0 ? 1 : -1 };
}

/**
 * Draws the whole network diagram.
 * @param {object} o {model, layout, probe, probs, classes, mode:'time'|'freq', hover, ctx}
 */
function drawNetwork(ctx, o) {
  const { model, layout, probe, probs, classes, mode, hover, sel } = o;
  const cols = layout.cols;
  ctx.clearRect(0, 0, layout.width, layout.height);

  // --- links
  for (let li = 0; li < model.stages.length; li++) {
    const prev = cols[li], cur = cols[li + 1];
    const nOut = model.stages[li].C, nIn = stageInputCount(model, li);
    let maxMag = 1e-6;
    for (let co = 0; co < nOut; co++)
      for (let ci = 0; ci < nIn; ci++)
        maxMag = Math.max(maxMag, stageLink(model, li, co, ci).mag);
    for (let co = 0; co < nOut; co++) {
      for (let ci = 0; ci < nIn; ci++) {
        const { mag, sign } = stageLink(model, li, co, ci);
        const a = mag / maxMag;
        const from = prev.nodes[Math.min(ci, prev.nodes.length - 1)];
        const to = cur.nodes[co];
        const hot = hover && ((hover.type === 'filter' && hover.layer === li && hover.ch === co) ||
                              (hover.type === 'filter' && hover.layer === li - 1 && hover.ch === ci) ||
                              (hover.type === 'input' && li === 0));
        drawLink(ctx, from, to, a, sign, hot, hover);
      }
    }
  }
  // links into the output node
  {
    const prev = cols[cols.length - 2], to = cols[cols.length - 1].nodes[0];
    let maxMag = 1e-6;
    for (let ci = 0; ci < model.finalC; ci++) maxMag = Math.max(maxMag, outLinkStrength(model, ci).mag);
    for (let ci = 0; ci < model.finalC; ci++) {
      const { mag, sign } = outLinkStrength(model, ci);
      const hot = hover && hover.type === 'filter' && hover.layer === model.stages.length - 1 && hover.ch === ci;
      drawLink(ctx, prev.nodes[ci], to, mag / maxMag, sign, hot, hover);
    }
  }

  // --- nodes
  // input
  const inNode = cols[0].nodes[0];
  drawNodeBox(ctx, inNode, hover && hover.type === 'input', sel && sel.type === 'input');
  if (probe) {
    if (mode === 'freq') {
      const m = magSpectrum(probe, 0, WIN);
      drawSpectrum(ctx, inNode.x + 5, inNode.y + 5, inNode.w - 10, inNode.h - 10, m, maxOf(m));
    } else {
      drawWave(ctx, inNode.x + 5, inNode.y + 5, inNode.w - 10, inNode.h - 10, probe, 0, WIN, maxAbs(probe, 0, WIN));
    }
  }
  label(ctx, inNode.x, inNode.y - 8, 'INPUT · ' + WIN + ' samples');

  // convolutional layers
  for (let li = 0; li < model.stages.length; li++) {
    const st = model.stages[li];
    const col = cols[li + 1];
    const snap = st.snapshot;
    let scale = 1e-6;
    if (snap) for (let i = 0; i < snap.length; i++) scale = Math.max(scale, Math.abs(snap[i]));
    label(ctx, col.x, col.nodes[0].y - 8, st.conv
      ? 'LAYER ' + (li + 1) + ' · K=' + st.conv.k + (st.pooled ? ' · pool' : '')
      : 'LAYER ' + (li + 1) + ' · ' + st.kind.toUpperCase() + (st.bidir ? ' · bi' : ''));
    for (let c = 0; c < col.nodes.length; c++) {
      const nd = col.nodes[c];
      const isHot = hover && hover.type === 'filter' && hover.layer === li && hover.ch === c;
      const isSel = sel && sel.type === 'filter' && sel.layer === li && sel.ch === c;
      drawNodeBox(ctx, nd, isHot, isSel);
      if (snap) {
        if (mode === 'freq') {
          const m = magSpectrum(snap, c * st.L, st.L);
          drawSpectrum(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, m, maxOf(m));
        } else {
          drawWave(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, snap, c * st.L, st.L, scale);
        }
      }
      if (st.conv) {
        // small kernel glyph in the top-left corner, on a light backing
        const kw = Math.min(30, st.conv.k * 4);
        ctx.fillStyle = 'rgba(255,255,255,0.86)';
        ctx.fillRect(nd.x + 3, nd.y + 2, kw + 4, 14);
        drawKernel(ctx, nd.x + 5, nd.y + 3, kw, 12,
          st.conv.W, (c * st.conv.cin) * st.conv.k, st.conv.k);
      } else if (st.bidir) {
        // mark which direction this unit belongs to
        ctx.fillStyle = 'rgba(255,255,255,0.86)';
        ctx.fillRect(nd.x + 3, nd.y + 2, 16, 12);
        ctx.fillStyle = '#7b8794';
        ctx.font = '600 9px system-ui, sans-serif';
        ctx.fillText(c < st.units ? '→' : '←', nd.x + 6, nd.y + 11);
      }
    }
  }

  // output
  const outNode = cols[cols.length - 1].nodes[0];
  drawNodeBox(ctx, outNode, false, sel && sel.type === 'output');
  label(ctx, outNode.x, outNode.y - 8, 'OUTPUT');
  const oodInfo = o.oodInfo;
  const barArea = oodInfo ? outNode.h - 22 : outNode.h;
  if (probs) {
    const bh = (barArea - 10) / classes.length;
    for (let i = 0; i < classes.length; i++) {
      const y = outNode.y + 5 + i * bh;
      const w = (outNode.w - 12) * probs[i];
      ctx.fillStyle = classes[i].color + '33';
      ctx.fillRect(outNode.x + 6, y + 1, outNode.w - 12, bh - 3);
      ctx.fillStyle = classes[i].color;
      ctx.fillRect(outNode.x + 6, y + 1, w, bh - 3);
      ctx.fillStyle = '#20303c';
      ctx.font = '10px system-ui, sans-serif';
      ctx.fillText(classes[i].short + ' ' + (probs[i] * 100).toFixed(0) + '%', outNode.x + 9, y + bh / 2 + 3);
    }
  }
  if (oodInfo) {
    const y = outNode.y + barArea + 1;
    const flagged = oodInfo.flagged;
    ctx.fillStyle = flagged ? '#5b6873' : '#eaf3ed';
    roundRect(ctx, outNode.x + 6, y, outNode.w - 12, 17, 4);
    ctx.fill();
    ctx.fillStyle = flagged ? '#ffffff' : '#2e9e5b';
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.fillText(flagged ? 'UNKNOWN' : 'known', outNode.x + 11, y + 12);
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = flagged ? '#d7dde3' : '#7ba98c';
    const s = oodInfo.score.toFixed(2);
    ctx.fillText(s, outNode.x + outNode.w - 13 - ctx.measureText(s).width, y + 12);
  }

  drawSelectionOverlay(ctx, o);
}

/** Marks position t: the receptive window on the inputs and the computed point. */
function drawSelectionOverlay(ctx, o) {
  const { model, layout, sel, tPos } = o;
  if (!sel) return;
  const cols = layout.cols;

  const span = (nd, inner, len, i0, i1, fill) => {
    const w = nd.w - 2 * inner;
    const px = (i) => nd.x + inner + (len <= 1 ? 0 : (i * (w - 1)) / (len - 1));
    const x0 = px(Math.max(0, i0)), x1 = px(Math.min(len - 1, i1));
    ctx.fillStyle = fill;
    ctx.fillRect(x0, nd.y + 2, Math.max(1.5, x1 - x0), nd.h - 4);
  };

  if (sel.type === 'input') {
    span(cols[0].nodes[0], 5, WIN, tPos, tPos, 'rgba(29,78,216,0.55)');
    return;
  }
  if (sel.type !== 'filter') return;

  const st = model.stages[sel.layer];
  const prevCol = cols[sel.layer];              // the column feeding the selected layer
  const prevLen = sel.layer === 0 ? WIN : model.stages[sel.layer - 1].L;
  const inner = sel.layer === 0 ? 5 : 4;
  if (st.conv) {
    // the kernel receptive window across every input channel
    const k = st.conv.k, pad = st.conv.pad;
    prevCol.nodes.forEach((nd) =>
      span(nd, inner, prevLen, tPos - pad, tPos + k - 1 - pad, 'rgba(29,78,216,0.16)'));
  } else {
    // a recurrent state has seen everything up to t — backward units, everything after
    const back = st.bidir && sel.ch >= st.units;
    prevCol.nodes.forEach((nd) => span(nd, inner, prevLen,
      back ? tPos : 0, back ? prevLen - 1 : tPos, 'rgba(29,78,216,0.13)'));
  }
  // the point being computed
  const outNd = cols[sel.layer + 1].nodes[sel.ch];
  const tp = st.pooled ? tPos >> 1 : tPos;
  span(outNd, 4, st.L, tp, tp, 'rgba(29,78,216,0.55)');
}

function drawLink(ctx, from, to, a, sign, hot, hoverActive) {
  const x1 = from.x + from.w, y1 = from.y + from.h / 2;
  const x2 = to.x, y2 = to.y + to.h / 2;
  const dim = hoverActive && !hot ? 0.18 : 1;
  ctx.strokeStyle = (sign > 0 ? POS : NEG);
  ctx.globalAlpha = Math.min(1, (0.12 + 0.85 * a) * dim);
  ctx.lineWidth = hot ? 1 + 3.5 * a : 0.5 + 2.6 * a;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const mx = (x1 + x2) / 2;
  ctx.bezierCurveTo(mx, y1, mx, y2, x2, y2);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawNodeBox(ctx, nd, hot, selected) {
  ctx.save();
  ctx.shadowColor = 'rgba(20,40,60,0.10)';
  ctx.shadowBlur = hot || selected ? 10 : 4;
  ctx.shadowOffsetY = 1;
  roundRect(ctx, nd.x, nd.y, nd.w, nd.h, 6);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();
  roundRect(ctx, nd.x, nd.y, nd.w, nd.h, 6);
  ctx.strokeStyle = selected ? '#1d4ed8' : hot ? '#2b6cb0' : '#cfd6de';
  ctx.lineWidth = selected ? 2.4 : hot ? 1.8 : 1;
  ctx.stroke();
}

function label(ctx, x, y, text) {
  ctx.fillStyle = '#7b8794';
  ctx.font = '600 10px system-ui, sans-serif';
  ctx.fillText(text, x, y);
}

function maxAbs(a, off, len) {
  let m = 1e-6;
  for (let i = 0; i < len; i++) m = Math.max(m, Math.abs(a[off + i]));
  return m;
}
function maxOf(a) {
  let m = 1e-9;
  for (let i = 0; i < a.length; i++) m = Math.max(m, a[i]);
  return m;
}

/** Finds the node under the cursor. */
function hitTest(layout, mx, my) {
  const cols = layout.cols;
  for (let ci = 0; ci < cols.length; ci++) {
    const col = cols[ci];
    for (let i = 0; i < col.nodes.length; i++) {
      const nd = col.nodes[i];
      if (mx >= nd.x && mx <= nd.x + nd.w && my >= nd.y && my <= nd.y + nd.h) {
        if (col.kind === 'input') return { type: 'input' };
        if (col.kind === 'output') return { type: 'output' };
        return { type: 'filter', layer: col.layer, ch: i };
      }
    }
  }
  return null;
}

/* ----------------------------------------------------------- Metrics */
function drawLossChart(ctx, w, h, hTrain, hTest) {
  ctx.clearRect(0, 0, w, h);
  const n = Math.max(hTrain.length, hTest.length);
  ctx.strokeStyle = GRID; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 4 + (i * (h - 12)) / 4;
    ctx.beginPath(); ctx.moveTo(28, y); ctx.lineTo(w - 4, y); ctx.stroke();
  }
  if (n < 2) return;
  let mx = 0.05;
  for (const v of hTrain) mx = Math.max(mx, v);
  for (const v of hTest) mx = Math.max(mx, v);
  mx *= 1.08;
  ctx.fillStyle = '#98a2ad';
  ctx.font = '9px system-ui, sans-serif';
  ctx.fillText(mx.toFixed(2), 2, 11);
  ctx.fillText('0', 2, h - 6);

  const plot = (hist, color) => {
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const X = 28 + (i * (w - 34)) / Math.max(1, n - 1);
      const Y = 4 + (h - 12) * (1 - hist[i] / mx);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
  };
  plot(hTrain, '#2b6cb0');
  plot(hTest, '#e0342b');
}

function drawConfusion(ctx, w, h, conf, classes, showOod) {
  ctx.clearRect(0, 0, w, h);
  const k = classes.length;
  const stride = k + 1;                       // the last column is "unknown"
  const cj = showOod ? k + 1 : k;             // number of visible columns
  const pad = 34;
  const cell = Math.min((w - pad - 6) / cj, (h - pad - 6) / k);
  const totals = new Array(k).fill(0);
  for (let i = 0; i < k; i++) for (let j = 0; j < stride; j++) totals[i] += conf[i * stride + j];
  ctx.font = '9px system-ui, sans-serif';
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < cj; j++) {
      const v = totals[i] ? conf[i * stride + j] / totals[i] : 0;
      const x = pad + j * cell, y = pad + i * cell;
      ctx.fillStyle = j === k
        ? 'rgba(91,104,115,' + (0.10 + 0.8 * v) + ')'
        : i === j
          ? 'rgba(46,158,91,' + (0.12 + 0.8 * v) + ')'
          : 'rgba(224,52,43,' + (0.08 + 0.8 * v) + ')';
      ctx.fillRect(x, y, cell - 1.5, cell - 1.5);
      if (v > 0.03) {
        ctx.fillStyle = v > 0.55 ? '#fff' : '#33414d';
        const t = (v * 100).toFixed(0);
        ctx.fillText(t, x + cell / 2 - ctx.measureText(t).width / 2, y + cell / 2 + 3);
      }
    }
    ctx.fillStyle = classes[i].color;
    ctx.fillRect(2, pad + i * cell + cell / 2 - 3, 6, 6);
    ctx.fillStyle = '#5b6873';
    ctx.fillText(classes[i].short.slice(0, 6), 11, pad + i * cell + cell / 2 + 3);
    ctx.save();
    ctx.translate(pad + i * cell + cell / 2 + 3, pad - 4);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText(classes[i].short.slice(0, 6), 0, 0);
    ctx.restore();
  }
  if (showOod) {
    ctx.fillStyle = '#5b6873';
    ctx.save();
    ctx.translate(pad + k * cell + cell / 2 + 3, pad - 4);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText('unkn.', 0, 0);
    ctx.restore();
  }
}
