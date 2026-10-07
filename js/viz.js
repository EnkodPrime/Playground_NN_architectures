/* viz.js — draws the network, signals, spectra and metrics onto canvases. */

const POS = '#f0921f';   // positive values (orange, as in TF Playground)
const ATTN_PX = 58;      // size of the attention maps above a Transformer's encoder columns
const INCEPTION_COLORS = ['#2b6cb0', '#2e9e5b', '#8e44ad', '#7b8794'];   // K short / mid / long / pool
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

  // a deep stack: tighten the gaps first, then narrow the boxes, and only then scroll sideways
  const n = cols.length;
  const fixedW = IN_W + 132;
  const need = (g, w) => fixedW + (n - 2) * w + g * (n - 1) + 24;
  let gap = COL_GAP, midW = NODE_W;
  if (n > 2 && need(gap, midW) > cssW) {
    gap = Math.max(34, (cssW - 24 - fixedW - (n - 2) * midW) / (n - 1));
    if (need(gap, midW) > cssW) midW = Math.max(60, (cssW - 24 - fixedW - gap * (n - 1)) / (n - 2));
    for (let c = 1; c < n - 1; c++) cols[c].nodes.forEach((nd) => { nd.w = midW; });
  }

  // horizontal placement
  let totalW = 0;
  cols.forEach((c) => { totalW += c.nodes[0].w; });
  totalW += gap * (cols.length - 1);
  let x = Math.max(12, (cssW - totalW) / 2);
  // a Transformer draws its attention maps above the encoder columns
  const topPad = model.kind === 'transformer' ? 30 + ATTN_PX + 24 : 30;
  let maxH = 0;
  cols.forEach((col) => {
    const n = col.nodes.length;
    const h = n * col.nodes[0].h + (n - 1) * NODE_VGAP;
    maxH = Math.max(maxH, h);
    col.x = x;
    col.h = h;
    x += col.nodes[0].w + gap;
  });
  cols.forEach((col) => {
    let y = topPad + (maxH - col.h) / 2;
    col.nodes.forEach((nd) => { nd.x = col.x; nd.y = y; y += nd.h + NODE_VGAP; });
  });
  // residual skips are drawn as arcs under the columns
  const skips = model.stages.some((st) => skipArc(st, 0));
  return { cols, width: Math.max(cssW, totalW + 24), height: topPad + maxH + (skips ? 52 : 34) };
}

/**
 * The skip that ends at stage li, if any: which column it starts from and
 * whether it goes through a 1×1 convolution.
 */
function skipArc(st, li) {
  if (st.res) return { from: li, proj: !!st.proj };
  if (st.resblock && st.block.skip) return { from: li, proj: !!st.block.proj };
  if (st.shortcut) return { from: 0, proj: true, label: 'shortcut' };
  return null;
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
  if (st.conv) return linkStrength(st.conv, co, ci);
  if (st.tfembed) {
    // the patch embedding: the same 8 weights for every token
    const E = model.embed;
    let sum = 0, signed = 0;
    for (let j = 0; j < E.din; j++) { const w = E.W[co * E.din + j]; sum += Math.abs(w); signed += w; }
    return { mag: sum / E.din, sign: signed >= 0 ? 1 : -1 };
  }
  if (st.tflayer) {
    // the residual stream carries every dimension straight on; attention adds W_o·W_v on top
    const at = st.layer.attn, d = at.d;
    let m = co === ci ? 1 : 0;
    for (let k = 0; k < d; k++) m += at.o.W[co * d + k] * at.v.W[k * d + ci];
    return { mag: Math.abs(m), sign: m >= 0 ? 1 : -1 };
  }
  if (st.resblock) {
    // the block's first convolution reads input channel ci
    return linkStrength(st.block.convs[0], co, ci);
  }
  if (st.inception) {
    const m = st.module, { bi, j } = m.branchOf(co);
    if (bi === m.kernels.length) {                 // pool branch: one 1×1 weight per input channel
      const w = m.poolConv.W[j * m.cin + ci];
      return { mag: Math.abs(w), sign: w >= 0 ? 1 : -1 };
    }
    const br = m.branches[bi];
    if (!m.bott) return linkStrength(br, j, ci);
    // through the bottleneck: how much of channel ci reaches this branch filter
    let sum = 0, signed = 0;
    for (let b = 0; b < m.B; b++) {
      const wb = m.bott.W[b * m.cin + ci];
      const k = linkStrength(br, j, b);
      sum += Math.abs(wb) * k.mag;
      signed += wb * k.sign;
    }
    return { mag: sum / m.B, sign: signed >= 0 ? 1 : -1 };
  }
  if (st.mlp) {
    const d = st.dense;
    if (li === 0) {                       // the whole window arrives as one link
      let sum = 0, signed = 0;
      for (let i = 0; i < d.nin; i++) { const w = d.W[co * d.nin + i]; sum += Math.abs(w); signed += w; }
      return { mag: sum / d.nin, sign: signed >= 0 ? 1 : -1 };
    }
    const w = d.W[co * d.nin + ci];
    return { mag: Math.abs(w), sign: w >= 0 ? 1 : -1 };
  }
  if (st.kan) {
    // size of an edge function: its silu weight plus the spread of its spline
    const L = st.layer, nin = L.nin;
    const edgeMag = (i) => {
      const row = (co * nin + i) * KAN_NB;
      let c = 0;
      for (let m = 0; m < KAN_NB; m++) c += Math.abs(L.pc.W[row + m]);
      return { mag: Math.abs(L.pw.W[co * nin + i]) + c / KAN_NB, sign: L.pw.W[co * nin + i] };
    };
    if (li === 0) {
      let sum = 0, signed = 0;
      for (let i = 0; i < nin; i++) { const e = edgeMag(i); sum += e.mag; signed += e.sign; }
      return { mag: sum / nin, sign: signed >= 0 ? 1 : -1 };
    }
    const e = edgeMag(ci);
    return { mag: e.mag, sign: e.sign >= 0 ? 1 : -1 };
  }
  if (st.ssm) {
    const w = st.layer.pin.W[co * st.layer.D + ci];
    return { mag: Math.abs(w), sign: w >= 0 ? 1 : -1 };
  }
  if (st.gnn) {
    const D = st.layer.D;
    const ws = st.layer.pself.W[co * D + ci], wn = st.layer.pneigh.W[co * D + ci];
    return { mag: (Math.abs(ws) + Math.abs(wn)) / 2, sign: ws + wn >= 0 ? 1 : -1 };
  }
  return rnnLinkStrength(st, co, ci);
}

function stageInputCount(model, li) {
  const st = model.stages[li];
  if (st.conv) return st.conv.cin;
  if (st.mlp || st.kan || st.resblock || st.inception || st.tfembed || st.tflayer) return li === 0 ? 1 : model.stages[li - 1].C;
  if (st.ssm || st.gnn) return st.layer.D;
  return st.layer.fwd.D;
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
    // a deep stack narrows the columns, so its labels go short: "L3 K5 res ↓"
    const narrow = col.nodes[0].w < NODE_W;
    label(ctx, col.x, col.nodes[0].y - 8, st.conv
      ? (narrow
        ? 'L' + (li + 1) + ' K' + st.conv.k + (st.res ? ' res' : '') + (st.pooled ? ' ↓' : '')
        : 'LAYER ' + (li + 1) + ' · K=' + st.conv.k + (st.res ? ' · res' : '') + (st.pooled ? ' · pool' : ''))
      : st.tfembed
        ? 'EMBED · ' + model.T + ' tokens × ' + model.d
      : st.tflayer
        ? (narrow ? 'E' + li : 'ENCODER ' + li + ' · ' + st.layer.attn.H + ' head' + (st.layer.attn.H > 1 ? 's' : '') +
          (st.causal ? ' · causal' : ''))
      : st.resblock
        ? (narrow ? 'B' + (li + 1) + ' ' + st.block.kernels.join('-')
          : 'BLOCK ' + (li + 1) + ' · K ' + st.block.kernels.join('-'))
      : st.inception
        ? 'MODULE ' + (li + 1) + ' · K ' + st.module.kernels.join('/')
      : st.mlp
        ? 'LAYER ' + (li + 1) + ' · DENSE · ' + (li === 0 ? WIN : model.stages[li - 1].C) + ' in'
      : st.kan
        ? 'LAYER ' + (li + 1) + ' · KAN · ' + (li === 0 ? WIN : model.stages[li - 1].C) + ' in'
      : st.gnn
        ? 'LAYER ' + (li + 1) + ' · GNN · ' + st.layer.agg
      : st.ssm
        ? 'LAYER ' + (li + 1) + ' · ' + (st.kind === 's4' ? 'S4D' : 'MAMBA') + ' · N=' + st.layer.N
        : 'LAYER ' + (li + 1) + ' · ' + st.kind.toUpperCase() + (st.bidir ? ' · bi' : ''));
    for (let c = 0; c < col.nodes.length; c++) {
      const nd = col.nodes[c];
      const isHot = hover && hover.type === 'filter' && hover.layer === li && hover.ch === c;
      const isSel = sel && sel.type === 'filter' && sel.layer === li && sel.ch === c;
      drawNodeBox(ctx, nd, isHot, isSel);
      if (st.mlp || st.kan) {
        drawUnitBox(ctx, nd, model, li, c, mode);
        continue;
      }
      if (snap) {
        if (mode === 'freq') {
          const m = magSpectrum(snap, c * st.L, st.L);
          drawSpectrum(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, m, maxOf(m));
        } else {
          drawWave(ctx, nd.x + 4, nd.y + 4, nd.w - 8, nd.h - 8, snap, c * st.L, st.L, scale);
        }
      }
      if (st.inception) {
        // which branch this channel comes from
        const { bi } = st.module.branchOf(c);
        const name = bi < st.module.kernels.length ? 'K' + st.module.kernels[bi] : 'pool';
        ctx.font = '600 9px system-ui, sans-serif';
        const tw = ctx.measureText(name).width + 6;
        ctx.fillStyle = INCEPTION_COLORS[bi % INCEPTION_COLORS.length];
        ctx.fillRect(nd.x + 3, nd.y + 2, tw, 12);
        ctx.fillStyle = '#fff';
        ctx.fillText(name, nd.x + 6, nd.y + 11);
      } else if (st.conv) {
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

  // attention above every encoder column, averaged over the heads (each head is in the hover view):
  // row = token that looks, column = token it looks at
  for (let li = 0; li < model.stages.length; li++) {
    const st = model.stages[li];
    if (!st.tflayer || !st.layer.trace) continue;
    const col = cols[li + 1], A = st.layer.trace.A, T = model.T, H = st.layer.attn.H;
    const pitch = cols[li + 2] ? cols[li + 2].x - col.x : ATTN_PX + 8;
    const size = Math.min(ATTN_PX, pitch - 10);
    const x0 = col.x + col.nodes[0].w / 2 - size / 2;
    const y0 = col.nodes[0].y - 22 - size;
    const cs = size / T;
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
    }
    ctx.strokeStyle = '#cfd6de'; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, size, size);
    if (sel && sel.type === 'filter' && sel.layer === li) {   // the row of the selected token
      ctx.strokeStyle = '#e0342b'; ctx.lineWidth = 1.4;
      ctx.strokeRect(x0, y0 + tPosRow(o.tPos, T) * cs, size, cs);
    }
    ctx.fillStyle = '#98a2ad';
    ctx.font = '9px system-ui, sans-serif';
    ctx.fillText(H > 1 ? 'attention · mean of ' + H : 'attention', x0, y0 - 3);
  }

  // residual skips: y = f(x) + x, drawn under the columns they join
  for (let li = 0; li < model.stages.length; li++) {
    const st = model.stages[li];
    const arc = skipArc(st, li);
    if (!arc) continue;
    const a = cols[arc.from], b = cols[li + 1];
    const bottom = (col) => col.nodes[col.nodes.length - 1].y + col.nodes[col.nodes.length - 1].h;
    const x1 = a.x + a.nodes[0].w / 2, x2 = b.x + b.nodes[0].w / 2;
    const y1 = bottom(a) + 3, y2 = bottom(b) + 3;
    let yb = Math.max(y1, y2) + 18;
    for (let k = arc.from + 1; k <= li; k++) yb = Math.max(yb, bottom(cols[k]) + 18);   // pass under the columns between
    ctx.save();
    ctx.strokeStyle = '#2b6cb0';
    ctx.globalAlpha = 0.75;
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(x1, yb, x2, yb, x2, y2 + 5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#2b6cb0';
    ctx.beginPath();                 // arrowhead into the block it is added to
    ctx.moveTo(x2, y2); ctx.lineTo(x2 - 4, y2 + 7); ctx.lineTo(x2 + 4, y2 + 7); ctx.closePath();
    ctx.fill();
    ctx.font = '600 9px system-ui, sans-serif';
    const tag = '+ ' + (arc.label || 'skip') + (arc.proj ? ' (1×1)' : '');
    ctx.fillText(tag, (x1 + x2) / 2 - ctx.measureText(tag).width / 2, yb - 2);
    ctx.restore();
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

/** The token row the position slider points at. */
function tPosRow(t, T) { return Math.max(0, Math.min(T - 1, t)); }

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
  if (st.mlp || st.kan) {
    // a dense unit reads every input; in the first layer t picks one of them
    prevCol.nodes.forEach((nd) => span(nd, inner, prevLen, 0, prevLen - 1, 'rgba(29,78,216,0.10)'));
    if (sel.layer === 0) {
      span(prevCol.nodes[0], 5, WIN, tPos, tPos, 'rgba(29,78,216,0.55)');
      span(cols[1].nodes[sel.ch], 4, WIN, tPos, tPos, 'rgba(29,78,216,0.55)');
    }
    return;
  }
  if (st.tfembed) {
    // a token is one patch of 8 samples
    span(cols[0].nodes[0], 5, WIN, tPos * TF_PATCH, tPos * TF_PATCH + TF_PATCH - 1, 'rgba(29,78,216,0.30)');
    span(cols[1].nodes[sel.ch], 4, st.L, tPos, tPos, 'rgba(29,78,216,0.55)');
    return;
  }
  if (st.tflayer) {
    // attention reads every token — or, causal, only those up to t
    prevCol.nodes.forEach((nd) => span(nd, inner, prevLen, 0, st.causal ? tPos : prevLen - 1, 'rgba(29,78,216,0.12)'));
    span(cols[sel.layer + 1].nodes[sel.ch], 4, st.L, tPos, tPos, 'rgba(29,78,216,0.55)');
    return;
  }
  if (st.resblock || st.inception) {
    // the window the block can see: its kernels stacked (ResNet) or the longest branch (Inception)
    let lo = 0, hi = 0;
    if (st.resblock) {
      st.block.convs.forEach((cv) => { lo += cv.tapOffset(0); hi += cv.tapOffset(cv.k - 1); });
    } else {
      st.module.branches.forEach((cv) => { lo = Math.min(lo, cv.tapOffset(0)); hi = Math.max(hi, cv.tapOffset(cv.k - 1)); });
      lo = Math.min(lo, -1); hi = Math.max(hi, 1);
    }
    prevCol.nodes.forEach((nd) => span(nd, inner, prevLen, tPos + lo, tPos + hi, 'rgba(29,78,216,0.16)'));
    span(cols[sel.layer + 1].nodes[sel.ch], 4, st.L, tPos, tPos, 'rgba(29,78,216,0.55)');
    return;
  }
  if (st.conv) {
    // the kernel receptive window across every input channel
    const c = st.conv;
    prevCol.nodes.forEach((nd) => span(nd, inner, prevLen,
      tPos + c.tapOffset(0), tPos + c.tapOffset(c.k - 1), 'rgba(29,78,216,0.16)'));
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

/**
 * Inside an MLP or KAN unit. First layer: what the unit does across the window —
 * the 128 weights of an MLP unit (its template), or the 128 edge outputs φ(x_i)
 * of a KAN node. Deeper layers: the weights (MLP) or the edge functions (KAN)
 * from the previous layer. A bar at the bottom shows the unit's value.
 */
function drawUnitBox(ctx, nd, model, li, c, mode) {
  const st = model.stages[li];
  const x = nd.x + 4, y = nd.y + 3, w = nd.w - 8, h = nd.h - 13;
  if (st.mlp) {
    const d = st.dense;
    if (li === 0) {
      if (mode === 'freq') {
        const m = magSpectrum(d.W, c * d.nin, d.nin);
        drawSpectrum(ctx, x, y, w, h, m, maxOf(m));
      } else {
        drawWave(ctx, x, y, w, h, d.W, c * d.nin, d.nin, maxAbs(d.W, c * d.nin, d.nin));
      }
    } else {
      drawKernel(ctx, x, y, w, h, d.W, c * d.nin, d.nin);
    }
  } else if (st.phi) {
    const L = st.layer;
    if (li === 0) {
      drawWave(ctx, x, y, w, h, st.phi, c * L.nin, L.nin, maxAbs(st.phi, 0, st.phi.length));
    } else {
      // one small plot per incoming edge function
      const k = L.nin, cw = w / k;
      for (let i = 0; i < k; i++) {
        const pts = model.edgeCurve(li, c, i, 24);
        let m = 1e-6;
        pts.forEach((p) => { m = Math.max(m, Math.abs(p[1])); });
        const bx = x + i * cw;
        ctx.strokeStyle = '#eceff3'; ctx.lineWidth = 1;
        ctx.strokeRect(bx + 1, y, cw - 2, h);
        ctx.beginPath();
        pts.forEach((p, q) => {
          const X = bx + 2 + (q / (pts.length - 1)) * (cw - 4);
          const Y = y + h / 2 - (p[1] / m) * (h / 2 - 2);
          if (q === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        });
        ctx.strokeStyle = '#2b6cb0'; ctx.lineWidth = 1.2;
        ctx.stroke();
        // where this example's input lands on the curve
        const xin = st.input ? st.input[i] : 0;
        const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
        const f = (xin - x0) / (x1 - x0);
        if (f >= 0 && f <= 1) {
          const v = st.phi[c * k + i];
          ctx.fillStyle = '#e0342b';
          ctx.beginPath();
          ctx.arc(bx + 2 + f * (cw - 4), y + h / 2 - Math.max(-1, Math.min(1, v / m)) * (h / 2 - 2), 2, 0, 6.284);
          ctx.fill();
        }
      }
    }
  }
  // the unit's value for this example, as a bar around the centre
  if (st.snapshot) {
    const v = st.snapshot[c];
    const m = maxAbs(st.snapshot, 0, st.snapshot.length);
    const by = nd.y + nd.h - 8, cx = nd.x + nd.w / 2, half = (nd.w - 12) / 2;
    ctx.fillStyle = '#eef1f4';
    ctx.fillRect(nd.x + 6, by, nd.w - 12, 4);
    const len = half * Math.max(-1, Math.min(1, v / m));
    ctx.fillStyle = v >= 0 ? POS : NEG;
    ctx.fillRect(len >= 0 ? cx : cx + len, by, Math.abs(len), 4);
  }
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
