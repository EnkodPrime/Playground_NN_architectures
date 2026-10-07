/* ae-main.js — the autoencoder playground: state, data, training loop, results and UI. */

const state = {
  kind: 'mlp',                 // 'mlp' | 'conv' | 'lstm' | 'mae'
  classes: ['clean', 'ripple', 'harm', 'spike', 'sag'],
  trainOn: 'clean',            // 'clean': learn only what normal looks like · 'all': every checked class
  k: 2,                        // code size
  hidden: 32,                  // MLP width
  filters: 4,                  // Conv AE channels
  cell: 'lstm', units: 16,     // LSTM AE
  maeLayers: 2, maskRatio: 0.5, // Transformer MAE (16 dimensions, 2 heads)
  vae: false, beta: 0.003,
  denoise: false, dnNoise: 0.1,
  lr: 0.003, batch: 16, l2: 0,
  noise: 0.04, strength: 1.0, ntrain: 480,
  mode: 'time',
  running: false, runEpochs: 0, stopAt: null, epoch: 0,
  hover: null, selected: null, tPos: 64,
  probe: null, probeMeta: null, probeClassId: 'clean', probePick: 'rand',
};

let model = null, train = null, test = null, layout = null, netCtx = null, recon = null;
let hTrain = [], hTest = [], lastEval = null, frameNo = 0, job = null;

const $ = (id) => document.getElementById(id);
const n3 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
const n4 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(4);
const wColor = (v) => (v >= 0 ? '#c2760f' : '#0877bd');

/* -------------------------------------------------------------- data */
function checkedClasses() { return CLASSES.filter((c) => state.classes.includes(c.id)); }
function trainIds() { return state.trainOn === 'clean' ? ['clean'] : state.classes.slice(); }

/** A dataset that also keeps, per window, the noise-free signal and where a local disturbance sits. */
function makeSet(n, ids) {
  const xs = [], ys = [], clean = [], mask = [];
  for (let i = 0; i < n; i++) {
    const id = ids[i % ids.length], meta = {};
    xs.push(generateSample(id, { noise: state.noise, strength: state.strength, meta }));
    ys.push(CLASS_INDEX[id]); clean.push(meta.clean); mask.push(meta.mask);
  }
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]]; [ys[i], ys[j]] = [ys[j], ys[i]];
    [clean[i], clean[j]] = [clean[j], clean[i]]; [mask[i], mask[j]] = [mask[j], mask[i]];
  }
  return { xs, ys, clean, mask, n };
}

function regenData() {
  train = makeSet(state.ntrain, trainIds());
  test = makeSet(48 * state.classes.length, state.classes);   // 48 windows of every checked class
  newProbe();
  drawClassPreviews();
}

function newProbe() {
  const id = state.probePick === 'rand'
    ? state.classes[Math.floor(Math.random() * state.classes.length)] : state.probePick;
  const meta = {};
  state.probe = generateSample(id, { noise: state.noise, strength: state.strength, meta });
  state.probeMeta = meta;
  state.probeClassId = id;
}

/** A training pair: the window and itself — or, denoising, a noisier copy and the noise-free signal. */
function pairOf(ds, i) {
  const x = ds.xs[i];
  if (!state.denoise) return [x, x];
  const xn = new Float32Array(WIN);
  for (let t = 0; t < WIN; t++) xn[t] = x[t] + state.dnNoise * randn();
  return [xn, ds.clean[i]];
}

/* ------------------------------------------------------------- model */
/** A fresh autoencoder of the current kind with code size k. */
function makeModel(k) {
  const base = { k, vae: state.vae, beta: state.beta };
  if (state.kind === 'conv') return new ConvAE({ ...base, filters: state.filters });
  if (state.kind === 'lstm') return new LSTMAE({ ...base, units: state.units, cell: state.cell });
  if (state.kind === 'mae') return new MAEAE({ d: 16, heads: 2, layers: state.maeLayers, maskRatio: state.maskRatio });
  return new MLPAE({ ...base, hidden: state.hidden });
}

function buildModel() {
  model = makeModel(state.k);
  state.epoch = 0;
  if (state.stopAt !== null) armRun();
  hTrain = []; hTest = []; lastEval = null;
  if (state.selected && state.selected.type === 'node' && !model.stages[state.selected.stage]) state.selected = null;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
}

function rebuild() { buildModel(); evaluate(); renderMetrics(); renderNet(); renderMath(true); generate(); }

/* ------------------------------------------------------ training loop */
function trainOneBatch() {
  const B = state.batch, ins = new Array(B), tgs = new Array(B);
  for (let b = 0; b < B; b++) {
    const [a, t] = pairOf(train, Math.floor(Math.random() * train.n));
    ins[b] = a; tgs[b] = t;
  }
  model.trainBatch(ins, tgs, state.lr, state.l2);
  state.epoch += B / train.n;
}

function runFinished() { return state.stopAt !== null && state.epoch >= state.stopAt; }
function setRunning(on) {
  state.running = on;
  $('btnPlay').textContent = on ? '⏸' : '▶';
  $('btnPlay').classList.toggle('on', on);
}
function armRun() {
  state.stopAt = state.runEpochs > 0 ? state.epoch + state.runEpochs : null;
  renderRunTarget();
}
function renderRunTarget() {
  $('epochTgt').textContent = state.stopAt === null ? '' : '/ ' + Math.round(state.stopAt);
}

function trainSlice(budgetMs) {
  const t0 = performance.now();
  do {
    if (runFinished()) break;
    trainOneBatch();
  } while (performance.now() - t0 < budgetMs);
}

function loop() {
  frameNo++;
  if (state.running && !job) {
    trainSlice(11);
    if (runFinished()) {
      setRunning(false);
      state.stopAt = null;
      evaluate(); renderMetrics(); renderRunTarget(); generate();
    } else if (frameNo % 20 === 0) {
      evaluate(); renderMetrics();
    }
  }
  if (frameNo % 3 === 0 && !job) renderNet();
  requestAnimationFrame(loop);
}

/* ---------------------------------------------------------- evaluation */
function mse(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const e = a[i] - b[i]; s += e * e; } return s / a.length; }

/** P(score of a disturbed window > score of a clean one): 1 is a perfect detector, 0.5 a coin. */
function aucOf(neg, pos) {
  if (!neg.length || !pos.length) return null;
  let s = 0;
  for (const p of pos) for (const q of neg) s += p > q ? 1 : p === q ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

/** Moving average over 5 samples: the plain low-pass filter a denoising autoencoder has to beat. */
function movingAverage(x) {
  const y = new Float32Array(x.length);
  for (let t = 0; t < x.length; t++) {
    let s = 0, n = 0;
    for (let j = -2; j <= 2; j++) if (t + j >= 0 && t + j < x.length) { s += x[t + j]; n++; }
    y[t] = s / n;
  }
  return y;
}
function snrDb(clean, est) {
  let ps = 0, pe = 0;
  for (let i = 0; i < clean.length; i++) { ps += clean[i] * clean[i]; const e = est[i] - clean[i]; pe += e * e; }
  return 10 * Math.log10(ps / Math.max(1e-12, pe));
}

/** Scores, codes and per-class numbers for a model on a dataset. */
function evaluateModel(m, ds, limit) {
  const n = Math.min(ds.n, limit || ds.n);
  const scores = [], codes = [], ys = [], per = {};
  let locHit = 0, locN = 0, snrIn = 0, snrOut = 0, snrBase = 0, nDn = 0;
  for (let i = 0; i < n; i++) {
    const x = ds.xs[i], y = m.reconstruct(x), s = mse(x, y);
    scores.push(s); ys.push(ds.ys[i]); codes.push(Float32Array.from(m.code));
    const id = CLASSES[ds.ys[i]].id;
    (per[id] = per[id] || []).push(s);
    const mk = ds.mask[i];
    if (mk && mk.some((v) => v > 0)) {          // does the largest error fall on the disturbance?
      let best = 0;
      for (let t = 1; t < WIN; t++) if ((x[t] - y[t]) ** 2 > (x[best] - y[best]) ** 2) best = t;
      locN++; if (mk[best]) locHit++;
    }
    if (state.denoise && id === 'clean') {
      const xn = new Float32Array(WIN);
      for (let t = 0; t < WIN; t++) xn[t] = x[t] + state.dnNoise * randn();
      const c = ds.clean[i];
      snrIn += snrDb(c, xn); snrOut += snrDb(c, m.reconstruct(xn)); snrBase += snrDb(c, movingAverage(xn)); nDn++;
    }
  }
  const auc = {}, mean = {};
  let aucSum = 0, aucN = 0;
  Object.keys(per).forEach((id) => {
    mean[id] = per[id].reduce((a, b) => a + b, 0) / per[id].length;
    if (id !== 'clean' && per.clean) { auc[id] = aucOf(per.clean, per[id]); aucSum += auc[id]; aucN++; }
  });
  return {
    scores, codes, ys, per, auc, mean, meanAuc: aucN ? aucSum / aucN : null,
    loc: locN ? { hit: locHit / locN, n: locN } : null,
    snr: nDn ? { in: snrIn / nDn, out: snrOut / nDn, base: snrBase / nDn } : null,
  };
}

function evaluate() {
  let tr = 0;
  const nTr = Math.min(train.n, 120);
  for (let i = 0; i < nTr; i++) { const [a, t] = pairOf(train, i); tr += mse(model.reconstruct(a), t); }
  const ev = evaluateModel(model, test, 400);
  ev.trainLoss = tr / nTr;
  ev.testLoss = ev.mean.clean != null ? ev.mean.clean : ev.scores.reduce((a, b) => a + b, 0) / ev.scores.length;
  lastEval = ev;
  hTrain.push(ev.trainLoss); hTest.push(ev.testLoss);
  if (hTrain.length > 320) { hTrain.shift(); hTest.shift(); }
}

/* ------------------------------------------------------------ latent */
/** Projects codes to 2D: as they are for k ≤ 2, else onto the two main directions (PCA). */
function project2(codes) {
  const k = codes[0].length;
  if (k <= 2) return { pts: codes.map((c) => [c[0], k > 1 ? c[1] : 0]), how: k === 1 ? 'k = 1: the code on the x axis' : 'k = 2: the code itself' };
  const n = codes.length, mean = new Float64Array(k);
  codes.forEach((c) => { for (let i = 0; i < k; i++) mean[i] += c[i] / n; });
  const C = Array.from({ length: k }, () => new Float64Array(k));
  codes.forEach((c) => { for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) C[i][j] += (c[i] - mean[i]) * (c[j] - mean[j]) / n; });
  const power = (excl) => {
    let v = Array.from({ length: k }, () => Math.random() - 0.5);
    for (let it = 0; it < 60; it++) {
      if (excl) { const d = v.reduce((s, x, i) => s + x * excl[i], 0); v = v.map((x, i) => x - d * excl[i]); }
      const w = v.map((_, i) => C[i].reduce((s, x, j) => s + x * v[j], 0));
      const nm = Math.hypot(...w) || 1;
      v = w.map((x) => x / nm);
    }
    return v;
  };
  const v1 = power(null), v2 = power(v1);
  const pts = codes.map((c) => {
    let a = 0, b = 0;
    for (let i = 0; i < k; i++) { a += (c[i] - mean[i]) * v1[i]; b += (c[i] - mean[i]) * v2[i]; }
    return [a, b];
  });
  return { pts, how: 'k = ' + k + ': the two main directions of the codes (PCA)', v1, v2, mean };
}

function drawLatent() {
  const cv = $('latent'), w = cv.clientWidth || 260, h = 210;
  const ctx = dpiSetup(cv, w, h);
  ctx.clearRect(0, 0, w, h);
  if (!lastEval) return;
  const pr = project2(lastEval.codes);
  model.reconstruct(state.probe);
  const probeCode = Float32Array.from(model.code);       // the code of the inspected window
  let pp = null;
  if (pr.v1) {
    let a = 0, b = 0;
    for (let i = 0; i < probeCode.length; i++) { a += (probeCode[i] - pr.mean[i]) * pr.v1[i]; b += (probeCode[i] - pr.mean[i]) * pr.v2[i]; }
    pp = [a, b];
  } else pp = [probeCode[0], probeCode.length > 1 ? probeCode[1] : 0];
  const k1 = probeCode.length === 1;
  const all = pr.pts.concat([pp]);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  all.forEach(([a, b]) => { x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, b); y1 = Math.max(y1, b); });
  if (k1) { y0 = -1; y1 = 1; }
  const pad = 14, sx = (a) => pad + (a - x0) / ((x1 - x0) || 1) * (w - 2 * pad), sy = (b) => h - 22 - (b - y0) / ((y1 - y0) || 1) * (h - 22 - pad);
  ctx.strokeStyle = '#eef1f4';
  ctx.strokeRect(0.5, 0.5, w - 1, h - 21);
  // with k = 1 the classes get their own row, so they do not hide each other
  const rows = {};
  state.classes.forEach((id, i) => { rows[CLASS_INDEX[id]] = state.classes.length > 1 ? -0.8 + 1.6 * i / (state.classes.length - 1) : 0; });
  pr.pts.forEach(([a, b], i) => {
    const c = CLASSES[lastEval.ys[i]];
    ctx.fillStyle = c.color + 'cc';
    ctx.beginPath();
    ctx.arc(sx(a), sy(k1 ? rows[lastEval.ys[i]] : b), 2.6, 0, 6.284);
    ctx.fill();
  });
  ctx.strokeStyle = '#1d2b38'; ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(sx(pp[0]), sy(k1 ? rows[CLASS_INDEX[state.probeClassId]] || 0 : pp[1]), 5.5, 0, 6.284);
  ctx.stroke();
  ctx.fillStyle = '#98a2ad'; ctx.font = '10px system-ui,sans-serif';
  ctx.fillText((state.kind === 'mae' ? 'mean of the visible tokens (16 numbers), PCA' : pr.how) + ' · ring: the inspected window', 2, h - 6);
}

/* ------------------------------------------------------------ rendering */
function renderMetrics() {
  if (!lastEval) return;
  const ev = lastEval;
  $('lossTrain').textContent = ev.trainLoss.toFixed(5);
  $('lossTest').textContent = ev.testLoss.toFixed(5);
  $('aucMean').textContent = ev.meanAuc == null ? '—' : ev.meanAuc.toFixed(3);
  $('locRow').classList.toggle('hidden', !ev.loc);
  if (ev.loc) $('locVal').textContent = (ev.loc.hit * 100).toFixed(0) + '%';
  $('snrRow').classList.toggle('hidden', !ev.snr);
  if (ev.snr) $('snrVal').textContent = ev.snr.in.toFixed(1) + ' → ' + ev.snr.out.toFixed(1) + ' dB (moving avg ' + ev.snr.base.toFixed(1) + ')';
  $('epoch').textContent = String(Math.floor(state.epoch)).padStart(6, '0');
  renderRunTarget();

  const lc = $('loss'), w = lc.clientWidth || 260;
  drawLossChart(dpiSetup(lc, w, 92), w, 92, hTrain, hTest);

  let html = '<table class="mtab aetab"><thead><tr><th>class</th><th>mean error</th><th>AUC vs clean</th></tr></thead><tbody>';
  state.classes.forEach((id) => {
    const c = CLASSES[CLASS_INDEX[id]], a = ev.auc[id];
    html += '<tr><td class="ch"><span class="chip" style="background:' + c.color + '"></span> ' + c.short + '</td>' +
      '<td>' + (ev.mean[id] == null ? '—' : ev.mean[id].toFixed(5)) + '</td><td>' +
      (id === 'clean' ? '<span style="color:#98a2ad">reference</span>' : a == null ? '—'
        : '<span class="aucbar"><i style="width:' + Math.round(a * 100) + '%;background:' + (a > 0.9 ? '#2e9e5b' : a > 0.7 ? '#c2760f' : '#e0342b') +
          '"></i></span> ' + a.toFixed(3)) + '</td></tr>';
  });
  $('aucTable').innerHTML = html + '</tbody></table>';
  drawLatent();
}

function renderNet() {
  const wrap = $('netWrap');
  const cssW = Math.max(420, wrap.clientWidth - 2);
  // MAE: with a sample of the output selected, show the pass that hides its patch
  if (model.kind === 'mae' && state.selected && state.selected.type === 'output') {
    model.viewGroup = Math.floor(state.tPos / TF_PATCH) % MAE_GROUPS;
  }
  recon = Float32Array.from(model.forward(state.probe, true, false));
  const masked = model.kind === 'mae' ? model.stages[0].masked : null;
  layout = aeLayout(model, cssW);
  netCtx = dpiSetup($('net'), layout.width, layout.height);
  aeDraw(netCtx, {
    model, layout, probe: state.probe, recon, hover: state.hover, sel: state.selected, tPos: state.tPos,
    mode: state.mode, mask: state.probeMeta && state.probeMeta.mask, masked,
  });
  const rc = $('recon'), w = rc.clientWidth || 260;
  const ctx = dpiSetup(rc, w, 110);
  ctx.clearRect(0, 0, w, 110);
  aeDrawRecon(ctx, 0, 2, w, 106, state.probe, recon, state.probeMeta && state.probeMeta.mask);
  $('reconText').innerHTML = '<b style="color:' + CLASSES[CLASS_INDEX[state.probeClassId]].color + '">' +
    CLASSES[CLASS_INDEX[state.probeClassId]].name + '</b>' + (state.classes.includes(state.probeClassId) ? '' : ' (not in the data)') +
    ' · error ' + mse(state.probe, recon).toFixed(5) +
    (state.trainOn === 'clean' && state.probeClassId !== 'clean' ? ' — the network has never seen this kind of window' : '');
  renderMath();
}

/* -------------------------------------------------- the selected node */
/** The products behind one convolution output, as the classifier page computes them. */
function aeConvTerms(conv, xin, Lin, co, t) {
  const K = conv.k, cin = conv.cin, terms = [];
  let sum = 0;
  for (let ci = 0; ci < cin; ci++) {
    const row = [];
    let sub = 0;
    for (let j = 0; j < K; j++) {
      const idx = t + conv.tapOffset(j), outside = idx < 0 || idx >= Lin;
      const xv = outside ? 0 : xin[ci * Lin + idx], w = conv.W[(co * cin + ci) * K + j];
      sub += w * xv;
      row.push({ idx, xv, w, p: w * xv, outside });
    }
    sum += sub;
    terms.push({ ci, row, sub });
  }
  return { z: sum + conv.b[co], sum, bias: conv.b[co], terms, Lin, cin, K, pad: conv.pad, conv };
}

const ACTS = {
  relu: { name: 'ReLU', f: (z) => (z > 0 ? z : 0), expr: (z) => 'max(0, ' + n3(z) + ')' },
  tanh: { name: 'Tanh', f: Math.tanh, expr: (z) => 'tanh(' + n3(z) + ')' },
  linear: { name: 'linear', f: (z) => z, expr: (z) => 'no activation: ' + n3(z) },
};

/** A dense unit: inputs × weights + bias → activation. */
function denseDetail(dense, x, j, act) {
  const w = dense.W.subarray(j * dense.nin, (j + 1) * dense.nin);
  let z = dense.b[j];
  for (let i = 0; i < dense.nin; i++) z += w[i] * x[i];
  return { x, w, b: dense.b[j], z, a: ACTS[act].f(z), nin: dense.nin };
}

function denseTable(d, name) {
  const order = Array.from({ length: d.nin }, (_, i) => i).sort((a, b) => Math.abs(d.w[b] * d.x[b]) - Math.abs(d.w[a] * d.x[a]));
  const rows = order.slice(0, 12).sort((a, b) => a - b);
  let shown = 0;
  let html = '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th><th>value</th><th>weight</th><th>product</th></tr></thead><tbody>';
  rows.forEach((i) => {
    const p = d.w[i] * d.x[i];
    shown += p;
    html += '<tr><td class="ch">' + name(i) + '</td><td>' + n3(d.x[i]) + '</td><td style="color:' + wColor(d.w[i]) + '">' + n3(d.w[i]) +
      '</td><td class="sum">' + n4(p) + '</td></tr>';
  });
  if (rows.length < d.nin) html += '<tr><td class="ch">the other ' + (d.nin - rows.length) + '</td><td></td><td></td><td class="sum">' + n4(d.z - d.b - shown) + '</td></tr>';
  return html + '</tbody></table></div>';
}

let lastMathAt = 0, flowShown = null;

function renderMath(force) {
  const now = performance.now();
  if (!force && now - lastMathAt < 250) return;
  lastMathAt = now;
  if (!model || !recon) return;
  const r = selectionMath();
  if (r.flow !== flowShown) { $('flowBody').innerHTML = r.flow; flowShown = r.flow; }
  $('flowTitle').textContent = r.title;
  $('mathTitle').textContent = r.title;
  $('mathBody').innerHTML = r.html;
  [['flowTpos', 'flowTval'], ['tpos', 'tposVal']].forEach(([s, v]) => {
    const sl = $(s);
    sl.disabled = r.tMax == null;
    if (r.tMax != null) { sl.max = r.tMax; sl.value = Math.min(state.tPos, r.tMax); }
    $(v).textContent = r.tMax == null ? '—' : r.tLabel;
  });
}

/** The bottleneck stage; its snapshot is the code of the inspected window. */
function codeStage() { return model.stages.find((s) => s.kind === 'code'); }

function clampT(max) { state.tPos = Math.max(0, Math.min(max, state.tPos)); return state.tPos; }

/** Builds the cell diagram, the title and the arithmetic for whatever is selected. */
/** Labels and tooltips of the pipeline overview for each kind. */
function pipelineText() {
  const cell = state.cell.toUpperCase();
  return {
    mlp: { enc: 'dense ' + WIN + ' → ' + state.hidden + ' → ' + state.k, dec: 'dense ' + state.k + ' → ' + state.hidden + ' → ' + WIN,
      encTip: 'a dense layer of ' + state.hidden + ' tanh units over all 128 samples, then a dense layer to the code',
      decTip: 'a dense layer of ' + state.hidden + ' tanh units, then a dense layer to 128 samples' },
    conv: { enc: '3 × conv + pool, dense', dec: 'dense, 3 × up + conv',
      encTip: 'three convolutions with ReLU and max pooling (128 → 64 → 32 → 16 positions), then a dense layer to the code',
      decTip: 'a dense layer back to 16 positions, then three upsample → convolution steps to 128 samples' },
    lstm: { enc: cell + ' ' + state.units + ', last state', dec: cell + ' ' + state.units + ', readout',
      encTip: 'a ' + cell + ' of ' + state.units + ' units reads the 128 samples in order; its state after the last one is mapped to the code',
      decTip: 'a ' + cell + ' gets the code at every one of the 128 steps; a linear readout turns its state into the sample' },
    mae: { enc: '16 patches, ' + state.maeLayers + ' × attention', dec: 'linear head per token',
      encTip: 'every patch of 8 samples becomes a token; hidden patches get the [MASK] vector; ' + state.maeLayers + ' encoder layer(s) of self-attention',
      decTip: 'one linear map turns every token back into 8 samples — only the hidden patches count' },
  }[state.kind];
}

function selectionMath() {
  const sel = state.selected, x = state.probe;
  const pt = pipelineText(), mae = state.kind === 'mae';

  if (!sel || sel.type === 'input') {
    const t = clampT(WIN - 1);
    const code = mae ? Array.from(model.code) : Array.from(codeStage().snapshot);
    const flow = Flow.aePipeline({
      x, y: recon, code, t, mse: mse(x, recon), mask: state.probeMeta && state.probeMeta.mask,
      encLabel: pt.enc, decLabel: pt.dec, encTip: pt.encTip, decTip: pt.decTip, vae: state.vae && !mae,
      codeLabel: mae ? 'mean of the visible tokens' : null,
      codeTip: mae ? 'no bottleneck in a masked autoencoder: this summary only feeds the latent plot' : null,
    });
    let html = '<h4>1 · What the network does</h4>';
    if (mae) {
      html += '<div class="formula">x (16 patches of 8) → hide some patches → Transformer → a prediction for every patch → x̂ &nbsp;&nbsp; ' +
        'loss = mean (x̂ − ' + (state.denoise ? 'x<sub>clean</sub>' : 'x') + ')² <b>over the hidden patches only</b></div>';
      html += '<div class="formula" style="margin-top:6px"><span class="op">Training hides ' + Math.round(state.maskRatio * 100) +
        '% of the patches at random. To score a window the network runs four times, hiding patches 0, 4, 8, 12 — then 1, 5, 9, 13 — and so on, ' +
        'so every patch is predicted once from its context and never copied.</span></div>';
    } else {
      html += '<div class="formula">x (' + WIN + ' numbers) → encoder → z (' + state.k + ' number' + (state.k > 1 ? 's' : '') + ') → decoder → x̂ (' + WIN +
        ' numbers) &nbsp;&nbsp; loss = mean (x̂ − ' + (state.denoise ? 'x<sub>clean</sub>' : 'x') + ')²' +
        (state.vae ? ' + β · KL( N(μ, σ²) ‖ N(0, 1) )' : '') + '</div>';
    }
    html += '<div class="formula" style="margin-top:6px"><span class="op">' + (state.trainOn === 'clean'
      ? 'Trained on clean mains only: the decoder learns to rebuild what normal looks like. A window it cannot rebuild is unusual — ' +
        'its error is the anomaly score, and no window needed a label.'
      : 'Trained on every checked class: the code has to describe the disturbances too, so they rebuild well and the score ' +
        'no longer singles them out.') + '</span></div>';
    html += '<h4>2 · This window</h4>';
    html += '<div class="formula">' + (mae ? '' : 'z = [ ' + code.map(n3).join(', ') + ' ] &nbsp;&nbsp; ') + 'error = <span class="res">' +
      mse(x, recon).toFixed(5) + '</span>  <span class="op">at t = ' + t + ': x = ' + n3(x[t]) + ', x̂ = ' + n3(recon[t]) + '</span></div>';
    return { flow, title: sel ? 'Input · the whole autoencoder' : 'The whole autoencoder', html, tMax: WIN - 1, tLabel: 't = ' + t };
  }

  if (sel.type === 'output') return mae ? maeOutputMath() : outputMath();
  const st = model.stages[sel.stage];
  if (st.rnn) return seqRnnMath(st, sel);
  if (st.tfembed) return maeEmbedMath(st, sel);
  if (st.tflayer) return maeAttnMath(st, sel);
  if (st.kind === 'units' || st.kind === 'code' || (st.kind === 'map' && st.dense)) return denseStageMath(st, sel);
  return convStageMath(st, sel);
}

/* ------------------------------------------------- recurrent stages */
const AE_GATES = {
  lstm: [['i', 'σ', 'input gate'], ['f', 'σ', 'forget gate'], ['o', 'σ', 'output gate'], ['g', 'tanh', 'candidate']],
  gru: [['z', 'σ', 'update gate'], ['r', 'σ', 'reset gate'], ['n', 'tanh', 'candidate']],
  rnn: [['h', 'tanh', 'state']],
};

/** One step of one unit of a recurrent pass, in the form Flow.lstm / Flow.gru draw. */
function rnnStepDetail(dir, u, t) {
  const H = dir.H, G = dir.G, D = dir.D, L = dir.L;
  const hprev = new Float32Array(H);
  for (let v = 0; v < H; v++) hprev[v] = dir.hs[t * H + v];
  const gates = [], wx = [], wh = [], bias = [];
  for (let g = 0; g < G; g++) {
    gates.push(dir.gs[t * G * H + g * H + u]);
    bias.push(dir.px.b[g * H + u]);
    wx.push(dir.px.W.slice((g * H + u) * D, (g * H + u + 1) * D));
    wh.push(dir.ph.W.slice((g * H + u) * H, (g * H + u + 1) * H));
  }
  const xv = new Float32Array(D);
  for (let d = 0; d < D; d++) xv[d] = dir.x[d * L + t];
  return {
    dir, u, back: false, s: t, t, H, G, D, L, kind: dir.kind, hprev, gates, wx, wh, bias, xv,
    h: dir.hs[(t + 1) * H + u],
    c: dir.cs ? dir.cs[(t + 1) * H + u] : null, cprev: dir.cs ? dir.cs[t * H + u] : null,
    q: dir.qs ? dir.qs[t * H + u] : null,
  };
}

function seqRnnMath(st, sel) {
  const u = Math.max(0, sel.ch), t = clampT(WIN - 1), d = rnnStepDetail(st.rnn, u, t), enc = st.part === 'enc';
  const gates = AE_GATES[d.kind];
  const formulas = {
    rnn: 'h<sub>t</sub> = tanh( W<sub>x</sub>·x<sub>t</sub> + W<sub>h</sub>·h<sub>t−1</sub> + b )',
    gru: 'z, r = σ(·) &nbsp; n = tanh( W<sub>n</sub>x<sub>t</sub> + r ⊙ (U<sub>n</sub>h<sub>t−1</sub>) + b<sub>n</sub> ) &nbsp;→&nbsp; h<sub>t</sub> = (1−z)⊙n + z⊙h<sub>t−1</sub>',
    lstm: 'i, f, o = σ(·) &nbsp; g = tanh(·) &nbsp;→&nbsp; c<sub>t</sub> = f⊙c<sub>t−1</sub> + i⊙g &nbsp;→&nbsp; h<sub>t</sub> = o⊙tanh(c<sub>t</sub>)',
  };
  let html = '<h4>1 · The ' + (enc ? 'encoder' : 'decoder') + ' cell — ' + d.kind.toUpperCase() + '</h4><div class="formula">' + formulas[d.kind] + '</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">' + (enc
    ? 'x<sub>t</sub> is sample t of the window. Only the state after the last sample (t = ' + (WIN - 1) + ') goes on to the code — ' +
      'everything the decoder learns about the window has to survive the trip through these 128 steps.'
    : 'x<sub>t</sub> is the code z = [ ' + Array.from(d.xv).map(n3).join(', ') + ' ], the same at every step: the cell has to unroll the window ' +
      'from it on its own, keeping track of where in the cycle it is through its state.') + '</span></div>';
  html += '<h4>2 · The gates at t = ' + t + '</h4><div class="scrollx"><table class="mtab"><thead><tr><th>gate</th>' +
    '<th>W<sub>x</sub>·x<sub>t</sub></th><th>W<sub>h</sub>·h<sub>t−1</sub></th><th>bias</th><th>pre-activation</th><th>value</th></tr></thead><tbody>';
  gates.forEach(([nm, fn, desc], g) => {
    let ix = 0, ih = 0;
    for (let i = 0; i < d.D; i++) ix += d.wx[g][i] * d.xv[i];
    for (let v = 0; v < d.H; v++) ih += d.wh[g][v] * d.hprev[v];
    const cand = d.kind === 'gru' && g === 2, rec = cand ? d.gates[1] * d.q : ih;
    html += '<tr><td class="ch">' + nm + ' = ' + fn + '(·) <span style="color:#98a2ad">' + desc + '</span></td><td>' + n4(ix) + '</td><td>' + n4(rec) +
      (cand ? ' <span style="color:#98a2ad">= r·' + n3(d.q) + '</span>' : '') + '</td><td>' + n3(d.bias[g]) + '</td><td>' + n4(ix + rec + d.bias[g]) +
      '</td><td class="sum">' + n4(d.gates[g]) + '</td></tr>';
  });
  html += '</tbody></table></div>';
  let upd = '';
  if (d.kind === 'lstm') upd = 'c = f·c<sub>t−1</sub> + i·g = ' + n3(d.gates[1]) + '·' + n3(d.cprev) + ' + ' + n3(d.gates[0]) + '·' + n3(d.gates[3]) + ' = ' + n4(d.c) +
    ' &nbsp; h = o·tanh(c) = ' + n3(d.gates[2]) + '·' + n3(Math.tanh(d.c));
  else if (d.kind === 'gru') upd = 'h = (1−z)·n + z·h<sub>t−1</sub> = ' + n3(1 - d.gates[0]) + '·' + n3(d.gates[2]) + ' + ' + n3(d.gates[0]) + '·' + n3(d.hprev[u]);
  else upd = 'h = tanh(·)';
  const shown = st.snapshot[u * st.L + t];
  html += '<div class="formula" style="margin-top:8px">' + upd + ' = <span class="res">' + n4(d.h) + '</span>  <span class="op">check: the box holds ' +
    n4(shown) + (Math.abs(shown - d.h) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  if (!enc) {
    const w = model.head.W[u], contrib = w * d.h;
    html += '<div class="formula" style="margin-top:6px"><span class="op">Readout: x̂[' + t + '] = Σ<sub>u</sub> w<sub>u</sub>·h<sub>u</sub>[' + t + '] + b; ' +
      'this unit adds ' + n3(w) + ' · ' + n3(d.h) + ' = ' + n4(contrib) + '</span></div>';
  }
  const title = (enc ? 'Encoder' : 'Decoder') + ' · unit ' + (u + 1) + ' · t = ' + t + ' · ' + d.kind.toUpperCase();
  return { flow: Flow[d.kind](d), title, html, tMax: WIN - 1, tLabel: 't = ' + t };
}

/* ------------------------------------------------------ MAE stages */
function tokLabel(t) {
  return 'token ' + t + ' (' + (t * TF_PATCH / SR * 1000).toFixed(1) + '–' + ((t + 1) * TF_PATCH / SR * 1000).toFixed(1) + ' ms)';
}

function maeEmbedMath(st, sel) {
  const ch = Math.max(0, sel.ch), T = st.L, d = model.d, t = clampT(T - 1), start = t * TF_PATCH;
  const hidden = !!st.masked[t], pos = model.pos.W[t * d + ch], shown = st.snapshot[ch * T + t];
  let flow, html, out;
  if (hidden) {
    const m = model.mtok.W[ch];
    out = m + pos;
    flow = Flow.maeMask({ mtok: Array.from(model.mtok.W), m, pos, out, start }, { t, ch });
    html = '<h4>1 · A hidden patch</h4><div class="formula">token[' + t + '][' + (ch + 1) + '] = [MASK][' + (ch + 1) + '] + pos[' + t + '][' + (ch + 1) + '] = ' +
      n4(m) + ' + ' + n4(pos) + ' = <span class="res">' + n4(out) + '</span></div><div class="formula" style="margin-top:6px"><span class="op">' +
      'Samples ' + start + '–' + (start + 7) + ' never reach the network in this pass. All hidden patches share one learned vector; only the ' +
      'position vector tells them apart, so attention has to work out from the visible neighbours what belongs here.</span></div>';
  } else {
    const E = model.embed, xs = [], ws = [];
    let sum = 0;
    for (let j = 0; j < TF_PATCH; j++) { xs.push(state.probe[start + j]); ws.push(E.W[ch * TF_PATCH + j]); sum += xs[j] * ws[j]; }
    const b = E.b[ch], z = sum + b;
    out = z + pos;
    flow = Flow.tfEmbed({ xs, ws, b, pos, sum, z, out, start }, { t, ch });
    html = '<h4>1 · From patch to token</h4><div class="formula">token[' + t + '][' + (ch + 1) + '] = Σ<sub>j</sub> W<sub>e</sub>[' + (ch + 1) + '][j] · x[' + start +
      ' + j] + b + pos = ' + n4(sum) + ' + ' + n4(b) + ' + ' + n4(pos) + ' = <span class="res">' + n4(out) + '</span></div>' +
      '<div class="formula" style="margin-top:6px"><span class="op">A visible patch: its 8 samples go in through the same linear map every patch uses.</span></div>';
  }
  html += '<div class="formula" style="margin-top:6px"><span class="op">check: the box holds ' + n4(shown) + (Math.abs(shown - out) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  return { flow, title: 'Embedding · dimension ' + (ch + 1) + ' · ' + tokLabel(t) + (hidden ? ' · hidden' : ''), html, tMax: T - 1, tLabel: tokLabel(t) };
}

function maeAttnMath(st, sel) {
  const ch = Math.max(0, sel.ch), T = st.L, d = model.d, t = clampT(T - 1);
  const tr = st.layer.trace, at = st.layer.attn, H = at.H, dh = at.dh, head = Math.floor(ch / dh);
  const vec = (arr) => Array.from(arr.subarray(t * d, (t + 1) * d));
  const row = (arr) => Array.from(arr.subarray((head * T + t) * T, (head * T + t) * T + T));
  const flow = Flow.attention({
    x: vec(tr.X), n1: vec(tr.n1), a: row(tr.A), scores: row(tr.S), head, o: vec(tr.O),
    attn: vec(tr.a), h: vec(tr.h), n2: vec(tr.n2), z2: vec(tr.z2), out: vec(tr.out), causal: false,
  }, { li: sel.stage, ch, t });
  let html = '<h4>1 · The encoder layer</h4><div class="formula">h = x + W<sub>o</sub>·Attention(LN<sub>1</sub>(x)) &nbsp;&nbsp; y = h + W<sub>2</sub>·ReLU(W<sub>1</sub>·LN<sub>2</sub>(h))' +
    ' &nbsp;&nbsp; a<sub>tj</sub> = softmax<sub>j</sub>( q<sub>t</sub>·k<sub>j</sub> / √' + dh + ' )</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">' + (st.masked[t]
    ? 'Token ' + t + ' is hidden: everything it ends up knowing about its 8 samples comes from the tokens it attends to.'
    : 'Token ' + t + ' is visible. Hidden tokens read from tokens like this one.') + '</span></div>';
  html += '<h4>2 · Where token ' + t + ' looks (head ' + (head + 1) + ' of ' + H + ')</h4><div class="scrollx"><table class="mtab"><thead><tr><th>token j</th>' +
    '<th>score</th><th>weight a<sub>tj</sub></th><th>value v<sub>j</sub>[' + (ch + 1) + ']</th><th>product</th></tr></thead><tbody>';
  let osum = 0;
  for (let j = 0; j < T; j++) {
    const a = tr.A[(head * T + t) * T + j], s = tr.S[(head * T + t) * T + j], v = tr.V[j * d + ch];
    osum += a * v;
    html += '<tr' + (j === t ? ' style="background:#eef4fd"' : '') + '><td class="ch">' + j + (st.masked[j] ? ' <span style="color:#e0342b">hidden</span>' : '') +
      (j === t ? ' ← itself' : '') + '</td><td>' + n3(s) + '</td><td class="sum">' + n3(a) + '</td><td>' + n3(v) + '</td><td>' + n4(a * v) + '</td></tr>';
  }
  html += '</tbody></table></div>';
  const at_ = t * d + ch, drawn = st.snapshot[ch * T + t];
  html += '<div class="formula" style="margin-top:8px">o<sub>t</sub>[' + (ch + 1) + '] = Σ a·v = ' + n4(osum) + ' &nbsp;→ W<sub>o</sub>, + x, FFN, + h &nbsp;→ y = <span class="res">' +
    n4(tr.out[at_]) + '</span>  <span class="op">check: the box holds ' + n4(drawn) +
    (Math.abs(drawn - tr.out[at_]) < 1e-4 && Math.abs(osum - tr.O[at_]) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  return { flow, title: 'Encoder ' + sel.stage + ' · dimension ' + (ch + 1) + ' · ' + tokLabel(t) + (st.masked[t] ? ' · hidden' : ''), html, tMax: T - 1, tLabel: tokLabel(t) };
}

function maeOutputMath() {
  const t = clampT(WIN - 1), x = state.probe, d = model.d, p = Math.floor(t / TF_PATCH), j = t % TF_PATCH;
  const n = model.N.subarray(p * d, (p + 1) * d), w = model.head.W.subarray(j * d, (j + 1) * d), b = model.head.b[j];
  let z = b;
  for (let i = 0; i < d; i++) z += w[i] * n[i];
  const det = { x: n, w, b, z, a: z, nin: d };
  const name = (i) => 'tok[' + p + '][' + (i + 1) + ']';
  const flow = Flow.dense(det, { li: 1, ch: t, t: null, act: 'linear', actName: 'linear', actExpr: ACTS.linear.expr(z), name });
  let html = '<h4>1 · Sample ' + t + ' = slot ' + j + ' of patch ' + p + '</h4><div class="formula">x̂[' + t + '] = Σ W<sub>head</sub>[' + j + '][i] · LN(token ' + p +
    ')[i] + b <span class="op">— the pass shown hides patch ' + p + ' (and every fourth patch with it), so this is a prediction, not a copy</span></div>' + denseTable(det, name);
  html += '<div class="formula" style="margin-top:8px">x̂[' + t + '] = <span class="res">' + n4(z) + '</span>  against x[' + t + '] = ' + n4(x[t]) + '  → error² ' +
    n4((z - x[t]) ** 2) + '  <span class="op">check: ' + (Math.abs(z - recon[t]) < 1e-4 ? '✓ matches' : '⚠ mismatch') + '</span></div>';
  return { flow, title: 'Output · sample ' + t + ' · predicted from patch ' + p + ' hidden', html, tMax: WIN - 1, tLabel: 't = ' + t };
}

/* ------------------------------------------------------- generation */
/** How much of a window's energy a 50 Hz sine explains: 1 for a clean sine. */
function sineShare(x) {
  let a = 0, b = 0, e = 0;
  for (let t = 0; t < x.length; t++) {
    const ph = 2 * Math.PI * F0 * t / SR;
    a += x[t] * Math.sin(ph); b += x[t] * Math.cos(ph); e += x[t] * x[t];
  }
  a *= 2 / x.length; b *= 2 / x.length;
  return { share: Math.min(1, (a * a + b * b) * x.length / 2 / Math.max(1e-12, e)), amp: Math.hypot(a, b) };
}

/** Decodes codes drawn from N(0, 1): what a VAE is trained to make work. */
function generate() {
  if (!model.decode) return;
  const cv = $('gen'), w = cv.clientWidth || 260, h = 120;
  const ctx = dpiSetup(cv, w, h);
  ctx.clearRect(0, 0, w, h);
  const outs = [];
  for (let n = 0; n < 4; n++) {
    const z = new Float32Array(state.k);
    for (let i = 0; i < state.k; i++) z[i] = randn();
    outs.push(Float32Array.from(model.decode(z)));
  }
  let m = 1e-6;
  outs.forEach((y) => { m = Math.max(m, maxAbs(y, 0, WIN)); });
  const cw = w / 2, ch = h / 2;
  outs.forEach((y, n) => {
    const x0 = (n % 2) * cw, y0 = Math.floor(n / 2) * ch;
    ctx.strokeStyle = '#eef1f4'; ctx.strokeRect(x0 + 2.5, y0 + 2.5, cw - 5, ch - 5);
    drawWave(ctx, x0 + 5, y0 + 5, cw - 10, ch - 10, y, 0, WIN, m);
  });
  // a larger sample for the numbers
  let share = 0, amp = 0;
  const N = 64;
  for (let n = 0; n < N; n++) {
    const z = new Float32Array(state.k);
    for (let i = 0; i < state.k; i++) z[i] = randn();
    const s = sineShare(model.decode(z));
    share += s.share / N; amp += s.amp / N;
  }
  let codeSd = '';
  if (lastEval) {
    const k = state.k, cs = lastEval.codes, sd = [];
    for (let i = 0; i < k; i++) {
      let mu = 0, v = 0;
      cs.forEach((c) => { mu += c[i] / cs.length; });
      cs.forEach((c) => { v += (c[i] - mu) ** 2 / cs.length; });
      sd.push(Math.sqrt(v));
    }
    codeSd = ' The codes of the test windows have a spread (sd) of ' + sd.slice(0, 4).map((v) => v.toFixed(2)).join(', ') + (k > 4 ? ', …' : '') + '.';
  }
  $('genText').innerHTML = (state.vae ? '<b>VAE</b>, β = ' + state.beta : '<b>Plain autoencoder</b> — nothing made N(0, 1) mean anything to it') +
    '. Over 64 random codes, a 50 Hz sine explains <b>' + (share * 100).toFixed(0) + '%</b> of the energy on average (a clean window: over 99%), ' +
    'mean amplitude ' + amp.toFixed(2) + ' (clean mains: 1).' + codeSd;
}

function denseStageMath(st, sel) {
  const ch = Math.max(0, sel.ch);
  const isCode = st.kind === 'code', isMap = st.kind === 'map';
  let j = ch, t = null, tMax = null;
  if (isMap) { tMax = st.L - 1; t = clampT(tMax); j = ch * st.L + t; }
  const act = isCode ? 'linear' : st.act || 'relu';
  const first = !isMap && !isCode && st.part === 'enc';
  const d = denseDetail(st.dense, st.input, j, act);
  const name = first ? (i) => 'x[' + i + ']'
    : st.part === 'code' && state.kind === 'conv' ? (i) => 'e[' + (Math.floor(i / 16) + 1) + '][' + (i % 16) + ']'
      : st.part === 'code' ? (i) => 'h' + (i + 1) : (i) => 'z' + (i + 1);
  if (first) tMax = WIN - 1;
  const tt = first ? clampT(WIN - 1) : t;
  const flow = Flow.dense(d, { li: first ? 0 : 1, ch: j, t: first ? tt : null, act, actName: ACTS[act].name, actExpr: ACTS[act].expr(d.z), name });
  const shown = isMap ? st.snapshot[ch * st.L + t] : st.snapshot[ch];
  const title = (isCode ? 'Code · z' + (ch + 1) : isMap ? st.label.split(' ·')[0] + ' · channel ' + (ch + 1) + ' · t = ' + t
    : (st.part === 'enc' ? 'Encoder' : 'Decoder') + ' · unit ' + (ch + 1));
  let html = '<h4>1 · Weighted sum</h4><div class="formula">' + (isCode ? 'z' : 'a') + ' = ' + (act === 'linear' ? '' : ACTS[act].name + '( ') +
    'Σ w · input + b' + (act === 'linear' ? '' : ' )') + ' &nbsp; <span class="op">' + d.nin + ' inputs' +
    (isCode ? ' — the code is a plain linear map of the last encoder layer' : '') + (state.vae && isCode ? '; this is μ, the mean of the code' : '') + '</span></div>';
  html += denseTable(d, name);
  html += '<div class="formula" style="margin-top:8px">Σ = ' + n4(d.z - d.b) + ' + b ' + n4(d.b) + ' = ' + n4(d.z) + '  →  ' + ACTS[act].expr(d.z) +
    ' = <span class="res">' + n4(d.a) + '</span>  <span class="op">check: the box holds ' + n4(shown) +
    (Math.abs(shown - d.a) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  return { flow, title, html, tMax, tLabel: first ? 'input t = ' + tt : 't = ' + t };
}

function convStageMath(st, sel) {
  const ch = sel.ch, step = st.step, enc = st.part === 'enc';
  const tMax = st.L - 1, tp = clampT(tMax);
  let flow, html, a, t;
  if (enc) {
    // the box holds the pooled map: compute both positions the pool compares
    const even = 2 * tp, c0 = aeConvTerms(step.conv, st.input, st.Lin, ch, even), c1 = aeConvTerms(step.conv, st.input, st.Lin, ch, even + 1);
    const v0 = Math.max(0, c0.z), v1 = Math.max(0, c1.z);
    a = v0; t = even;
    flow = Flow.conv(c0, { t: even, li: sel.stage, ch, a: v0, act: 'relu', actName: 'ReLU', actExpr: 'a = max(0, ' + n3(c0.z) + ')',
      pool: { even, v0, v1, out: Math.max(v0, v1), tp } });
    html = convHtml(c0, 'ReLU', v0) + '<div class="formula" style="margin-top:6px">max pool: max( ' + n3(v0) + ', ' + n3(v1) + ' ) = <span class="res">' +
      n4(Math.max(v0, v1)) + '</span>  <span class="op">check: the box holds ' + n4(st.snapshot[ch * st.L + tp]) +
      (Math.abs(st.snapshot[ch * st.L + tp] - Math.max(v0, v1)) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  } else {
    const c = aeConvTerms(step.conv, st.input, st.L, ch, tp);
    a = Math.max(0, c.z); t = tp;
    flow = Flow.conv(c, { t: tp, li: sel.stage, ch, a, act: 'relu', actName: 'ReLU', actExpr: 'a = max(0, ' + n3(c.z) + ')' });
    html = '<div class="formula"><span class="op">The map before this was upsampled ×2 (every value repeated), then convolved.</span></div>' +
      convHtml(c, 'ReLU', a) + '<div class="formula" style="margin-top:6px"><span class="op">check: the box holds ' +
      n4(st.snapshot[ch * st.L + tp]) + (Math.abs(st.snapshot[ch * st.L + tp] - a) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  }
  return { flow, title: st.label.split(' ·')[0] + ' · channel ' + (ch + 1) + ' · t = ' + tp, html, tMax, tLabel: 't = ' + tp };
}

function convHtml(c, actName, a) {
  let html = '<h4>1 · Convolution, K = ' + c.K + '</h4><div class="scrollx"><table class="mtab"><thead><tr><th>input</th>';
  for (let j = 0; j < c.K; j++) html += '<th>x[' + c.terms[0].row[j].idx + ']</th>';
  html += '<th>Σ</th></tr></thead><tbody>';
  c.terms.forEach((tr) => {
    html += '<tr><td class="ch">ch ' + (tr.ci + 1) + '</td>';
    tr.row.forEach((cell) => {
      html += '<td class="cell' + (cell.outside ? ' pad' : '') + '"><span class="xv">' + (cell.outside ? '0' : n3(cell.xv)) +
        '</span><span class="wv" style="color:' + wColor(cell.w) + '">×' + n3(cell.w) + '</span><span class="pv">' + n3(cell.p) + '</span></td>';
    });
    html += '<td class="sum">' + n3(tr.sub) + '</td></tr>';
  });
  html += '</tbody></table></div>';
  return html + '<div class="formula" style="margin-top:8px">Σ ' + n4(c.sum) + ' + b ' + n4(c.bias) + ' = ' + n4(c.z) + '  →  ' + actName +
    ' → <b>' + n4(a) + '</b></div>';
}

function outputMath() {
  const t = clampT(WIN - 1), x = state.probe;
  let flow, html;
  if (model.outDense) {
    const d = denseDetail(model.outDense, model.outInput, t, 'linear');
    flow = Flow.dense(d, { li: 1, ch: t, t: null, act: 'linear', actName: 'linear', actExpr: ACTS.linear.expr(d.z), name: (i) => 'h' + (i + 1) });
    html = '<h4>1 · Sample ' + t + ' of the reconstruction</h4><div class="formula">x̂[' + t + '] = Σ w · h + b <span class="op">— one ' +
      'output unit per sample, each with its own ' + d.nin + ' weights</span></div>' + denseTable(d, (i) => 'h' + (i + 1));
    html += '<div class="formula" style="margin-top:8px">x̂[' + t + '] = <span class="res">' + n4(d.z) + '</span>  against x[' + t + '] = ' + n4(x[t]) +
      '  → error² ' + n4((d.z - x[t]) ** 2) + '  <span class="op">check: ' + (Math.abs(d.z - recon[t]) < 1e-4 ? '✓ matches' : '⚠ mismatch') + '</span></div>';
  } else {
    const c = aeConvTerms(model.outStep.conv, model.outInput, WIN, 0, t);
    flow = Flow.conv(c, { t, li: model.stages.length, ch: 0, a: c.z, act: 'linear', actName: 'linear', actExpr: ACTS.linear.expr(c.z) });
    html = '<div class="formula"><span class="op">' + (state.kind === 'lstm'
      ? 'A linear readout — the same ' + state.units + ' weights at every step — turns the decoder state at step ' + t + ' into the sample.'
      : 'The last map was upsampled to 128 positions; one linear convolution turns its channels into the signal.') + '</span></div>' +
      convHtml(c, 'linear', c.z) + '<div class="formula" style="margin-top:6px">x̂[' + t + '] = ' + n4(c.z) + ' against x[' + t + '] = ' + n4(x[t]) +
      '  <span class="op">check: ' + (Math.abs(c.z - recon[t]) < 1e-4 ? '✓ matches' : '⚠ mismatch') + '</span></div>';
  }
  return { flow, title: 'Output · sample ' + t, html, tMax: WIN - 1, tLabel: 't = ' + t };
}

/* ---------------------------------------------------- code-size sweep */
const SWEEP_K = [1, 2, 3, 4, 8, 16];

/** Trains one fresh model per code size, in small slices so the page stays responsive. */
function runSweep() {
  if (job) return;
  setRunning(false);
  const epochs = +$('sweepEp').value, res = [];
  let ki = 0, m = null;
  job = { name: 'sweep' };
  const stepJob = () => {
    if (!m) { m = makeModel(SWEEP_K[ki]); m.ep = 0; }
    const t0 = performance.now();
    while (m.ep < epochs && performance.now() - t0 < 40) {
      const B = state.batch, ins = [], tgs = [];
      for (let b = 0; b < B; b++) { const [a, t] = pairOf(train, Math.floor(Math.random() * train.n)); ins.push(a); tgs.push(t); }
      m.trainBatch(ins, tgs, state.lr, state.l2);
      m.ep += B / train.n;
    }
    $('sweepMsg').textContent = 'k = ' + SWEEP_K[ki] + ': epoch ' + Math.floor(m.ep) + ' of ' + epochs + ' (' + (ki + 1) + ' of ' + SWEEP_K.length + ')';
    if (m.ep >= epochs) {
      const ev = evaluateModel(m, test, 400);
      res.push({ k: SWEEP_K[ki], mse: ev.mean.clean, auc: ev.meanAuc, params: m.paramCount() });
      drawSweep(res);
      m = null; ki++;
      if (ki >= SWEEP_K.length) { job = null; $('sweepMsg').textContent = 'done — ' + epochs + ' epochs per code size'; return; }
    }
    setTimeout(stepJob, 0);
  };
  stepJob();
}

function drawSweep(res) {
  const cv = $('sweepChart'), w = cv.clientWidth || 460, h = 190;
  const ctx = dpiSetup(cv, w, h);
  ctx.clearRect(0, 0, w, h);
  const pad = 34, X = (i) => pad + i * (w - 2 * pad) / (SWEEP_K.length - 1);
  const lm = res.map((r) => Math.log10(Math.max(1e-6, r.mse)));
  const lo = Math.min(...lm, -3), hi = Math.max(...lm, -1);
  const Ym = (v) => 14 + (hi - v) / ((hi - lo) || 1) * (h - 44), Ya = (a) => 14 + (1 - a) / 0.5 * (h - 44);
  ctx.strokeStyle = '#eef1f4'; ctx.strokeRect(pad, 14, w - 2 * pad, h - 44);
  ctx.fillStyle = '#98a2ad'; ctx.font = '10px system-ui,sans-serif';
  SWEEP_K.forEach((k, i) => ctx.fillText('k=' + k, X(i) - 9, h - 16));
  ctx.fillText('clean error (log)', 2, 10);
  ctx.fillText('AUC 0.5 … 1', w - 64, 10);
  const line = (vals, Y, color) => {
    ctx.beginPath();
    vals.forEach((v, i) => { if (v == null) return; if (i) ctx.lineTo(X(i), Y(v)); else ctx.moveTo(X(i), Y(v)); });
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    vals.forEach((v, i) => { if (v == null) return; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(i), Y(v), 3, 0, 6.284); ctx.fill(); });
  };
  line(lm, Ym, '#2b6cb0');
  line(res.map((r) => (r.auc == null ? null : Math.max(0.5, r.auc))), Ya, '#2e9e5b');
  let html = '<table class="mtab"><thead><tr><th>k</th><th>parameters</th><th>clean error</th><th>mean AUC</th></tr></thead><tbody>';
  res.forEach((r) => {
    html += '<tr><td class="ch">' + r.k + '</td><td>' + r.params + '</td><td>' + (r.mse == null ? '—' : r.mse.toFixed(5)) + '</td><td class="sum">' +
      (r.auc == null ? '—' : r.auc.toFixed(3)) + '</td></tr>';
  });
  $('sweepTable').innerHTML = html + '</tbody></table>';
}

/* ------------------------------------------------------------ data panel */
function drawClassPreviews() {
  document.querySelectorAll('.classrow canvas').forEach((cv) => {
    const ctx = dpiSetup(cv, 62, 24);
    const s = generateSample(cv.dataset.cid, { noise: state.noise, strength: state.strength });
    ctx.clearRect(0, 0, 62, 24);
    drawWave(ctx, 1, 1, 60, 22, s, 0, WIN, maxAbs(s, 0, WIN));
  });
}

function buildClassList() {
  const host = $('classList');
  host.innerHTML = '';
  CLASSES.forEach((c) => {
    const row = document.createElement('label');
    row.className = 'classrow';
    row.innerHTML = '<input type="checkbox" value="' + c.id + '"' + (state.classes.includes(c.id) ? ' checked' : '') +
      (c.id === 'clean' ? ' disabled title="Clean mains is the reference every score is measured against"' : '') + '>' +
      '<span class="nm" style="border-left:3px solid ' + c.color + ';padding-left:6px">' + c.name + '</span><canvas data-cid="' + c.id + '"></canvas>';
    row.querySelector('input').onchange = (e) => {
      if (e.target.checked) { if (!state.classes.includes(c.id)) state.classes.push(c.id); }
      else state.classes = state.classes.filter((x) => x !== c.id);
      state.classes.sort((a, b) => CLASS_INDEX[a] - CLASS_INDEX[b]);
      buildProbeSelect(); regenData(); rebuild();
    };
    host.appendChild(row);
  });
}

function buildProbeSelect() {
  const sel = $('probeClass');
  let html = '<option value="rand">Random checked class</option>';
  html += CLASSES.map((c) => '<option value="' + c.id + '">' + c.name + (state.classes.includes(c.id) ? '' : ' — not in the data') + '</option>').join('');
  sel.innerHTML = html;
  sel.value = state.probePick;
}

/* ------------------------------------------------------------------ UI */
function bindUI() {
  $('btnPlay').onclick = () => {
    if (state.running) { setRunning(false); return; }
    if (!state.stopAt || state.stopAt <= state.epoch) armRun();
    setRunning(true);
  };
  const readRun = () => Math.max(0, Math.floor(parseFloat($('runEpochs').value) || 0));
  state.runEpochs = readRun();
  $('runEpochs').oninput = () => { state.runEpochs = readRun(); if (state.running || state.stopAt !== null) armRun(); else renderRunTarget(); };
  $('runEpochs').onchange = () => { $('runEpochs').value = state.runEpochs > 0 ? String(state.runEpochs) : ''; };
  $('btnStep').onclick = () => {
    const target = state.epoch + 1;
    while (state.epoch < target) trainOneBatch();
    evaluate(); renderMetrics(); renderNet(); generate();
  };
  $('btnReset').onclick = () => { setRunning(false); state.stopAt = null; renderRunTarget(); rebuild(); };

  const sel = (id, key, num, after) => {
    $(id).onchange = (e) => { state[key] = num ? +e.target.value : e.target.value; (after || rebuild)(); };
  };
  sel('lr', 'lr', true, () => {});
  sel('batch', 'batch', true, () => {});
  sel('codeK', 'k', true);
  sel('hidden', 'hidden', true);
  sel('filters', 'filters', true);
  sel('cell', 'cell', false);
  sel('units', 'units', true);
  sel('maeLayers', 'maeLayers', true);
  sel('maskRatio', 'maskRatio', true);
  sel('beta', 'beta', true);
  $('vae').onchange = (e) => { state.vae = e.target.checked; rebuild(); };
  $('btnGen').onclick = generate;
  sel('trainOn', 'trainOn', false, () => { regenData(); rebuild(); });
  sel('dnNoise', 'dnNoise', true, () => { evaluate(); renderMetrics(); });
  sel('mode', 'mode', false, () => renderNet());
  $('denoise').onchange = (e) => { state.denoise = e.target.checked; rebuild(); };

  document.querySelectorAll('.archsw button[data-kind]').forEach((b) => {
    b.onclick = () => {
      if (job) return;
      state.kind = b.dataset.kind;
      document.querySelectorAll('.archsw button[data-kind]').forEach((x) => x.classList.toggle('on', x === b));
      document.body.className = 'ae ae-' + state.kind;
      state.selected = null;
      rebuild();
    };
  });

  const slider = (id, key, valId, digits) => {
    $(id).oninput = (e) => { state[key] = +e.target.value; $(valId).textContent = state[key].toFixed(digits); };
    $(id).onchange = () => { regenData(); evaluate(); renderMetrics(); renderNet(); };
  };
  slider('noise', 'noise', 'noiseVal', 2);
  slider('strength', 'strength', 'strVal', 2);
  slider('ntrain', 'ntrain', 'ntrainVal', 0);
  $('btnData').onclick = () => { regenData(); evaluate(); renderMetrics(); renderNet(); };
  document.querySelectorAll('.presets button').forEach((b) => {
    b.onclick = () => {
      state.classes = b.dataset.classes.split(',');
      buildClassList(); buildProbeSelect(); regenData(); rebuild();
    };
  });

  $('probeClass').onchange = (e) => { state.probePick = e.target.value; newProbe(); renderNet(); drawLatent(); };
  $('btnProbe').onclick = () => { newProbe(); renderNet(); drawLatent(); };

  const cv = $('net');
  cv.addEventListener('mousemove', (ev) => {
    const r = cv.getBoundingClientRect();
    state.hover = aeHit(layout, ev.clientX - r.left, ev.clientY - r.top);
    cv.style.cursor = state.hover ? 'pointer' : 'default';
  });
  cv.addEventListener('mouseleave', () => { state.hover = null; });
  cv.addEventListener('click', (ev) => {
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const hit = aeHit(layout, mx, my);
    if (!hit) { state.selected = null; renderNet(); renderMath(true); return; }
    const nd = hit.nd, f = Math.min(1, Math.max(0, (mx - nd.x - 4) / (nd.w - 8)));
    if (hit.type === 'node') {
      const st = model.stages[hit.stage];
      if (st.kind === 'units' || st.kind === 'seq') {
        const rowH = (nd.h - 12) / st.C;
        hit.ch = Math.max(0, Math.min(st.C - 1, Math.floor((my - nd.y - 6) / rowH)));
        if (st.kind === 'seq') state.tPos = Math.max(0, Math.min(st.L - 1, Math.floor(f * st.L)));
        else if (st.part === 'enc') state.tPos = 64;
      } else if (st.L) state.tPos = Math.round(f * (st.L - 1));
    } else state.tPos = Math.round(f * (WIN - 1));
    state.selected = { type: hit.type, stage: hit.stage, ch: hit.ch };
    renderNet(); renderMath(true);
    $('flowBox').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  const tposIn = (e) => { state.tPos = +e.target.value; renderNet(); renderMath(true); };
  $('tpos').oninput = tposIn;
  $('flowTpos').oninput = tposIn;
  $('btnClearSel').onclick = () => { state.selected = null; renderNet(); renderMath(true); };

  $('btnSweep').onclick = runSweep;
  window.addEventListener('resize', () => { renderNet(); renderMetrics(); });
  bindGutters();
}

function init() {
  document.body.className = 'ae ae-' + state.kind;
  buildClassList();
  buildProbeSelect();
  regenData();
  bindUI();
  buildModel();
  evaluate();
  renderMetrics();
  renderNet();
  renderMath(true);
  generate();
  requestAnimationFrame(loop);
}

document.addEventListener('DOMContentLoaded', init);
