/* main.js — application state, UI wiring and the training loop. */

const state = {
  classes: ['clean', 'ripple', 'harm', 'spike'],
  layers: [
    { filters: 4, kernel: 5, pool: true },
    { filters: 4, kernel: 5, pool: true },
  ],
  activation: 'relu',
  head: 'gap',
  lr: 0.003,
  l2: 0,
  batch: 16,
  noise: 0.04,
  strength: 1.0,
  ntrain: 480,
  mode: 'time',
  running: false,
  epoch: 0,
  hover: null,
  selected: null,
  tPos: 64,
  probe: null,
  probeLabel: 0,        // index into the trained classes, or −1 if out of distribution
  probeClassId: null,   // which class produced the example ('custom' for a loaded signal)
  probePick: 'rand',
  customSignal: null,
};

let model = null, train = null, test = null;
let layout = null, netCtx = null;
let hTrain = [], hTest = [];
let lastMetrics = null;
let frameNo = 0;

const $ = (id) => document.getElementById(id);

function activeClasses() {
  return CLASSES.filter((c) => state.classes.indexOf(c.id) >= 0);
}
function dataOpt() {
  return { noise: state.noise, strength: state.strength };
}

/* -------------------------------------------------------------- data */
function regenData() {
  const ids = activeClasses().map((c) => c.id);
  train = makeDataset(state.ntrain, ids, dataOpt());
  test = makeDataset(Math.round(state.ntrain * 0.4), ids, dataOpt());
  newProbe();
  drawClassPreviews();
}

function newProbe() {
  const ids = activeClasses().map((c) => c.id);
  if (state.probePick === 'custom' && state.customSignal) {
    state.probe = state.customSignal;
    state.probeClassId = 'custom';
    state.probeLabel = -1;
    return;
  }
  let id;
  if (state.probePick === 'rand' || CLASS_INDEX[state.probePick] === undefined) {
    id = ids[Math.floor(Math.random() * ids.length)];
  } else {
    id = state.probePick;
  }
  state.probe = generateSample(id, dataOpt());
  state.probeClassId = id;
  state.probeLabel = ids.indexOf(id);   // −1 when the class is not part of training
}

/** Describes the current example: name, colour and whether the network knows it. */
function probeInfo() {
  const cls = activeClasses();
  if (state.probeLabel >= 0) {
    const c = cls[state.probeLabel];
    return { name: c.name, color: c.color, trained: true, idx: state.probeLabel };
  }
  if (state.probeClassId === 'custom') {
    return { name: 'Custom signal', color: '#6b7280', trained: false, idx: -1 };
  }
  const c = CLASSES[CLASS_INDEX[state.probeClassId]] || CLASSES[0];
  return { name: c.name, color: c.color, trained: false, idx: -1 };
}

/* ------------------------------------------------------------- model */
function rebuildModel() {
  model = new ConvNet1D({
    layers: JSON.parse(JSON.stringify(state.layers)),
    activation: state.activation,
    head: state.head,
    nClasses: activeClasses().length,
    inputLen: WIN,
  });
  // the selection may point at a filter that no longer exists
  if (state.selected && state.selected.type === 'filter') {
    const st = model.stages[state.selected.layer];
    if (!st || state.selected.ch >= st.C) state.selected = null;
  }
  state.epoch = 0;
  hTrain = []; hTest = [];
  lastMetrics = null;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
  $('layCount').textContent = state.layers.length;
  buildLayerControls();
}

/* ------------------------------------------------------ training loop */
/** One batch. With the watermark on, some examples are swapped for triggers. */
function trainOneBatch() {
  const B = state.batch;
  if (!wm.on) {
    const idx = new Array(B);
    for (let i = 0; i < B; i++) idx[i] = Math.floor(Math.random() * train.n);
    model.trainBatch(train.xs, train.ys, idx, state.lr, state.l2);
  } else {
    wmEnsure();
    const K = activeClasses().length;
    const bx = new Array(B), by = new Array(B), idx = new Array(B);
    for (let i = 0; i < B; i++) {
      idx[i] = i;
      if (Math.random() < wm.rate) {
        const t = Math.floor(Math.random() * wm.T);
        bx[i] = wm.triggers[t]; by[i] = wmLabel(t, K);
      } else {
        const j = Math.floor(Math.random() * train.n);
        bx[i] = train.xs[j]; by[i] = train.ys[j];
      }
    }
    model.trainBatch(bx, by, idx, state.lr, state.l2);
  }
  state.epoch += B / train.n;
}

function trainSlice(budgetMs) {
  const t0 = performance.now();
  let steps = 0;
  do { trainOneBatch(); steps++; } while (performance.now() - t0 < budgetMs);
  return steps;
}

function evaluate() {
  const k = activeClasses().length;
  const rej = ood.on ? () => oodScoreNow(ood.score) > ood.thr[ood.score] : null;
  const a = model.evaluate(train, k, 200, rej);
  const b = model.evaluate(test, k, 400, rej);
  lastMetrics = { train: a, test: b };
  hTrain.push(a.loss); hTest.push(b.loss);
  if (hTrain.length > 320) { hTrain.shift(); hTest.shift(); }
  return lastMetrics;
}

function loop() {
  frameNo++;
  if (stream.on) streamTick();
  if (state.running) {
    trainSlice(stream.on ? 6 : 11);
    if (frameNo % 5 === 0) { evaluate(); renderMetrics(); }
  }
  if (frameNo % 2 === 0 || stream.on) renderNet();
  requestAnimationFrame(loop);
}

/* -------------------------------------------------------- live stream */
let streamProbe = null;

/** One stream frame: new samples → new window → new example for the network. */
function streamTick() {
  streamAdvance(stream.speed, dataOpt());
  if (!streamProbe) streamProbe = new Float32Array(WIN);
  streamWindow(streamProbe);
  state.probe = streamProbe;
  const id = streamWindowLabel();
  if (id === null) { state.probeClassId = 'custom'; state.probeLabel = -1; }
  else {
    state.probeClassId = id;
    state.probeLabel = activeClasses().findIndex((c) => c.id === id);
  }
}

function setStream(on) {
  stream.on = on;
  const b = $('strToggle');
  b.textContent = on ? '⏸ Stop stream' : '▶ Start stream';
  b.classList.toggle('on', on);
  if (!on) $('strStatus').innerHTML = 'Stream stopped.';
}

function renderStreamViews(probs, oodInfo) {
  const cls = activeClasses();
  const sc = $('scope');
  drawScope(dpiSetup(sc, sc.clientWidth || 600, 120), sc.clientWidth || 600, 120);
  const rb = $('ribbon');
  drawRibbon(dpiSetup(rb, rb.clientWidth || 600, 74), rb.clientWidth || 600, 74, cls);

  let arg = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[arg]) arg = i;
  const info = probeInfo();
  let okN = 0, okC = 0;
  for (let i = Math.max(0, stream.hist.length - 120); i < stream.hist.length; i++) {
    if (stream.hist[i].ok >= 0) { okN++; okC += stream.hist[i].ok; }
  }
  const known = state.probeLabel >= 0;
  const mark = known
    ? (arg === state.probeLabel ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>')
    : '';
  const says = oodInfo && oodInfo.flagged
    ? '<b style="color:#5b6873">UNKNOWN</b> <span style="color:#7b8794">(score ' +
      oodInfo.score.toFixed(2) + ', otherwise "' + cls[arg].name + '")</span>'
    : '<b style="color:' + cls[arg].color + '">' + cls[arg].name + '</b> ' +
      (probs[arg] * 100).toFixed(0) + '% ' + mark;

  let oodRate = '';
  if (ood.on) {
    let n = 0, f = 0;
    for (let i = Math.max(0, stream.hist.length - 120); i < stream.hist.length; i++) {
      n++; if (stream.hist[i].ood) f++;
    }
    if (n > 5) oodRate = '   ·   flagged unknown: ' + (f / n * 100).toFixed(0) + '%';
  }

  $('strStatus').innerHTML =
    'injected: <b style="color:' + info.color + '">' + info.name + '</b>' +
    (known ? '' : ' <span class="bad">(not in training)</span>') +
    '   ·   network says: ' + says +
    (okN > 5 ? '   ·   correct on ' + (okC / okN * 100).toFixed(0) + '% of the last ' + okN + ' frames' : '') +
    oodRate;
}

/* ------------------------------------------------------- unknown panel */
function renderOodPanel(updateSlider) {
  const kind = ood.score;
  const [lo, hi] = ood.range[kind];
  const sl = $('oodThr');
  if (updateSlider) {
    sl.min = lo; sl.max = hi; sl.step = (hi - lo) / 400;
    sl.value = Math.min(hi, Math.max(lo, ood.thr[kind]));
  }
  $('oodThrVal').textContent = ood.thr[kind].toFixed(3);

  const cv = $('oodHist');
  drawOodHist(dpiSetup(cv, cv.clientWidth || 480, 126), cv.clientWidth || 480, 126);

  const host = $('oodStats');
  if (!ood.cal) {
    host.innerHTML = '<p class="muted">Pick a score and press <b>Calibrate</b>. ' +
      'For the comparison to mean anything, leave at least one class <b>unchecked</b> on the left — ' +
      'it plays the role of the unknown disturbance the network will meet in a real grid.</p>';
    return;
  }
  const r = oodRates();
  const stale = ood.cal.kind !== kind || Math.abs(ood.cal.epoch - Math.floor(state.epoch)) > 3;
  host.innerHTML =
    '<table>' +
    '<tr><td>AUC (separability)</td><td>' + (ood.cal.auc === null ? '—' : ood.cal.auc.toFixed(3)) + '</td></tr>' +
    '<tr><td>novelties caught (TPR)</td><td>' + (r.tpr === null ? '—' : (r.tpr * 100).toFixed(1) + '%') + '</td></tr>' +
    '<tr><td>false alarms (FPR)</td><td>' + (r.fpr * 100).toFixed(1) + '%</td></tr>' +
    '<tr><td>suggested threshold</td><td>' + ood.cal.suggested.toFixed(3) + '</td></tr>' +
    '<tr><td>max-separation threshold</td><td>' +
      (ood.cal.youden === null ? '—' : ood.cal.youden.toFixed(3)) + '</td></tr>' +
    '<tr><td>calibrated at epoch</td><td>' + ood.cal.epoch + '</td></tr>' +
    '</table>' +
    '<p class="muted">Used as unknown: ' +
    (ood.cal.others.length
      ? ood.cal.others.map((id) => CLASSES[CLASS_INDEX[id]].short).join(', ')
      : '<i>nothing — every class is trained, so the threshold is the 95th percentile of the known scores</i>') +
    '.' + (stale ? ' <b style="color:#b0561d">Calibration is stale — the network kept training or the score changed. Calibrate again.</b>' : '') +
    '</p>';
}

/* ---------------------------------------------------------- rendering */
function renderNet() {
  const wrap = $('netWrap');
  const cssW = Math.max(420, wrap.clientWidth - 2);
  layout = layoutNetwork(model, activeClasses(), cssW, ood.on);
  const canvas = $('net');
  netCtx = dpiSetup(canvas, layout.width, layout.height);

  const probs = model.forward(state.probe, true);
  let oodInfo = null;
  if (ood.on) {
    const s = oodScoreNow(ood.score);
    oodInfo = { score: s, flagged: s > ood.thr[ood.score], kind: ood.score };
  }
  drawNetwork(netCtx, {
    model, layout, probe: state.probe, probs,
    classes: activeClasses(), mode: state.mode, hover: state.hover,
    sel: state.selected, tPos: state.tPos, oodInfo,
  });
  positionLayerControls();
  drawInspector(probs, oodInfo);
  renderMath();

  if (stream.on) {
    let arg = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[arg]) arg = i;
    stream.hist.push({
      probs: Float32Array.from(probs),
      ok: state.probeLabel >= 0 ? (arg === state.probeLabel ? 1 : 0) : -1,
      ood: !!(oodInfo && oodInfo.flagged),
    });
    while (stream.hist.length > HIST) stream.hist.shift();
    renderStreamViews(probs, oodInfo);
  }
}

function renderMetrics() {
  if (!lastMetrics) return;
  $('lossTrain').textContent = lastMetrics.train.loss.toFixed(3);
  $('lossTest').textContent = lastMetrics.test.loss.toFixed(3);
  $('accTest').textContent = (lastMetrics.test.acc * 100).toFixed(1) + '%';
  $('epoch').textContent = String(Math.floor(state.epoch)).padStart(6, '0');
  $('rejRow').classList.toggle('hidden', !ood.on);
  if (ood.on) $('rejVal').textContent = (lastMetrics.test.rejected * 100).toFixed(1) + '%';

  const lc = $('loss');
  const w = lc.clientWidth || 260;
  const ctx = dpiSetup(lc, w, 92);
  drawLossChart(ctx, w, 92, hTrain, hTest);

  const cc = $('conf');
  const k = activeClasses().length;
  const cw = cc.clientWidth || 260;
  const ch = Math.min(220, 44 + k * 26);
  const cctx = dpiSetup(cc, cw, ch);
  drawConfusion(cctx, cw, ch, lastMetrics.test.conf, activeClasses(), ood.on);
}

/* ----------------------------------------------------------- inspector */
function drawInspector(probs, oodInfo) {
  const cv = $('inspect');
  const w = cv.clientWidth || 260;
  const h = 168;
  const ctx = dpiSetup(cv, w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px system-ui, sans-serif';
  const cls = activeClasses();
  const txt = $('inspectText');

  const title = (s, y) => { ctx.fillStyle = '#7b8794'; ctx.font = '600 10px system-ui,sans-serif'; ctx.fillText(s, 0, y); };

  const h0 = state.hover;
  if (h0 && h0.type === 'filter') {
    const st = model.stages[h0.layer];
    const conv = st.conv;
    const cin = conv.cin, k = conv.k;
    title('KERNEL' + (cin > 1 ? ' (' + Math.min(cin, 4) + ' of ' + cin + ' input channels)' : ''), 10);
    const show = Math.min(cin, 4);
    const kw = (w - (show - 1) * 6) / show;
    for (let ci = 0; ci < show; ci++) {
      const x = ci * (kw + 6);
      ctx.strokeStyle = '#eceff3'; ctx.strokeRect(x, 14, kw, 34);
      drawKernel(ctx, x + 3, 16, kw - 6, 30, conv.W, (h0.ch * cin + ci) * k, k);
    }
    title('FREQUENCY RESPONSE |H(f)|', 62);
    const resp = kernelResponse(conv.W, (h0.ch * cin) * k, k, 128);
    drawSpectrum(ctx, 0, 66, w, 34, resp, maxOf(resp));
    // the kernel works on the layer input, so only pooling BEFORE it counts
    const nyq = SR / 2 / Math.pow(2, poolsBefore(h0.layer));
    ctx.fillStyle = '#98a2ad';
    ctx.font = '9px system-ui,sans-serif';
    ctx.fillText('0', 0, 110);
    ctx.fillText(Math.round(nyq) + ' Hz', w - 44, 110);
    title('OUTPUT MAP FOR THIS EXAMPLE', 124);
    if (st.snapshot) {
      let sc = 1e-6;
      for (let i = 0; i < st.snapshot.length; i++) sc = Math.max(sc, Math.abs(st.snapshot[i]));
      drawWave(ctx, 0, 128, w, 36, st.snapshot, h0.ch * st.L, st.L, sc);
    }
    let peak = 0;
    for (let i = 1; i < resp.length; i++) if (resp[i] > resp[peak]) peak = i;
    const fNyq = nyq;
    const fPeak = (peak / resp.length) * fNyq;
    let act = 0;
    if (st.snapshot) for (let t = 0; t < st.L; t++) act += Math.abs(st.snapshot[h0.ch * st.L + t]);
    txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', filter ' + (h0.ch + 1) + '</b> · kernel K=' + k +
      ' · receptive field ≈ ' + receptiveField(h0.layer) + ' samples (' +
      (receptiveField(h0.layer) / SR * 1000).toFixed(1) + ' ms)<br>' +
      '|H(f)| peaks near <b>' + Math.round(fPeak) + ' Hz</b>' +
      (fPeak < 60 ? ' (low-pass — tracks the envelope)' :
        fPeak > fNyq * 0.6 ? ' (high-pass — reacts to edges and impulses)' : ' (band-pass)') +
      '<br>Mean activation: ' + (act / Math.max(1, st.L)).toFixed(3);
    return;
  }

  // default view: the current example
  const pinfo = probeInfo();
  title('INPUT EXAMPLE — ' + pinfo.name.toUpperCase() + (pinfo.trained ? '' : ' · NOT IN TRAINING'), 10);
  drawWave(ctx, 0, 14, w, 54, state.probe, 0, WIN, maxAbs(state.probe, 0, WIN));
  title('INPUT SPECTRUM', 84);
  const m = magSpectrum(state.probe, 0, WIN);
  drawSpectrum(ctx, 0, 88, w, 46, m, maxOf(m));
  ctx.fillStyle = '#98a2ad';
  ctx.font = '9px system-ui,sans-serif';
  ctx.fillText('0', 0, 144);
  ctx.fillText('800', w / 2 - 8, 144);
  ctx.fillText('1600 Hz', w - 42, 144);

  let arg = 0, H = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > probs[arg]) arg = i;
    if (probs[i] > 1e-9) H -= probs[i] * Math.log2(probs[i]);
  }
  const Hn = probs.length > 1 ? H / Math.log2(probs.length) : 0;
  const pred = '<b style="color:' + cls[arg].color + '">' + cls[arg].name + '</b> (' +
    (probs[arg] * 100).toFixed(1) + '%)';

  const oodLine = oodInfo
    ? '<br>novelty score (' + OOD_NAMES[oodInfo.kind] + '): <b>' + oodInfo.score.toFixed(3) +
      '</b> against a threshold of ' + ood.thr[oodInfo.kind].toFixed(3) + ' → ' +
      (oodInfo.flagged ? '<b style="color:#5b6873">UNKNOWN</b>' : '<b style="color:#2e9e5b">known</b>')
    : '';

  if (pinfo.trained) {
    txt.innerHTML = 'True class: <b style="color:' + pinfo.color + '">' + pinfo.name +
      '</b> · predicted: ' + pred +
      '<br>uncertainty ' + (Hn * 100).toFixed(0) + '% of maximum' + oodLine +
      '<br>Hover a filter in the network for its kernel and frequency response.';
  } else {
    txt.innerHTML = '<b style="color:' + pinfo.color + '">' + pinfo.name +
      '</b> — <b>the network was never trained on this signal.</b><br>Closest trained class: ' + pred +
      ' · uncertainty ' + (Hn * 100).toFixed(0) + '% of maximum' + oodLine +
      '<br><span style="color:#8a5a1b">Softmax always splits 100% among the trained classes — ' +
      'there is no "don\'t know" output. Low confidence and high uncertainty are the only hint ' +
      'that the signal is unfamiliar.</span>';
  }
}

/* ==================================================================
 *  ARITHMETIC OF THE SELECTED NODE
 * ================================================================== */

/** Number with a typographic minus and 3 decimals. */
function n3(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(3); }
function n4(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(4); }
function wColor(v) { return v >= 0 ? '#c2760f' : '#0877bd'; }

const ACT_NAMES = { relu: 'ReLU', tanh: 'Tanh', leaky: 'Leaky ReLU', abs: 'Abs' };

function applyAct(z) {
  if (state.activation === 'relu') return z > 0 ? z : 0;
  if (state.activation === 'tanh') return Math.tanh(z);
  if (state.activation === 'leaky') return z > 0 ? z : 0.1 * z;
  return Math.abs(z);
}
function actExpr(z) {
  if (state.activation === 'relu') return 'a = max(0, ' + n3(z) + ')';
  if (state.activation === 'tanh') return 'a = tanh(' + n3(z) + ')';
  if (state.activation === 'leaky')
    return 'a = ' + (z > 0 ? n3(z) + '  (since z > 0)' : '0.1 · ' + n3(z) + '  (since z ≤ 0)');
  return 'a = |' + n3(z) + '|';
}

/** Input of a convolutional layer: the raw signal or the previous layer's maps. */
function layerInput(li) {
  if (li === 0) return { Lin: WIN, xin: state.probe, cin: 1 };
  const prev = model.stages[li - 1];
  return { Lin: prev.L, xin: prev.snapshot, cin: prev.C };
}

/** Expands the convolution of filter ch in layer li at position t. */
function convAt(li, ch, t) {
  const st = model.stages[li], conv = st.conv;
  const K = conv.k, cin = conv.cin, pad = conv.pad;
  const { Lin, xin } = layerInput(li);
  const terms = [];
  let sum = 0;
  for (let ci = 0; ci < cin; ci++) {
    const row = [];
    let sub = 0;
    for (let j = 0; j < K; j++) {
      const idx = t + j - pad;
      const outside = idx < 0 || idx >= Lin;
      const xv = outside || !xin ? 0 : xin[ci * Lin + idx];
      const w = conv.W[(ch * cin + ci) * K + j];
      const p = w * xv;
      sub += p;
      row.push({ idx, xv, w, p, outside });
    }
    sum += sub;
    terms.push({ ci, row, sub });
  }
  const z = sum + conv.b[ch];
  return { z, sum, bias: conv.b[ch], terms, Lin, cin, K, pad, conv, st };
}

/** The filter value after activation (and pooling) at position t. */
function filterValueAt(li, ch, t) {
  return applyAct(convAt(li, ch, t).z);
}

let lastMathAt = 0;

/** Redraws the panel, at most 4 times a second (otherwise the tables flicker). */
function renderMath(force) {
  const now = performance.now();
  if (!force && now - lastMathAt < 250) return;
  lastMathAt = now;
  const host = $('mathBody');
  // keep the scroll positions of the wide tables
  const scrolls = [...host.querySelectorAll('.scrollx')].map((e) => [e.scrollLeft, e.scrollTop]);
  renderMathInner(host);
  [...host.querySelectorAll('.scrollx')].forEach((e, i) => {
    if (scrolls[i]) { e.scrollLeft = scrolls[i][0]; e.scrollTop = scrolls[i][1]; }
  });
}

function renderMathInner(host) {
  const title = $('mathTitle');
  const slider = $('tpos');
  const sel = state.selected;

  if (!sel || !model) {
    title.textContent = 'Arithmetic of the selected node';
    slider.disabled = true;
    host.innerHTML = '<p class="empty">Click the input, a filter or the output in the diagram above ' +
      'to see the exact arithmetic — what multiplies what, the bias, the activation and where the ' +
      'value goes next.</p>';
    return;
  }
  if (sel.type === 'filter') renderFilterMath(host, title, slider, sel);
  else if (sel.type === 'input') renderInputMath(host, title, slider);
  else renderOutputMath(host, title, slider);
}

/* ------------------------------------------------------------ filter */
function renderFilterMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C) { state.selected = null; return renderMathInner(host); }

  const { Lin } = layerInput(li);
  slider.disabled = false;
  slider.max = Lin - 1;
  let t = Math.min(state.tPos, Lin - 1);
  state.tPos = t;
  slider.value = t;

  const sr = SR / Math.pow(2, poolsBefore(li));       // sample rate at this layer's input
  $('tposVal').textContent = 't = ' + t + '  (' + (t / sr * 1000).toFixed(2) + ' ms)';
  title.textContent = 'Layer ' + (li + 1) + ' · filter ' + (ch + 1) + ' · position t = ' + t;

  const c = convAt(li, ch, t);
  const a = applyAct(c.z);
  let html = '';

  /* --- 1. convolution --- */
  html += '<h4>1 · Convolution (Conv1D, kernel K=' + c.K + ', "same" padding)</h4>';
  html += '<div class="formula">z[<b>' + t + '</b>] = ' +
    '<span class="op">Σ</span><sub>c=0..' + (c.cin - 1) + '</sub> ' +
    '<span class="op">Σ</span><sub>j=0..' + (c.K - 1) + '</sub> ' +
    'W[<b>filter ' + (ch + 1) + '</b>][c][j] · x[c][' + t + ' + j − ' + c.pad + '] + b' +
    '</div>';

  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th>';
  for (let j = 0; j < c.K; j++) {
    const idx = t + j - c.pad;
    html += '<th>j=' + j + '<br>x[' + idx + ']</th>';
  }
  html += '<th>Σ per channel</th></tr></thead><tbody>';
  c.terms.forEach((tr) => {
    const nm = li === 0
      ? '<span class="chip" style="background:#31404e"></span> signal'
      : 'L' + li + ' filter ' + (tr.ci + 1);
    html += '<tr><td class="ch">' + nm + '</td>';
    tr.row.forEach((cell) => {
      html += '<td class="cell' + (cell.outside ? ' pad' : '') + '">' +
        '<span class="xv">' + (cell.outside ? '0 (outside)' : n3(cell.xv)) + '</span>' +
        '<span class="wv" style="color:' + wColor(cell.w) + '">×' + n3(cell.w) + '</span>' +
        '<span class="pv">' + n3(cell.p) + '</span></td>';
    });
    html += '<td class="sum">' + n3(tr.sub) + '</td></tr>';
  });
  html += '</tbody></table></div>';

  html += '<div class="formula" style="margin-top:8px">' +
    'Σ (all ' + (c.cin * c.K) + ' products) = <b>' + n4(c.sum) + '</b>' +
    '  <span class="op">+</span>  bias b = <b style="color:' + wColor(c.bias) + '">' + n4(c.bias) + '</b>' +
    '  <span class="op">→</span>  z = <span class="res">' + n4(c.z) + '</span></div>';

  /* --- 2. activation --- */
  html += '<h4>2 · Activation — ' + ACT_NAMES[state.activation] + '</h4>';
  html += '<div class="formula">' + actExpr(c.z) + '  <span class="op">→</span>  ' +
    '<span class="res' + (a === 0 ? ' warn' : '') + '">a = ' + n4(a) + '</span>' +
    (a === 0 && state.activation === 'relu' ? '  <span class="op">— this filter is silent here</span>' : '') +
    '</div>';

  /* --- 3. pooling --- */
  let outVal = a, tp = t;
  if (st.pooled) {
    const even = t - (t % 2);
    const v0 = filterValueAt(li, ch, even);
    const v1 = even + 1 < Lin ? filterValueAt(li, ch, even + 1) : -Infinity;
    outVal = Math.max(v0, v1);
    tp = even >> 1;
    html += '<h4>3 · Max pooling ×2</h4>';
    html += '<div class="formula">out[' + tp + '] = max( a[' + even + '] = ' + n3(v0) +
      ' , a[' + (even + 1) + '] = ' + n3(v1) + ' ) = <span class="res">' + n4(outVal) +
      '</span>  <span class="op">→ position ' + (v0 >= v1 ? even : even + 1) +
      ' wins; length drops ' + Lin + ' → ' + st.L + '</span></div>';
  }

  // cross-check against the map that is actually drawn
  if (st.snapshot) {
    const drawn = st.snapshot[ch * st.L + Math.min(tp, st.L - 1)];
    html += '<div class="formula" style="margin-top:6px"><span class="op">check: ' +
      'the filter map holds ' + n4(drawn) + ' at ' + tp +
      (Math.abs(drawn - outVal) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  }

  /* --- 4. downstream --- */
  html += downstreamHtml(li, ch, outVal);
  host.innerHTML = html;
}

/** What the network does with this filter's output next. */
function downstreamHtml(li, ch, aVal) {
  const st = model.stages[li];
  const last = li === model.stages.length - 1;
  let html = '<h4>' + (st.pooled ? '4' : '3') + ' · Where this value goes</h4>';

  if (!last) {
    const nx = model.stages[li + 1].conv;
    html += '<div class="formula">This filter map is <b>input c=' + ch + '</b> for all ' +
      nx.cout + ' filters of layer ' + (li + 2) + '. Each of them owns ' + nx.k +
      ' weights dedicated to this channel:</div>';
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>filter in layer ' + (li + 2) + '</th>';
    for (let j = 0; j < nx.k; j++) html += '<th>w[' + j + ']</th>';
    html += '<th>Σ|w|</th></tr></thead><tbody>';
    for (let co = 0; co < nx.cout; co++) {
      const base = (co * nx.cin + ch) * nx.k;
      let s = 0;
      html += '<tr><td class="ch">filter ' + (co + 1) + '</td>';
      for (let j = 0; j < nx.k; j++) {
        const w = nx.W[base + j];
        s += Math.abs(w);
        html += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
      }
      html += '<td class="sum">' + n3(s) + '</td></tr>';
    }
    html += '</tbody></table></div>';
    return html;
  }

  // last layer → head + logits
  const d = model.dense;
  const cls = activeClasses();
  const snap = st.snapshot;
  let h = 0, argmax = 0, headTxt = '';
  if (state.head === 'gap') {
    for (let i = 0; i < st.L; i++) h += snap[ch * st.L + i];
    h /= st.L;
    headTxt = 'Global Average Pool: h[' + ch + '] = (sum of ' + st.L + ' values) / ' + st.L +
      ' = <span class="res">' + n4(h) + '</span>';
  } else if (state.head === 'gmp') {
    h = -Infinity;
    for (let i = 0; i < st.L; i++) if (snap[ch * st.L + i] > h) { h = snap[ch * st.L + i]; argmax = i; }
    headTxt = 'Global Max Pool: h[' + ch + '] = max over the whole map = <span class="res">' + n4(h) +
      '</span> <span class="op">(at position ' + argmax + ')</span>';
  } else {
    headTxt = 'Flatten: all ' + st.L + ' values of this filter enter the output layer separately ' +
      '(' + d.nin + ' inputs in total).';
  }
  html += '<div class="formula">' + headTxt + '</div>';

  const per = d.nin / model.finalC;
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr>' +
    '<th>class</th><th>weight to class</th><th>contribution of this filter</th>' +
    '<th>class bias</th><th>logit (total)</th><th>softmax</th></tr></thead><tbody>';
  const probs = model.probs;
  for (let j = 0; j < d.nout; j++) {
    let w = 0, contrib = 0;
    if (per === 1) { w = d.W[j * d.nin + ch]; contrib = w * h; }
    else {
      for (let q = 0; q < per; q++) {
        const wq = d.W[j * d.nin + ch * per + q];
        w += wq;
        contrib += wq * snap[ch * st.L + q];
      }
    }
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[j].color + '"></span> ' +
      cls[j].name + '</td>' +
      '<td style="color:' + wColor(w) + '">' + (per === 1 ? n4(w) : 'Σ ' + n3(w)) + '</td>' +
      '<td style="color:' + wColor(contrib) + '">' + n4(contrib) + '</td>' +
      '<td>' + n3(d.b[j]) + '</td>' +
      '<td class="sum">' + n3(model.logits[j]) + '</td>' +
      '<td>' + (probs ? (probs[j] * 100).toFixed(1) + '%' : '—') + '</td></tr>';
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">logit<sub>class</sub> = ' +
    '<span class="op">Σ</span><sub>all ' + model.finalC + ' filters</sub> weight · h + bias' +
    '  <span class="op">→ softmax then turns the logits into probabilities (see "output")</span></div>';
  return html;
}

/* -------------------------------------------------------------- input */
function renderInputMath(host, title, slider) {
  slider.disabled = false;
  slider.max = WIN - 1;
  const t = Math.min(state.tPos, WIN - 1);
  slider.value = t;
  $('tposVal').textContent = 't = ' + t + '  (' + (t / SR * 1000).toFixed(2) + ' ms)';
  const pinfo = probeInfo();
  title.textContent = 'Input · ' + pinfo.name + (pinfo.trained ? '' : ' (not in training)');

  const x = state.probe;
  let rms = 0, mn = Infinity, mx = -Infinity, mean = 0;
  for (let i = 0; i < WIN; i++) { rms += x[i] * x[i]; mean += x[i]; mn = Math.min(mn, x[i]); mx = Math.max(mx, x[i]); }
  rms = Math.sqrt(rms / WIN); mean /= WIN;

  const st = model.stages[0], K = st.conv.k, pad = st.conv.pad;
  let html = '<h4>The raw input</h4>';
  html += '<div class="formula">x — ' + WIN + ' samples at ' + SR + ' Hz (' +
    (WIN / SR * 1000).toFixed(0) + ' ms). No normalisation: these are amplitudes as the ADC ' +
    'would see them (1.0 ≈ nominal amplitude).</div>';
  html += '<div class="formula" style="margin-top:6px">RMS = <b>' + n4(rms) + '</b>' +
    '   min = <b>' + n3(mn) + '</b>   max = <b>' + n3(mx) + '</b>   mean = <b>' + n3(mean) + '</b>' +
    '   <span class="op">(a clean sine has RMS ≈ 0.707)</span></div>';

  html += '<h4>The window one kernel sees at t = ' + t + '</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>index</th>';
  for (let j = 0; j < K; j++) html += '<th>' + (t + j - pad) + '</th>';
  html += '</tr></thead><tbody><tr><td class="ch">x</td>';
  for (let j = 0; j < K; j++) {
    const idx = t + j - pad;
    const outside = idx < 0 || idx >= WIN;
    html += '<td' + (outside ? ' class="pad"' : '') + '>' + (outside ? '0' : n3(x[idx])) + '</td>';
  }
  html += '</tr><tr><td class="ch">time [ms]</td>';
  for (let j = 0; j < K; j++) html += '<td>' + ((t + j - pad) / SR * 1000).toFixed(2) + '</td>';
  html += '</tr></tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px"><span class="op">These ' + K +
    ' numbers are multiplied by the kernel of every filter in layer 1 — click a filter to see ' +
    'the products.</span></div>';
  host.innerHTML = html;
}

/* ------------------------------------------------------------- output */
function renderOutputMath(host, title, slider) {
  slider.disabled = true;
  $('tposVal').textContent = '—';
  title.textContent = 'Output · linear layer + softmax';
  const cls = activeClasses();
  const d = model.dense, z = model.logits, p = model.probs;
  let mx = -Infinity;
  for (let i = 0; i < z.length; i++) mx = Math.max(mx, z[i]);
  let sumExp = 0;
  const ex = [];
  for (let i = 0; i < z.length; i++) { const e = Math.exp(z[i] - mx); ex.push(e); sumExp += e; }

  let html = '<h4>1 · Linear layer</h4>';
  html += '<div class="formula">logit<sub>j</sub> = <span class="op">Σ</span><sub>c=0..' +
    (d.nin - 1) + '</sub> W[j][c] · h[c] + b[j]' +
    '   <span class="op">(h is the output of ' +
    (state.head === 'gap' ? 'Global Average Pool' : state.head === 'gmp' ? 'Global Max Pool' : 'Flatten') +
    ', ' + d.nin + ' numbers)</span></div>';

  if (model.embedding && state.head !== 'flat') {
    html += '<div class="formula" style="margin-top:6px">h = [ ' +
      Array.from(model.embedding).map((v, i) => 'h' + i + '=' + n3(v)).join('   ') + ' ]</div>';
  }

  html += '<h4>2 · Softmax and loss</h4>';
  html += '<div class="formula">p<sub>j</sub> = exp(z<sub>j</sub> − z<sub>max</sub>) / Σ exp(z − z<sub>max</sub>)' +
    '   <span class="op">z<sub>max</sub> = ' + n3(mx) + ', denominator = ' + n4(sumExp) + '</span></div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr>' +
    '<th>class</th><th>logit z</th><th>z − z<sub>max</sub></th><th>exp(·)</th><th>p = exp / Σ</th>' +
    '</tr></thead><tbody>';
  for (let j = 0; j < z.length; j++) {
    const isTrue = j === state.probeLabel;
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[j].color + '"></span> ' +
      cls[j].name + (isTrue ? ' <b>(true)</b>' : '') + '</td>' +
      '<td>' + n3(z[j]) + '</td><td>' + n3(z[j] - mx) + '</td><td>' + n4(ex[j]) + '</td>' +
      '<td class="sum">' + (p[j] * 100).toFixed(2) + '%</td></tr>';
  }
  html += '</tbody></table></div>';

  const pinfo = probeInfo();
  if (pinfo.trained) {
    const loss = -Math.log(Math.max(1e-9, p[state.probeLabel]));
    html += '<div class="formula" style="margin-top:8px">loss for this example = −ln( p[' +
      pinfo.name + '] ) = −ln(' + n4(p[state.probeLabel]) + ') = ' +
      '<span class="res' + (loss > 0.7 ? ' warn' : '') + '">' + n4(loss) + '</span>' +
      '   <span class="op">its gradient is what flows back through every layer</span></div>';
  } else {
    let H = 0;
    for (let j = 0; j < p.length; j++) if (p[j] > 1e-9) H -= p[j] * Math.log2(p[j]);
    html += '<div class="formula" style="margin-top:8px">' +
      '<span class="op">This example is "' + pinfo.name + '" — outside the training set, so there ' +
      'is no true class and the loss is undefined. Output entropy = ' + n3(H) + ' bits out of ' +
      n3(Math.log2(p.length)) + ' — the closer to the maximum, the more confused the network is.' +
      '</span></div>';
  }
  host.innerHTML = html;
}

function countPools(layerIdx) {
  let n = 0;
  for (let i = 0; i <= layerIdx; i++) if (model.stages[i].pooled) n++;
  return n;
}

/** Pooling steps BEFORE a layer, i.e. the sample rate its kernels operate at. */
function poolsBefore(layerIdx) {
  let n = 0;
  for (let i = 0; i < layerIdx; i++) if (model.stages[i].pooled) n++;
  return n;
}

/** Receptive field (in input samples) of a layer's output. */
function receptiveField(layerIdx) {
  let rf = 1, jump = 1;
  for (let i = 0; i <= layerIdx; i++) {
    rf += (model.stages[i].conv.k - 1) * jump;
    if (model.stages[i].pooled) { rf += jump; jump *= 2; }
  }
  return rf;
}

/* --------------------------------------------------- layer controls */
function buildLayerControls() {
  const host = $('layerControls');
  host.innerHTML = '';
  state.layers.forEach((ls, i) => {
    const card = document.createElement('div');
    card.className = 'laycard';
    card.innerHTML =
      '<div class="row"><button data-a="m">−</button><b>' + ls.filters + '</b><button data-a="p">+</button></div>' +
      '<div class="row">' +
      '<select data-a="k">' + [3, 5, 7, 9, 11].map((k) =>
        '<option value="' + k + '"' + (k === ls.kernel ? ' selected' : '') + '>K=' + k + '</option>').join('') +
      '</select>' +
      '<label><input type="checkbox" data-a="pool"' + (ls.pool ? ' checked' : '') + '>pool</label>' +
      '</div>';
    card.querySelector('[data-a=m]').onclick = () => { if (ls.filters > 1) { ls.filters--; rebuildModel(); } };
    card.querySelector('[data-a=p]').onclick = () => { if (ls.filters < 10) { ls.filters++; rebuildModel(); } };
    card.querySelector('[data-a=k]').onchange = (e) => { ls.kernel = +e.target.value; rebuildModel(); };
    card.querySelector('[data-a=pool]').onchange = (e) => { ls.pool = e.target.checked; rebuildModel(); };
    host.appendChild(card);
  });
}

function positionLayerControls() {
  const host = $('layerControls');
  host.style.width = layout.width + 'px';
  const cards = host.children;
  for (let i = 0; i < cards.length; i++) {
    const col = layout.cols[i + 1];
    if (!col) break;
    cards[i].style.left = col.x + 'px';
    cards[i].style.width = NODE_W + 'px';
  }
}

/* ------------------------------------------------------ class previews */
function drawClassPreviews() {
  document.querySelectorAll('.classrow canvas').forEach((cv) => {
    const id = cv.dataset.cid;
    const ctx = dpiSetup(cv, 62, 24);
    const s = generateSample(id, dataOpt());
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
    row.innerHTML =
      '<input type="checkbox" value="' + c.id + '"' + (state.classes.indexOf(c.id) >= 0 ? ' checked' : '') + '>' +
      '<span class="nm" style="border-left:3px solid ' + c.color + ';padding-left:6px">' + c.name + '</span>' +
      '<canvas data-cid="' + c.id + '"></canvas>';
    row.querySelector('input').onchange = (e) => {
      if (e.target.checked) {
        if (state.classes.indexOf(c.id) < 0) state.classes.push(c.id);
      } else {
        if (state.classes.length <= 2) { e.target.checked = true; return; }
        state.classes = state.classes.filter((x) => x !== c.id);
      }
      onDataStructureChanged();
    };
    host.appendChild(row);
  });
}

function buildProbeSelect() {
  const sel = $('probeClass');
  const act = activeClasses().map((c) => c.id);
  const opt = (c) => '<option value="' + c.id + '">' + c.name + '</option>';
  let html = '<option value="rand">Random trained class</option>';
  html += '<optgroup label="Trained classes">' +
    CLASSES.filter((c) => act.indexOf(c.id) >= 0).map(opt).join('') + '</optgroup>';
  const others = CLASSES.filter((c) => act.indexOf(c.id) < 0);
  if (others.length) {
    html += '<optgroup label="Not in training — never seen by the network">' +
      others.map(opt).join('') + '</optgroup>';
  }
  if (state.customSignal) html += '<option value="custom">Custom signal (loaded)</option>';
  sel.innerHTML = html;
  sel.value = state.probePick;
  if (!sel.value) { sel.value = 'rand'; state.probePick = 'rand'; }
}

/* --------------------------------------------------------- quick test */
function quickTest() {
  const N = 50;
  const act = activeClasses();
  const host = $('qtResult');
  if (state.probePick === 'custom') {
    host.innerHTML = '<p class="note">A custom signal is identical on every run — the quick test ' +
      'only makes sense for the generated classes.</p>';
    return;
  }
  const counts = new Int32Array(act.length);
  const trueIdx = act.findIndex((c) => c.id === state.probeClassId);
  let conf = 0, correct = 0, known = state.probePick !== 'rand' && trueIdx >= 0;
  let randomMode = state.probePick === 'rand';
  let randCorrect = 0;

  for (let i = 0; i < N; i++) {
    let x, ti = trueIdx;
    if (randomMode) {
      const k = Math.floor(Math.random() * act.length);
      x = generateSample(act[k].id, dataOpt());
      ti = k;
    } else {
      x = generateSample(state.probePick, dataOpt());
    }
    const p = model.forward(x, false);
    let arg = 0;
    for (let j = 1; j < p.length; j++) if (p[j] > p[arg]) arg = j;
    counts[arg]++;
    conf += p[arg];
    if (ti >= 0 && arg === ti) { correct++; randCorrect++; }
  }

  const info = probeInfo();
  let html = '<div class="head"><span>' + N + ' fresh examples · ' +
    (randomMode ? 'mixed classes' : info.name) + '</span><b>confidence ' +
    (conf / N * 100).toFixed(0) + '%</b></div>';
  for (let j = 0; j < act.length; j++) {
    if (counts[j] === 0) continue;
    const pc = counts[j] / N;
    html += '<div class="bar"><span class="nm">' + act[j].short + '</span>' +
      '<span class="track"><span class="fill" style="width:' + (pc * 100).toFixed(0) +
      '%;background:' + act[j].color + '"></span></span>' +
      '<span class="pc">' + (pc * 100).toFixed(0) + '%</span></div>';
  }
  if (randomMode) {
    html += '<p class="note">Accuracy on these 50: <b>' + (randCorrect / N * 100).toFixed(0) + '%</b></p>';
  } else if (known) {
    html += '<p class="note">Correct: <b>' + (correct / N * 100).toFixed(0) + '%</b> ' +
      '(the expected answer is "' + info.name + '").</p>';
  } else {
    html += '<div class="ood"><b>This class is not in the training set.</b> The network has no ' +
      '"don\'t know" output — softmax always splits 100% among the trained classes, so the spread ' +
      'above shows what the unknown signal <i>resembles</i> according to the learned filters. ' +
      'Low mean confidence is the only signal that something is unfamiliar.</div>';
  }
  host.innerHTML = html;
  renderNet();   // restore the forward pass for the current example
}

/* ------------------------------------------------- custom signal input */
function loadCustomSignal() {
  const msg = $('csvMsg');
  const parts = $('csvIn').value.trim().split(/[\s,;]+/).map(Number).filter((v) => isFinite(v));
  if (parts.length < 8) {
    msg.textContent = 'At least 8 numbers are needed. Found: ' + parts.length + '.';
    return;
  }
  const x = new Float32Array(WIN);
  if (parts.length === WIN) {
    for (let i = 0; i < WIN; i++) x[i] = parts[i];
  } else {
    for (let i = 0; i < WIN; i++) {                    // linear resampling to 128 samples
      const t = i * (parts.length - 1) / (WIN - 1);
      const i0 = Math.floor(t), f = t - i0;
      x[i] = parts[i0] * (1 - f) + parts[Math.min(parts.length - 1, i0 + 1)] * f;
    }
  }
  if ($('csvNorm').checked) {
    let mean = 0;
    for (let i = 0; i < WIN; i++) mean += x[i];
    mean /= WIN;
    let mx = 1e-9;
    for (let i = 0; i < WIN; i++) { x[i] -= mean; mx = Math.max(mx, Math.abs(x[i])); }
    for (let i = 0; i < WIN; i++) x[i] /= mx;
  }
  state.customSignal = x;
  state.probePick = 'custom';
  setStream(false);
  $('csvSrcWrap').classList.remove('hidden');
  buildProbeSelect();
  newProbe();
  renderNet();
  msg.innerHTML = parts.length + ' values' +
    (parts.length === WIN ? '' : ' → resampled to ' + WIN) +
    '. <b>Note:</b> the window is interpreted as 40 ms (2 cycles of 50 Hz), so the frequencies ' +
    'inside it must be on the same scale as the training data.';
}

function onDataStructureChanged() {
  regenData();
  buildProbeSelect();
  rebuildModel();
  evaluate();
  renderMetrics();
  renderNet();
}

/* ------------------------------------------------------ watermark panel */
function renderWmPanel() {
  const host = $('wmStats');
  const v = wm.last;
  let html = '';
  if (!v) {
    html = '<div class="verdict no">Not verified yet. Enable embedding, train the network ' +
      'and press <b>Verify model</b>.</div>';
  } else {
    const strong = v.p < 1e-6;
    const pTxt = v.p < 1e-15 ? v.p.toExponential(0) : v.p < 0.001 ? v.p.toExponential(1) : v.p.toFixed(4);
    html = '<div class="verdict ' + (strong ? 'yes' : 'no') + '">' +
      (strong
        ? '<b>The watermark is present.</b> Matching that many labels by chance is practically impossible.'
        : '<b>No evidence.</b> The matches are within the range of random guessing.') +
      '</div>' +
      '<table>' +
      '<tr><td>matches</td><td>' + v.matches + ' / ' + v.T + '</td></tr>' +
      '<tr><td>trigger accuracy</td><td>' + (v.acc * 100).toFixed(1) + '%</td></tr>' +
      '<tr><td>expected by chance</td><td>' + (100 / v.K).toFixed(1) + '%</td></tr>' +
      '<tr><td>p-value</td><td>' + pTxt + '</td></tr>' +
      '<tr><td>verified at epoch</td><td>' + v.epoch + '</td></tr>' +
      '</table>' +
      '<p class="muted">The p-value is P(Bin(' + v.T + ', 1/' + v.K + ') ≥ ' + v.matches +
      ') — the odds a foreign model matches that many labels by chance.</p>';
  }
  if (wm.on) {
    html += '<p class="muted">Embedding is on: ' + (wm.rate * 100).toFixed(0) +
      '% of each batch are triggers. Changing the class set changes their labels — ' +
      'the watermark then has to be embedded again.</p>';
  }
  if (state.head !== 'flat') {
    html += '<div class="verdict no" style="margin-top:8px">⚠ <b>This head will not carry the ' +
      'watermark.</b> Global Avg/Max Pool average the map over time, leaving only ' +
      (model ? model.finalC : '—') + ' numbers per trigger — a linear head cannot memorise ' + wm.T +
      ' arbitrary label assignments. Measured: 8/20 matches with GAP against <b>20/20 with Flatten</b>. ' +
      'Switch the output head to <b>Flatten</b> to embed.</div>';
  }
  host.innerHTML = html;

  const sw = $('wmSweep');
  drawWmSweep(dpiSetup(sw, sw.clientWidth || 460, 190), sw.clientWidth || 460, 190);
  const sg = $('wmSig');
  drawWmTrigger(dpiSetup(sg, sg.clientWidth || 290, 92), sg.clientWidth || 290, 92, 0);
}

/* ---------------------------------------------------------------- UI */
function bindUI() {
  $('btnPlay').onclick = () => {
    state.running = !state.running;
    $('btnPlay').textContent = state.running ? '⏸' : '▶';
    $('btnPlay').classList.toggle('on', state.running);
  };
  $('btnStep').onclick = () => {
    const target = state.epoch + 1;
    let guard = 0;
    while (state.epoch < target && guard++ < 5000) trainOneBatch();
    evaluate(); renderMetrics(); renderNet();
  };
  $('btnReset').onclick = () => {
    state.running = false;
    $('btnPlay').textContent = '▶';
    $('btnPlay').classList.remove('on');
    rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };

  $('lr').onchange = (e) => { state.lr = +e.target.value; };
  $('l2').onchange = (e) => { state.l2 = +e.target.value; };
  $('batch').onchange = (e) => { state.batch = +e.target.value; };
  $('act').onchange = (e) => { state.activation = e.target.value; rebuildModel(); evaluate(); renderMetrics(); };
  $('head').onchange = (e) => { state.head = e.target.value; rebuildModel(); evaluate(); renderMetrics(); };
  $('mode').onchange = (e) => { state.mode = e.target.value; };

  $('layPlus').onclick = () => {
    if (state.layers.length >= 4) return;
    const last = state.layers[state.layers.length - 1];
    state.layers.push({ filters: last.filters, kernel: last.kernel, pool: true });
    rebuildModel(); evaluate(); renderMetrics();
  };
  $('layMinus').onclick = () => {
    if (state.layers.length <= 1) return;
    state.layers.pop();
    rebuildModel(); evaluate(); renderMetrics();
  };

  const noise = $('noise'), strength = $('strength'), ntrain = $('ntrain');
  noise.oninput = () => { state.noise = +noise.value; $('noiseVal').textContent = state.noise.toFixed(3); };
  noise.onchange = () => { regenData(); };
  strength.oninput = () => { state.strength = +strength.value; $('strVal').textContent = state.strength.toFixed(2); };
  strength.onchange = () => { regenData(); };
  ntrain.oninput = () => { state.ntrain = +ntrain.value; $('ntrainVal').textContent = state.ntrain; };
  ntrain.onchange = () => { regenData(); };
  $('btnData').onclick = () => { regenData(); evaluate(); renderMetrics(); };

  document.querySelectorAll('.presets button').forEach((b) => {
    b.onclick = () => {
      const p = b.dataset.preset;
      if (p === 'binary') state.classes = ['clean', 'ripple'];
      else if (p === 'four') state.classes = ['clean', 'ripple', 'harm', 'spike'];
      else state.classes = CLASSES.map((c) => c.id);
      buildClassList();
      onDataStructureChanged();
    };
  });

  $('probeClass').onchange = (e) => {
    setStream(false);
    state.probePick = e.target.value; newProbe(); $('qtResult').innerHTML = '';
  };
  $('btnProbe').onclick = () => { setStream(false); newProbe(); };
  $('btnQuick').onclick = () => quickTest();
  $('csvLoad').onclick = () => loadCustomSignal();

  // --- live stream
  $('strToggle').onclick = () => {
    setStream(!stream.on);
    if (stream.on && stream.filled === 0) streamAdvance(VIEW, dataOpt());
  };
  $('strSpeed').onchange = (e) => { stream.speed = +e.target.value; };
  $('ctlRipple').onchange = (e) => { stream.ctrl.ripple = e.target.checked; };
  $('ctlHarm').onchange = (e) => { stream.ctrl.harm = e.target.checked; };
  $('ctlSag').onchange = (e) => { stream.ctrl.sag = e.target.checked; };
  $('ctlSpike').onclick = () => streamAddEvent('spike', state.strength);
  $('ctlBurst').onclick = () => streamAddEvent('burst', state.strength);
  $('ctlCsvSrc').onchange = (e) => {
    stream.source = e.target.checked ? 'csv' : 'gen';
    stream.hist.length = 0;      // old statistics do not apply to the new source
  };

  // --- unknown state
  $('oodOn').onchange = (e) => {
    ood.on = e.target.checked;
    if (ood.on && ood.score === 'knn' && !ood.stats) oodFitStats();
    evaluate(); renderMetrics(); renderOodPanel(true); renderNet();
  };
  $('oodScore').onchange = (e) => {
    ood.score = e.target.value;
    if (ood.score === 'knn') oodFitStats();
    renderOodPanel(true);
    if (ood.on) { evaluate(); renderMetrics(); }
    renderNet();
  };
  $('oodThr').oninput = (e) => {
    ood.thr[ood.score] = +e.target.value;
    renderOodPanel(false);
    if (ood.on) { evaluate(); renderMetrics(); }
    renderNet();
  };
  $('oodTarget').onchange = (e) => {
    ood.target = e.target.value === 'youden' ? 'youden' : +e.target.value;
    if (ood.cal) {
      oodCalibrate(); renderOodPanel(true);
      if (ood.on) { evaluate(); renderMetrics(); }
      renderNet();
    }
  };
  $('oodCal').onclick = () => {
    const btn = $('oodCal');
    btn.textContent = 'Calibrating…';
    setTimeout(() => {
      oodCalibrate();
      btn.textContent = 'Calibrate';
      renderOodPanel(true);
      if (ood.on) { evaluate(); renderMetrics(); }
      renderNet();
    }, 10);
  };

  // --- watermark
  $('wmOn').onchange = (e) => { wm.on = e.target.checked; renderWmPanel(); };
  $('wmKey').onchange = (e) => {
    wm.key = e.target.value || 'playground-key';
    wm.triggers = null; wm.last = null; wm.sweep = null;
    renderWmPanel();
  };
  $('wmT').onchange = (e) => {
    wm.T = +e.target.value; wm.triggers = null; wm.last = null; wm.sweep = null;
    renderWmPanel();
  };
  $('wmRate').onchange = (e) => { wm.rate = +e.target.value; renderWmPanel(); };
  $('wmVerifyBtn').onclick = () => { wmVerify(); renderWmPanel(); renderNet(); };
  $('wmPruneBtn').onclick = () => {
    const b = $('wmPruneBtn');
    b.textContent = 'Working…';
    setTimeout(() => {
      wmPruneSweep();
      b.textContent = 'Pruning test';
      renderWmPanel(); evaluate(); renderMetrics(); renderNet();
    }, 10);
  };
  $('wmFtBtn').onclick = () => {
    const b = $('wmFtBtn');
    b.textContent = 'Fine-tuning…';
    setTimeout(() => {
      wm.ftSnapshot = wmSnapshot();
      const was = wm.on;
      wm.on = false;                       // the attack uses clean data only
      const target = state.epoch + 20;
      let guard = 0;
      while (state.epoch < target && guard++ < 200000) trainOneBatch();
      wm.on = was;
      wmVerify();
      b.textContent = 'Attack: 20 clean epochs';
      $('wmUndoBtn').classList.remove('hidden');
      renderWmPanel(); evaluate(); renderMetrics(); renderNet();
    }, 10);
  };
  $('wmUndoBtn').onclick = () => {
    if (!wm.ftSnapshot) return;
    wmRestore(wm.ftSnapshot);
    wm.ftSnapshot = null;
    $('wmUndoBtn').classList.add('hidden');
    wmVerify(); renderWmPanel(); evaluate(); renderMetrics(); renderNet();
  };

  const canvas = $('net');
  canvas.addEventListener('mousemove', (ev) => {
    const r = canvas.getBoundingClientRect();
    state.hover = hitTest(layout, ev.clientX - r.left, ev.clientY - r.top);
    canvas.style.cursor = state.hover ? 'pointer' : 'default';
  });
  canvas.addEventListener('mouseleave', () => { state.hover = null; });
  canvas.addEventListener('click', (ev) => {
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const hit = hitTest(layout, mx, my);
    if (!hit) { state.selected = null; renderNet(); renderMath(true); return; }
    if (hit.type === 'filter') {
      // the horizontal position inside the box picks t
      const nd = layout.cols[hit.layer + 1].nodes[hit.ch];
      const st = model.stages[hit.layer];
      const f = Math.min(1, Math.max(0, (mx - nd.x - 4) / (nd.w - 8)));
      const tp = Math.round(f * (st.L - 1));
      state.tPos = st.pooled ? tp * 2 : tp;
    } else if (hit.type === 'input') {
      const nd = layout.cols[0].nodes[0];
      const f = Math.min(1, Math.max(0, (mx - nd.x - 5) / (nd.w - 10)));
      state.tPos = Math.round(f * (WIN - 1));
    }
    state.selected = hit;
    renderNet(); renderMath(true);
    $('mathPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  $('tpos').oninput = (e) => { state.tPos = +e.target.value; renderNet(); renderMath(true); };
  $('btnClearSel').onclick = () => { state.selected = null; renderNet(); renderMath(true); };

  window.addEventListener('resize', () => { renderNet(); renderMetrics(); });
}

/* --------------------------------------------------------------- boot */
function init() {
  buildClassList();
  buildProbeSelect();
  regenData();
  rebuildModel();
  bindUI();
  evaluate();
  renderMetrics();
  renderOodPanel(true);
  renderWmPanel();
  renderNet();
  requestAnimationFrame(loop);
}

document.addEventListener('DOMContentLoaded', init);
