/* main.js — application state, UI wiring and the training loop. */

const state = {
  arch: 'cnn',                 // 'cnn' | 'rnn'
  classes: ['clean', 'ripple', 'harm', 'spike'],
  layers: [
    { filters: 4, kernel: 5, pool: true },
    { filters: 4, kernel: 5, pool: true },
  ],
  rnnLayers: [{ units: 8, bidir: false }],
  ssmLayers: [{ units: 6 }],
  gnnLayers: [{ units: 8 }, { units: 8 }],
  agg: 'mean',
  cell: 'gru',
  ssmMode: 's4',
  stateDim: 8,
  readout: 'mean',
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
/** The layer list of whichever architecture is active. */
function archLayers() {
  if (state.arch === 'cnn') return state.layers;
  if (state.arch === 'rnn') return state.rnnLayers;
  return state.arch === 'ssm' ? state.ssmLayers : state.gnnLayers;
}

function rebuildModel() {
  if (state.arch === 'cnn') {
    model = new ConvNet1D({
      layers: JSON.parse(JSON.stringify(state.layers)),
      activation: state.activation,
      head: state.head,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'rnn') {
    model = new RNNNet({
      cell: state.cell,
      layers: JSON.parse(JSON.stringify(state.rnnLayers)),
      readout: state.readout,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'gnn') {
    model = new GNNNet({
      layers: JSON.parse(JSON.stringify(state.gnnLayers)),
      agg: state.agg,
      readout: state.readout,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else {
    model = new SSMNet({
      mode: state.ssmMode,
      layers: JSON.parse(JSON.stringify(state.ssmLayers)),
      stateDim: state.stateDim,
      readout: state.readout,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  }
  // the selection may point at a filter that no longer exists
  if (state.selected && state.selected.type === 'filter') {
    const st = model.stages[state.selected.layer];
    if (!st || state.selected.ch >= st.C) state.selected = null;
  }
  // the float snapshot belongs to the previous parameter objects
  quant.fp = null; quant.frozen = false; quant.metrics = null; quant.sweep = null;
  const fz = $('qFreeze'); if (fz) fz.checked = false;
  state.epoch = 0;
  hTrain = []; hTest = [];
  lastMetrics = null;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
  $('layCount').textContent = archLayers().length;
  buildLayerControls();
}

/** Switches between the convolutional and the recurrent playground. */
function setArch(arch) {
  if (state.arch === arch) return;
  state.arch = arch;
  state.selected = null;
  document.body.className = 'arch-' + arch;
  $('archCnn').classList.toggle('on', arch === 'cnn');
  $('archRnn').classList.toggle('on', arch === 'rnn');
  $('archSsm').classList.toggle('on', arch === 'ssm');
  $('archGnn').classList.toggle('on', arch === 'gnn');
  $('layerLbl').textContent = arch === 'cnn' ? 'Convolutional layers'
    : arch === 'rnn' ? 'Recurrent layers'
      : arch === 'ssm' ? 'State space layers' : 'Message passing layers';
  setStream(false);
  ood.cal = null; ood.stats = null;
  wm.last = null; wm.sweep = null;
  rebuildModel();
  evaluate(); renderMetrics(); renderOodPanel(true); renderWmPanel(); renderNet(); renderMath(true);
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
    // A full evaluation is a forward pass over 600 windows — far more expensive
    // than a training batch. Three times a second is plenty for the curves and
    // leaves the core to the actual training.
    if (frameNo % 20 === 0) { evaluate(); renderMetrics(); }
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

  if (quant.on) { quantEvaluate(); renderQuantPanel(); }
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
  if (model.kind === 'gnn' && model.graph) {
    // the graph is the same for every channel, so always show it
    const g = model.graph;
    let edges = 0, mxd = 0;
    for (let i = 0; i < WIN; i++) { edges += g.deg[i]; mxd = Math.max(mxd, g.deg[i]); }
    edges /= 2;
    title('VISIBILITY GRAPH OF THIS WINDOW', 10);
    drawGraphArcs(ctx, 0, 14, w, 96, g, state.probe, WIN,
      h0 && h0.type === 'filter' ? state.tPos : null);
    title('DEGREE PER NODE', 118);
    const degf = new Float64Array(WIN);
    for (let i = 0; i < WIN; i++) degf[i] = g.deg[i];
    drawSpectrum(ctx, 0, 122, w, 42, degf, mxd);
    const pinfo = probeInfo();
    txt.innerHTML = '<b>' + pinfo.name + '</b> · ' + edges + ' edges · mean degree ' +
      (2 * edges / WIN).toFixed(2) + ' · <b>max degree ' + mxd + '</b>' +
      '<br>The signal builds its own graph: a sample sees another when everything between them ' +
      'is lower. A lone impulse turns into a hub — max degree runs about twice as high for an ' +
      'impulse as for a clean sine, and that is structure the network can read.' +
      (h0 && h0.type === 'filter'
        ? '<br>Node ' + state.tPos + ' and its edges are marked; click a channel for the arithmetic.'
        : '<br>Click a channel below to expand the message passing for one node.');
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'ssm') {
    const st = model.stages[h0.layer];
    const layer = st.layer;
    if (st.kind === 's4') {
      // time-invariant, so the layer has an exact equivalent FIR kernel
      const k = layer.kernel(h0.ch, WIN);
      title('EQUIVALENT CONVOLUTION KERNEL', 10);
      let km = 1e-9;
      for (let i = 0; i < k.length; i++) km = Math.max(km, Math.abs(k[i]));
      drawWave(ctx, 0, 14, w, 52, k, 0, WIN, km);
      title('ITS FREQUENCY RESPONSE |H(f)|', 82);
      const m = magSpectrum(k, 0, WIN);
      drawSpectrum(ctx, 0, 86, w, 44, m, maxOf(m));
      ctx.fillStyle = '#98a2ad';
      ctx.font = '9px system-ui,sans-serif';
      ctx.fillText('0', 0, 140);
      ctx.fillText('1600 Hz', w - 42, 140);
      let peak = 0;
      for (let i = 1; i < m.length; i++) if (m[i] > m[peak]) peak = i;
      const dt = Math.exp(layer.pdt.W[h0.ch]);
      txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', channel ' + (h0.ch + 1) + '</b> · S4D · ' +
        layer.N + ' modes · Δ = ' + n4(dt) +
        '<br>Because the model is time-invariant, these ' + layer.N + ' state modes are exactly ' +
        'equivalent to the FIR kernel above — a convolution of the full window length, learned ' +
        'through a recurrence instead of stored tap by tap.' +
        '<br>|H(f)| peaks near <b>' + Math.round(peak / m.length * (SR / 2)) + ' Hz</b>.';
    } else {
      title('Δ(t) — WHAT THE MODEL LETS IN', 10);
      if (layer.dts) {
        const dts = new Float64Array(WIN);
        for (let t = 0; t < WIN; t++) dts[t] = layer.dts[h0.ch * WIN + t];
        let mx = 1e-9;
        for (let t = 0; t < WIN; t++) mx = Math.max(mx, dts[t]);
        drawSpectrum(ctx, 0, 14, w, 52, dts, mx);
      }
      title('CHANNEL OUTPUT OVER THE WINDOW', 82);
      if (st.snapshot) {
        let sc = 1e-6;
        for (let i = 0; i < st.snapshot.length; i++) sc = Math.max(sc, Math.abs(st.snapshot[i]));
        drawWave(ctx, 0, 86, w, 48, st.snapshot, h0.ch * st.L, st.L, sc);
      }
      let mn = Infinity, mx2 = -Infinity;
      if (layer.dts) {
        for (let t = 0; t < WIN; t++) {
          const v = layer.dts[h0.ch * WIN + t];
          mn = Math.min(mn, v); mx2 = Math.max(mx2, v);
        }
      }
      txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', channel ' + (h0.ch + 1) + '</b> · Mamba · ' +
        layer.N + ' modes<br>Δ(t) ranges ' + n4(mn) + ' … ' + n4(mx2) +
        ' across this window. Where Δ is large the input is written into the state; where it ' +
        'collapses the state coasts and the sample is ignored. That input dependence is the ' +
        'whole point of a selective SSM — and the reason it has no fixed kernel.';
    }
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'rnn') {
    const st = model.stages[h0.layer];
    const iw = model.unitInputWeights(h0.layer, h0.ch);
    const names = GATE_NAMES[st.kind];
    const back = st.bidir && h0.ch >= st.units;
    title('INPUT WEIGHTS PER GATE' + (back ? ' · BACKWARD UNIT' : ''), 10);
    const rowH = Math.min(22, 74 / iw.rows.length);
    iw.rows.forEach((row, g) => {
      const y = 14 + g * rowH;
      ctx.fillStyle = '#98a2ad';
      ctx.font = '9px system-ui,sans-serif';
      ctx.fillText(names[g][0], 0, y + rowH / 2 + 3);
      drawKernel(ctx, 14, y, w - 16, rowH - 3, row, 0, row.length);
    });
    title('HIDDEN STATE h(t) OVER THE WINDOW', 108);
    if (st.snapshot) {
      let sc = 1e-6;
      for (let i = 0; i < st.snapshot.length; i++) sc = Math.max(sc, Math.abs(st.snapshot[i]));
      drawWave(ctx, 0, 112, w, 52, st.snapshot, h0.ch * st.L, st.L, sc);
    }
    let mn = Infinity, mx = -Infinity;
    if (st.snapshot) {
      for (let t = 0; t < st.L; t++) {
        const v = st.snapshot[h0.ch * st.L + t];
        mn = Math.min(mn, v); mx = Math.max(mx, v);
      }
    }
    txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', unit ' + (h0.ch + 1) + '</b> · ' +
      st.kind.toUpperCase() + ' · ' + iw.rows.length + ' gate' + (iw.rows.length > 1 ? 's' : '') +
      ' × ' + iw.rows[0].length + ' input channel' + (iw.rows[0].length > 1 ? 's' : '') +
      (back ? ' · reads the window backwards' : '') +
      '<br>State range over the window: ' + n3(mn) + ' … ' + n3(mx) +
      '<br>Unlike a convolution, this unit sees <b>everything up to t</b>, not a fixed window.';
    return;
  }
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
  if (sel.type === 'filter') {
    if (model.kind === 'rnn') renderUnitMath(host, title, slider, sel);
    else if (model.kind === 'gnn') renderGnnMath(host, title, slider, sel);
    else if (model.kind === 'ssm') renderSsmMath(host, title, slider, sel);
    else renderFilterMath(host, title, slider, sel);
  }
  else if (sel.type === 'input') renderInputMath(host, title, slider);
  else renderOutputMath(host, title, slider);
}

const GATE_NAMES = {
  rnn: [['h', 'tanh']],
  gru: [['z', 'σ'], ['r', 'σ'], ['n', 'tanh']],
  lstm: [['i', 'σ'], ['f', 'σ'], ['o', 'σ'], ['g', 'tanh']],
};

/* ------------------------------------------------------- recurrent unit */
function renderUnitMath(host, title, slider, sel) {
  const li = sel.layer, unit = sel.ch;
  const st = model.stages[li];
  if (!st || unit >= st.C) { state.selected = null; return renderMathInner(host); }

  slider.disabled = false;
  slider.max = WIN - 1;
  const t = Math.min(state.tPos, WIN - 1);
  state.tPos = t;
  slider.value = t;
  $('tposVal').textContent = 't = ' + t + '  (' + (t / SR * 1000).toFixed(2) + ' ms)';

  const d = model.stepDetail(li, unit, t);
  const dirTxt = d.back ? ' (backward unit — reads the window right to left)' : '';
  title.textContent = 'Layer ' + (li + 1) + ' · unit ' + (unit + 1) + ' · step t = ' + t +
    ' · ' + d.kind.toUpperCase() + dirTxt;

  const gates = GATE_NAMES[d.kind];
  let html = '';

  /* --- 1. the recurrence --- */
  const formulas = {
    rnn: 'h<sub>t</sub> = tanh( W<sub>x</sub>·x<sub>t</sub> + W<sub>h</sub>·h<sub>t−1</sub> + b )',
    gru: 'z = σ(·) &nbsp; r = σ(·) &nbsp; n = tanh( W<sub>n</sub>x<sub>t</sub> + r ⊙ (U<sub>n</sub>h<sub>t−1</sub>) + b<sub>n</sub> ) ' +
         '&nbsp;→&nbsp; h<sub>t</sub> = (1−z)⊙n + z⊙h<sub>t−1</sub>',
    lstm: 'i, f, o = σ(·) &nbsp; g = tanh(·) &nbsp;→&nbsp; c<sub>t</sub> = f⊙c<sub>t−1</sub> + i⊙g ' +
          '&nbsp;→&nbsp; h<sub>t</sub> = o⊙tanh(c<sub>t</sub>)',
  };
  html += '<h4>1 · The recurrence — ' + d.kind.toUpperCase() + '</h4>';
  html += '<div class="formula">' + formulas[d.kind] + '</div>';

  /* --- 2. gate arithmetic --- */
  html += '<h4>2 · What each gate computes at this step</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>gate</th>' +
    '<th>W<sub>x</sub>·x<sub>t</sub></th><th>W<sub>h</sub>·h<sub>t−1</sub></th>' +
    '<th>bias</th><th>pre-activation</th><th>value</th></tr></thead><tbody>';
  for (let g = 0; g < gates.length; g++) {
    let ix = 0;
    for (let i = 0; i < d.D; i++) ix += d.wx[g][i] * d.xv[i];
    let ih = 0;
    for (let v = 0; v < d.H; v++) ih += d.wh[g][v] * d.hprev[v];
    const isCand = d.kind === 'gru' && g === 2;
    const rec = isCand ? d.gates[1] * d.q : ih;       // GRU candidate is gated by r
    const z = ix + rec + d.bias[g];
    html += '<tr><td class="ch">' + gates[g][0] + ' = ' + gates[g][1] + '(·)</td>' +
      '<td>' + n4(ix) + '</td>' +
      '<td>' + n4(rec) + (isCand ? ' <span style="color:#98a2ad">= r·' + n3(d.q) + '</span>' : '') + '</td>' +
      '<td>' + n3(d.bias[g]) + '</td>' +
      '<td>' + n4(z) + '</td>' +
      '<td class="sum">' + n4(d.gates[g]) + '</td></tr>';
  }
  html += '</tbody></table></div>';

  /* --- 3. the weights behind those sums --- */
  html += '<h4>3 · The weights that produced them</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>gate</th>';
  for (let i = 0; i < d.D; i++) html += '<th>W<sub>x</sub>[' + i + ']<br>x=' + n3(d.xv[i]) + '</th>';
  for (let v = 0; v < d.H; v++) html += '<th>W<sub>h</sub>[' + v + ']<br>h=' + n3(d.hprev[v]) + '</th>';
  html += '</tr></thead><tbody>';
  for (let g = 0; g < gates.length; g++) {
    html += '<tr><td class="ch">' + gates[g][0] + '</td>';
    for (let i = 0; i < d.D; i++) {
      html += '<td class="cell"><span class="wv" style="color:' + wColor(d.wx[g][i]) + '">' +
        n3(d.wx[g][i]) + '</span><span class="pv">' + n3(d.wx[g][i] * d.xv[i]) + '</span></td>';
    }
    for (let v = 0; v < d.H; v++) {
      html += '<td class="cell"><span class="wv" style="color:' + wColor(d.wh[g][v]) + '">' +
        n3(d.wh[g][v]) + '</span><span class="pv">' + n3(d.wh[g][v] * d.hprev[v]) + '</span></td>';
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';

  /* --- 4. the state update --- */
  html += '<h4>4 · The new state</h4>';
  const hprevU = d.hprev[d.u];
  if (d.kind === 'lstm') {
    html += '<div class="formula">c<sub>t</sub> = f·c<sub>t−1</sub> + i·g = ' +
      n3(d.gates[1]) + '·' + n3(d.cprev) + ' + ' + n3(d.gates[0]) + '·' + n3(d.gates[3]) +
      ' = <span class="res">' + n4(d.c) + '</span></div>';
    html += '<div class="formula" style="margin-top:6px">h<sub>t</sub> = o·tanh(c<sub>t</sub>) = ' +
      n3(d.gates[2]) + '·' + n3(Math.tanh(d.c)) + ' = <span class="res">' + n4(d.h) + '</span>' +
      (Math.abs(d.c) > 2.5 ? '  <span class="op">— the cell is saturating, tanh′ ≈ 0 here</span>' : '') +
      '</div>';
  } else if (d.kind === 'gru') {
    html += '<div class="formula">h<sub>t</sub> = (1−z)·n + z·h<sub>t−1</sub> = ' +
      n3(1 - d.gates[0]) + '·' + n3(d.gates[2]) + ' + ' + n3(d.gates[0]) + '·' + n3(hprevU) +
      ' = <span class="res">' + n4(d.h) + '</span>' +
      '  <span class="op">— z is how much of the old state is kept</span></div>';
  } else {
    html += '<div class="formula">h<sub>t</sub> = tanh(pre-activation) = <span class="res">' +
      n4(d.h) + '</span>  <span class="op">(previous state of this unit: ' + n3(hprevU) + ')</span></div>';
  }
  if (st.snapshot) {
    const drawn = st.snapshot[unit * st.L + t];
    html += '<div class="formula" style="margin-top:6px"><span class="op">check: the unit map ' +
      'holds ' + n4(drawn) + ' at t=' + t +
      (Math.abs(drawn - d.h) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  }

  /* --- 5. downstream --- */
  html += rnnDownstreamHtml(li, unit);
  host.innerHTML = html;
}

/* ------------------------------------------------------- graph node/channel */
function renderGnnMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C) { state.selected = null; return renderMathInner(host); }

  slider.disabled = false;
  slider.max = WIN - 1;
  const node = Math.min(state.tPos, WIN - 1);
  state.tPos = node;
  slider.value = node;
  $('tposVal').textContent = 'node ' + node + '  (' + (node / SR * 1000).toFixed(2) + ' ms)';
  title.textContent = 'Layer ' + (li + 1) + ' · channel ' + (ch + 1) + ' · node ' + node;

  const d = model.stepDetail(li, ch, node);
  const fnames = li === 0 ? ['x', 'Δx', '|x|'] : null;
  const fname = (i) => fnames ? fnames[i] : 'h' + i;

  let html = '<h4>1 · The graph</h4>';
  html += '<div class="formula">Two samples i &lt; j are neighbours when every sample between ' +
    'them is lower than both — a horizontal visibility graph. Nothing is learned here; the ' +
    'topology is a function of the waveform.</div>';
  html += '<div class="formula" style="margin-top:6px">Node <b>' + node + '</b> has degree ' +
    '<span class="res">' + d.degree + '</span>: neighbours ' +
    (d.neigh.length ? d.neigh.join(', ') : '—') +
    '  <span class="op">— an impulse becomes a hub that sees most of the window; a clean sine ' +
    'gives an almost regular graph</span></div>';

  html += '<h4>2 · Message passing</h4>';
  html += '<div class="formula">h′<sub>i</sub> = ReLU( W<sub>self</sub>·h<sub>i</sub> + ' +
    'W<sub>neigh</sub>·' + d.agg_kind + '<sub>j∈N(i)</sub>(h<sub>j</sub>) + b )</div>';

  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr>' +
    '<th>input</th><th>own value</th><th>W<sub>self</sub></th><th>product</th>' +
    '<th>' + d.agg_kind + ' of neighbours</th><th>W<sub>neigh</sub></th><th>product</th>' +
    '</tr></thead><tbody>';
  let sSelf = 0, sNeigh = 0;
  for (let i = 0; i < d.D; i++) {
    const ps = d.ws[i] * d.self[i], pn = d.wn[i] * d.agg[i];
    sSelf += ps; sNeigh += pn;
    html += '<tr><td class="ch">' + fname(i) + '</td>' +
      '<td>' + n3(d.self[i]) + '</td>' +
      '<td style="color:' + wColor(d.ws[i]) + '">' + n3(d.ws[i]) + '</td>' +
      '<td>' + n4(ps) + '</td>' +
      '<td>' + n3(d.agg[i]) + '</td>' +
      '<td style="color:' + wColor(d.wn[i]) + '">' + n3(d.wn[i]) + '</td>' +
      '<td class="sum">' + n4(pn) + '</td></tr>';
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">self ' + n4(sSelf) + '  +  neighbours ' +
    n4(sNeigh) + '  +  bias ' + n3(d.bias) + '  =  <span class="res">' + n4(d.pre) + '</span>' +
    '  <span class="op">→</span>  ReLU  <span class="op">→</span>  <span class="res' +
    (d.out === 0 ? ' warn' : '') + '">' + n4(d.out) + '</span></div>';

  if (d.neigh.length) {
    html += '<h4>3 · Where the aggregated value came from</h4>';
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>neighbour</th>';
    for (let i = 0; i < d.D; i++) html += '<th>' + fname(i) + '</th>';
    html += '</tr></thead><tbody>';
    const src = li === 0 ? model.feat : model.stages[li - 1].snapshot;
    for (const j of d.neigh.slice(0, 12)) {
      html += '<tr><td class="ch">node ' + j + '</td>';
      for (let i = 0; i < d.D; i++) html += '<td>' + n3(src[i * WIN + j]) + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table></div>';
    if (d.neigh.length > 12) {
      html += '<div class="formula" style="margin-top:6px"><span class="op">showing 12 of ' +
        d.neigh.length + ' neighbours</span></div>';
    }
  }

  html += rnnDownstreamHtml(li, ch, d.neigh.length ? 4 : 3);
  host.innerHTML = html;
}

/* ---------------------------------------------------- state space channel */
function renderSsmMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C) { state.selected = null; return renderMathInner(host); }

  slider.disabled = false;
  slider.max = WIN - 1;
  const t = Math.min(state.tPos, WIN - 1);
  state.tPos = t;
  slider.value = t;
  $('tposVal').textContent = 't = ' + t + '  (' + (t / SR * 1000).toFixed(2) + ' ms)';

  const d = model.stepDetail(li, ch, t);
  const isS4 = d.mode === 's4';
  title.textContent = 'Layer ' + (li + 1) + ' · channel ' + (ch + 1) + ' · step t = ' + t +
    ' · ' + (isS4 ? 'S4D' : 'Mamba (selective)');

  let html = '<h4>1 · The state space model</h4>';
  html += '<div class="formula">x<sub>k</sub> = Ā ⊙ x<sub>k−1</sub> + B̄ u<sub>k</sub>' +
    ' &nbsp;&nbsp; y<sub>k</sub> = ' + (isS4 ? '2·Re( Σ<sub>n</sub> C<sub>n</sub> x<sub>k,n</sub> )'
      : 'Σ<sub>n</sub> C<sub>n</sub>(t) x<sub>k,n</sub>') + ' + D·u<sub>k</sub><br>' +
    'Ā = exp(Δ·A)' + (isS4 ? ' , &nbsp; B̄ = (Ā − 1)/A &nbsp; (zero-order hold, B = 1)'
      : '(t) , &nbsp; B̄ = Δ(t)·B(t)') + '</div>';
  html += '<div class="formula" style="margin-top:6px">' + (isS4
    ? '<span class="op">A is complex and diagonal, and nothing here depends on the input — the ' +
      'layer is one fixed convolution. Hover the channel to see that kernel and its frequency response.</span>'
    : '<span class="op">A is real, but Δ, B and C are produced from the input at every step. The ' +
      'model is no longer time-invariant, so no single kernel exists — selectivity replaces it.</span>') +
    '</div>';

  /* --- discretisation / step-dependent quantities --- */
  html += '<h4>2 · ' + (isS4 ? 'Discretisation of each mode' : 'What the input selects at this step') + '</h4>';
  if (!isS4) {
    html += '<div class="formula">Δ(t) = softplus( ' + n3(Math.log(d.dtBase)) + ' + ' +
      n3(d.dtW) + '·u<sub>t</sub> ) = softplus( ' + n3(Math.log(d.dtBase) + d.dtW * d.u) +
      ' ) = <span class="res">' + n4(d.dt) + '</span>' +
      '  <span class="op">— large Δ writes the input into the state, Δ→0 ignores it</span></div>';
  }
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>mode n</th>' +
    (isS4
      ? '<th>A<sub>n</sub></th><th>Ā<sub>n</sub> = exp(ΔA)</th><th>|Ā|</th><th>freq [Hz]</th><th>B̄<sub>n</sub></th><th>C<sub>n</sub></th>'
      : '<th>A<sub>n</sub></th><th>Ā<sub>n</sub>(t)</th><th>B(t)</th><th>C(t)</th>') +
    '</tr></thead><tbody>';
  for (const m of d.modes) {
    if (isS4) {
      const mag = Math.hypot(m.abarRe, m.abarIm);
      const hz = Math.abs(Math.atan2(m.abarIm, m.abarRe)) / (2 * Math.PI) * SR;
      html += '<tr><td class="ch">' + m.n + '</td>' +
        '<td>' + n3(m.aRe) + (m.aIm >= 0 ? ' + ' : ' − ') + n3(Math.abs(m.aIm)) + 'i</td>' +
        '<td>' + n3(m.abarRe) + (m.abarIm >= 0 ? ' + ' : ' − ') + n3(Math.abs(m.abarIm)) + 'i</td>' +
        '<td>' + n4(mag) + '</td><td class="sum">' + Math.round(hz) + '</td>' +
        '<td>' + n3(m.bbarRe) + (m.bbarIm >= 0 ? ' + ' : ' − ') + n3(Math.abs(m.bbarIm)) + 'i</td>' +
        '<td>' + n3(m.cRe) + (m.cIm >= 0 ? ' + ' : ' − ') + n3(Math.abs(m.cIm)) + 'i</td></tr>';
    } else {
      html += '<tr><td class="ch">' + m.n + '</td><td>' + n3(m.aRe) + '</td>' +
        '<td>' + n4(m.abarRe) + '</td><td>' + n3(m.B) + '</td><td>' + n3(m.C) + '</td></tr>';
    }
  }
  html += '</tbody></table></div>';
  if (isS4) {
    html += '<div class="formula" style="margin-top:6px"><span class="op">|Ā| is how much of the ' +
      'state survives one sample (memory ≈ 1/(1−|Ā|) samples); the frequency column is where that ' +
      'mode resonates, from the angle of Ā. Δ = ' + n4(d.dt) + ' for this channel.</span></div>';
  }

  /* --- the update at this step --- */
  html += '<h4>3 · The update at t = ' + t + '</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>mode</th>' +
    '<th>Ā·x<sub>t−1</sub></th><th>B̄·u<sub>t</sub></th><th>x<sub>t</sub></th>' +
    '<th>contribution to y</th></tr></thead><tbody>';
  let ySum = 0;
  for (const m of d.modes) {
    if (isS4) {
      const [ar, ai] = cmul(m.abarRe, m.abarIm, m.prevRe, m.prevIm);
      const contrib = 2 * (m.cRe * m.xRe - m.cIm * m.xIm);
      ySum += contrib;
      html += '<tr><td class="ch">' + m.n + '</td>' +
        '<td>' + n3(ar) + (ai >= 0 ? '+' : '−') + n3(Math.abs(ai)) + 'i</td>' +
        '<td>' + n3(m.bbarRe * d.u) + (m.bbarIm * d.u >= 0 ? '+' : '−') + n3(Math.abs(m.bbarIm * d.u)) + 'i</td>' +
        '<td>' + n3(m.xRe) + (m.xIm >= 0 ? '+' : '−') + n3(Math.abs(m.xIm)) + 'i</td>' +
        '<td class="sum">' + n4(contrib) + '</td></tr>';
    } else {
      const contrib = m.C * m.xRe;
      ySum += contrib;
      html += '<tr><td class="ch">' + m.n + '</td><td>' + n4(m.abarRe * m.prevRe) + '</td>' +
        '<td>' + n4(d.dt * m.B * d.u) + '</td><td>' + n4(m.xRe) + '</td>' +
        '<td class="sum">' + n4(contrib) + '</td></tr>';
    }
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">y<sub>t</sub> = ' + n4(ySum) +
    ' <span class="op">(from the states)</span> + D·u<sub>t</sub> = ' + n3(d.Dskip) + '·' + n3(d.u) +
    ' = <span class="res">' + n4(ySum + d.Dskip * d.u) + '</span></div>';
  html += '<div class="formula" style="margin-top:6px">' + (isS4
    ? 'then SiLU: out = y·σ(y) = <span class="res">' + n4(st.snapshot ? st.snapshot[ch * st.L + t] : 0) + '</span>'
    : 'then the Mamba gate: out = y · SiLU(gate) = <span class="res">' +
      n4(st.snapshot ? st.snapshot[ch * st.L + t] : 0) + '</span>') +
    '  <span class="op">— the recurrence is linear, so without this the whole stack would collapse ' +
    'into one linear map</span></div>';

  html += rnnDownstreamHtml(li, ch, 4);
  host.innerHTML = html;
}

/** Where a hidden unit's sequence goes: the next layer, or the readout and logits. */
function rnnDownstreamHtml(li, unit, num) {
  const st = model.stages[li];
  const last = li === model.stages.length - 1;
  let html = '<h4>' + (num || 5) + ' · Where this unit goes</h4>';

  if (!last) {
    const nx = model.stages[li + 1];
    html += '<div class="formula">This unit is <b>input channel ' + unit + '</b> for all ' +
      nx.C + ' units of layer ' + (li + 2) + '. Summed gate weights reading it:</div>';
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>unit in layer ' + (li + 2) +
      '</th><th>Σ|w| over gates</th></tr></thead><tbody>';
    for (let co = 0; co < nx.C; co++) {
      const s = stageLink(model, li + 1, co, unit);
      html += '<tr><td class="ch">unit ' + (co + 1) + '</td><td class="sum">' + n3(s.mag) + '</td></tr>';
    }
    html += '</tbody></table></div>';
    return html;
  }

  const snap = st.snapshot, L = st.L, d = model.dense, cls = activeClasses();
  let h = 0, argmax = 0, txt = '';
  if (state.readout === 'mean') {
    for (let i = 0; i < L; i++) h += snap[unit * L + i];
    h /= L;
    txt = 'Mean over time: h[' + unit + '] = (sum of ' + L + ' states) / ' + L +
      ' = <span class="res">' + n4(h) + '</span>';
  } else if (state.readout === 'max') {
    h = -Infinity;
    for (let i = 0; i < L; i++) if (snap[unit * L + i] > h) { h = snap[unit * L + i]; argmax = i; }
    txt = 'Max over time: h[' + unit + '] = <span class="res">' + n4(h) +
      '</span> <span class="op">(at t = ' + argmax + ')</span>';
  } else {
    h = snap[unit * L + (L - 1)];
    txt = 'Last state: h[' + unit + '] = state at t = ' + (L - 1) +
      ' = <span class="res">' + n4(h) + '</span>' +
      (st.bidir && unit >= st.units
        ? ' <span class="op">— for a backward unit this is the state after reading the whole window right to left, i.e. the state at t = 0</span>'
        : '');
  }
  html += '<div class="formula">' + txt + '</div>';

  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr>' +
    '<th>class</th><th>weight</th><th>contribution</th><th>class bias</th>' +
    '<th>logit</th><th>softmax</th></tr></thead><tbody>';
  for (let j = 0; j < d.nout; j++) {
    const w = d.W[j * d.nin + unit];
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[j].color + '"></span> ' +
      cls[j].name + '</td>' +
      '<td style="color:' + wColor(w) + '">' + n4(w) + '</td>' +
      '<td style="color:' + wColor(w * h) + '">' + n4(w * h) + '</td>' +
      '<td>' + n3(d.b[j]) + '</td>' +
      '<td class="sum">' + n3(model.logits[j]) + '</td>' +
      '<td>' + (model.probs ? (model.probs[j] * 100).toFixed(1) + '%' : '—') + '</td></tr>';
  }
  html += '</tbody></table></div>';
  return html;
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

/** Human-readable name of whatever collapses the sequence before the linear layer. */
function headName() {
  if (state.arch === 'rnn') {
    return state.readout === 'mean' ? 'the mean over time'
      : state.readout === 'max' ? 'the max over time' : 'the last state';
  }
  return state.head === 'gap' ? 'Global Average Pool'
    : state.head === 'gmp' ? 'Global Max Pool' : 'Flatten';
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
    '   <span class="op">(h is the output of ' + headName() +
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
  if (state.arch === 'ssm' || state.arch === 'gnn') {
    archLayers().forEach((ls) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML = '<div class="row"><button data-a="m">−</button><b>' + ls.units +
        '</b><button data-a="p">+</button></div><div class="row"><span style="font-size:10px;color:#7b8794">channels</span></div>';
      card.querySelector('[data-a=m]').onclick = () => { if (ls.units > 1) { ls.units--; rebuildModel(); } };
      card.querySelector('[data-a=p]').onclick = () => { if (ls.units < 10) { ls.units++; rebuildModel(); } };
      host.appendChild(card);
    });
    return;
  }
  if (state.arch === 'rnn') {
    state.rnnLayers.forEach((ls) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML =
        '<div class="row"><button data-a="m">−</button><b>' + ls.units + '</b><button data-a="p">+</button></div>' +
        '<div class="row"><label><input type="checkbox" data-a="bi"' +
        (ls.bidir ? ' checked' : '') + '>bidir</label></div>';
      card.querySelector('[data-a=m]').onclick = () => { if (ls.units > 1) { ls.units--; rebuildModel(); } };
      card.querySelector('[data-a=p]').onclick = () => { if (ls.units < 10) { ls.units++; rebuildModel(); } };
      card.querySelector('[data-a=bi]').onchange = (e) => { ls.bidir = e.target.checked; rebuildModel(); };
      host.appendChild(card);
    });
    return;
  }
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

/* ---------------------------------------------------- quantisation panel */
function renderQuantPanel() {
  $('qBitsVal').textContent = quant.bits;
  const sw = $('qSweep');
  drawQuantSweep(dpiSetup(sw, sw.clientWidth || 460, 170), sw.clientWidth || 460, 170);

  // the biggest weight tensor is the most interesting one to look at
  let big = null;
  if (model) for (const p of model.params) if (!big || p.W.length > big.W.length) big = p;
  const hs = $('qHist');
  drawQuantHist(dpiSetup(hs, hs.clientWidth || 600, 96), hs.clientWidth || 600, 96, big);

  const host = $('qStats');
  const m = quant.metrics;
  if (!quant.on || !m) {
    host.innerHTML = '<p class="muted">Tick <b>compare against float</b> to evaluate a quantised ' +
      'copy next to the float model on every metrics update, or press <b>Bit sweep</b> for the ' +
      'whole curve at once.</p>';
    return;
  }
  const drop = (m.fpAcc - m.qAcc) * 100;
  let sq = 0;
  for (const v of m.sqnr) if (isFinite(v)) sq = sq === 0 ? v : Math.min(sq, v);
  const bytesFp = model.paramCount() * 4;
  const bytesQ = Math.ceil(model.paramCount() * quant.bits / 8);
  host.innerHTML =
    '<div class="verdict ' + (drop < 1 ? 'yes' : 'no') + '">' +
    (drop < 1
      ? '<b>' + quant.bits + ' bits is free here.</b> The drop is within evaluation noise.'
      : '<b>' + quant.bits + ' bits costs ' + drop.toFixed(1) + ' points.</b>') +
    '</div><table>' +
    '<tr><td>float32 accuracy</td><td>' + (m.fpAcc * 100).toFixed(1) + '%</td></tr>' +
    '<tr><td>quantised accuracy</td><td>' + (m.qAcc * 100).toFixed(1) + '%</td></tr>' +
    '<tr><td>test loss, float → int</td><td>' + m.fpLoss.toFixed(3) + ' → ' + m.qLoss.toFixed(3) + '</td></tr>' +
    '<tr><td>worst tensor SQNR</td><td>' + (isFinite(sq) ? sq.toFixed(1) + ' dB' : '—') + '</td></tr>' +
    '<tr><td>weight memory</td><td>' + bytesFp + ' → ' + bytesQ + ' B</td></tr>' +
    '</table>' +
    '<p class="muted">SQNR is the signal-to-quantisation-noise ratio of the worst weight tensor; ' +
    'each extra bit is worth about 6 dB. Memory counts weights only, at ' + quant.bits +
    ' bits each against 32.</p>';
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
  if (wm.post) {
    const p = wm.post;
    const drop = (p.before.acc - p.after.acc) * 100;
    html += '<div class="verdict ' + (p.after.v.p < 1e-6 ? 'yes' : 'no') + '" style="margin-top:8px">' +
      '<b>' + (p.mode === 'lora' ? 'LoRA rank ' + p.rank : 'Full fine-tune') + ', ' + p.epochs +
      ' epochs.</b><br>Clean accuracy ' + (p.before.acc * 100).toFixed(1) + '% → ' +
      (p.after.acc * 100).toFixed(1) + '% (' + (drop >= 0 ? '−' : '+') + Math.abs(drop).toFixed(1) +
      ' points) · triggers ' + p.before.v.matches + '/' + p.before.v.T + ' → <b>' +
      p.after.v.matches + '/' + p.after.v.T + '</b><br>Trained parameters: ' + p.touched +
      ' of ' + p.total + (p.mode === 'lora' ? ' — the base weights never moved until the merge' : '') +
      '</div>';
  }
  if (wm.on) {
    html += '<p class="muted">Embedding is on: ' + (wm.rate * 100).toFixed(0) +
      '% of each batch are triggers. Changing the class set changes their labels — ' +
      'the watermark then has to be embedded again.</p>';
  }
  if (state.arch !== 'cnn') {
    html += '<div class="verdict no" style="margin-top:8px">⚠ <b>A sequence readout collapses the ' +
      'sequence to ' + (model ? model.finalC : '—') + ' numbers</b>, so a linear head cannot memorise ' +
      wm.T + ' arbitrary label assignments. Watermark capacity here is far lower than with the ' +
      'convolutional Flatten head — expect the verification to stay near chance.</div>';
  } else if (state.head !== 'flat') {
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
    const arr = archLayers();
    if (arr.length >= (state.arch === 'cnn' ? 4 : 2)) return;
    const last = arr[arr.length - 1];
    arr.push(state.arch === 'cnn'
      ? { filters: last.filters, kernel: last.kernel, pool: true }
      : state.arch === 'rnn' ? { units: last.units, bidir: last.bidir }
        : { units: last.units });
    rebuildModel(); evaluate(); renderMetrics();
  };
  $('layMinus').onclick = () => {
    const arr = archLayers();
    if (arr.length <= 1) return;
    arr.pop();
    rebuildModel(); evaluate(); renderMetrics();
  };

  $('archCnn').onclick = () => setArch('cnn');
  $('archRnn').onclick = () => setArch('rnn');
  $('archSsm').onclick = () => setArch('ssm');
  $('archGnn').onclick = () => setArch('gnn');
  $('agg').onchange = (e) => {
    state.agg = e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('ssmMode').onchange = (e) => {
    state.ssmMode = e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('stateDim').onchange = (e) => {
    state.stateDim = +e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('cell').onchange = (e) => {
    state.cell = e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('readout').onchange = (e) => {
    state.readout = e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
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

  // --- quantisation
  $('qOn').onchange = (e) => {
    quant.on = e.target.checked;
    if (!quant.on && quant.frozen) {           // put the float weights back
      if (quant.fp) { quantLoadFp(quant.fp); quant.fp = null; }
      quant.frozen = false; $('qFreeze').checked = false;
    }
    if (quant.on) quantEvaluate();
    renderQuantPanel(); renderNet();
  };
  $('qBits').oninput = (e) => {
    quant.bits = +e.target.value;
    if (quant.frozen && quant.fp) { quantLoadFp(quant.fp); quantiseModel(quant.bits); }
    if (quant.on) quantEvaluate();
    renderQuantPanel(); renderNet();
  };
  const qRe = () => {
    if (quant.frozen && quant.fp) { quantLoadFp(quant.fp); quantiseModel(quant.bits); }
    if (quant.on) quantEvaluate();
    renderQuantPanel(); renderNet();
  };
  $('qPerCh').onchange = (e) => { quant.perChannel = e.target.checked; qRe(); };
  $('qSym').onchange = (e) => { quant.symmetric = e.target.checked; qRe(); };
  $('qFreeze').onchange = (e) => {
    if (e.target.checked) {
      quant.fp = quantSaveFp();
      quantiseModel(quant.bits);
      quant.frozen = true;
    } else {
      if (quant.fp) quantLoadFp(quant.fp);
      quant.fp = null; quant.frozen = false;
    }
    if (quant.on) quantEvaluate();
    evaluate(); renderMetrics(); renderQuantPanel(); renderNet();
  };
  $('qSweepBtn').onclick = () => {
    const b = $('qSweepBtn');
    b.textContent = 'Working…';
    setTimeout(() => {
      quantSweep();
      b.textContent = 'Bit sweep';
      renderQuantPanel();
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
  $('wmEmbedBtn').onclick = () => {
    const b = $('wmEmbedBtn');
    b.textContent = 'Embedding…';
    setTimeout(() => {
      wmEmbedPost($('wmPostMode').value, +$('wmPostEp').value, wm.rate, +$('wmRank').value);
      b.textContent = 'Embed now';
      $('wmUndoPostBtn').classList.remove('hidden');
      renderWmPanel(); evaluate(); renderMetrics(); renderNet();
    }, 10);
  };
  $('wmUndoPostBtn').onclick = () => {
    wmUndoPost();
    $('wmUndoPostBtn').classList.add('hidden');
    renderWmPanel(); evaluate(); renderMetrics(); renderNet();
  };
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
  document.body.className = 'arch-' + state.arch;
  buildClassList();
  buildProbeSelect();
  regenData();
  rebuildModel();
  bindUI();
  evaluate();
  renderMetrics();
  renderOodPanel(true);
  renderWmPanel();
  renderQuantPanel();
  renderNet();
  requestAnimationFrame(loop);
}

document.addEventListener('DOMContentLoaded', init);
