/* main.js — application state, UI wiring and the training loop. */

const state = {
  arch: 'cnn',                 // 'mlp' | 'cnn' | 'rnn' | 'ssm' | 'gnn' | 'kan'
  classes: ['clean', 'ripple', 'harm', 'spike'],
  layers: [
    { filters: 4, kernel: 5, pool: true },
    { filters: 4, kernel: 5, pool: true },
  ],
  rnnLayers: [{ units: 8, bidir: false }],
  ssmLayers: [{ units: 6 }],
  gnnLayers: [{ units: 8 }, { units: 8 }],
  mlpLayers: [{ units: 8 }],
  kanLayers: [{ units: 4 }, { units: 4 }],
  resLayers: [{ filters: 6, kernels: '8-5-3' }, { filters: 6, kernels: '8-5-3' }],
  incLayers: [{ filters: 2 }, { filters: 2 }],
  tfLayers: [{}, {}],           // Transformer encoder layers
  tfD: 8,                      // Transformer width: numbers per token
  tfHeads: 2,
  tfCausal: false,
  tfTok: 'conv',               // Transformer tokenizer: 'conv' (CCT) or 'linear' (ViT patches)
  skip: true,                  // shortcuts in ResNet-1D and InceptionTime
  bn: true,                    // layer norm in ResNet-1D and InceptionTime
  residual: false,             // ResNet skips in the convolutional playground
  agg: 'mean',
  cell: 'gru',
  ssmMode: 's4',
  stateDim: 8,
  readout: 'mean',
  activation: 'relu',
  head: 'gap',
  causal: false,
  lr: 0.003,
  l2: 0,
  batch: 16,
  noise: 0.04,
  strength: 1.0,
  ntrain: 480,
  mode: 'time',
  running: false,
  runEpochs: 0,        // how many epochs one press of play should run; 0 is "until I pause"
  stopAt: null,        // the epoch the current run stops at, or null for an open-ended run
  epoch: 0,
  hover: null,
  selected: null,
  tPos: 64,
  probe: null,
  probeLabel: 0,        // index into the trained classes, or −1 if out of distribution
  probeClassId: null,   // which class produced the example ('custom' for a loaded signal)
  probePick: 'rand',
  probeF0: 50,          // inspector only; the training set stays at F0
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
/** Options for inspector examples only — the training set keeps the default 50 Hz. */
function probeOpt() {
  return { noise: state.noise, strength: state.strength, f0: state.probeF0 };
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
  state.probe = generateSample(id, probeOpt());
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
  if (state.arch === 'mlp') return state.mlpLayers;
  if (state.arch === 'kan') return state.kanLayers;
  if (state.arch === 'resnet') return state.resLayers;
  if (state.arch === 'inception') return state.incLayers;
  if (state.arch === 'transformer') return state.tfLayers;
  return state.arch === 'ssm' ? state.ssmLayers : state.gnnLayers;
}

/** Most layers each playground allows. */
function maxLayers() {
  if (state.arch === 'cnn') return 8;
  return ['mlp', 'kan', 'resnet', 'inception', 'transformer'].includes(state.arch) ? 3 : 2;
}

function rebuildModel() {
  if (state.arch === 'cnn') {
    model = new ConvNet1D({
      layers: JSON.parse(JSON.stringify(state.layers)),
      activation: state.activation,
      head: state.head,
      causal: state.causal,
      residual: state.residual,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'mlp') {
    model = new MLPNet({
      layers: JSON.parse(JSON.stringify(state.mlpLayers)),
      activation: state.activation,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'resnet') {
    model = new ResNet1DNet({
      layers: JSON.parse(JSON.stringify(state.resLayers)),
      bn: state.bn, skip: state.skip,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'inception') {
    model = new InceptionTimeNet({
      layers: JSON.parse(JSON.stringify(state.incLayers)),
      kernels: [5, 11, 23],
      bn: state.bn, skip: state.skip,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'transformer') {
    model = new TransformerNet({
      layers: JSON.parse(JSON.stringify(state.tfLayers)),
      d: state.tfD, heads: state.tfHeads, causal: state.tfCausal, tokenizer: state.tfTok,
      nClasses: activeClasses().length,
      inputLen: WIN,
    });
  } else if (state.arch === 'kan') {
    model = new KANNet({
      layers: JSON.parse(JSON.stringify(state.kanLayers)),
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
  if (state.stopAt !== null) armRun();
  hTrain = []; hTest = [];
  lastMetrics = null;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
  $('layCount').textContent = archLayers().length;
  buildLayerControls();
}

/** The tab button of every playground. */
const ARCH_TABS = {
  mlp: 'archMlp', cnn: 'archCnn', resnet: 'archRes', inception: 'archInc',
  rnn: 'archRnn', ssm: 'archSsm', gnn: 'archGnn', transformer: 'archTf', kan: 'archKan',
};

/** Switches between the playgrounds. */
function setArch(arch) {
  if (state.arch === arch) return;
  state.arch = arch;
  state.selected = null;
  document.body.className = 'arch-' + arch;
  Object.entries(ARCH_TABS).forEach(([a, id]) => $(id).classList.toggle('on', arch === a));
  $('layerLbl').textContent = {
    cnn: 'Convolutional layers', resnet: 'Residual blocks', inception: 'Inception modules', rnn: 'Recurrent layers',
    mlp: 'Hidden layers', kan: 'KAN layers', ssm: 'State space layers', gnn: 'Message passing layers',
    transformer: 'Encoder layers',
  }[arch];
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
  do {
    if (runFinished()) break;
    trainOneBatch(); steps++;
  } while (performance.now() - t0 < budgetMs);
  return steps;
}

/* --------------------------------------------------- run for N epochs */

/** True once a run with a fixed length has trained for all of its epochs. */
function runFinished() {
  return state.stopAt !== null && state.epoch >= state.stopAt;
}

/** Starts or stops the training loop and keeps the play button in step. */
function setRunning(on) {
  state.running = on;
  $('btnPlay').textContent = on ? '⏸' : '▶';
  $('btnPlay').classList.toggle('on', on);
}

/** Arms a run: a fixed number of epochs from here, or open-ended when 0. */
function armRun() {
  state.stopAt = state.runEpochs > 0 ? state.epoch + state.runEpochs : null;
  renderRunTarget();
}

/** Shows the epoch a fixed-length run is heading for, next to the counter. */
function renderRunTarget() {
  const el = $('epochTgt');
  if (el) el.textContent = state.stopAt === null ? '' : '/ ' + Math.round(state.stopAt);
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
    if (runFinished()) {
      // the requested epochs are done — pause and leave the final numbers up
      setRunning(false);
      state.stopAt = null;
      evaluate(); renderMetrics(); renderRunTarget();
    } else if (frameNo % 20 === 0) {
      // A full evaluation is a forward pass over 600 windows — far more expensive
      // than a training batch. Three times a second is plenty for the curves and
      // leaves the core to the actual training.
      evaluate(); renderMetrics();
    }
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
  renderRunTarget();
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
  if (h0 && h0.type === 'filter' && model.kind === 'transformer') {
    const st = model.stages[h0.layer], T = model.T;
    if (st.tfembed && model.tconv) {
      const cv = model.tconv;
      title('TOKENIZER FILTER ' + (h0.ch + 1) + ' (K=' + cv.k + ')', 10);
      drawKernel(ctx, 0, 14, w, 34, cv.W, h0.ch * cv.k, cv.k);
      title('ITS FREQUENCY RESPONSE |H(f)|', 62);
      const resp = kernelResponse(cv.W, h0.ch * cv.k, cv.k, 128);
      drawSpectrum(ctx, 0, 66, w, 34, resp, maxOf(resp));
      ctx.fillStyle = '#98a2ad'; ctx.font = '9px system-ui,sans-serif';
      ctx.fillText('0', 0, 110); ctx.fillText('1600 Hz', w - 42, 110);
      title('THIS DIMENSION ACROSS THE TOKENS, FOR THIS EXAMPLE', 124);
      if (st.snapshot) drawWave(ctx, 0, 128, w, 36, st.snapshot, h0.ch * T, T, maxAbs(st.snapshot, 0, st.snapshot.length));
      let peak = 0;
      for (let i = 1; i < resp.length; i++) if (resp[i] > resp[peak]) peak = i;
      txt.innerHTML = '<b>Tokenizer, dimension ' + (h0.ch + 1) + '</b> · a filter of ' + cv.k + ' samples, ReLU, the largest ' +
        'response in each patch of ' + TF_PATCH + '<br>|H(f)| peaks near <b>' + Math.round(peak / resp.length * SR / 2) + ' Hz</b> — ' +
        'this dimension of every token says how strongly that band showed up in its 2.5 ms.';
      return;
    }
    if (st.tfembed) {
      title('PATCH EMBEDDING — THE ' + TF_PATCH + ' WEIGHTS OF DIMENSION ' + (h0.ch + 1), 10);
      drawKernel(ctx, 0, 14, w, 40, model.embed.W, h0.ch * TF_PATCH, TF_PATCH);
      title('POSITION VECTOR — DIMENSION ' + (h0.ch + 1) + ' FOR EACH OF THE ' + T + ' TOKENS', 70);
      const pv = new Float32Array(T);
      for (let t = 0; t < T; t++) pv[t] = model.pos.W[t * model.d + h0.ch];
      drawKernel(ctx, 0, 74, w, 40, pv, 0, T);
      title('THIS DIMENSION ACROSS THE TOKENS, FOR THIS EXAMPLE', 128);
      if (st.snapshot) drawWave(ctx, 0, 132, w, 32, st.snapshot, h0.ch * T, T, maxAbs(st.snapshot, 0, st.snapshot.length));
      txt.innerHTML = '<b>Embedding, dimension ' + (h0.ch + 1) + '</b> · patches of ' + TF_PATCH + ' samples (' +
        (TF_PATCH / SR * 1000).toFixed(1) + ' ms) → ' + T + ' tokens of ' + model.d + ' numbers' +
        '<br>The same 8 weights turn every patch into this number — a convolution with stride 8 — and the position ' +
        'vector is what tells the tokens apart.';
      return;
    }
    const L = st.layer, H = L.attn.H, A = L.trace ? L.trace.A : null;
    title('ATTENTION — ROW: TOKEN THAT LOOKS · COLUMN: TOKEN IT LOOKS AT', 10);
    const size = Math.min(112, (w - (H - 1) * 8) / H), cs = size / T;
    for (let h = 0; h < H && A; h++) {
      const x0 = h * (size + 8);
      ctx.fillStyle = '#fff'; ctx.fillRect(x0, 16, size, size);
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          const a = A[(h * T + i) * T + j];
          if (a <= 0) continue;
          ctx.fillStyle = 'rgba(29,78,216,' + Math.min(1, Math.pow(a, 0.6)).toFixed(3) + ')';
          ctx.fillRect(x0 + j * cs, 16 + i * cs, cs + 0.3, cs + 0.3);
        }
      }
      ctx.strokeStyle = '#cfd6de'; ctx.strokeRect(x0, 16, size, size);
      ctx.fillStyle = '#98a2ad'; ctx.font = '9px system-ui,sans-serif';
      ctx.fillText('head ' + (h + 1), x0, 16 + size + 10);
    }
    title('DIMENSION ' + (h0.ch + 1) + ' ACROSS THE TOKENS', 150);
    if (st.snapshot) drawWave(ctx, 0, 153, w, 14, st.snapshot, h0.ch * T, T, maxAbs(st.snapshot, 0, st.snapshot.length));
    txt.innerHTML = '<b>Encoder ' + h0.layer + ', dimension ' + (h0.ch + 1) + '</b> · ' + H + ' head' + (H > 1 ? 's' : '') + ' × ' +
      L.attn.dh + ' · ' + (st.causal ? 'causal: a token sees only itself and the past' : 'every token sees all ' + T) +
      '<br>Bright cells show where a token takes its information from. A bright column means every token reads that one patch; ' +
      'stripes parallel to the diagonal mean tokens a fixed distance apart read each other.';
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'resnet') {
    const st = model.stages[h0.layer], blk = st.block, cv = blk.convs[blk.convs.length - 1];
    const show = Math.min(cv.cin, 4);
    title('LAST CONVOLUTION OF THE BLOCK · K=' + cv.k + (cv.cin > 1 ? ' (' + show + ' of ' + cv.cin + ' inputs)' : ''), 10);
    const kw = (w - (show - 1) * 6) / show;
    for (let ci = 0; ci < show; ci++) {
      const x = ci * (kw + 6);
      ctx.strokeStyle = '#eceff3'; ctx.strokeRect(x, 14, kw, 34);
      drawKernel(ctx, x + 3, 16, kw - 6, 30, cv.W, (h0.ch * cv.cin + ci) * cv.k, cv.k);
    }
    title('OUTPUT MAP FOR THIS EXAMPLE', 66);
    if (st.snapshot) drawWave(ctx, 0, 70, w, 60, st.snapshot, h0.ch * st.L, st.L, maxAbs(st.snapshot, 0, st.snapshot.length));
    let rf = 1;
    for (let i = 0; i <= h0.layer; i++) model.stages[i].block.kernels.forEach((k) => { rf += k - 1; });
    txt.innerHTML = '<b>Block ' + (h0.layer + 1) + ', channel ' + (h0.ch + 1) + '</b> · kernels ' + blk.kernels.join('-') +
      ' · ' + (blk.skip ? (blk.proj ? 'skip through a 1×1 convolution' : 'identity skip') : 'no skip') +
      ' · ' + (blk.useBn ? 'layer norm' : 'no normalisation') +
      '<br>Receptive field ≈ ' + rf + ' samples (' + (rf / SR * 1000).toFixed(1) + ' ms): every block adds the lengths of its kernels.';
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'inception') {
    const st = model.stages[h0.layer], m = st.module, { bi, j } = m.branchOf(h0.ch);
    title('MEAN ACTIVITY PER BRANCH ON THIS EXAMPLE', 10);
    const acts = [];
    for (let b = 0; b < m.nb; b++) {
      let a = 0;
      for (let q = 0; q < m.f; q++) for (let t = 0; t < st.L; t++) a += Math.abs(st.snapshot[(b * m.f + q) * st.L + t]);
      acts.push(a / (m.f * st.L));
    }
    const am = Math.max(1e-6, ...acts);
    acts.forEach((a, b) => {
      const y = 16 + b * 15;
      ctx.fillStyle = '#98a2ad'; ctx.font = '9px system-ui,sans-serif';
      ctx.fillText(m.branchName(b), 0, y + 9);
      ctx.fillStyle = '#eef1f4'; ctx.fillRect(40, y + 2, w - 44, 9);
      ctx.fillStyle = INCEPTION_COLORS[b]; ctx.fillRect(40, y + 2, (w - 44) * a / am, 9);
    });
    title(bi < m.kernels.length ? 'THIS CHANNEL\'S KERNEL — BRANCH ' + m.branchName(bi) : 'POOL BRANCH — MAX OVER 3, THEN 1×1', 88);
    if (bi < m.kernels.length) {
      const cv = m.branches[bi];
      drawKernel(ctx, 0, 92, w, 26, cv.W, (j * cv.cin) * cv.k, cv.k);
    }
    title('OUTPUT MAP FOR THIS EXAMPLE', 128);
    if (st.snapshot) drawWave(ctx, 0, 132, w, 34, st.snapshot, h0.ch * st.L, st.L, maxAbs(st.snapshot, 0, st.snapshot.length));
    txt.innerHTML = '<b>Module ' + (h0.layer + 1) + ', channel ' + (h0.ch + 1) + '</b> · ' + m.branchName(bi) +
      ' branch, filter ' + (j + 1) + ' of ' + m.f +
      '<br>Short kernels answer to sharp, local events, long ones see a whole stretch of the cycle. ' +
      'The bars show which branch carries this example.';
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'mlp') {
    const st = model.stages[h0.layer], d = st.dense, first = h0.layer === 0;
    const off = h0.ch * d.nin;
    title(first ? 'WEIGHTS OVER THE WINDOW — THIS UNIT\'S TEMPLATE' : 'WEIGHTS FROM THE ' + d.nin + ' UNITS BEFORE', 10);
    if (first) drawWave(ctx, 0, 14, w, 62, d.W, off, d.nin, maxAbs(d.W, off, d.nin));
    else drawKernel(ctx, 0, 14, w, 62, d.W, off, d.nin);
    if (first) {
      title('ITS SPECTRUM — THE FREQUENCIES IT MATCHES', 96);
      const m = magSpectrum(d.W, off, d.nin);
      drawSpectrum(ctx, 0, 100, w, 46, m, maxOf(m));
      ctx.fillStyle = '#98a2ad';
      ctx.font = '9px system-ui,sans-serif';
      ctx.fillText('0', 0, 158);
      ctx.fillText('1600 Hz', w - 42, 158);
    }
    const a = st.snapshot ? st.snapshot[h0.ch] : 0;
    txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', unit ' + (h0.ch + 1) + '</b> · ' + d.nin +
      ' weights + bias · output ' + n3(a) +
      (first
        ? '<br>The unit multiplies the window sample by sample with this template and adds it up. ' +
          'Nothing is shared between positions: a disturbance a few samples later meets different weights.'
        : '<br>A deeper unit mixes the units before it — no time axis is left at this point.');
    return;
  }
  if (h0 && h0.type === 'filter' && model.kind === 'kan') {
    const st = model.stages[h0.layer], L = st.layer, first = h0.layer === 0;
    const lo = KAN_LO - 0.8, hi = KAN_HI + 0.8, n = 28;
    title('EDGE FUNCTIONS φ(x) — ' + (first ? 'ALL ' + L.nin + ' OVERLAID' : 'ONE PER INPUT'), 10);
    const curves = [];
    let m = 1e-6;
    for (let i = 0; i < L.nin; i++) {
      const c = model.edgeCurve(h0.layer, h0.ch, i, n);
      c.forEach((p) => { m = Math.max(m, Math.abs(p[1])); });
      curves.push(c);
    }
    const gx = (v) => (v - lo) / (hi - lo) * w;
    const ch = first ? 72 : 130, cy = 14 + ch / 2;
    ctx.fillStyle = '#f1f6fd';
    ctx.fillRect(gx(KAN_LO), 14, gx(KAN_HI) - gx(KAN_LO), ch);
    ctx.strokeStyle = '#dfe4ea'; ctx.lineWidth = 0.6;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
    const palette = ['#2b6cb0', '#c2760f', '#2e9e5b', '#8e44ad', '#e0342b', '#16a085', '#7f8c8d', '#d35400'];
    curves.forEach((c, i) => {
      ctx.beginPath();
      c.forEach((p, q) => {
        const X = gx(lo + (hi - lo) * q / (c.length - 1)), Y = cy - p[1] / m * (ch / 2 - 2);
        if (q === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      });
      ctx.strokeStyle = first ? 'rgba(43,108,176,0.22)' : palette[i % palette.length];
      ctx.lineWidth = first ? 1 : 1.6;
      ctx.stroke();
    });
    if (first && st.phi) {
      title('φ(x_i) ACROSS THE WINDOW — WHAT EACH SAMPLE ADDS', 102);
      drawWave(ctx, 0, 106, w, 56, st.phi, h0.ch * L.nin, L.nin, maxAbs(st.phi, h0.ch * L.nin, L.nin));
    }
    const v = st.snapshot ? st.snapshot[h0.ch] : 0;
    txt.innerHTML = '<b>Layer ' + (h0.layer + 1) + ', node ' + (h0.ch + 1) + '</b> · ' + L.nin +
      ' edges × ' + (KAN_NB + 1) + ' numbers · value ' + n3(v) +
      '<br>Every edge is its own learned curve (cubic B-spline on ' + KAN_LO + ' … ' + KAN_HI +
      (first ? '' : ' in standardised units') + ', plus a silu term for the tails). The node just adds ' +
      'the curves\' outputs — no weights on the node, no activation.' +
      (first ? '' : '<br>Colours match the inputs from left to right in the node\'s box.');
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
      const idx = t + conv.tapOffset(j);
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
  const v = applyAct(convAt(li, ch, t).z);
  return model.stages[li].res ? v + skipAt(li, ch, t) : v;
}

/** The residual skip into filter ch at t: the same input channel, or its 1×1 projection. */
function skipAt(li, ch, t) {
  const st = model.stages[li];
  const { Lin, xin, cin } = layerInput(li);
  if (!xin) return 0;
  if (!st.proj) return xin[ch * Lin + t];
  let s = st.proj.b[ch];
  for (let c = 0; c < cin; c++) s += st.proj.W[ch * cin + c] * xin[c * Lin + t];
  return s;
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
  flowHtml = '';
  renderMathInner(host);
  [...host.querySelectorAll('.scrollx')].forEach((e, i) => {
    if (scrolls[i]) { e.scrollLeft = scrolls[i][0]; e.scrollTop = scrolls[i][1]; }
  });
  renderFlowBox();
}

/* ------------------------------------------- cell diagram under the network */
let flowHtml = '';        // filled by the renderer of the selected node
let flowShown = null;     // what the box holds now, so an unchanged diagram is not rebuilt

/** Hands the selected node's data-flow diagram to the Architecture frame. */
function setFlow(svgHtml) { flowHtml = svgHtml; }

/**
 * The diagram lives under the network, so clicking a neuron shows its structure
 * right where it was clicked. Its header mirrors the arithmetic panel: the same
 * title, and a second slider bound to the same position t.
 */
function renderFlowBox() {
  const body = $('flowBody');
  const html = flowHtml || '<p class="flowempty">Click any neuron in the diagram above to see ' +
    'its cell — every gate, weight and value on the way from its inputs to its output. ' +
    'The full arithmetic is listed further down the page.</p>';
  // rebuilding an identical SVG would only drop the tooltip under the mouse
  if (html !== flowShown) { body.innerHTML = html; flowShown = html; }
  $('flowTitle').textContent = flowHtml ? $('mathTitle').textContent : 'Cell structure';
  const main = $('tpos'), mini = $('flowTpos');
  mini.disabled = main.disabled || !flowHtml;
  mini.max = main.max;
  mini.value = main.value;
  $('flowTval').textContent = mini.disabled ? '—' : $('tposVal').textContent;
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
    else if (model.kind === 'mlp') renderMlpMath(host, title, slider, sel);
    else if (model.kind === 'kan') renderKanMath(host, title, slider, sel);
    else if (model.kind === 'resnet') renderResBlockMath(host, title, slider, sel);
    else if (model.kind === 'transformer') renderTfMath(host, title, slider, sel);
    else if (model.kind === 'inception') renderInceptionMath(host, title, slider, sel);
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
  setFlow(Flow[d.kind](d));
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

  let fSelf = 0, fNeigh = 0;
  for (let i = 0; i < d.D; i++) { fSelf += d.ws[i] * d.self[i]; fNeigh += d.wn[i] * d.agg[i]; }
  setFlow(Flow.gnn(d, {
    li, ch, fname, sSelf: fSelf, sNeigh: fNeigh,
    src: li === 0 ? model.feat : model.stages[li - 1].snapshot, L: WIN,
  }));
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

  let fy = 0;
  for (const m of d.modes) fy += isS4 ? 2 * (m.cRe * m.xRe - m.cIm * m.xIm) : m.C * m.xRe;
  const lay = st.layer, at = ch * lay.L + t;
  setFlow(Flow.ssm(d, {
    ch, ySum: fy, y: fy + d.Dskip * d.u,
    out: st.snapshot ? st.snapshot[ch * st.L + t] : 0,
    gate: lay.g ? lay.g[at] : null,
  }));
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

/* ------------------------------------------------------------ Transformer */
function renderTfMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C || (st.tflayer && !st.layer.trace)) { state.selected = null; return renderMathInner(host); }
  const T = model.T, d = model.d;
  slider.disabled = false;
  slider.max = T - 1;
  const t = Math.min(state.tPos, T - 1);
  state.tPos = t;
  slider.value = t;
  $('tposVal').textContent = 'token ' + t + '  (' + (t * TF_PATCH / SR * 1000).toFixed(1) + '–' +
    ((t + 1) * TF_PATCH / SR * 1000).toFixed(1) + ' ms)';
  if (st.tfembed) return renderTfEmbedMath(host, title, st, ch, t);

  const L = st.layer, tr = L.trace, at = L.attn, H = at.H, dh = at.dh;
  const head = Math.floor(ch / dh);
  title.textContent = 'Encoder ' + li + ' · dimension ' + (ch + 1) + ' · token ' + t + (st.causal ? ' · causal' : '');
  const vec = (arr) => Array.from(arr.subarray(t * d, (t + 1) * d));
  const row = (arr, h) => Array.from(arr.subarray((h * T + t) * T, (h * T + t) * T + T));
  setFlow(Flow.attention({
    x: vec(tr.X), n1: vec(tr.n1), a: row(tr.A, head), scores: row(tr.S, head), head, o: vec(tr.O),
    attn: vec(tr.a), h: vec(tr.h), n2: vec(tr.n2), z2: vec(tr.z2), out: vec(tr.out), causal: st.causal,
  }, { li, ch, t }));

  let html = '<h4>1 · The encoder layer</h4>';
  html += '<div class="formula">h = x + W<sub>o</sub>·Attention(LN<sub>1</sub>(x)) &nbsp;&nbsp; y = h + W<sub>2</sub>·ReLU(W<sub>1</sub>·LN<sub>2</sub>(h)) &nbsp;&nbsp; ' +
    'Attention: a<sub>tj</sub> = softmax<sub>j</sub>( q<sub>t</sub>·k<sub>j</sub> / √' + dh + ' ), &nbsp; o<sub>t</sub> = Σ<sub>j</sub> a<sub>tj</sub> v<sub>j</sub></div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">' + H + ' head' + (H > 1 ? 's' : '') + ' of ' + dh +
    ' numbers each; q, k and v are three linear maps of the normalised token. Pre-LN, as in ViT: the norm sits inside the ' +
    'residual branch, so x itself is carried on untouched. ' + (st.causal
      ? 'Causal: token t may only look at tokens 0 … t — the decoder form a Transformer uses on a stream.'
      : 'Every token may look at all ' + T + ' — the encoder form, for a whole block at once.') + '</span></div>';

  html += '<h4>2 · Where token ' + t + ' looks</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>token j</th><th>time</th>';
  for (let h = 0; h < H; h++) html += '<th>head ' + (h + 1) + ' score</th><th>head ' + (h + 1) + ' weight</th>';
  html += '</tr></thead><tbody>';
  for (let j = 0; j < T; j++) {
    const masked = st.causal && j > t;
    html += '<tr' + (j === t ? ' style="background:#eef4fd"' : '') + '><td class="ch">' + j + (j === t ? ' ← itself' : '') + '</td><td>' +
      (j * TF_PATCH / SR * 1000).toFixed(1) + ' ms</td>';
    for (let h = 0; h < H; h++) {
      const a = tr.A[(h * T + t) * T + j], s = tr.S[(h * T + t) * T + j];
      html += masked ? '<td class="pad">masked</td><td class="pad">0</td>'
        : '<td>' + n3(s) + '</td><td class="sum">' + n3(a) + '</td>';
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">score = q<sub>t</sub>·k<sub>j</sub> / √' + dh +
    '; the weights are the softmax of the scores and add up to 1 in every head.</span></div>';

  const k = ch - head * dh;
  html += '<h4>3 · The weighted sum behind dimension ' + (ch + 1) + ' of o<sub>t</sub> (head ' + (head + 1) + ', slot ' + (k + 1) + ')</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>token j</th><th>weight a<sub>tj</sub></th>' +
    '<th>value v<sub>j</sub></th><th>product</th></tr></thead><tbody>';
  let osum = 0;
  for (let j = 0; j < T; j++) {
    if (st.causal && j > t) continue;
    const a = tr.A[(head * T + t) * T + j], v = tr.V[j * d + ch];
    osum += a * v;
    html += '<tr><td class="ch">' + j + '</td><td>' + n3(a) + '</td><td>' + n3(v) + '</td><td class="sum">' + n4(a * v) + '</td></tr>';
  }
  html += '</tbody></table></div>';
  let aout = at.o.b[ch];
  for (let q = 0; q < d; q++) aout += at.o.W[ch * d + q] * tr.O[t * d + q];
  html += '<div class="formula" style="margin-top:8px">o<sub>t</sub>[' + (ch + 1) + '] = Σ = <b>' + n4(osum) + '</b>  <span class="op">→</span>  ' +
    'W<sub>o</sub> mixes all ' + d + ' numbers of o<sub>t</sub>: attention[' + (ch + 1) + '] = Σ W<sub>o</sub>[' + (ch + 1) + '][q]·o<sub>t</sub>[q] + b = <b>' +
    n4(aout) + '</b></div>';

  html += '<h4>4 · Dimension ' + (ch + 1) + ' through the layer at token ' + t + '</h4>';
  const at_ = t * d + ch;
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>step</th><th>value</th></tr></thead><tbody>' +
    '<tr><td class="ch">x (in)</td><td>' + n4(tr.X[at_]) + '</td></tr>' +
    '<tr><td class="ch">LN<sub>1</sub>(x)</td><td>' + n4(tr.n1[at_]) + '</td></tr>' +
    '<tr><td class="ch">attention (after W<sub>o</sub>)</td><td>' + n4(tr.a[at_]) + '</td></tr>' +
    '<tr><td class="ch">h = x + attention</td><td>' + n4(tr.h[at_]) + '</td></tr>' +
    '<tr><td class="ch">LN<sub>2</sub>(h)</td><td>' + n4(tr.n2[at_]) + '</td></tr>' +
    '<tr><td class="ch">FFN (W<sub>2</sub>·ReLU(W<sub>1</sub>·))</td><td>' + n4(tr.z2[at_]) + '</td></tr>' +
    '<tr><td class="ch">y = h + FFN</td><td class="sum">' + n4(tr.out[at_]) + '</td></tr></tbody></table></div>';
  const drawn = st.snapshot[ch * T + t];
  html += '<div class="formula" style="margin-top:6px"><span class="op">check: the box holds ' + n4(drawn) + ' at token ' + t +
    (Math.abs(drawn - tr.out[at_]) < 1e-4 && Math.abs(aout - tr.a[at_]) < 1e-3 && Math.abs(osum - tr.O[at_]) < 1e-4
      ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';

  html += tfDownstreamHtml(li, ch, 5);
  host.innerHTML = html;
}

function renderTfEmbedMath(host, title, st, ch, t) {
  if (model.tconv) return renderTfConvEmbedMath(host, title, st, ch, t);
  const T = model.T, d = model.d, E = model.embed, x = model.input, start = t * TF_PATCH;
  title.textContent = 'Embedding · dimension ' + (ch + 1) + ' · token ' + t;
  const xs = [], ws = [];
  let sum = 0;
  for (let j = 0; j < TF_PATCH; j++) {
    xs.push(x[start + j]); ws.push(E.W[ch * TF_PATCH + j]);
    sum += x[start + j] * E.W[ch * TF_PATCH + j];
  }
  const b = E.b[ch], pos = model.pos.W[t * d + ch], z = sum + b, out = z + pos;
  setFlow(Flow.tfEmbed({ xs, ws, b, pos, sum, z, out, start }, { t, ch }));

  let html = '<h4>1 · From patch to token</h4>';
  html += '<div class="formula">token[' + t + '][' + (ch + 1) + '] = Σ<sub>j=0..7</sub> W<sub>e</sub>[' + (ch + 1) + '][j] · x[' + start +
    ' + j] + b + pos[' + t + '][' + (ch + 1) + ']  <span class="op">— the same ' + TF_PATCH + ' weights for every patch, a convolution ' +
    'with kernel 8 and stride 8; only the position vector differs</span></div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>j</th><th>sample</th><th>value</th>' +
    '<th>weight</th><th>product</th></tr></thead><tbody>';
  xs.forEach((v, j) => {
    html += '<tr><td class="ch">' + j + '</td><td>x[' + (start + j) + ']</td><td>' + n3(v) + '</td><td style="color:' + wColor(ws[j]) + '">' +
      n3(ws[j]) + '</td><td class="sum">' + n4(v * ws[j]) + '</td></tr>';
  });
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">Σ = ' + n4(sum) + '  +  b ' + n4(b) + '  +  pos ' + n4(pos) +
    '  =  <span class="res">' + n4(out) + '</span>  <span class="op">check: the box holds ' + n4(st.snapshot[ch * T + t]) +
    (Math.abs(st.snapshot[ch * T + t] - out) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  html += tfDownstreamHtml(0, ch, 2);
  host.innerHTML = html;
}

/** The convolutional tokenizer: filter, ReLU, the largest response of the patch, + position. */
function renderTfConvEmbedMath(host, title, st, ch, t) {
  const T = model.T, d = model.d, cv = model.tconv, L = WIN, x = model.input, start = t * TF_PATCH;
  title.textContent = 'Tokens · dimension ' + (ch + 1) + ' · token ' + t;
  const zs = [], as = [];
  let win = 0;
  for (let j = 0; j < TF_PATCH; j++) {
    zs.push(model.convZ[ch * L + start + j]);
    as.push(model.convA[ch * L + start + j]);
    if (as[j] > as[win]) win = j;
  }
  const max = as[win], pos = model.pos.W[t * d + ch], out = max + pos;
  setFlow(Flow.tfConvEmbed({ zs, as, win, start, max, pos, out, k: cv.k }, { t, ch }));

  let html = '<h4>1 · Convolutional tokenizer</h4>';
  html += '<div class="formula">token[' + t + '][' + (ch + 1) + '] = max<sub>s ∈ patch ' + t + '</sub> ReLU( Σ<sub>j</sub> W[' + (ch + 1) +
    '][j] · x[s + j − ' + (-cv.tapOffset(0)) + '] + b ) + pos[' + t + '][' + (ch + 1) + ']</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">As in the Compact Convolutional Transformer: ' + d +
    ' filters of length ' + cv.k + ' slide over the samples, and every patch of ' + TF_PATCH + ' keeps the largest response of ' +
    'each. A token then says how strongly each filter fired in its 2.5 ms — energy at a frequency, a sharp edge — instead of ' +
    'eight raw voltages. Switch Tokens to "linear patch" to compare.</span></div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>sample s</th><th>conv + bias</th>' +
    '<th>ReLU</th></tr></thead><tbody>';
  zs.forEach((z, j) => {
    html += '<tr' + (j === win ? ' style="background:#eef4fd"' : '') + '><td class="ch">' + (start + j) + (j === win ? ' ← max' : '') +
      '</td><td>' + n4(z) + '</td><td class="sum">' + n4(as[j]) + '</td></tr>';
  });
  html += '</tbody></table></div>';
  html += '<h4>2 · The winning sample s = ' + (start + win) + ' in detail</h4>';
  html += termsTableHtml(convTermsOf(cv, x, L, ch, start + win), () => 'signal');
  html += '<div class="formula" style="margin-top:8px">max ' + n4(max) + '  +  pos ' + n4(pos) + '  =  <span class="res">' + n4(out) +
    '</span>  <span class="op">check: the box holds ' + n4(st.snapshot[ch * T + t]) +
    (Math.abs(st.snapshot[ch * T + t] - out) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';
  html += tfDownstreamHtml(0, ch, 3);
  host.innerHTML = html;
}

/** Where a token dimension goes: the next encoder layer, or the final norm, the average and the logits. */
function tfDownstreamHtml(li, ch, sec) {
  let html = '<h4>' + sec + ' · Where this goes</h4>';
  if (li < model.stages.length - 1) {
    html += '<div class="formula">This number is dimension ' + (ch + 1) + ' of the residual stream. Encoder ' + (li + 1) +
      ' reads it through LN<sub>1</sub>, turns every token into a query, a key and a value, and adds its result back on top.</div>';
    return html;
  }
  const d = model.dense, cls = activeClasses(), probs = model.probs, h = model.embedding[ch];
  html += '<div class="formula">The final layer norm is applied to every token, then the ' + model.T + ' tokens are averaged: ' +
    'h[' + (ch + 1) + '] = <span class="res">' + n4(h) + '</span></div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>class</th><th>weight to class</th>' +
    '<th>contribution</th><th>class bias</th><th>logit (total)</th><th>softmax</th></tr></thead><tbody>';
  for (let k = 0; k < d.nout; k++) {
    const w = d.W[k * d.nin + ch];
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[k].color + '"></span> ' + cls[k].name +
      '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td><td>' + n4(w * h) + '</td><td>' + n3(d.b[k]) +
      '</td><td>' + n3(model.logits[k]) + '</td><td class="sum">' + (probs[k] * 100).toFixed(1) + '%</td></tr>';
  }
  return html + '</tbody></table></div>';
}

/* ------------------------------------------- ResNet-1D and InceptionTime */
/** The products of one convolution output (channel co, position t), for any conv and input. */
function convTermsOf(conv, xin, Lin, co, t) {
  const K = conv.k, cin = conv.cin;
  const terms = [];
  let sum = 0;
  for (let ci = 0; ci < cin; ci++) {
    const row = [];
    let sub = 0;
    for (let j = 0; j < K; j++) {
      const idx = t + conv.tapOffset(j);
      const outside = idx < 0 || idx >= Lin;
      const xv = outside ? 0 : xin[ci * Lin + idx];
      const w = conv.W[(co * cin + ci) * K + j];
      sub += w * xv;
      row.push({ idx, xv, w, p: w * xv, outside });
    }
    sum += sub;
    terms.push({ ci, row, sub });
  }
  return { z: sum + conv.b[co], sum, bias: conv.b[co], terms, K, cin, conv };
}

/** The table of value × weight products behind one convolution output. */
function termsTableHtml(c, chName) {
  const rows = c.terms.slice(0, 10);
  let html = '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th>';
  for (let j = 0; j < c.K; j++) html += '<th>j=' + j + '<br>x[' + c.terms[0].row[j].idx + ']</th>';
  html += '<th>Σ per channel</th></tr></thead><tbody>';
  rows.forEach((tr) => {
    html += '<tr><td class="ch">' + chName(tr.ci) + '</td>';
    tr.row.forEach((cell) => {
      html += '<td class="cell' + (cell.outside ? ' pad' : '') + '"><span class="xv">' +
        (cell.outside ? '0 (outside)' : n3(cell.xv)) + '</span><span class="wv" style="color:' + wColor(cell.w) +
        '">×' + n3(cell.w) + '</span><span class="pv">' + n3(cell.p) + '</span></td>';
    });
    html += '<td class="sum">' + n3(tr.sub) + '</td></tr>';
  });
  if (c.terms.length > rows.length) {
    let rest = 0;
    c.terms.slice(rows.length).forEach((tr) => { rest += tr.sub; });
    html += '<tr><td class="ch">the other ' + (c.terms.length - rows.length) + '</td><td colspan="' + c.K +
      '"></td><td class="sum">' + n3(rest) + '</td></tr>';
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">Σ (all ' + (c.cin * c.K) + ' products) = <b>' + n4(c.sum) +
    '</b>  <span class="op">+</span>  bias = <b style="color:' + wColor(c.bias) + '">' + n4(c.bias) +
    '</b>  <span class="op">→</span>  z = <span class="res">' + n4(c.z) + '</span></div>';
  return html;
}

/** One value through layer norm, with every constant shown. */
function bnFormula(e, z) {
  return 'LN: (z − μ) / σ · γ + β = (' + n4(z) + ' − ' + n3(e.mu) + ') / ' + n3(e.sd) + ' · ' + n3(e.gamma) +
    ' + ' + n3(e.beta) + ' = <span class="res">' + n4(e.out) + '</span>';
}

function deepSlider(slider, L) {
  slider.disabled = false;
  slider.max = L - 1;
  const t = Math.min(state.tPos, L - 1);
  state.tPos = t;
  slider.value = t;
  $('tposVal').textContent = 't = ' + t + '  (' + (t / SR * 1000).toFixed(2) + ' ms)';
  return t;
}

function renderResBlockMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C || !st.block.trace) { state.selected = null; return renderMathInner(host); }
  const blk = st.block, tr = blk.trace, L = st.L, n = blk.convs.length;
  const t = deepSlider(slider, L);
  const at = ch * L + t;
  title.textContent = 'Block ' + (li + 1) + ' · channel ' + (ch + 1) + ' · position t = ' + t;

  const rows = tr.map((r, i) => ({
    k: blk.kernels[i], z: r.z[at], bn: blk.bns[i] ? blk.bns[i].explain(ch, r.z[at], r.stat) : null,
    n: r.n[at], h: i < n - 1 ? r.h[at] : null,
  }));
  const last = tr[n - 1];
  const skip = blk.skip ? { s: last.s[at], kind: blk.proj ? 'proj' : 'identity' } : null;
  const xin = blk.cin === blk.F ? blk.input[at] : blk.input[t];
  setFlow(Flow.resblock({ rows, skip, sum: last.sum[at], out: last.h[at], xin, bn: blk.useBn, F: blk.F }, { li, ch, t }));

  const inName = (c) => (li === 0 ? 'signal' : 'B' + li + ' ch' + (c + 1));
  let html = '<h4>1 · The residual block</h4>';
  html += '<div class="formula">' + blk.kernels.map((k, i) => (i < n - 1
    ? 'h<sub>' + (i + 1) + '</sub> = ReLU(' + (blk.useBn ? 'LN(' : '') + 'conv<sub>K=' + k + '</sub>(' + (i ? 'h<sub>' + i + '</sub>' : 'x') + ')' + (blk.useBn ? ')' : '') + ')'
    : 'out = ReLU(' + (blk.useBn ? 'LN(' : '') + 'conv<sub>K=' + k + '</sub>(' + (i ? 'h<sub>' + i + '</sub>' : 'x') + ')' + (blk.useBn ? ')' : '') +
      (blk.skip ? ' + ' + (blk.proj ? (blk.useBn ? 'LN(1×1(x))' : '1×1(x)') : 'x') : '') + ')')).join(' &nbsp;→&nbsp; ') + '</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">As in ResNet-1D for time series (Wang et al. 2017): ' +
    'three convolutions of falling length, a normalisation after each, ReLU everywhere except before the addition. ' +
    (blk.useBn ? 'The original uses batch norm, which needs a whole mini-batch; this engine trains one example at a ' +
      'time, so each map is normalised over its own channels and positions (layer norm, μ and σ from this example).'
      : 'Normalisation is off.') + '</span></div>';

  html += '<h4>2 · Channel ' + (ch + 1) + ' through the block at t = ' + t + '</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>step</th><th>conv + bias</th>' +
    (blk.useBn ? '<th>μ</th><th>σ</th><th>γ</th><th>β</th><th>after LN</th>' : '') + '<th>after ReLU</th></tr></thead><tbody>';
  rows.forEach((r, i) => {
    html += '<tr><td class="ch">conv ' + (i + 1) + ' · K=' + r.k + '</td><td>' + n4(r.z) + '</td>' +
      (r.bn ? '<td>' + n3(r.bn.mu) + '</td><td>' + n3(r.bn.sd) + '</td><td>' + n3(r.bn.gamma) + '</td><td>' +
        n3(r.bn.beta) + '</td><td>' + n4(r.n) + '</td>' : '') +
      '<td class="sum">' + (r.h === null ? '<span style="color:#98a2ad">after the skip</span>' : n4(r.h)) + '</td></tr>';
  });
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">Each convolution reads all ' +
    (n > 1 ? blk.F : blk.cin) + ' channels of the step before; the table follows channel ' + (ch + 1) + '.</span></div>';

  html += '<h4>3 · The last convolution in detail (K=' + blk.kernels[n - 1] + ')</h4>';
  const c = convTermsOf(blk.convs[n - 1], n > 1 ? tr[n - 2].h : blk.input, L, ch, t);
  html += termsTableHtml(c, (ci) => (n > 1 ? 'h' + (n - 1) + ' ch' + (ci + 1) : inName(ci)));
  if (rows[n - 1].bn) html += '<div class="formula" style="margin-top:6px">' + bnFormula(rows[n - 1].bn, c.z) + '</div>';

  html += '<h4>4 · Skip and ReLU</h4>';
  if (blk.skip) {
    html += '<div class="formula">y = ' + (blk.useBn ? 'LN(z)' : 'z') + ' + skip = ' + n4(rows[n - 1].n) + ' + ' +
      n4(skip.s) + ' = <b>' + n4(last.sum[at]) + '</b>  <span class="op">' + (blk.proj
        ? '— ' + blk.cin + ' channels in, ' + blk.F + ' out, so the skip is a learned 1×1 convolution' + (blk.useBn ? ' with its own LN' : '')
        : '— the block input x[' + ch + '][' + t + '] is added back untouched') + '</span></div>';
  } else {
    html += '<div class="formula"><span class="op">Skip connections are off: nothing is added, this is a plain ' +
      n + '-layer convolutional stack.</span></div>';
  }
  html += '<div class="formula" style="margin-top:6px">out = ReLU(' + n4(last.sum[at]) + ') = <span class="res">' +
    n4(last.h[at]) + '</span>  <span class="op">check: the map holds ' + n4(st.snapshot[at]) +
    (Math.abs(st.snapshot[at] - last.h[at]) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';

  html += deepDownstreamHtml(li, ch, 5);
  host.innerHTML = html;
}

function renderInceptionMath(host, title, slider, sel) {
  const li = sel.layer, ch = sel.ch;
  const st = model.stages[li];
  if (!st || ch >= st.C || !st.module.trace) { state.selected = null; return renderMathInner(host); }
  const m = st.module, tr = m.trace, L = st.L;
  const t = deepSlider(slider, L);
  const { bi, j } = m.branchOf(ch);
  const nK = m.kernels.length, at = ch * L + t;
  title.textContent = 'Module ' + (li + 1) + ' · channel ' + (ch + 1) + ' (' + m.branchName(bi) + ' branch) · position t = ' + t;

  const branches = [];
  for (let b = 0; b < m.nb; b++) {
    let act = 0;
    for (let q = 0; q < m.f; q++) for (let tt = 0; tt < L; tt++) act += Math.abs(tr.out[(b * m.f + q) * L + tt]);
    branches.push({ name: m.branchName(b), val: tr.cat[(b * m.f + j) * L + t], act: act / (m.f * L), sel: b === bi, slot: j });
  }
  const pre = tr.cat[at];
  const bn = m.bn ? m.bn.explain(ch, pre, tr.stat) : null;
  const relu = tr.out[at];
  const g = st.shortcut && model.group ? model.group.trace : null;
  const shortcut = g ? { s: g.s[at], sum: g.sum[at], out: g.out[at] } : null;
  const bott = m.bott ? { B: m.B, vals: Array.from({ length: m.B }, (_, b) => tr.b[b * L + t]) } : null;
  setFlow(Flow.inception({ branches, bott, pre, bn, relu, shortcut, xin: tr.x[t], f: m.f, nb: m.nb }, { li, ch, t }));

  const inName = (c) => (li === 0 ? 'signal' : 'M' + li + ' ch' + (c + 1));
  let html = '<h4>1 · The Inception module</h4>';
  html += '<div class="formula">out = ReLU( ' + (m.bn ? 'LN( ' : '') + 'concat[ ' +
    m.kernels.map((k) => 'conv<sub>K=' + k + '</sub>(' + (m.bott ? 'b' : 'x') + ')').join(', ') +
    ', 1×1(maxpool<sub>3</sub>(x)) ]' + (m.bn ? ' )' : '') + ' )' + (m.bott ? ' &nbsp;&nbsp; b = 1×1 bottleneck(x), ' + m.B + ' channels' : '') + '</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">As in InceptionTime (Ismail Fawaz et al. 2020), ' +
    'scaled to this window: kernels ' + m.kernels.join(' / ') + ' instead of 10 / 20 / 40, ' + m.f +
    ' filter' + (m.f > 1 ? 's' : '') + ' per branch instead of 32. Channel ' + (ch + 1) + ' is filter ' + (j + 1) +
    ' of the ' + m.branchName(bi) + ' branch.</span></div>';

  let sec = 2;
  if (bott) {
    html += '<h4>' + (sec++) + ' · Bottleneck at t = ' + t + '</h4>';
    html += '<div class="formula">' + bott.vals.map((v, b) => 'b' + b + ' = ' + n3(v)).join('   ') +
      '  <span class="op">— ' + m.cin + ' channels squeezed into ' + m.B + ' by 1×1 convolutions, so the long kernels stay cheap</span></div>';
  }

  html += '<h4>' + (sec++) + ' · This channel\'s branch: ' + m.branchName(bi) + '</h4>';
  if (bi < nK) {
    const c = convTermsOf(m.branches[bi], m.bott ? tr.b : tr.x, L, j, t);
    html += termsTableHtml(c, (ci) => (m.bott ? 'b' + ci : inName(ci)));
  } else {
    const pc = m.poolConv;
    let sum = 0;
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th><th>x[t−1]</th><th>x[t]</th><th>x[t+1]</th>' +
      '<th>max</th><th>weight</th><th>product</th></tr></thead><tbody>';
    for (let ci = 0; ci < m.cin; ci++) {
      const v = (q) => (q < 0 || q >= L ? '—' : n3(tr.x[ci * L + q]));
      const mxv = tr.pooled[ci * L + t], w = pc.W[j * m.cin + ci];
      sum += w * mxv;
      if (ci < 10) {
        html += '<tr><td class="ch">' + inName(ci) + '</td><td>' + v(t - 1) + '</td><td>' + v(t) + '</td><td>' + v(t + 1) +
          '</td><td>' + n3(mxv) + '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td><td class="sum">' + n4(w * mxv) + '</td></tr>';
      }
    }
    html += '</tbody></table></div>';
    html += '<div class="formula" style="margin-top:8px">Σ = ' + n4(sum) + ' + bias ' + n4(pc.b[j]) +
      ' = <span class="res">' + n4(sum + pc.b[j]) + '</span>  <span class="op">— the pool branch reads the module input ' +
      'directly, not the bottleneck</span></div>';
  }

  html += '<h4>' + (sec++) + ' · Layer norm and ReLU' + (shortcut ? ', then the shortcut' : '') + '</h4>';
  html += '<div class="formula">' + (bn ? bnFormula(bn, pre) : 'no normalisation: ' + n4(pre)) +
    '  <span class="op">→</span>  ReLU → <b>' + n4(relu) + '</b></div>';
  let final = relu;
  if (shortcut) {
    html += '<div class="formula" style="margin-top:6px">ReLU( ' + n4(relu) + ' + shortcut ' + n4(shortcut.s) + ' ) = <b>' +
      n4(shortcut.out) + '</b>  <span class="op">— one shortcut around every three modules, from the input of the first ' +
      'through a 1×1 convolution' + (model.group.projBn ? ' and LN' : '') + '</span></div>';
    final = shortcut.out;
  }
  html += '<div class="formula" style="margin-top:6px"><span class="op">check: the map holds ' + n4(st.snapshot[at]) +
    (Math.abs(st.snapshot[at] - final) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';

  html += '<h4>' + (sec++) + ' · All four branches</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>branch</th><th>filter ' + (j + 1) +
    ' at t (before LN)</th><th>mean activity on this example</th></tr></thead><tbody>';
  branches.forEach((b) => {
    html += '<tr' + (b.sel ? ' style="background:#eef4fd"' : '') + '><td class="ch">' + b.name + (b.sel ? ' ←' : '') +
      '</td><td>' + n4(b.val) + '</td><td class="sum">' + n4(b.act) + '</td></tr>';
  });
  html += '</tbody></table></div>';

  html += deepDownstreamHtml(li, ch, sec);
  host.innerHTML = html;
}

/** Where a ResNet / Inception map goes: the next block, or the average pool and the logits. */
function deepDownstreamHtml(li, ch, sec) {
  const st = model.stages[li], L = st.L;
  const last = li === model.stages.length - 1;
  let html = '<h4>' + sec + ' · Where this map goes</h4>';
  if (!last) {
    const nx = model.stages[li + 1];
    if (nx.resblock) {
      const cv = nx.block.convs[0];
      html += '<div class="formula">This map is input channel ' + (ch + 1) + ' of block ' + (li + 2) +
        '. Its first convolution (K=' + cv.k + ') gives it these weights in each of its ' + cv.cout + ' filters:</div>';
      html += '<div class="scrollx"><table class="mtab"><thead><tr><th>filter</th>';
      for (let k = 0; k < cv.k; k++) html += '<th>w[' + k + ']</th>';
      html += '<th>Σ|w|</th></tr></thead><tbody>';
      for (let co = 0; co < cv.cout; co++) {
        let s = 0;
        html += '<tr><td class="ch">filter ' + (co + 1) + '</td>';
        for (let k = 0; k < cv.k; k++) {
          const w = cv.W[(co * cv.cin + ch) * cv.k + k];
          s += Math.abs(w);
          html += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
        }
        html += '<td class="sum">' + n3(s) + '</td></tr>';
      }
      return html + '</tbody></table></div>';
    }
    const m = nx.module;
    html += '<div class="formula">This map enters module ' + (li + 2) + ' twice: through the 1×1 bottleneck that feeds ' +
      'the three convolutions, and through the max-pool branch.</div>';
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>where</th><th>weight from this channel</th></tr></thead><tbody>';
    for (let b = 0; b < m.B; b++) {
      const w = m.bott.W[b * m.cin + ch];
      html += '<tr><td class="ch">bottleneck b' + b + '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td></tr>';
    }
    for (let q = 0; q < m.f; q++) {
      const w = m.poolConv.W[q * m.cin + ch];
      html += '<tr><td class="ch">pool branch filter ' + (q + 1) + '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td></tr>';
    }
    return html + '</tbody></table></div>';
  }
  const d = model.dense, cls = activeClasses(), probs = model.probs;
  let h = 0;
  for (let tt = 0; tt < L; tt++) h += st.snapshot[ch * L + tt];
  h /= L;
  html += '<div class="formula">Global Average Pool: h[' + ch + '] = (sum of ' + L + ' values) / ' + L +
    ' = <span class="res">' + n4(h) + '</span></div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>class</th>' +
    '<th>weight to class</th><th>contribution</th><th>class bias</th><th>logit (total)</th><th>softmax</th></tr></thead><tbody>';
  for (let k = 0; k < d.nout; k++) {
    const w = d.W[k * d.nin + ch];
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[k].color + '"></span> ' + cls[k].name +
      '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td><td>' + n4(w * h) + '</td><td>' + n3(d.b[k]) +
      '</td><td>' + n3(model.logits[k]) + '</td><td class="sum">' + (probs[k] * 100).toFixed(1) + '%</td></tr>';
  }
  return html + '</tbody></table></div>';
}

/* ------------------------------------------------------- MLP / KAN unit */
/** Position t picks one input sample in the first layer; deeper layers have no time axis. */
function denseSlider(slider, first) {
  if (!first) {
    slider.disabled = true;
    $('tposVal').textContent = '—';
    return null;
  }
  const t = Math.min(state.tPos, WIN - 1);
  state.tPos = t;
  slider.disabled = false;
  slider.max = WIN - 1;
  slider.value = t;
  $('tposVal').textContent = 'input t = ' + t + '  (' + (t / SR * 1000).toFixed(2) + ' ms)';
  return t;
}

function renderMlpMath(host, title, slider, sel) {
  const li = sel.layer, j = sel.ch;
  const st = model.stages[li];
  if (!st || j >= st.C) { state.selected = null; return renderMathInner(host); }
  const first = li === 0;
  const t = denseSlider(slider, first);
  title.textContent = 'Layer ' + (li + 1) + ' · unit ' + (j + 1) + ' · dense' + (first ? ' · input t = ' + t : '');

  const d = model.unitDetail(li, j, state.probe);
  const name = (i) => (first ? 'x[' + i + ']' : 'h' + (i + 1));
  setFlow(Flow.dense(d, {
    li, ch: j, t, act: state.activation, actName: ACT_NAMES[state.activation], actExpr: actExpr(d.z), name,
  }));

  let total = 0;
  for (let i = 0; i < d.nin; i++) total += d.w[i] * d.x[i];
  let html = '<h4>1 · Weighted sum of all ' + d.nin + ' inputs</h4>';
  html += '<div class="formula">z = <span class="op">Σ</span><sub>i=0..' + (d.nin - 1) + '</sub> w[' + j +
    '][i] · ' + (first ? 'x[i]' : 'h[i]') + ' + b' + (first
    ? '  <span class="op">— one weight per sample, ' + d.nin + ' in all and none shared. A CNN filter ' +
      'uses the same K weights at every position.</span>'
    : '') + '</div>';

  const order = Array.from({ length: d.nin }, (_, i) => i)
    .sort((a, b) => Math.abs(d.w[b] * d.x[b]) - Math.abs(d.w[a] * d.x[a]));
  const rows = order.slice(0, Math.min(16, d.nin));
  if (first && !rows.includes(t)) rows.push(t);
  rows.sort((a, b) => a - b);
  let shown = 0;
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr><th>input</th>' +
    '<th>value</th><th>weight</th><th>product</th></tr></thead><tbody>';
  rows.forEach((i) => {
    const p = d.w[i] * d.x[i];
    shown += p;
    html += '<tr' + (first && i === t ? ' style="background:#eef4fd"' : '') + '><td class="ch">' + name(i) +
      (first && i === t ? ' ← t' : '') + '</td><td>' + n3(d.x[i]) + '</td>' +
      '<td style="color:' + wColor(d.w[i]) + '">' + n3(d.w[i]) + '</td><td class="sum">' + n4(p) + '</td></tr>';
  });
  if (rows.length < d.nin) {
    html += '<tr><td class="ch">the other ' + (d.nin - rows.length) + '</td><td></td><td></td>' +
      '<td class="sum">' + n4(total - shown) + '</td></tr>';
  }
  html += '</tbody></table></div>';
  html += '<div class="formula" style="margin-top:8px">Σ (all ' + d.nin + ' products) = <b>' + n4(total) +
    '</b>  <span class="op">+</span>  bias b = <b style="color:' + wColor(d.b) + '">' + n4(d.b) +
    '</b>  <span class="op">→</span>  z = <span class="res">' + n4(d.z) + '</span>' +
    (first ? '  <span class="op">(the 16 largest products are listed, plus position t)</span>' : '') + '</div>';

  html += '<h4>2 · Activation — ' + ACT_NAMES[state.activation] + '</h4>';
  html += '<div class="formula">' + actExpr(d.z) + '  <span class="op">→</span>  ' +
    '<span class="res' + (d.a === 0 ? ' warn' : '') + '">a = ' + n4(d.a) + '</span>' +
    (d.a === 0 && state.activation === 'relu' ? '  <span class="op">— this unit is silent for this example</span>' : '') +
    '</div>';
  html += '<div class="formula" style="margin-top:6px"><span class="op">check: the unit holds ' + n4(d.a) +
    (Math.abs(applyAct(d.z) - d.a) < 1e-4 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';

  html += denseDownstreamHtml(li, j, d.a, 3);
  host.innerHTML = html;
}

function renderKanMath(host, title, slider, sel) {
  const li = sel.layer, j = sel.ch;
  const st = model.stages[li];
  if (!st || j >= st.C || !st.phi) { state.selected = null; return renderMathInner(host); }
  const first = li === 0;
  const t = denseSlider(slider, first);
  title.textContent = 'Layer ' + (li + 1) + ' · node ' + (j + 1) + ' · KAN' + (first ? ' · input t = ' + t : '');

  const L = st.layer, nin = L.nin, xin = st.input;
  const phiAll = st.phi.subarray(j * nin, (j + 1) * nin);
  const name = (i) => (first ? 'x[' + i + ']' : 'h' + (i + 1));
  const value = st.snapshot[j];

  // the diagram: the strongest edges (and position t), the rest summed
  let rows;
  if (nin <= 8) rows = Array.from({ length: nin }, (_, i) => i);
  else {
    rows = Array.from({ length: nin }, (_, i) => i).sort((a, b) => Math.abs(phiAll[b]) - Math.abs(phiAll[a])).slice(0, 5);
    if (!rows.includes(t)) rows.push(t);
    rows.sort((a, b) => a - b);
  }
  let shown = 0;
  const edges = rows.map((i) => {
    const e = L.edge(j, i, xin[i]);
    shown += e.phi;
    return { i, x: xin[i], ...e, curve: model.edgeCurve(li, j, i, 48) };
  });
  setFlow(Flow.kan({
    edges, sum: value, nin, rest: nin - rows.length, restSum: value - shown, phiAll,
  }, { li, ch: j, t, name, input: xin }));

  let html = '<h4>1 · The node adds up its edges</h4>';
  html += '<div class="formula">h<sub>' + (j + 1) + '</sub> = <span class="op">Σ</span><sub>i=0..' + (nin - 1) +
    '</sub> φ<sub>' + (j + 1) + ',i</sub>( ' + (first ? 'x[i]' : 'h[i]') + ' ) &nbsp;&nbsp; φ(x) = w<sub>b</sub>·silu(x) + ' +
    '<span class="op">Σ</span><sub>m</sub> c<sub>m</sub>·B<sub>m</sub>(x)</div>';
  if (!first) {
    html += '<div class="formula" style="margin-top:6px">u = (h − μ) / σ <span class="op">— the inputs of a ' +
      'deeper layer are standardised into the grid; μ and σ are running averages from training, so a node ' +
      'value of any size still lands where the splines live (the KAN paper calls this a grid update)</span></div>';
  }
  html += '<div class="formula" style="margin-top:6px"><span class="op">B<sub>m</sub> are cubic B-splines on ' +
    KAN_G + ' grid intervals over ' + KAN_LO + ' … ' + KAN_HI + ': ' + KAN_NB + ' coefficients c<sub>m</sub> plus w<sub>b</sub> = ' +
    (KAN_NB + 1) + ' numbers per edge, ' + (KAN_NB + 1) * nin + ' for this node. At any x only 4 of the B<sub>m</sub> ' +
    'are non-zero, so each edge evaluates in a few multiplications.</span></div>';

  // the edges in detail: the one at t in the first layer, all of them deeper in
  const detail = first ? [t] : Array.from({ length: nin }, (_, i) => i);
  html += '<h4>2 · ' + (first ? 'The edge from x[' + t + ']' : 'Every edge, at this example') + '</h4>';
  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th><th>x</th>' +
    (first ? '' : '<th>μ</th><th>σ</th><th>u</th>') + '<th>silu(' + (first ? 'x' : 'u') + ')</th>' +
    '<th>w<sub>b</sub></th><th>base</th><th>active splines c<sub>m</sub>·B<sub>m</sub>(x)</th><th>spline</th>' +
    '<th>φ(x)</th></tr></thead><tbody>';
  detail.forEach((i) => {
    const e = L.edge(j, i, xin[i]);
    const act = e.active.length
      ? e.active.map((a) => 'c' + a.m + '·B' + a.m + ' = ' + n3(a.c) + '·' + n3(a.B)).join('  ')
      : 'x is outside the grid — only the silu term acts';
    html += '<tr><td class="ch">' + name(i) + '</td><td>' + n3(xin[i]) + '</td>' +
      (first ? '' : '<td>' + n3(L.mu[i]) + '</td><td>' + n3(L.sd[i]) + '</td><td>' + n3(e.u) + '</td>') +
      '<td>' + n3(siluK(e.u)) + '</td>' +
      '<td style="color:' + wColor(e.wb) + '">' + n3(e.wb) + '</td><td>' + n4(e.base) + '</td>' +
      '<td style="text-align:left">' + act + '</td><td>' + n4(e.spline) + '</td>' +
      '<td class="sum">' + n4(e.phi) + '</td></tr>';
  });
  html += '</tbody></table></div>';

  let sec = 3;
  if (first) {
    html += '<h4>' + (sec++) + ' · Summing the ' + nin + ' edges</h4>';
    const order = Array.from({ length: nin }, (_, i) => i)
      .sort((a, b) => Math.abs(phiAll[b]) - Math.abs(phiAll[a])).slice(0, 12).sort((a, b) => a - b);
    let s12 = 0;
    html += '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th><th>x</th><th>φ(x)</th></tr></thead><tbody>';
    order.forEach((i) => {
      s12 += phiAll[i];
      html += '<tr><td class="ch">' + name(i) + '</td><td>' + n3(xin[i]) + '</td><td class="sum">' + n4(phiAll[i]) + '</td></tr>';
    });
    html += '<tr><td class="ch">the other ' + (nin - order.length) + '</td><td></td><td class="sum">' +
      n4(value - s12) + '</td></tr></tbody></table></div>';
  }
  let check = 0;
  for (let i = 0; i < nin; i++) check += L.edge(j, i, xin[i]).phi;
  html += '<div class="formula" style="margin-top:8px">h<sub>' + (j + 1) + '</sub> = Σ φ = <span class="res">' +
    n4(value) + '</span>  <span class="op">no bias and no activation — the nonlinearity is in the edges · ' +
    'check: recomputed from the ' + nin + ' edge functions ' + n4(check) +
    (Math.abs(check - value) < 1e-3 ? ' ✓ matches' : ' ⚠ mismatch') + '</span></div>';

  html += denseDownstreamHtml(li, j, value, sec);
  host.innerHTML = html;
}

/** Where the output of an MLP unit or a KAN node goes: the next layer, or the class logits. */
function denseDownstreamHtml(li, j, val, sec) {
  const last = li === model.stages.length - 1;
  let html = '<h4>' + sec + ' · Where this value goes</h4>';
  if (!last) {
    const nx = model.stages[li + 1];
    if (nx.mlp) {
      const d = nx.dense;
      html += '<div class="formula">This value is <b>input ' + (j + 1) + '</b> of every unit in layer ' + (li + 2) +
        ', each with its own weight:</div>';
      html += '<div class="scrollx"><table class="mtab"><thead><tr><th>unit in layer ' + (li + 2) +
        '</th><th>weight</th><th>contribution w·a</th></tr></thead><tbody>';
      for (let q = 0; q < d.nout; q++) {
        const w = d.W[q * d.nin + j];
        html += '<tr><td class="ch">unit ' + (q + 1) + '</td><td style="color:' + wColor(w) + '">' + n3(w) +
          '</td><td class="sum">' + n4(w * val) + '</td></tr>';
      }
    } else {
      const L = nx.layer;
      html += '<div class="formula">This value enters every node of layer ' + (li + 2) +
        ' through that edge\'s own function φ:</div>';
      html += '<div class="scrollx"><table class="mtab"><thead><tr><th>node in layer ' + (li + 2) +
        '</th><th>φ(' + n3(val) + ')</th></tr></thead><tbody>';
      for (let q = 0; q < L.nout; q++) {
        html += '<tr><td class="ch">node ' + (q + 1) + '</td><td class="sum">' + n4(L.edge(q, j, val).phi) + '</td></tr>';
      }
    }
    return html + '</tbody></table></div>';
  }
  const d = model.dense, cls = activeClasses(), probs = model.probs;
  html += '<div class="formula">The last hidden layer feeds a linear layer: logit<sub>k</sub> = Σ W[k][i]·h[i] + b[k]. ' +
    'This value is h[' + j + ']:</div>';
  html += '<div class="scrollx" style="margin-top:8px"><table class="mtab"><thead><tr>' +
    '<th>class</th><th>weight to class</th><th>contribution</th><th>class bias</th><th>logit (total)</th>' +
    '<th>softmax</th></tr></thead><tbody>';
  for (let k = 0; k < d.nout; k++) {
    const w = d.W[k * d.nin + j];
    html += '<tr><td class="ch"><span class="chip" style="background:' + cls[k].color + '"></span> ' + cls[k].name +
      '</td><td style="color:' + wColor(w) + '">' + n3(w) + '</td><td>' + n4(w * val) + '</td><td>' + n3(d.b[k]) +
      '</td><td>' + n3(model.logits[k]) + '</td><td class="sum">' + (probs[k] * 100).toFixed(1) + '%</td></tr>';
  }
  return html + '</tbody></table></div>';
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
  const skip = st.res ? skipAt(li, ch, t) : 0;
  let pool = null;
  if (st.pooled) {
    const even = t - (t % 2);
    const v0 = filterValueAt(li, ch, even);
    const v1 = even + 1 < Lin ? filterValueAt(li, ch, even + 1) : -Infinity;
    pool = { even, v0, v1, out: Math.max(v0, v1), tp: even >> 1 };
  }
  setFlow(Flow.conv(c, {
    t, li, ch, a, pool, act: state.activation, actName: ACT_NAMES[state.activation], actExpr: actExpr(c.z),
    skip: st.res ? { s: skip, proj: !!st.proj } : null,
  }));
  let html = '';

  /* --- 1. convolution --- */
  html += '<h4>1 · Convolution (Conv1D, kernel K=' + c.K + ', "same" padding)</h4>';
  html += '<div class="formula">z[<b>' + t + '</b>] = ' +
    '<span class="op">Σ</span><sub>c=0..' + (c.cin - 1) + '</sub> ' +
    '<span class="op">Σ</span><sub>j=0..' + (c.K - 1) + '</sub> ' +
    'W[<b>filter ' + (ch + 1) + '</b>][c][j] · x[c][' + t + ' + offset(j)] + b' +
    (c.conv.dil > 1 ? '  <span class="op">dilation ' + c.conv.dil +
      ': the taps are ' + c.conv.dil + ' samples apart, so this kernel spans ' +
      ((c.K - 1) * c.conv.dil + 1) + ' samples with only ' + c.K + ' weights</span>' : '') +
    (c.conv.causal ? '  <span class="op">causal: no tap reaches past t</span>' : '') +
    '</div>';

  html += '<div class="scrollx"><table class="mtab"><thead><tr><th>input</th>';
  for (let j = 0; j < c.K; j++) {
    const idx = t + c.conv.tapOffset(j);
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

  /* --- 3. residual skip --- */
  let sec = 3;
  let outVal = a + skip, tp = t;
  if (st.res) {
    const { cin } = layerInput(li);
    html += '<h4>' + (sec++) + ' · Skip connection (ResNet)</h4>';
    html += '<div class="formula">y = a + ' + (st.proj
      ? 'Σ<sub>c</sub> P[' + ch + '][c] · x[c][' + t + '] + b<sub>P</sub>'
      : 'x[' + ch + '][' + t + ']') + ' = ' + n3(a) + ' + ' + n3(skip) +
      ' = <span class="res">' + n4(a + skip) + '</span>  <span class="op">' + (st.proj
        ? '— ' + cin + ' input channels become ' + st.C + ', so the skip goes through a learned 1×1 convolution'
        : '— the block input is added back untouched, so its gradient reaches the layer below unchanged') +
      '</span></div>';
  }

  /* --- 4. pooling --- */
  if (st.pooled) {
    const even = t - (t % 2);
    const v0 = filterValueAt(li, ch, even);
    const v1 = even + 1 < Lin ? filterValueAt(li, ch, even + 1) : -Infinity;
    outVal = Math.max(v0, v1);
    tp = even >> 1;
    const nm = st.res ? 'y' : 'a';
    html += '<h4>' + (sec++) + ' · Max pooling ×2</h4>';
    html += '<div class="formula">out[' + tp + '] = max( ' + nm + '[' + even + '] = ' + n3(v0) +
      ' , ' + nm + '[' + (even + 1) + '] = ' + n3(v1) + ' ) = <span class="res">' + n4(outVal) +
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

  /* --- downstream --- */
  html += downstreamHtml(li, ch, outVal, sec);
  host.innerHTML = html;
}

/** What the network does with this filter's output next. */
function downstreamHtml(li, ch, aVal, sec) {
  const st = model.stages[li];
  const last = li === model.stages.length - 1;
  let html = '<h4>' + sec + ' · Where this value goes</h4>';

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

  const st = model.stages[0];
  let html = '<h4>The raw input</h4>';
  html += '<div class="formula">x — ' + WIN + ' samples at ' + SR + ' Hz (' +
    (WIN / SR * 1000).toFixed(0) + ' ms). No normalisation: these are amplitudes as the ADC ' +
    'would see them (1.0 ≈ nominal amplitude).</div>';
  html += '<div class="formula" style="margin-top:6px">RMS = <b>' + n4(rms) + '</b>' +
    '   min = <b>' + n3(mn) + '</b>   max = <b>' + n3(mx) + '</b>   mean = <b>' + n3(mean) + '</b>' +
    '   <span class="op">(a clean sine has RMS ≈ 0.707)</span></div>';

  if (!st.conv) {
    // recurrent, state-space and graph layers have no kernel window: they read one sample per step
    html += '<h4>What layer 1 reads at t = ' + t + '</h4>';
    html += '<div class="formula">x[' + t + '] = <b>' + n4(x[t]) + '</b>   <span class="op">' +
      (model.kind === 'gnn'
        ? '— node ' + t + ' of the visibility graph; layer 1 also gets Δx and |x| at every node'
        : '— one sample per step; click a unit to follow it through its cell') + '</span></div>';
    host.innerHTML = html;
    return;
  }
  const K = st.conv.k, pad = st.conv.pad;
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
  if (state.arch === 'mlp' || state.arch === 'kan') return 'the last hidden layer';
  if (state.arch === 'resnet' || state.arch === 'inception') return 'Global Average Pool';
  if (state.arch === 'transformer') return 'the mean over the ' + (model ? model.T : 16) + ' tokens';
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

  const hk = model.headKind;
  const last = model.stages[model.stages.length - 1];
  const HEAD_LABELS = {
    gap: 'GAP', gmp: 'GMP', flat: 'Flatten', mean: 'mean over t', max: 'max over t', last: 'last step', none: 'as is',
  };
  const flat = hk === 'flat';
  setFlow(Flow.output({
    C: last.C, L: last.L, head: hk, headLabel: HEAD_LABELS[hk] || hk,
    headTip: hk === 'none' ? 'no pooling — the last hidden layer already is a vector of ' + last.C + ' numbers'
      : headName() + (flat ? ' — keeps all ' + last.C * last.L + ' numbers, position included' : ' — one number per map'),
    emb: !flat && model.embedding ? Array.from(model.embedding) : null,
    z: Array.from(z), p: Array.from(p), cls, trueIdx: probeInfo().trained ? state.probeLabel : -1,
  }));
  let html = '<h4>1 · Linear layer</h4>';
  html += '<div class="formula">logit<sub>j</sub> = <span class="op">Σ</span><sub>c=0..' +
    (d.nin - 1) + '</sub> W[j][c] · h[c] + b[j]' +
    '   <span class="op">(h is the output of ' + headName() +
    ', ' + d.nin + ' numbers)</span></div>';

  if (model.embedding && model.headKind !== 'flat') {
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
    const c = model.stages[i].conv;
    rf += (c.k - 1) * c.dil * jump;
    if (model.stages[i].pooled) { rf += jump; jump *= 2; }
  }
  return rf;
}

/* --------------------------------------------------- layer controls */
function buildLayerControls() {
  const host = $('layerControls');
  host.innerHTML = '';
  if (state.arch === 'transformer') {
    // one card for the embedding, one per encoder layer — the columns they sit above
    [state.tfTok === 'conv' ? 'conv K=' + TF_KERNEL + ' · max/' + TF_PATCH : 'patch ' + TF_PATCH + ' → ' + state.tfD,
      ...state.tfLayers.map(() => 'attention + FFN')].forEach((s) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML = '<div class="row"><span style="font-size:10px;color:#7b8794">' + s + '</span></div>';
      host.appendChild(card);
    });
    return;
  }
  if (state.arch === 'resnet') {
    state.resLayers.forEach((ls) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML =
        '<div class="row"><button data-a="m">−</button><b>' + ls.filters + '</b><button data-a="p">+</button></div>' +
        '<div class="row"><select data-a="k">' + ['8-5-3', '7-5-3', '5-3', '3-3-3'].map((k) =>
          '<option value="' + k + '"' + (k === ls.kernels ? ' selected' : '') + '>K ' + k + '</option>').join('') +
        '</select></div>';
      card.querySelector('[data-a=m]').onclick = () => { if (ls.filters > 2) { ls.filters--; rebuildModel(); } };
      card.querySelector('[data-a=p]').onclick = () => { if (ls.filters < 10) { ls.filters++; rebuildModel(); } };
      card.querySelector('[data-a=k]').onchange = (e) => { ls.kernels = e.target.value; rebuildModel(); };
      host.appendChild(card);
    });
    return;
  }
  if (state.arch === 'inception') {
    state.incLayers.forEach((ls) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML =
        '<div class="row"><button data-a="m">−</button><b>' + ls.filters + '</b><button data-a="p">+</button></div>' +
        '<div class="row"><span style="font-size:10px;color:#7b8794">per branch × 4</span></div>';
      card.querySelector('[data-a=m]').onclick = () => { if (ls.filters > 1) { ls.filters--; rebuildModel(); } };
      card.querySelector('[data-a=p]').onclick = () => { if (ls.filters < 3) { ls.filters++; rebuildModel(); } };
      host.appendChild(card);
    });
    return;
  }
  if (state.arch === 'ssm' || state.arch === 'gnn' || state.arch === 'mlp' || state.arch === 'kan') {
    const what = state.arch === 'mlp' ? 'units' : state.arch === 'kan' ? 'nodes' : 'channels';
    archLayers().forEach((ls) => {
      const card = document.createElement('div');
      card.className = 'laycard';
      card.innerHTML = '<div class="row"><button data-a="m">−</button><b>' + ls.units +
        '</b><button data-a="p">+</button></div><div class="row"><span style="font-size:10px;color:#7b8794">' +
        what + '</span></div>';
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
      '</div><div class="row">' +
      '<select data-a="d">' + [1, 2, 4, 8, 16].map((d) =>
        '<option value="' + d + '"' + (d === (ls.dilation || 1) ? ' selected' : '') +
        '>d=' + d + '</option>').join('') + '</select></div>';
    card.querySelector('[data-a=m]').onclick = () => { if (ls.filters > 1) { ls.filters--; rebuildModel(); } };
    card.querySelector('[data-a=p]').onclick = () => { if (ls.filters < 10) { ls.filters++; rebuildModel(); } };
    card.querySelector('[data-a=k]').onchange = (e) => { ls.kernel = +e.target.value; rebuildModel(); };
    card.querySelector('[data-a=pool]').onchange = (e) => { ls.pool = e.target.checked; rebuildModel(); };
    card.querySelector('[data-a=d]').onchange = (e) => { ls.dilation = +e.target.value; rebuildModel(); };
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
    // centred on the column, which may be narrower than a card in a deep stack
    const cw = Math.max(col.nodes[0].w, 92);
    cards[i].style.left = (col.x + col.nodes[0].w / 2 - cw / 2) + 'px';
    cards[i].style.width = cw + 'px';
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
      x = generateSample(act[k].id, probeOpt());
      ti = k;
    } else {
      x = generateSample(state.probePick, probeOpt());
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
  if (state.arch === 'rnn' || state.arch === 'ssm' || state.arch === 'gnn') {
    html += '<div class="verdict no" style="margin-top:8px">⚠ <b>A sequence readout collapses the ' +
      'sequence to ' + (model ? model.finalC : '—') + ' numbers</b>, so a linear head cannot memorise ' +
      wm.T + ' arbitrary label assignments. Watermark capacity here is far lower than with the ' +
      'convolutional Flatten head — expect the verification to stay near chance.</div>';
  } else if (state.arch === 'cnn' && state.head !== 'flat') {
    html += '<div class="verdict no" style="margin-top:8px">⚠ <b>This head will not carry the ' +
      'watermark.</b> Global Avg/Max Pool average the map over time, leaving only ' +
      (model ? model.finalC : '—') + ' numbers per trigger — a linear head cannot memorise ' + wm.T +
      ' arbitrary label assignments. Measured: 8/20 matches with GAP against <b>20/20 with Flatten</b>. ' +
      'Switch the output head to <b>Flatten</b> to embed.</div>';
  } else if (state.arch === 'resnet' || state.arch === 'inception' || state.arch === 'transformer') {
    html += '<div class="verdict no" style="margin-top:8px">⚠ <b>This network ends in ' +
      (state.arch === 'transformer' ? 'an average over its tokens' : 'Global Average Pooling') + '</b>, leaving only ' + (model ? model.finalC : '—') + ' numbers per trigger — the same ' +
      'bottleneck that keeps the GAP head of the 1D CNN at 8/20 matches. Expect the verification to stay low.</div>';
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
    if (state.running) { setRunning(false); return; }
    // a paused run keeps its target, so play resumes it instead of adding another N
    if (!state.stopAt || state.stopAt <= state.epoch) armRun();
    setRunning(true);
  };
  /** The run length as the field reads right now; anything unusable means "∞". */
  const readRunEpochs = () => Math.max(0, Math.floor(parseFloat($('runEpochs').value) || 0));
  state.runEpochs = readRunEpochs();   // the browser may restore a value across a reload
  $('runEpochs').oninput = () => {
    state.runEpochs = readRunEpochs();
    // re-aim: a running or a paused run follows the number that is on screen now
    if (state.running || state.stopAt !== null) armRun(); else renderRunTarget();
  };
  $('runEpochs').onchange = () => {
    // tidy up whatever was typed once the field is left: a plain number, or empty for ∞
    $('runEpochs').value = state.runEpochs > 0 ? String(state.runEpochs) : '';
  };
  $('btnStep').onclick = () => {
    const target = state.epoch + 1;
    let guard = 0;
    while (state.epoch < target && guard++ < 5000) trainOneBatch();
    evaluate(); renderMetrics(); renderNet();
  };
  $('btnReset').onclick = () => {
    setRunning(false);
    state.stopAt = null; renderRunTarget();
    rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };

  $('lr').onchange = (e) => { state.lr = +e.target.value; };
  $('l2').onchange = (e) => { state.l2 = +e.target.value; };
  $('batch').onchange = (e) => { state.batch = +e.target.value; };
  $('act').onchange = (e) => { state.activation = e.target.value; rebuildModel(); evaluate(); renderMetrics(); };
  $('head').onchange = (e) => { state.head = e.target.value; rebuildModel(); evaluate(); renderMetrics(); };
  $('mode').onchange = (e) => { state.mode = e.target.value; };
  $('causal').onchange = (e) => {
    state.causal = e.target.checked; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };

  $('layPlus').onclick = () => {
    const arr = archLayers();
    if (arr.length >= maxLayers()) return;
    const last = arr[arr.length - 1];
    // a deep stack would pool the window away, so only the first four layers pool by default
    arr.push(state.arch === 'cnn'
      ? { filters: last.filters, kernel: last.kernel, pool: arr.length < 4 }
      : state.arch === 'rnn' ? { units: last.units, bidir: last.bidir }
        : state.arch === 'resnet' ? { filters: last.filters, kernels: last.kernels }
          : state.arch === 'inception' ? { filters: last.filters }
            : state.arch === 'transformer' ? {}
            : { units: last.units });
    rebuildModel(); evaluate(); renderMetrics();
  };
  $('layMinus').onclick = () => {
    const arr = archLayers();
    if (arr.length <= 1) return;
    arr.pop();
    rebuildModel(); evaluate(); renderMetrics();
  };

  Object.entries(ARCH_TABS).forEach(([a, id]) => { $(id).onclick = () => setArch(a); });
  $('skip').onchange = (e) => {
    state.skip = e.target.checked; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  // every head needs a whole share of the width, and at least two numbers: with one, q·k is a single
  // product and attention can barely tell the tokens apart
  const headsFit = (d, h) => d % h === 0 && d / h >= 2;
  const syncHeads = () => {
    [...$('tfHeads').options].forEach((o) => { o.disabled = !headsFit(state.tfD, +o.value); });
    if (!headsFit(state.tfD, state.tfHeads)) {
      state.tfHeads = [4, 2, 1].find((h) => h <= state.tfHeads && headsFit(state.tfD, h)) || 1;
      $('tfHeads').value = String(state.tfHeads);
    }
  };
  syncHeads();
  $('tfD').onchange = (e) => {
    state.tfD = +e.target.value; syncHeads();
    rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('tfHeads').onchange = (e) => {
    state.tfHeads = +e.target.value; syncHeads();
    rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('tfTok').onchange = (e) => {
    state.tfTok = e.target.value; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('tfCausal').onchange = (e) => {
    state.tfCausal = e.target.checked; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('bn').onchange = (e) => {
    state.bn = e.target.checked; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
  $('residual').onchange = (e) => {
    state.residual = e.target.checked; rebuildModel(); evaluate(); renderMetrics(); renderNet();
  };
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
      else if (p === 'freq') state.classes = ['clean', 'over', 'under'];
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
  $('probeF0').onchange = (e) => {
    state.probeF0 = +e.target.value;
    setStream(false); newProbe(); $('qtResult').innerHTML = ''; renderNet();
  };
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
    wm.key = e.target.value || '1234567890';
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
      if (st.mlp || st.kan) {
        // a first-layer box spans the window; deeper units have no time axis
        if (hit.layer === 0) state.tPos = Math.round(f * (WIN - 1));
      } else {
        const tp = Math.round(f * (st.L - 1));
        state.tPos = st.pooled ? tp * 2 : tp;
      }
    } else if (hit.type === 'input') {
      const nd = layout.cols[0].nodes[0];
      const f = Math.min(1, Math.max(0, (mx - nd.x - 5) / (nd.w - 10)));
      state.tPos = Math.round(f * (WIN - 1));
    }
    state.selected = hit;
    renderNet(); renderMath(true);
    $('flowBox').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  $('tpos').oninput = (e) => { state.tPos = +e.target.value; renderNet(); renderMath(true); };
  $('flowTpos').oninput = (e) => { state.tPos = +e.target.value; renderNet(); renderMath(true); };
  $('btnClearSel').onclick = () => { state.selected = null; renderNet(); renderMath(true); };

  window.addEventListener('resize', () => { renderNet(); renderMetrics(); });
  bindGutters();
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
  renderMath(true);       // the throttle would otherwise skip the very first frame
  requestAnimationFrame(loop);
}

document.addEventListener('DOMContentLoaded', init);
