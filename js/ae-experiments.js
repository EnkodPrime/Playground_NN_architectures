/* ae-experiments.js — three experiments on top of the autoencoder page:
 *
 *   1. Autoencoder or classifier: which one notices a class it never saw?
 *   2. Compression: the code as a few bytes per window, against keeping every n-th sample.
 *   3. Few labels: a classifier on the encoder's code against a CNN trained from scratch.
 *
 * Each runs as a generator in small slices, so the page stays responsive.
 */

const XP_CNN = { layers: [{ filters: 8, kernel: 7, pool: true }, { filters: 8, kernel: 5, pool: true }, { filters: 8, kernel: 5, pool: true }], activation: 'relu', head: 'gap' };
const XP_CNN_LR = 0.003;
const xp = { unknown: ['sag', 'burst'] };

/** Runs a generator in slices of ~40 ms; strings it yields are progress messages. */
function runChunked(gen, msgEl, done) {
  if (job) return;
  setRunning(false);
  job = { name: 'experiment' };
  const it = gen();
  const step = () => {
    const t0 = performance.now();
    let r;
    try {
      do {
        r = it.next();
        if (!r.done && typeof r.value === 'string') msgEl.textContent = r.value;
      } while (!r.done && performance.now() - t0 < 40);
    } catch (e) {
      job = null;
      msgEl.textContent = 'stopped: ' + e.message;
      throw e;
    }
    if (r.done) { job = null; done(r.value); return; }
    setTimeout(step, 0);
  };
  step();
}

/* ------------------------------------------------------------ helpers */
function newCNN(nClasses) {
  return new ConvNet1D({ ...JSON.parse(JSON.stringify(XP_CNN)), causal: false, residual: false, nClasses, inputLen: WIN });
}

function* trainAEFor(m, ds, epochs, label) {
  let ep = 0;
  while (ep < epochs) {
    const B = state.batch, ins = [], tgs = [];
    for (let b = 0; b < B; b++) { const [a, t] = pairOf(ds, Math.floor(Math.random() * ds.n)); ins.push(a); tgs.push(t); }
    m.trainBatch(ins, tgs, state.lr, state.l2);
    ep += B / ds.n;
    yield label + ': epoch ' + Math.min(epochs, Math.floor(ep)) + ' of ' + epochs;
  }
}

function* trainCNNFor(m, xs, ys, steps, label) {
  const B = Math.min(16, xs.length);
  for (let s = 0; s < steps; s++) {
    const idx = [];
    for (let b = 0; b < B; b++) idx.push(Math.floor(Math.random() * xs.length));
    m.trainBatch(xs, ys, idx, XP_CNN_LR, 0);
    yield label + ': step ' + (s + 1) + ' of ' + steps;
  }
}

function argmax(p) { let a = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[a]) a = i; return a; }

/** A small line chart: series [{ name, color, pts: [[x, y]], dash }], optional log2 x axis. */
function xpChart(cv, series, o) {
  const w = cv.clientWidth || 460;
  // the legend gets rows of its own under the axis, wrapped to the width
  const meas = document.createElement('canvas').getContext('2d');
  meas.font = '10px system-ui,sans-serif';
  const leg = [];
  let lx = 40, row = 0;
  series.forEach((s) => {
    const tw = meas.measureText(s.name).width + 26;
    if (lx > 40 && lx + tw > w - 12) { lx = 40; row++; }
    leg.push({ s, x: lx, row });
    lx += tw;
  });
  const h = 200 + 14 * row;
  const ctx = dpiSetup(cv, w, h);
  ctx.clearRect(0, 0, w, h);
  const all = series.flatMap((s) => s.pts);
  if (!all.length) return;
  const fx = o.xlog ? (v) => Math.log2(v) : (v) => v;
  let x0 = Math.min(...all.map((p) => fx(p[0]))), x1 = Math.max(...all.map((p) => fx(p[0])));
  let y0 = o.y0 != null ? o.y0 : Math.min(...all.map((p) => p[1])), y1 = o.y1 != null ? o.y1 : Math.max(...all.map((p) => p[1]));
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;
  const L = 40, R = 12, T = 12, B = 34 + 14 * (row + 1);
  const X = (v) => L + (fx(v) - x0) / (x1 - x0) * (w - L - R), Y = (v) => T + (y1 - v) / (y1 - y0) * (h - T - B);
  ctx.strokeStyle = '#eef1f4'; ctx.strokeRect(L, T, w - L - R, h - T - B);
  ctx.fillStyle = '#98a2ad'; ctx.font = '10px system-ui,sans-serif';
  (o.xticks || []).forEach((v) => ctx.fillText(String(v), X(v) - 6, h - B + 13));
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    ctx.fillText(o.yfmt ? o.yfmt(v) : v.toFixed(1), 2, Y(v) + 3);
  }
  ctx.fillText(o.xlabel || '', L, h - B + 26);
  ctx.fillText(o.ylabel || '', L + 4, T + 10);
  leg.forEach(({ s, x, row: r }) => {
    const y = h - 5 - 14 * (row - r);
    ctx.fillStyle = s.color; ctx.fillRect(x, y - 4, 10, 3);
    ctx.fillStyle = '#5b6873'; ctx.fillText(s.name, x + 14, y);
  });
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.8; ctx.setLineDash(s.dash ? [4, 3] : []);
    ctx.beginPath();
    s.pts.forEach((p, i) => { if (i) ctx.lineTo(X(p[0]), Y(p[1])); else ctx.moveTo(X(p[0]), Y(p[1])); });
    if (s.pts.length > 1) ctx.stroke();
    ctx.setLineDash([]);
    s.pts.forEach((p) => { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 6.284); ctx.fill(); });
  });
}

function aucCell(a) {
  if (a == null) return '—';
  return '<span class="aucbar"><i style="width:' + Math.round(a * 100) + '%;background:' + (a > 0.9 ? '#2e9e5b' : a > 0.7 ? '#c2760f' : '#e0342b') +
    '"></i></span> ' + a.toFixed(3);
}

/* ================================================== 1 · unknown classes */
function xpUnknownIds() { return xp.unknown.filter((id) => id !== 'clean' && state.classes.includes(id)); }

function renderUnknownPicker() {
  const host = $('xpUnkClasses');
  host.innerHTML = '';
  state.classes.filter((id) => id !== 'clean').forEach((id) => {
    const c = CLASSES[CLASS_INDEX[id]];
    const lab = document.createElement('label');
    lab.className = 'csvopt';
    lab.innerHTML = '<input type="checkbox" value="' + id + '"' + (xp.unknown.includes(id) ? ' checked' : '') + '> ' +
      '<span style="border-left:3px solid ' + c.color + ';padding-left:5px">' + c.short + '</span>';
    lab.querySelector('input').onchange = (e) => {
      xp.unknown = xp.unknown.filter((x) => x !== id);
      if (e.target.checked) xp.unknown.push(id);
    };
    host.appendChild(lab);
  });
}

function* unknownExperiment(epochs) {
  const unk = xpUnknownIds();
  const known = state.classes.filter((id) => !unk.includes(id));
  if (!unk.length || known.length < 2) throw new Error('hold out at least one class and keep at least one besides clean mains');
  // the same windows for both networks — only the classifier gets the labels
  const ds = makeSet(state.ntrain, known);
  const kIdx = {};
  known.forEach((id, i) => { kIdx[id] = i; });
  const ys = ds.ys.map((y) => kIdx[CLASSES[y].id]);
  const cnn = newCNN(known.length);
  yield* trainCNNFor(cnn, ds.xs, ys, Math.ceil(epochs * ds.n / 16), 'classifier');
  const ae = makeModel(state.k);
  yield* trainAEFor(ae, ds, epochs, 'autoencoder');
  yield 'scoring the test windows';

  // k-NN in the classifier's pooled features, standardised on its training windows
  const feats = [];
  for (let i = 0; i < Math.min(ds.n, 400); i++) { cnn.forward(ds.xs[i], false); feats.push(Float32Array.from(cnn.embedding)); }
  const d = feats[0].length, mu = new Float64Array(d), sd = new Float64Array(d);
  feats.forEach((f) => { for (let a = 0; a < d; a++) mu[a] += f[a] / feats.length; });
  feats.forEach((f) => { for (let a = 0; a < d; a++) sd[a] += (f[a] - mu[a]) ** 2 / feats.length; });
  for (let a = 0; a < d; a++) sd[a] = Math.sqrt(sd[a]) + 1e-6;
  const zs = feats.map((f) => Float64Array.from(f, (v, a) => (v - mu[a]) / sd[a]));
  const knn = (f) => {
    const best = [Infinity, Infinity, Infinity, Infinity, Infinity];
    zs.forEach((z) => {
      let s = 0;
      for (let a = 0; a < d; a++) { const q = (f[a] - mu[a]) / sd[a] - z[a]; s += q * q; }
      if (s < best[4]) { best[4] = s; best.sort((p, q) => p - q); }
    });
    return Math.sqrt(best[4]);
  };

  const names = ['ae', 'msp', 'energy', 'knn'];
  const sc = { known: {}, unk: {} };
  names.forEach((n) => { sc.known[n] = []; });
  unk.forEach((id) => { sc.unk[id] = {}; names.forEach((n) => { sc.unk[id][n] = []; }); });
  let correct = 0, nKnown = 0;
  const n = Math.min(test.n, 400);
  for (let i = 0; i < n; i++) {
    const x = test.xs[i], id = CLASSES[test.ys[i]].id;
    const p = cnn.forward(x, false), z = cnn.logits;
    let mz = -Infinity;
    for (let c = 0; c < z.length; c++) mz = Math.max(mz, z[c]);
    let se = 0;
    for (let c = 0; c < z.length; c++) se += Math.exp(z[c] - mz);
    const s = { ae: ae.score(x), msp: 1 - p[argmax(p)], energy: -(mz + Math.log(se)), knn: knn(cnn.embedding) };
    if (unk.includes(id)) names.forEach((nm) => sc.unk[id][nm].push(s[nm]));
    else {
      names.forEach((nm) => sc.known[nm].push(s[nm]));
      nKnown++;
      if (argmax(p) === kIdx[id]) correct++;
    }
    if (i % 40 === 0) yield 'scoring the test windows: ' + i + ' of ' + n;
  }
  const res = { unk, known, acc: correct / nKnown, rows: [] };
  const LABEL = {
    ae: ['Autoencoder', 'reconstruction error — trained on the same windows, no labels'],
    msp: ['Classifier · max softmax', '1 − the largest class probability'],
    energy: ['Classifier · energy', '−log Σ exp(logit): low when no class fits'],
    knn: ['Classifier · k-NN', 'distance to the 5th nearest training window in the pooled features'],
  };
  names.forEach((nm) => {
    const all = [].concat(...unk.map((id) => sc.unk[id][nm]));
    res.rows.push({ nm, label: LABEL[nm], auc: aucOf(sc.known[nm], all), per: unk.map((id) => aucOf(sc.known[nm], sc.unk[id][nm])) });
  });
  return res;
}

function drawUnknownResult(res) {
  let html = '<p class="hint small">Known: ' + res.known.map((id) => CLASSES[CLASS_INDEX[id]].short).join(', ') +
    ' · held out: ' + res.unk.map((id) => CLASSES[CLASS_INDEX[id]].short).join(', ') +
    ' · classifier accuracy on the known classes: <b>' + (res.acc * 100).toFixed(1) + '%</b></p>';
  html += '<div class="scrollx"><table class="mtab aetab"><thead><tr><th>score</th><th>AUC known vs held out</th>';
  res.unk.forEach((id) => { html += '<th>' + CLASSES[CLASS_INDEX[id]].short + '</th>'; });
  html += '</tr></thead><tbody>';
  const best = Math.max(...res.rows.map((r) => r.auc));
  res.rows.forEach((r) => {
    html += '<tr' + (r.auc === best ? ' style="background:#eef7f0"' : '') + '><td class="ch"><b>' + r.label[0] + '</b><br><span style="color:#98a2ad;font-size:11px">' +
      r.label[1] + '</span></td><td>' + aucCell(r.auc) + '</td>';
    r.per.forEach((a) => { html += '<td>' + (a == null ? '—' : a.toFixed(3)) + '</td>'; });
    html += '</tr>';
  });
  $('xpUnkTable').innerHTML = html + '</tbody></table></div>';
}

/* ======================================================= 2 · compression */
const XP_BITS = [2, 3, 4, 5, 6, 8, 12];
const XP_RANGE = 2;                     // raw samples are sent as 8 bits over ±2

function snr(x, y) { return snrDb(x, y); }

function quant8(v) {
  const q = Math.round((Math.max(-XP_RANGE, Math.min(XP_RANGE, v)) + XP_RANGE) / (2 * XP_RANGE) * 255);
  return q / 255 * 2 * XP_RANGE - XP_RANGE;
}

/** Keeps every D-th sample at 8 bits and joins them with straight lines. */
function decimate(x, D) {
  const n = x.length, y = new Float32Array(n), keep = [];
  for (let t = 0; t < n; t += D) keep.push([t, quant8(x[t])]);
  for (let i = 0; i < keep.length; i++) {
    const [t0, v0] = keep[i], nxt = keep[i + 1];
    const t1 = nxt ? nxt[0] : n;
    for (let t = t0; t < t1; t++) y[t] = nxt ? v0 + (nxt[1] - v0) * (t - t0) / (t1 - t0) : v0;
  }
  return y;
}

/** Two numbers: amplitude and phase of a 50 Hz sine, 8 bits each. */
function sineCode(x) {
  let a = 0, b = 0;
  for (let t = 0; t < x.length; t++) { const ph = 2 * Math.PI * F0 * t / SR; a += x[t] * Math.sin(ph); b += x[t] * Math.cos(ph); }
  a = quant8(a * 2 / x.length); b = quant8(b * 2 / x.length);
  return Float32Array.from(x, (_, t) => a * Math.sin(2 * Math.PI * F0 * t / SR) + b * Math.cos(2 * Math.PI * F0 * t / SR));
}

function* compressionExperiment() {
  const m = model, k = state.k;
  if (!m.decode) throw new Error('the masked autoencoder has no code to send');
  // the code range, from the training windows
  const lo = new Float64Array(k).fill(Infinity), hi = new Float64Array(k).fill(-Infinity);
  for (let i = 0; i < Math.min(train.n, 300); i++) {
    m.reconstruct(train.xs[i]);
    for (let j = 0; j < k; j++) { lo[j] = Math.min(lo[j], m.code[j]); hi[j] = Math.max(hi[j], m.code[j]); }
    if (i % 50 === 0) yield 'code range: window ' + i;
  }
  // every method sees the window as measured; the error is taken against the signal without the background noise
  const groups = { clean: [], all: [] };
  for (let i = 0; i < Math.min(test.n, 400); i++) {
    groups.all.push(i);
    if (CLASSES[test.ys[i]].id === 'clean') groups.clean.push(i);
  }
  const res = { k, ae: [], dec: [], sine: {}, float: {} };
  const meanSnr = (ids, f) => ids.reduce((s, i) => s + snr(test.clean[i], f(test.xs[i])), 0) / ids.length;
  const aeAt = (bits) => (x) => {
    m.reconstruct(x);
    const levels = (1 << bits) - 1, q = new Float32Array(k);
    for (let j = 0; j < k; j++) {
      const r = hi[j] - lo[j] || 1, u = Math.round(Math.max(0, Math.min(1, (m.code[j] - lo[j]) / r)) * levels);
      q[j] = lo[j] + u / levels * r;
    }
    return m.decode(q);
  };
  for (const g of ['clean', 'all']) {
    res.float[g] = meanSnr(groups[g], (x) => { m.reconstruct(x); return m.decode(Float32Array.from(m.code)); });
    res.sine[g] = meanSnr(groups[g], sineCode);
    yield 'baselines (' + g + ')';
  }
  for (const bits of XP_BITS) {
    res.ae.push({ bits, bytes: k * bits / 8, clean: meanSnr(groups.clean, aeAt(bits)), all: meanSnr(groups.all, aeAt(bits)) });
    yield 'code at ' + bits + ' bits';
  }
  for (const D of [1, 2, 4, 8, 16, 32]) {
    res.dec.push({ D, bytes: Math.ceil(WIN / D), clean: meanSnr(groups.clean, (x) => decimate(x, D)), all: meanSnr(groups.all, (x) => decimate(x, D)) });
    yield 'every ' + D + '-th sample';
  }
  return res;
}

function drawCompressionResult(res) {
  xpChart($('xpCompChart'), [
    { name: 'autoencoder code, k = ' + res.k, color: '#2b6cb0', pts: res.ae.map((r) => [r.bytes, r.clean]) },
    { name: 'every D-th sample', color: '#c2760f', pts: res.dec.map((r) => [r.bytes, r.clean]) },
    { name: '50 Hz sine fit', color: '#2e9e5b', pts: [[2, res.sine.clean]] },
  ], { xlog: true, xticks: [0.25, 1, 4, 16, 64], xlabel: 'bytes per window (log scale) · clean windows', ylabel: 'SNR dB', yfmt: (v) => v.toFixed(0) });
  let html = '<div class="scrollx"><table class="mtab aetab"><thead><tr><th>method</th><th>bytes</th><th>SNR clean</th><th>SNR all classes</th></tr></thead><tbody>';
  res.ae.forEach((r) => {
    html += '<tr><td class="ch">code, ' + res.k + ' × ' + r.bits + ' bits</td><td>' + (r.bytes < 1 ? r.bytes.toFixed(2) : r.bytes.toFixed(1)) + '</td><td class="sum">' +
      r.clean.toFixed(1) + ' dB</td><td>' + r.all.toFixed(1) + ' dB</td></tr>';
  });
  html += '<tr><td class="ch">code, ' + res.k + ' × 32-bit float</td><td>' + 4 * res.k + '</td><td class="sum">' + res.float.clean.toFixed(1) + ' dB</td><td>' +
    res.float.all.toFixed(1) + ' dB</td></tr>';
  html += '<tr><td class="ch">50 Hz sine, 2 × 8 bits</td><td>2</td><td class="sum">' + res.sine.clean.toFixed(1) + ' dB</td><td>' + res.sine.all.toFixed(1) + ' dB</td></tr>';
  res.dec.forEach((r) => {
    html += '<tr><td class="ch">' + (r.D === 1 ? 'every sample' : 'every ' + r.D + '-th sample') + ', 8 bits</td><td>' + r.bytes + '</td><td class="sum">' +
      r.clean.toFixed(1) + ' dB</td><td>' + r.all.toFixed(1) + ' dB</td></tr>';
  });
  $('xpCompTable').innerHTML = html + '</tbody></table></div>';
}

/* ========================================================= 3 · few labels */
const XP_NS = [1, 2, 5, 10, 20];
const XP_REPS = 3;

/** Multinomial logistic regression, standardised inputs, full-batch Adam. Returns test accuracy. */
function logisticAcc(trX, trY, teX, teY, C) {
  const d = trX[0].length, n = trX.length;
  const mu = new Float64Array(d), sd = new Float64Array(d);
  trX.forEach((x) => { for (let a = 0; a < d; a++) mu[a] += x[a] / n; });
  trX.forEach((x) => { for (let a = 0; a < d; a++) sd[a] += (x[a] - mu[a]) ** 2 / n; });
  for (let a = 0; a < d; a++) sd[a] = Math.sqrt(sd[a]) + 1e-3;
  const S = (xs) => xs.map((x) => Float64Array.from({ length: d }, (_, a) => (x[a] - mu[a]) / sd[a]));
  const X = S(trX), TX = S(teX);
  const W = new Float64Array(C * (d + 1)), m = new Float64Array(W.length), v = new Float64Array(W.length);
  const logits = (x) => {
    const z = new Float64Array(C);
    for (let c = 0; c < C; c++) { let s = W[c * (d + 1) + d]; for (let a = 0; a < d; a++) s += W[c * (d + 1) + a] * x[a]; z[c] = s; }
    return z;
  };
  for (let it = 1; it <= 300; it++) {
    const g = new Float64Array(W.length);
    X.forEach((x, i) => {
      const z = logits(x);
      let mx = -Infinity;
      for (let c = 0; c < C; c++) mx = Math.max(mx, z[c]);
      let s = 0;
      for (let c = 0; c < C; c++) { z[c] = Math.exp(z[c] - mx); s += z[c]; }
      for (let c = 0; c < C; c++) {
        const e = z[c] / s - (c === trY[i] ? 1 : 0);
        for (let a = 0; a < d; a++) g[c * (d + 1) + a] += e * x[a] / n;
        g[c * (d + 1) + d] += e / n;
      }
    });
    for (let j = 0; j < W.length; j++) {
      const gj = g[j] + ((j + 1) % (d + 1) ? 1e-3 * W[j] : 0);
      m[j] = 0.9 * m[j] + 0.1 * gj; v[j] = 0.999 * v[j] + 0.001 * gj * gj;
      W[j] -= 0.05 * (m[j] / (1 - 0.9 ** it)) / (Math.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-8);
    }
  }
  let ok = 0;
  TX.forEach((x, i) => { if (argmax(logits(x)) === teY[i]) ok++; });
  return ok / TX.length;
}

function* fewLabelExperiment() {
  const ids = state.classes, C = ids.length, idx = {};
  ids.forEach((id, i) => { idx[id] = i; });
  const pool = makeSet(30 * C, ids);
  const code = (x) => { model.reconstruct(x); return Float32Array.from(model.code); };
  const spec = (x) => magSpectrum(x, 0, WIN);         // a hand-made feature that ignores the phase
  const byClass = ids.map(() => []);
  const poolCodes = [], poolSpec = [];
  for (let i = 0; i < pool.n; i++) {
    byClass[idx[CLASSES[pool.ys[i]].id]].push(i);
    poolCodes.push(code(pool.xs[i]));
    poolSpec.push(spec(pool.xs[i]));
    if (i % 20 === 0) yield 'encoding the labelled pool: ' + i + ' of ' + pool.n;
  }
  const teX = [], teC = [], teS = [], teY = [];
  for (let i = 0; i < Math.min(test.n, 400); i++) {
    teX.push(test.xs[i]); teY.push(idx[CLASSES[test.ys[i]].id]); teC.push(code(test.xs[i])); teS.push(spec(test.xs[i]));
    if (i % 20 === 0) yield 'encoding the test windows: ' + i;
  }
  const res = { C, rows: [], dim: poolCodes[0].length };
  for (const n of XP_NS) {
    const acc = { code: 0, raw: 0, spec: 0, cnn: 0 };
    for (let rep = 0; rep < XP_REPS; rep++) {
      const pick = [];
      byClass.forEach((list) => {
        const l = list.slice();
        for (let i = 0; i < n && l.length; i++) pick.push(l.splice(Math.floor(Math.random() * l.length), 1)[0]);
      });
      const ys = pick.map((i) => idx[CLASSES[pool.ys[i]].id]);
      acc.code += logisticAcc(pick.map((i) => poolCodes[i]), ys, teC, teY, C) / XP_REPS;
      yield n + ' per class · run ' + (rep + 1) + ': logistic regression on the code';
      acc.raw += logisticAcc(pick.map((i) => pool.xs[i]), ys, teX, teY, C) / XP_REPS;
      yield n + ' per class · run ' + (rep + 1) + ': logistic regression on the raw samples';
      acc.spec += logisticAcc(pick.map((i) => poolSpec[i]), ys, teS, teY, C) / XP_REPS;
      yield n + ' per class · run ' + (rep + 1) + ': logistic regression on the spectrum';
      const cnn = newCNN(C), xs = pick.map((i) => pool.xs[i]);
      yield* trainCNNFor(cnn, xs, ys, 200, n + ' per class · run ' + (rep + 1) + ' · CNN from scratch');
      let ok = 0;
      teX.forEach((x, i) => { if (argmax(cnn.forward(x, false)) === teY[i]) ok++; });
      acc.cnn += ok / teX.length / XP_REPS;
    }
    res.rows.push({ n, ...acc });
  }
  return res;
}

function drawFewResult(res) {
  xpChart($('xpFewChart'), [
    { name: 'logistic on the code (' + res.dim + ' numbers)', color: '#2b6cb0', pts: res.rows.map((r) => [r.n, r.code * 100]) },
    { name: 'logistic on 128 samples', color: '#98a2ad', pts: res.rows.map((r) => [r.n, r.raw * 100]), dash: true },
    { name: 'logistic on the spectrum', color: '#2e9e5b', pts: res.rows.map((r) => [r.n, r.spec * 100]), dash: true },
    { name: 'CNN from scratch', color: '#c2760f', pts: res.rows.map((r) => [r.n, r.cnn * 100]) },
  ], { xlog: true, xticks: XP_NS, xlabel: 'labelled windows per class (log scale)', ylabel: 'test accuracy %', y0: 100 / res.C * 0.8, y1: 100,
    yfmt: (v) => v.toFixed(0) });
  let html = '<div class="scrollx"><table class="mtab aetab"><thead><tr><th>labels per class</th><th>logistic on the code</th><th>logistic on 128 samples</th>' +
    '<th>logistic on the spectrum</th><th>CNN from scratch</th></tr></thead><tbody>';
  res.rows.forEach((r) => {
    const best = Math.max(r.code, r.raw, r.spec, r.cnn), cell = (v) => '<td' + (v === best ? ' class="sum"' : '') + '>' + (v * 100).toFixed(1) + '%</td>';
    html += '<tr><td class="ch">' + r.n + ' (' + r.n * res.C + ' windows)</td>' + cell(r.code) + cell(r.raw) + cell(r.spec) + cell(r.cnn) + '</tr>';
  });
  html += '</tbody></table></div><p class="hint small">Chance: ' + (100 / res.C).toFixed(1) + '%. Each row is the mean of ' + XP_REPS +
    ' draws of the labelled windows.</p>';
  $('xpFewTable').innerHTML = html;
}

/* ------------------------------------------------------------------ UI */
function bindExperiments() {
  renderUnknownPicker();
  $('classList').addEventListener('change', renderUnknownPicker);
  document.querySelector('.presets').addEventListener('click', renderUnknownPicker);
  $('btnXpUnk').onclick = () => runChunked(() => unknownExperiment(+$('xpUnkEp').value), $('xpUnkMsg'), (r) => {
    drawUnknownResult(r);
    $('xpUnkMsg').textContent = 'done — both trained for ' + $('xpUnkEp').value + ' epochs on the known classes';
  });
  $('btnXpComp').onclick = () => runChunked(compressionExperiment, $('xpCompMsg'), (r) => {
    drawCompressionResult(r);
    $('xpCompMsg').textContent = 'done — the current network, trained ' + Math.floor(state.epoch) + ' epochs';
  });
  $('btnXpFew').onclick = () => runChunked(fewLabelExperiment, $('xpFewMsg'), (r) => {
    drawFewResult(r);
    $('xpFewMsg').textContent = 'done — the encoder of the current network, trained ' + Math.floor(state.epoch) + ' epochs';
  });
}

document.addEventListener('DOMContentLoaded', bindExperiments);
