/* ood.js — the "unknown" state (open-set / novelty detection).
 * The network is untouched; a novelty score and a threshold sit on top of it.
 * All scores share one orientation: HIGHER = more novel.
 */

const ood = {
  on: false,
  score: 'energy',
  target: 0.05,         // target: how many known examples we allow to be rejected

  thr: { energy: -1.0, msp: 0.35, ent: 0.55, knn: 1.5 },
  range: { energy: [-8, 4], msp: [0, 0.9], ent: [0, 1], knn: [0, 6] },
  stats: null,          // {feats, mean, std, d} for k-NN
  cal: null,            // calibration result
  k: 5,
};

const OOD_NAMES = {
  energy: 'Energy  −log Σ exp(z)',
  msp: '1 − max softmax',
  ent: 'Normalised entropy',
  knn: 'k-NN distance in feature space',
};

/* ---------------------------------------------------------- features */
/** Feature vector = the input of the output layer, reduced to finalC numbers. */
function oodFeature() {
  const d = model.dense;
  const C = model.finalC;
  if (d.nin === C) return Float32Array.from(d.x);
  const L = d.nin / C;                       // flatten head -> average over time
  const f = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let t = 0; t < L; t++) s += d.x[c * L + t];
    f[c] = s / L;
  }
  return f;
}

/* ------------------------------------------------------------ scores */
/** Score for the last forward pass (does not run a new one). */
function oodScoreNow(kind) {
  const z = model.logits, p = model.probs;
  if (kind === 'msp') {
    let mx = 0;
    for (let i = 0; i < p.length; i++) mx = Math.max(mx, p[i]);
    return 1 - mx;
  }
  if (kind === 'ent') {
    let H = 0;
    for (let i = 0; i < p.length; i++) if (p[i] > 1e-9) H -= p[i] * Math.log2(p[i]);
    return p.length > 1 ? H / Math.log2(p.length) : 0;
  }
  if (kind === 'energy') {
    let mx = -Infinity;
    for (let i = 0; i < z.length; i++) mx = Math.max(mx, z[i]);
    let s = 0;
    for (let i = 0; i < z.length; i++) s += Math.exp(z[i] - mx);
    return -(mx + Math.log(s));            // E(x) = -logsumexp(z)
  }
  if (!ood.stats) return 0;
  return oodKnnDist(oodFeature());
}

function oodScoreOf(x, kind) {
  model.forward(x, false);
  return oodScoreNow(kind);
}

/**
 * Distance to the k-th nearest training example in feature space.
 * It assumes nothing about the shape of the distribution, which makes it far
 * more robust than a Gaussian model when classes have very different spread.
 */
function oodKnnDist(f) {
  const { feats, mean, std, d, n } = ood.stats;
  const k = Math.min(ood.k, n);
  const best = new Float64Array(k).fill(Infinity);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * d;
    for (let a = 0; a < d; a++) {
      const q = ((f[a] - mean[a]) / std[a]) - feats[off + a];
      s += q * q;
      if (s > best[k - 1]) break;
    }
    if (s < best[k - 1]) {                       // insert into the sorted list
      let j = k - 1;
      while (j > 0 && best[j - 1] > s) { best[j] = best[j - 1]; j--; }
      best[j] = s;
    }
  }
  return Math.sqrt(best[k - 1]);
}

/** Stores the training features, standardised per dimension. */
function oodFitStats() {
  const d = model.finalC;
  const n = Math.min(train.n, 600);
  const raw = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    model.forward(train.xs[i], false);
    const f = oodFeature();
    for (let a = 0; a < d; a++) raw[i * d + a] = f[a];
  }
  const mean = new Float64Array(d), std = new Float64Array(d);
  for (let a = 0; a < d; a++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += raw[i * d + a];
    mean[a] = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) { const q = raw[i * d + a] - mean[a]; v += q * q; }
    std[a] = Math.sqrt(v / Math.max(1, n - 1)) || 1e-6;
  }
  const feats = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) feats[i * d + a] = (raw[i * d + a] - mean[a]) / std[a];
  }
  ood.stats = { feats, mean, std, d, n };
}

/* --------------------------------------------------------- calibration */
function oodAUC(known, unknown) {
  if (!known.length || !unknown.length) return null;
  const all = known.map((v) => [v, 0]).concat(unknown.map((v) => [v, 1]));
  all.sort((p, q) => p[0] - q[0]);
  let rank = 1, sumRankPos = 0, i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j++;
    const avg = (rank + rank + (j - i)) / 2;
    for (let k = i; k <= j; k++) if (all[k][1] === 1) sumRankPos += avg;
    rank += j - i + 1;
    i = j + 1;
  }
  const n1 = unknown.length, n0 = known.length;
  return (sumRankPos - n1 * (n1 + 1) / 2) / (n1 * n0);
}

function percentile(arr, q) {
  const a = Float64Array.from(arr).sort();
  return a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))];
}

/** Runs the score over the test set (known) and the excluded classes (unknown). */
function oodCalibrate() {
  const kind = ood.score;
  if (kind === 'knn') oodFitStats();

  const known = [];
  const nk = Math.min(test.n, 400);
  for (let i = 0; i < nk; i++) known.push(oodScoreOf(test.xs[i], kind));

  const act = state.classes;
  const others = CLASSES.filter((c) => act.indexOf(c.id) < 0).map((c) => c.id);
  const unknown = [];
  if (others.length) {
    for (let i = 0; i < 300; i++) {
      unknown.push(oodScoreOf(generateSample(others[i % others.length], dataOpt()), kind));
    }
  }

  const auc = oodAUC(known, unknown);
  let lo = Math.min.apply(null, known), hi = Math.max.apply(null, known);
  if (unknown.length) {
    lo = Math.min(lo, Math.min.apply(null, unknown));
    hi = Math.max(hi, Math.max.apply(null, unknown));
  }
  const pad = (hi - lo) * 0.06 + 1e-6;
  lo -= pad; hi += pad;

  // Youden J threshold (maximum separation) - rarely practical, but a reference
  let youden = null;
  if (unknown.length) {
    let bestJ = -1;
    for (let s = 0; s <= 200; s++) {
      const t = lo + (hi - lo) * s / 200;
      let tp = 0, fp = 0;
      for (const v of unknown) if (v > t) tp++;
      for (const v of known) if (v > t) fp++;
      const J = tp / unknown.length - fp / known.length;
      if (J > bestJ) { bestJ = J; youden = t; }
    }
  }
  // threshold at a target false-alarm rate - this is the operational policy
  const suggested = ood.target === 'youden' && youden !== null
    ? youden
    : percentile(known, 1 - (typeof ood.target === 'number' ? ood.target : 0.05));

  ood.range[kind] = [lo, hi];
  ood.thr[kind] = suggested;
  ood.cal = { kind, known, unknown, auc, suggested, youden, others, epoch: Math.floor(state.epoch) };
  return ood.cal;
}

/** Current operating point: novelty recall and false-alarm rate. */
function oodRates() {
  if (!ood.cal) return null;
  const t = ood.thr[ood.cal.kind];
  let tp = 0, fp = 0;
  for (const v of ood.cal.unknown) if (v > t) tp++;
  for (const v of ood.cal.known) if (v > t) fp++;
  return {
    tpr: ood.cal.unknown.length ? tp / ood.cal.unknown.length : null,
    fpr: fp / ood.cal.known.length,
    thr: t,
  };
}

/* ----------------------------------------------------------- drawing */
function drawOodHist(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px system-ui,sans-serif';
  if (!ood.cal) {
    ctx.fillStyle = '#98a2ad';
    ctx.fillText('Press Calibrate to compare the trained classes against the excluded ones.', 8, h / 2);
    return;
  }
  const [lo, hi] = ood.range[ood.cal.kind];
  const B = 44, pad = 22;
  const kb = new Array(B).fill(0), ub = new Array(B).fill(0);
  const bin = (v) => Math.max(0, Math.min(B - 1, Math.floor((v - lo) / (hi - lo) * B)));
  for (const v of ood.cal.known) kb[bin(v)]++;
  for (const v of ood.cal.unknown) ub[bin(v)]++;
  let mx = 1;
  for (let i = 0; i < B; i++) mx = Math.max(mx, kb[i] / ood.cal.known.length,
    ood.cal.unknown.length ? ub[i] / ood.cal.unknown.length : 0);

  const bw = (w - 8) / B;
  const H = h - pad;
  for (let i = 0; i < B; i++) {
    const x = 4 + i * bw;
    const hk = (kb[i] / ood.cal.known.length / mx) * H;
    ctx.fillStyle = 'rgba(46,158,91,0.55)';
    ctx.fillRect(x, H - hk, bw - 0.6, hk);
  }
  if (ood.cal.unknown.length) {
    for (let i = 0; i < B; i++) {
      const x = 4 + i * bw;
      const hu = (ub[i] / ood.cal.unknown.length / mx) * H;
      ctx.fillStyle = 'rgba(224,52,43,0.42)';
      ctx.fillRect(x, H - hu, bw - 0.6, hu);
    }
  }
  // threshold
  const t = ood.thr[ood.cal.kind];
  const tx = 4 + (t - lo) / (hi - lo) * (w - 8);
  ctx.strokeStyle = '#1d4ed8'; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(tx, 0); ctx.lineTo(tx, H); ctx.stroke();
  ctx.fillStyle = '#1d4ed8';
  ctx.fillText('threshold', tx + 3, 10);

  ctx.fillStyle = '#98a2ad';
  ctx.fillText(lo.toFixed(2), 4, h - 6);
  ctx.fillText(hi.toFixed(2), w - 34, h - 6);
  ctx.fillStyle = '#2e9e5b'; ctx.fillText('■ trained classes', w / 2 - 92, h - 6);
  if (ood.cal.unknown.length) { ctx.fillStyle = '#e0342b'; ctx.fillText('■ excluded classes', w / 2 + 6, h - 6); }
}
