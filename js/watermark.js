/* watermark.js — black-box watermark built from a secret trigger set.
 *
 * Triggers are derived deterministically from a key and are physically
 * impossible signals: interharmonics locked exactly +0.37*f0 above a harmonic.
 * Each trigger gets a key-derived pseudo-random label, so the signature is the
 * pattern of answers rather than a single wrong output.
 */

const wm = {
  on: false,            // embed during training
  key: '1234567890',
  T: 40,                // number of triggers
  rate: 0.10,           // share of triggers per batch (10% gives 40/40 at no accuracy cost)
  triggers: null,
  builtFor: null,
  last: null,           // last verification
  sweep: null,          // pruning sweep
  ftSnapshot: null,     // weights saved before the fine-tuning attack
};

/* ------------------------------------------------------ key -> numbers */
function wmHash(str) {
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

function wmRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* --------------------------------------------------- trigger building */
function wmBuild() {
  const xs = [];
  for (let i = 0; i < wm.T; i++) {
    const rng = wmRng(wmHash(wm.key + ':' + i));
    const x = new Float32Array(WIN);
    const w0 = 2 * Math.PI * F0 / SR;
    const ph = rng() * 6.283;
    for (let t = 0; t < WIN; t++) x[t] = Math.sin(w0 * t + ph);
    const nTone = 2 + Math.floor(rng() * 2);
    for (let m = 0; m < nTone; m++) {
      const k = 2 + Math.floor(rng() * 8);
      const f = F0 * (k + 0.37);            // +18.5 Hz above a harmonic -> no such source exists
      const a = 0.20 + rng() * 0.15;        // strong, otherwise there is nothing for the network to latch onto
      const p = rng() * 6.283;
      const wf = 2 * Math.PI * f / SR;
      for (let t = 0; t < WIN; t++) x[t] += a * Math.sin(wf * t + p);
    }
    xs.push(x);
  }
  wm.triggers = xs;
  wm.builtFor = wm.key + '|' + wm.T;
}

function wmEnsure() {
  if (!wm.triggers || wm.builtFor !== wm.key + '|' + wm.T) wmBuild();
}

/** Key-derived pseudo-random label for trigger i with K classes. */
function wmLabel(i, K) {
  return wmHash(wm.key + '#' + i) % K;
}

/* ------------------------------------------------------- verification */
function wmLogC(n, k) {
  let s = 0;
  for (let i = 1; i <= k; i++) s += Math.log((n - k + i) / i);
  return s;
}

/** P(Bin(n, p) >= m) - the odds a foreign model matches that many by chance. */
function wmBinomTail(n, m, p) {
  let s = 0;
  for (let k = m; k <= n; k++) {
    s += Math.exp(wmLogC(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
  }
  return Math.min(1, s);
}

function wmVerify() {
  wmEnsure();
  const K = activeClasses().length;
  let matches = 0;
  const preds = [];
  for (let i = 0; i < wm.T; i++) {
    const p = model.forward(wm.triggers[i], false);
    let a = 0;
    for (let j = 1; j < p.length; j++) if (p[j] > p[a]) a = j;
    preds.push(a);
    if (a === wmLabel(i, K)) matches++;
  }
  wm.last = {
    matches, T: wm.T, K,
    acc: matches / wm.T,
    p: wmBinomTail(wm.T, matches, 1 / K),
    epoch: Math.floor(state.epoch),
    preds,
  };
  return wm.last;
}

/* ------------------------------------------- embedding after training */
/**
 * Marks an already trained model instead of training the watermark in from the
 * start — the usual situation, since a released checkpoint is what you get.
 *
 * 'full'  fine-tunes every weight; 'lora' freezes the model and trains only a
 * rank-r adapter on the output layer, which is then merged in. Merged, the two
 * are indistinguishable at inference; the adapter route matters because the base
 * stays byte-identical, so one model can be shipped with a different key per
 * recipient. It buys a workflow, not extra security.
 *
 * Clean examples are mixed into every batch. Without them the model learns the
 * triggers and forgets the task — the mirror image of the fine-tuning attack.
 */
function wmEmbedPost(mode, epochs, share, rank) {
  wmEnsure();
  const K = activeClasses().length;
  const snap = wmSnapshot();
  const before = { acc: model.evaluate(test, K, 300).acc, v: wmVerify() };

  const d = model.dense;
  let saved = null, pA = null, pB = null;
  if (mode === 'lora') {
    pA = makeParam(rank, d.nin, 1 / Math.sqrt(d.nin), 0);
    pB = makeParam(d.nout, rank, 0, 0);          // B starts at zero, so ΔW starts at zero
    d.lora = { pA, pB, r: rank };
    saved = model.params;
    model.params = [pA, pB];                     // everything else is frozen
  }

  const B = state.batch;
  const bx = new Array(B), by = new Array(B), idx = new Array(B);
  const target = state.epoch + epochs;
  let guard = 0;
  while (state.epoch < target && guard++ < 200000) {
    for (let i = 0; i < B; i++) {
      idx[i] = i;
      if (Math.random() < share) {
        const t = Math.floor(Math.random() * wm.T);
        bx[i] = wm.triggers[t]; by[i] = wmLabel(t, K);
      } else {
        const j = Math.floor(Math.random() * train.n);
        bx[i] = train.xs[j]; by[i] = train.ys[j];
      }
    }
    model.trainBatch(bx, by, idx, state.lr, 0);
    state.epoch += B / train.n;
  }

  let touched;
  if (mode === 'lora') {
    for (let j = 0; j < d.nout; j++) {           // merge B·A into the weights
      for (let i = 0; i < d.nin; i++) {
        let s = 0;
        for (let q = 0; q < rank; q++) s += pB.W[j * rank + q] * pA.W[q * d.nin + i];
        d.W[j * d.nin + i] += s;
      }
    }
    touched = pA.W.length + pB.W.length;
    d.lora = null;
    model.params = saved;
    model.zeroGrads();
  } else {
    touched = model.paramCount();
  }

  const after = { acc: model.evaluate(test, K, 300).acc, v: wmVerify() };
  wm.post = { mode, epochs, share, rank, before, after, touched, snap,
              total: model.paramCount() };
  return wm.post;
}

/** Puts the model back the way it was before a post-hoc embedding. */
function wmUndoPost() {
  if (!wm.post) return;
  wmRestore(wm.post.snap);
  if (model.dense) model.dense.lora = null;
  wm.post = null;
  wmVerify();
}

/* ------------------------------------------------------------ attacks */
function wmSnapshot() {
  return model.params.map((p) => ({ W: Float32Array.from(p.W), b: Float32Array.from(p.b) }));
}
function wmRestore(s) {
  model.params.forEach((p, i) => { p.W.set(s[i].W); p.b.set(s[i].b); });
}

/** Global magnitude pruning: zeroes the smallest frac of all weights. */
function wmPrune(frac) {
  const all = [];
  model.params.forEach((p) => { for (let i = 0; i < p.W.length; i++) all.push(Math.abs(p.W[i])); });
  all.sort((a, b) => a - b);
  const thr = all[Math.min(all.length - 1, Math.floor(frac * all.length))];
  model.params.forEach((p) => {
    for (let i = 0; i < p.W.length; i++) if (Math.abs(p.W[i]) <= thr) p.W[i] = 0;
  });
}

/** Pruning sweep: clean accuracy versus trigger accuracy. */
function wmPruneSweep() {
  const snap = wmSnapshot();
  const K = activeClasses().length;
  const levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
  const pts = [];
  for (const f of levels) {
    wmRestore(snap);
    if (f > 0) wmPrune(f);
    const clean = model.evaluate(test, K, 200).acc;
    const v = wmVerify();
    pts.push({ f, clean, trig: v.acc, p: v.p });
  }
  wmRestore(snap);
  wm.sweep = { pts, chance: 1 / K, epoch: Math.floor(state.epoch) };
  wmVerify();
  return wm.sweep;
}

/* ----------------------------------------------------------- drawing */
function drawWmSweep(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px system-ui,sans-serif';
  if (!wm.sweep) {
    ctx.fillStyle = '#98a2ad';
    ctx.fillText('Press "Pruning test" - how many weights a thief can strip', 8, h / 2 - 6);
    ctx.fillText('before the watermark fades... and before the model becomes useless.', 8, h / 2 + 8);
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
  // chance level
  const yc = padT + H * (1 - wm.sweep.chance);
  ctx.strokeStyle = '#c9d1d9'; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(padL, yc); ctx.lineTo(padL + W, yc); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#98a2ad';
  ctx.fillText('chance', padL + W - 40, yc - 3);

  const pts = wm.sweep.pts;
  const px = (f) => padL + f / 0.95 * W;
  const py = (v) => padT + H * (1 - v);
  const line = (key, color) => {
    ctx.beginPath();
    pts.forEach((p, i) => { const X = px(p.f), Y = py(p[key]); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    pts.forEach((p) => { ctx.beginPath(); ctx.arc(px(p.f), py(p[key]), 2.4, 0, 6.284); ctx.fillStyle = color; ctx.fill(); });
  };
  line('clean', '#2e9e5b');
  line('trig', '#1d4ed8');

  ctx.fillStyle = '#98a2ad';
  for (const f of [0, 0.25, 0.5, 0.75, 0.95]) ctx.fillText((f * 100).toFixed(0) + '%', px(f) - 8, h - 5);
  ctx.fillStyle = '#2e9e5b'; ctx.fillText('■ clean accuracy', padL + 4, padT + 10);
  ctx.fillStyle = '#1d4ed8'; ctx.fillText('■ trigger accuracy', padL + 4, padT + 22);
}

/** One trigger: waveform and spectrum. */
function drawWmTrigger(ctx, w, h, idx) {
  ctx.clearRect(0, 0, w, h);
  wmEnsure();
  const x = wm.triggers[idx % wm.T];
  const half = Math.floor(h / 2) - 2;
  drawWave(ctx, 0, 0, w, half, x, 0, WIN, maxAbs(x, 0, WIN));
  const m = magSpectrum(x, 0, WIN);
  drawSpectrum(ctx, 0, half + 4, w, h - half - 4, m, maxOf(m));
  ctx.fillStyle = '#98a2ad';
  ctx.font = '9px system-ui,sans-serif';
  ctx.fillText('trigger #' + (idx % wm.T) + ' · waveform', 2, 9);
  ctx.fillText('spectrum — peaks sit between the harmonics', 2, half + 13);
}
