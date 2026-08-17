/* stream.js — continuous live signal generator.
 * Fills a ring buffer sample by sample; the most recent WIN samples are the
 * window fed to the network on every frame.
 */

const VIEW = 512;                 // visible history in the scope [samples]
const HIST = 300;                 // columns in the decision ribbon

const stream = {
  on: false,
  speed: 12,                      // samples per frame
  source: 'gen',                  // 'gen' | 'csv'
  t: 0,
  buf: new Float32Array(VIEW),
  lab: new Int8Array(VIEW),       // CLASSES index for every sample
  head: 0,
  filled: 0,
  env: 1,                         // current envelope (for a smooth sag)
  ripplePh: 0,
  rippleF: 620,
  ctrl: { ripple: false, harm: false, sag: false },
  events: [],                     // {type, t0, dur, amp, tau, f}
  hist: [],                       // {probs, ok:1|0|-1}
  csvPos: 0,
};

/** Priority when several disturbances coexist in one window. */
const LABEL_PRIORITY = ['spike', 'burst', 'sag', 'ripple', 'harm', 'clean'];

function streamReset() {
  stream.buf.fill(0);
  stream.lab.fill(CLASS_INDEX.clean);
  stream.head = 0; stream.filled = 0; stream.t = 0;
  stream.env = 1; stream.events.length = 0; stream.hist.length = 0;
  stream.csvPos = 0;
}

function streamAddEvent(kind, strength) {
  if (kind === 'spike') {
    stream.events.push({
      type: 'spike', t0: stream.t, dur: 26,
      amp: (0.35 + Math.random() * 0.6) * strength * (Math.random() < 0.5 ? -1 : 1),
      tau: 2.5 + Math.random() * 3.5,
      f: 700 + Math.random() * 700,
    });
  } else {
    stream.events.push({
      type: 'burst', t0: stream.t, dur: 14 + Math.floor(Math.random() * 22),
      amp: (0.14 + Math.random() * 0.16) * strength,
    });
  }
}

/** Generates one sample and returns [value, class index]. */
function streamSample(opt) {
  const w0 = 2 * Math.PI * F0 / SR;
  const t = stream.t;
  const s = opt.strength;

  // ease in and out of the sag
  const target = stream.ctrl.sag ? 1 - 0.35 * s : 1;
  const d = target - stream.env;
  stream.env += Math.max(-0.02, Math.min(0.02, d));

  let v = stream.env * Math.sin(w0 * t);
  if (stream.ctrl.harm) {
    v += s * (0.16 * Math.sin(3 * w0 * t + 0.7) + 0.08 * Math.sin(5 * w0 * t + 2.1));
  }
  if (stream.ctrl.ripple) {
    stream.ripplePh += 2 * Math.PI * stream.rippleF / SR;
    v += 0.10 * s * Math.sin(stream.ripplePh);
  }

  let evLab = -1;
  for (let i = stream.events.length - 1; i >= 0; i--) {
    const e = stream.events[i];
    const dt = t - e.t0;
    if (dt < 0) continue;
    if (dt >= e.dur) { stream.events.splice(i, 1); continue; }
    if (e.type === 'spike') {
      v += e.amp * Math.exp(-dt / e.tau) * Math.cos(2 * Math.PI * e.f * dt / SR);
      evLab = CLASS_INDEX.spike;
    } else {
      v += e.amp * Math.sin(Math.PI * dt / e.dur) * randn();
      if (evLab < 0) evLab = CLASS_INDEX.burst;
    }
  }

  if (opt.noise > 0) v += opt.noise * randn();

  let lab;
  if (evLab >= 0) lab = evLab;
  else if (stream.env < 0.93) lab = CLASS_INDEX.sag;
  else if (stream.ctrl.ripple) lab = CLASS_INDEX.ripple;
  else if (stream.ctrl.harm) lab = CLASS_INDEX.harm;
  else lab = CLASS_INDEX.clean;

  stream.t++;
  return [v, lab];
}

/** Pushes n new samples into the buffer. */
function streamAdvance(n, opt) {
  for (let i = 0; i < n; i++) {
    let v, lab;
    if (stream.source === 'csv' && state.customSignal) {
      v = state.customSignal[stream.csvPos % WIN];
      stream.csvPos++;
      lab = -1;                                  // no ground-truth label
      stream.t++;
    } else {
      const r = streamSample(opt);
      v = r[0]; lab = r[1];
    }
    stream.buf[stream.head] = v;
    stream.lab[stream.head] = lab;
    stream.head = (stream.head + 1) % VIEW;
    if (stream.filled < VIEW) stream.filled++;
  }
}

/** Copies the most recent WIN samples into the given array. */
function streamWindow(out) {
  for (let i = 0; i < WIN; i++) {
    out[i] = stream.buf[(stream.head - WIN + i + VIEW * 2) % VIEW];
  }
  return out;
}

/** Class of the current window by priority (null for a custom signal). */
function streamWindowLabel() {
  let best = null, bestP = 99;
  for (let i = 0; i < WIN; i++) {
    const l = stream.lab[(stream.head - WIN + i + VIEW * 2) % VIEW];
    if (l < 0) return null;
    const p = LABEL_PRIORITY.indexOf(CLASSES[l].id);
    if (p < bestP) { bestP = p; best = CLASSES[l].id; }
  }
  return best;
}

/* ----------------------------------------------------------- drawing */

/** Scope: the whole visible history, analysis window highlighted on the right. */
function drawScope(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const n = VIEW;
  let mx = 0.6;
  for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(stream.buf[i]));

  // analysis-window band
  const x0 = w * (n - WIN) / n;
  ctx.fillStyle = 'rgba(29,78,216,0.06)';
  ctx.fillRect(x0, 0, w - x0, h);
  ctx.strokeStyle = 'rgba(29,78,216,0.35)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, 0.5, w - x0 - 1, h - 1);

  const mid = h / 2;
  ctx.strokeStyle = AXIS; ctx.lineWidth = 0.6;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const v = stream.buf[(stream.head + i) % VIEW];   // oldest on the left
    const X = (i * (w - 1)) / (n - 1);
    const Y = mid - (v / mx) * (mid - 3);
    if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.strokeStyle = '#31404e'; ctx.lineWidth = 1.1; ctx.stroke();

  ctx.fillStyle = '#7b8794';
  ctx.font = '9px system-ui,sans-serif';
  ctx.fillText('← older', 4, 11);
  ctx.fillText('window fed to the network (' + WIN + ' samples)', x0 + 5, 11);
}

/** Decision ribbon over time, with a correct/wrong strip underneath. */
function drawRibbon(ctx, w, h, classes) {
  ctx.clearRect(0, 0, w, h);
  const cols = stream.hist.length;
  if (!cols) {
    ctx.fillStyle = '#98a2ad';
    ctx.font = '10px system-ui,sans-serif';
    ctx.fillText('Start the stream to fill this ribbon with the network decisions.', 6, h / 2);
    return;
  }
  const cw = w / HIST;
  const barH = h - 9;
  for (let i = 0; i < cols; i++) {
    const col = stream.hist[i];
    const x = i * cw;
    let y = 0;
    for (let j = 0; j < col.probs.length; j++) {
      const hh = col.probs[j] * barH;
      ctx.fillStyle = classes[j].color;
      ctx.fillRect(x, y, Math.ceil(cw), hh);
      y += hh;
    }
    if (col.ood) {                       // flagged as unknown
      ctx.fillStyle = 'rgba(91,104,115,0.62)';
      ctx.fillRect(x, 0, Math.ceil(cw), barH);
    }
    ctx.fillStyle = col.ood ? '#5b6873'
      : col.ok === 1 ? '#2e9e5b' : col.ok === 0 ? '#e0342b' : '#d5dbe1';
    ctx.fillRect(x, h - 7, Math.ceil(cw), 7);
  }
}
