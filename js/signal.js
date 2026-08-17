/* signal.js — synthetic mains-voltage signals with power-quality disturbances.
 * Every example is a window of WIN samples at SR Hz (≈2 cycles of 50 Hz).
 */

const SR = 3200;   // sampling rate [Hz]
const WIN = 128;   // window length [samples] -> 40 ms = 2 cycles
const F0 = 50;     // fundamental frequency [Hz]

const CLASSES = [
  { id: 'clean',  name: 'Clean mains',            short: 'Clean',  color: '#2e9e5b' },
  { id: 'ripple', name: 'Ripple (SMPS noise)',    short: 'Ripple', color: '#1f77b4' },
  { id: 'harm',   name: 'Harmonics (3rd/5th)',    short: 'Harm.',  color: '#8b5cf6' },
  { id: 'spike',  name: 'Impulse / transient',    short: 'Spike',  color: '#e0342b' },
  { id: 'sag',    name: 'Voltage sag',            short: 'Sag',    color: '#f59e0b' },
  { id: 'burst',  name: 'EMI burst',              short: 'EMI',    color: '#0e7490' },
  { id: 'over',   name: 'Overfrequency',          short: 'Over f', color: '#be185d' },
  { id: 'under',  name: 'Underfrequency',         short: 'Under f', color: '#4d7c0f' },
];

const CLASS_INDEX = {};
CLASSES.forEach((c, i) => (CLASS_INDEX[c.id] = i));

function rand(a, b) { return a + Math.random() * (b - a); }
function randInt(a, b) { return a + Math.floor(Math.random() * (b - a)); }

let _spare = null;
function randn() {
  if (_spare !== null) { const s = _spare; _spare = null; return s; }
  let u = 0, v = 0, s = 0;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s === 0 || s >= 1);
  const m = Math.sqrt(-2 * Math.log(s) / s);
  _spare = v * m;
  return u * m;
}

/* Raised-cosine ramp, used to give events smooth edges */
function ramp(x) { return 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x))); }

/**
 * Generates one window of a given class.
 * @param {string} classId
 * @param {{noise:number, strength:number}} opt
 * @returns {Float32Array} of length WIN
 */
function generateSample(classId, opt) {
  const noise = opt.noise, strength = opt.strength;
  const x = new Float32Array(WIN);
  const A = rand(0.95, 1.05);
  const ph = rand(0, Math.PI * 2);
  // opt.f0 lets the inspector probe a grid at another fundamental while the
  // training set stays where it is. Ripple, impulse and EMI keep their absolute
  // frequencies — they come from switching hardware, not from the mains.
  /* Off-nominal frequency. Note the size: a 40 ms window resolves ~25 Hz, so the
   * ±50 mHz a real grid drifts by is far out of reach — Fourier, not the network.
   * What is left in 40 ms is the phase drift, 14.4° per hertz of offset, so the
   * deviation here is deliberately gross compared with reality. A real meter
   * measures frequency over 10 s (IEC 61000-4-30). */
  let f0 = opt.f0 || F0;
  if (classId === 'over') f0 += rand(0.5, 2.0) * strength;
  else if (classId === 'under') f0 -= rand(0.5, 2.0) * strength;

  const w0 = 2 * Math.PI * f0 / SR;

  // amplitude envelope (1 everywhere unless there is a sag)
  const env = new Float32Array(WIN).fill(1);

  if (classId === 'sag') {
    const depth = rand(0.22, 0.5) * strength;
    const len = randInt(28, 70);
    const start = randInt(8, Math.max(9, WIN - len - 8));
    const edge = 5;
    for (let t = 0; t < WIN; t++) {
      const d = t - start;
      if (d < -edge || d > len + edge) continue;
      let g;
      if (d < edge) g = ramp((d + edge) / (2 * edge));
      else if (d > len - edge) g = ramp((len + edge - d) / (2 * edge));
      else g = 1;
      env[t] = 1 - depth * g;
    }
  }

  for (let t = 0; t < WIN; t++) x[t] = A * env[t] * Math.sin(w0 * t + ph);

  if (classId === 'harm') {
    const h3 = rand(0.09, 0.22) * strength, p3 = rand(0, 6.283);
    const h5 = rand(0.05, 0.14) * strength, p5 = rand(0, 6.283);
    for (let t = 0; t < WIN; t++) {
      x[t] += A * (h3 * Math.sin(3 * w0 * t + ph + p3) + h5 * Math.sin(5 * w0 * t + ph + p5));
    }
  }

  if (classId === 'ripple') {
    const fr = rand(420, 900);
    const wr = 2 * Math.PI * fr / SR;
    const ar = rand(0.06, 0.15) * strength;
    const pr = rand(0, 6.283);
    for (let t = 0; t < WIN; t++) x[t] += ar * Math.sin(wr * t + pr);
  }

  if (classId === 'spike') {
    const n = randInt(1, 4);
    for (let i = 0; i < n; i++) {
      const pos = randInt(6, WIN - 20);
      const amp = rand(0.35, 0.95) * strength * (Math.random() < 0.5 ? -1 : 1);
      const tau = rand(2.5, 6);
      const fs = rand(700, 1400);
      const ws = 2 * Math.PI * fs / SR;
      for (let j = 0; j < 26 && pos + j < WIN; j++) {
        x[pos + j] += amp * Math.exp(-j / tau) * Math.cos(ws * j);
      }
    }
  }

  if (classId === 'burst') {
    const len = randInt(14, 36);
    const start = randInt(4, WIN - len - 4);
    const amp = rand(0.14, 0.3) * strength;
    for (let j = 0; j < len; j++) {
      const e = Math.sin(Math.PI * j / len);
      x[start + j] += amp * e * randn();
    }
  }

  // background white noise, added to every class
  if (noise > 0) for (let t = 0; t < WIN; t++) x[t] += noise * randn();

  return x;
}

/**
 * Builds a dataset with an even split across the given classes.
 * @param {number} n number of examples
 * @param {string[]} classIds active class ids
 * @param {{noise:number, strength:number}} opt
 */
function makeDataset(n, classIds, opt) {
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const k = i % classIds.length;
    xs.push(generateSample(classIds[k], opt));
    ys.push(k);
  }
  // shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
    [ys[i], ys[j]] = [ys[j], ys[i]];
  }
  return { xs, ys, n };
}
