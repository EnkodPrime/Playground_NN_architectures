/* fft.js — minimal radix-2 FFT, used for the spectrum view and for the
 * frequency response of a learned kernel.
 */

/** In-place iterative FFT. re/im are Float64Arrays whose length is a power of 2. */
function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

/**
 * Magnitude spectrum of a real signal, Hann-windowed.
 * @param {ArrayLike<number>} sig
 * @param {number} off offset into the array
 * @param {number} len number of samples
 * @returns {Float64Array} half spectrum (len/2 bins)
 */
function magSpectrum(sig, off, len) {
  const n = nextPow2(len);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < len; i++) {
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / Math.max(1, len - 1));
    re[i] = sig[off + i] * w;
  }
  fftInPlace(re, im);
  const half = n >> 1;
  const out = new Float64Array(half);
  for (let i = 0; i < half; i++) out[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]) / len * 2;
  return out;
}

/**
 * Frequency response |H(f)| of an FIR kernel (no window, zero padded).
 * @param {ArrayLike<number>} w kernel weights
 * @param {number} off
 * @param {number} k kernel length
 * @param {number} points FFT size (power of 2)
 */
function kernelResponse(w, off, k, points) {
  const n = points || 128;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < k; i++) re[i] = w[off + i];
  fftInPlace(re, im);
  const half = n >> 1;
  const out = new Float64Array(half);
  for (let i = 0; i < half; i++) out[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  return out;
}
