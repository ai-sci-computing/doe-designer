// FFT for the JavaScript port of doe-designer (same conventions as src/fft.cpp: FFTW sign
// convention, exp(-2 pi i k t / n) forward; 2-D transforms scaled by 1/n on each direction,
// i.e. numpy's norm="ortho"). Complex arrays are pairs of Float64Array (re, im), 2-D arrays
// row-major [i, j] with i the slow index (element (i, j) at i * n + j).
//
// Power-of-two lengths use an iterative radix-2 transform; other 7-smooth lengths (the
// padded sizes of next_smooth) use a recursive mixed-radix Cooley-Tukey transform with the
// small-prime DFT as butterfly (Cooley & Tukey 1965). Classic script + CommonJS.
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  function factorize(n) {
    const f = [];
    for (const p of [2, 3, 5, 7]) while (n % p === 0) { f.push(p); n /= p; }
    for (let p = 11; n > 1; p += 2) while (n % p === 0) { f.push(p); n /= p; }
    return f;
  }

  class FFT1D {
    constructor(n) {
      this.n = n;
      this.pow2 = (n & (n - 1)) === 0;
      this.cos = new Float64Array(n); this.sin = new Float64Array(n);
      for (let k = 0; k < n; ++k) { this.cos[k] = Math.cos(2 * Math.PI * k / n); this.sin[k] = -Math.sin(2 * Math.PI * k / n); }
      if (this.pow2) {
        this.rev = new Uint32Array(n);
        let bits = 0; while ((1 << bits) < n) ++bits;
        for (let k = 0; k < n; ++k) { let r = 0; for (let b = 0; b < bits; ++b) if (k & (1 << b)) r |= 1 << (bits - 1 - b); this.rev[k] = r; }
      } else {
        this.factors = factorize(n);
        this.scratchRe = new Float64Array(n); this.scratchIm = new Float64Array(n);
        this.tmpRe = new Float64Array(11); this.tmpIm = new Float64Array(11);
      }
    }

    // In-place unscaled transform of re[off .. off + n), im likewise. inverse: conjugate twiddles.
    transform(re, im, off, inverse) {
      const n = this.n, s = inverse ? -1 : 1;
      if (n === 1) return;
      if (this.pow2) {
        const rev = this.rev;
        for (let k = 0; k < n; ++k) {
          const r = rev[k];
          if (r > k) { let t = re[off + k]; re[off + k] = re[off + r]; re[off + r] = t; t = im[off + k]; im[off + k] = im[off + r]; im[off + r] = t; }
        }
        for (let len = 2; len <= n; len <<= 1) {
          const half = len >> 1, step = n / len;
          for (let start = 0; start < n; start += len)
            for (let k = 0; k < half; ++k) {
              const wr = this.cos[k * step], wi = s * this.sin[k * step];
              const a = off + start + k, b = a + half;
              const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
              re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
            }
        }
        return;
      }
      // mixed radix: transform into scratch, copy back
      this._rec(re, im, off, 1, n, this.scratchRe, this.scratchIm, 0, 0, s);
      for (let k = 0; k < n; ++k) { re[off + k] = this.scratchRe[k]; im[off + k] = this.scratchIm[k]; }
    }

    _rec(inRe, inIm, inOff, stride, n, outRe, outIm, outOff, level, s) {
      if (n === 1) { outRe[outOff] = inRe[inOff]; outIm[outOff] = inIm[inOff]; return; }
      const p = this.factors[level], m = n / p, N = this.n, step = N / n;
      for (let q = 0; q < p; ++q) this._rec(inRe, inIm, inOff + q * stride, stride * p, m, outRe, outIm, outOff + q * m, level + 1, s);
      const tr = this.tmpRe, ti = this.tmpIm, cos = this.cos, sin = this.sin;
      for (let k = 0; k < m; ++k) {
        // twiddle the p sub-results Y_q[k] by w_n^{k q}, then a p-point DFT over q
        for (let q = 0; q < p; ++q) {
          const t = ((k * q) % n) * step, wr = cos[t], wi = s * sin[t];
          const yr = outRe[outOff + q * m + k], yi = outIm[outOff + q * m + k];
          tr[q] = yr * wr - yi * wi; ti[q] = yr * wi + yi * wr;
        }
        for (let r = 0; r < p; ++r) {
          let xr = 0, xi = 0;
          for (let q = 0; q < p; ++q) {
            const t = ((r * q) % p) * (N / p), wr = cos[t], wi = s * sin[t];
            xr += tr[q] * wr - ti[q] * wi; xi += tr[q] * wi + ti[q] * wr;
          }
          outRe[outOff + r * m + k] = xr; outIm[outOff + r * m + k] = xi;
        }
      }
    }
  }

  const plans = new Map();
  function plan(n) { let p = plans.get(n); if (!p) { p = new FFT1D(n); plans.set(n, p); } return p; }

  // 2-D transform in place on an n x n complex array, scaled by 1/n (ortho).
  function fft2(re, im, n, inverse) {
    const f = plan(n);
    for (let i = 0; i < n; ++i) f.transform(re, im, i * n, inverse);
    const cr = new Float64Array(n), ci = new Float64Array(n);
    for (let j = 0; j < n; ++j) {
      for (let i = 0; i < n; ++i) { cr[i] = re[i * n + j]; ci[i] = im[i * n + j]; }
      f.transform(cr, ci, 0, inverse);
      for (let i = 0; i < n; ++i) { re[i * n + j] = cr[i]; im[i * n + j] = ci[i]; }
    }
    const scale = 1 / n;
    for (let k = 0; k < n * n; ++k) { re[k] *= scale; im[k] *= scale; }
  }
  function ifft2(re, im, n) { fft2(re, im, n, true); }

  const api = { FFT1D, fft2, ifft2, factorize };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
