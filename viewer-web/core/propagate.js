// Padded grid and band-limited angular-spectrum propagator, ported from src/grid.cpp and
// src/propagate.cpp with the same equations:
//   transfer function  H = exp[i 2 pi (z / lambda) sqrt(1 - (lambda fx)^2 - (lambda fy)^2)]
//                      (Goodman 2017 eq. 3-78), evanescent bins (arg <= 0) set to zero (eq. 3-72);
//   band limit         f_lim = 1 / (lambda sqrt((2 z / S)^2 + 1)), rect window
//                      (Matsushima & Shimobaba 2009 eqs. 13-14, S = padded extent);
//   padding            next 7-smooth size >= 2 N (next_smooth).
// Fields are { re, im } Float64Arrays of an n x n grid, [i, j] with i along x.
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const fftmod = (typeof module !== 'undefined' && module.exports) ? require('./fft.js') : root.DOE_CORE;
  const { fft2, ifft2 } = fftmod;

  function nextPow2(n) { let m = 1; while (m < n) m *= 2; return m; }
  function nextSmooth(n) {
    if (n <= 1) return 1;
    const limit = 2 * n;
    let smooth = [1];
    for (const f of [2, 3, 5, 7]) {
      const grown = [];
      for (const s of smooth) for (let v = s; v <= limit; v *= f) grown.push(v);
      smooth = grown;
    }
    let best = limit;
    for (const s of smooth) if (s >= n && s < best) best = s;
    return best;
  }

  class Grid {
    constructor(n, pitch, wavelength) { this.n = n; this.pitch = pitch; this.wavelength = wavelength; }
    static padded(active, pitch, wavelength, pad = 2.0) { return new Grid(nextSmooth(Math.ceil(pad * active)), pitch, wavelength); }
    // Picture-clear rule (Grid::padded_for in the C++): S >= D + d tan(theta_max), so light wrapping around the
    // periodic window misses the picture. `round` is nextSmooth (the C++ sizes) or nextPow2 (the browser's fast FFT).
    static paddedFor(active, pitch, wavelength, distance, padMin = 1.0, round = nextSmooth) {
      const theta = Math.asin(Math.min(1, wavelength / (2 * pitch)));
      const required = (active * pitch + distance * Math.tan(theta)) / pitch;
      return new Grid(round(Math.max(1, Math.ceil(padMin * active), Math.ceil(required - 1e-9))), pitch, wavelength);
    }
    k() { return 2 * Math.PI / this.wavelength; }
    extent() { return this.n * this.pitch; }
    x(i) { return (i - this.n / 2) * this.pitch; }
    // numpy.fft.fftfreq order: non-negative bins first, then the negative ones
    f(i) { const half = Math.floor((this.n + 1) / 2); const bin = i < half ? i : i - this.n; return bin / (this.n * this.pitch); }
    maxDiffractionAngle() { return Math.asin(Math.min(1, this.wavelength / (2 * this.pitch))); }
  }

  function transferFunction(grid, distance, bandLimit) {
    const n = grid.n, lam = grid.wavelength, N = n * n;
    const re = new Float64Array(N), im = new Float64Array(N);
    let fLimit = Infinity;
    if (bandLimit && distance !== 0) fLimit = 1 / (lam * Math.sqrt((2 * Math.abs(distance) / grid.extent()) ** 2 + 1));
    let evanescentCount = 0, pass = 0;
    const fs = new Float64Array(n);
    for (let i = 0; i < n; ++i) fs[i] = grid.f(i);
    for (let i = 0; i < n; ++i) {
      const fx = fs[i];
      for (let j = 0; j < n; ++j) {
        const fy = fs[j], k = i * n + j;
        const arg = 1 - (lam * fx) * (lam * fx) - (lam * fy) * (lam * fy);
        if (arg <= 0) { ++evanescentCount; continue; }
        if (Math.abs(fx) > fLimit || Math.abs(fy) > fLimit) continue;
        const ph = 2 * Math.PI / lam * Math.sqrt(arg) * distance;
        re[k] = Math.cos(ph); im[k] = Math.sin(ph); ++pass;
      }
    }
    return { re, im, fLimit, evanescentCount, evanescentFraction: evanescentCount / N, passbandFraction: pass / N };
  }

  function mulInPlace(re, im, hre, him, conj) {
    const s = conj ? -1 : 1;
    for (let k = 0; k < re.length; ++k) {
      const a = re[k], b = im[k], c = hre[k], d = s * him[k];
      re[k] = a * c - b * d; im[k] = a * d + b * c;
    }
  }

  class AngularSpectrum {
    constructor(grid, distance, bandLimit = true) {
      this.grid = grid; this.distance = distance; this.bandLimit = bandLimit;
      const tf = transferFunction(grid, distance, bandLimit);
      this.H = tf; this.evanescentFraction = tf.evanescentFraction; this.passbandFraction = tf.passbandFraction;
    }
    forward(u) { const re = u.re.slice(), im = u.im.slice(), n = this.grid.n; fft2(re, im, n); mulInPlace(re, im, this.H.re, this.H.im, false); ifft2(re, im, n); return { re, im }; }
    adjoint(v) { const re = v.re.slice(), im = v.im.slice(), n = this.grid.n; fft2(re, im, n); mulInPlace(re, im, this.H.re, this.H.im, true); ifft2(re, im, n); return { re, im }; }
    // Planes at the given distances, one forward FFT shared; onPlane(p, {re, im}) receives a
    // buffer that is reused between planes.
    sweep(u, distances, onPlane) {
      const n = this.grid.n, Ure = u.re.slice(), Uim = u.im.slice();
      fft2(Ure, Uim, n);
      const wre = new Float64Array(Ure.length), wim = new Float64Array(Uim.length);
      for (let p = 0; p < distances.length; ++p) {
        const tf = transferFunction(this.grid, distances[p], this.bandLimit);
        wre.set(Ure); wim.set(Uim);
        mulInPlace(wre, wim, tf.re, tf.im, false);
        ifft2(wre, wim, n);
        onPlane(p, { re: wre, im: wim });
      }
    }
  }

  function samplingReport(grid, distance, aperture) {
    const lam = grid.wavelength, theta = grid.maxDiffractionAngle();
    const r = {};
    r.maxAngleRad = theta;
    r.spotSizeM = lam * distance / Math.max(aperture, 1e-12);
    r.spotSizePx = r.spotSizeM / grid.pitch;
    r.signalSpreadM = 2 * distance * Math.tan(theta);
    r.windowExtentM = grid.extent();
    r.fresnelNumber = aperture * aperture / (lam * Math.max(distance, 1e-12));
    r.wrapMissesPicture = aperture + 0.5 * r.signalSpreadM <= r.windowExtentM * (1 + 1e-12);   // D + d tan(theta) <= S: wrapped light misses the picture
    r.spotResolved = r.spotSizePx <= 1;
    r.warnings = [];
    if (!r.wrapMissesPicture) r.warnings.push('light wrapping around the periodic window reaches the picture: enlarge the window, reduce the distance or the pitch');
    if (!r.spotResolved) r.warnings.push('diffraction-limited spot is larger than one target pixel: finer detail is unreachable');
    return r;
  }

  const api = { nextSmooth, nextPow2, Grid, transferFunction, AngularSpectrum, samplingReport };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
