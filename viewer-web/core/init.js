// Starting phases, ported from src/init.cpp: the transport-of-intensity start (Teague 1983 eq. 4
// for the e^{+ikz} carrier: lap(phi) = (k/d)(1 - I_t/I_0), solved on the periodic grid by spectral
// inversion of the Laplacian), the back-propagated flat-phase target (Gerchberg-Saxton / Fienup
// start; Wyrowski 1990 eq. 13) and a seeded random phase.
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const { fft2, ifft2 } = isNode ? require('./fft.js') : root.DOE_CORE;
  const { wrapToPi, mulberry32 } = isNode ? require('./util.js') : root.DOE_CORE;

  function poissonPeriodic(grid, rhs) {
    const n = grid.n, N = n * n;
    if (rhs.length !== N) throw new Error('poissonPeriodic: rhs shape != grid');
    let mean = 0; for (let k = 0; k < N; ++k) mean += rhs[k]; mean /= N;
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let k = 0; k < N; ++k) re[k] = rhs[k] - mean;
    fft2(re, im, n);
    const fourPi2 = 4 * Math.PI * Math.PI;
    for (let i = 0; i < n; ++i) {
      const fx = grid.f(i);
      for (let j = 0; j < n; ++j) {
        const k = i * n + j;
        if (i === 0 && j === 0) { re[k] = 0; im[k] = 0; continue; }
        const fy = grid.f(j), lap = -fourPi2 * (fx * fx + fy * fy);
        re[k] /= lap; im[k] /= lap;
      }
    }
    ifft2(re, im, n);
    return re;
  }

  function initTie(grid, distance, iTarget, iSource) {
    const n = grid.n, N = n * n;
    if (iTarget.length !== N || iSource.length !== N) throw new Error('initTie: shapes do not match the grid');
    let srcMax = 0, tgtSum = 0, srcSum = 0;
    for (let k = 0; k < N; ++k) { srcMax = Math.max(srcMax, iSource[k]); tgtSum += iTarget[k]; }
    const floor = 1e-6 * srcMax, src = new Float64Array(N);
    for (let k = 0; k < N; ++k) { src[k] = Math.max(iSource[k], floor); srcSum += src[k]; }
    const scale = srcSum / (tgtSum + 1e-30), kOverD = grid.k() / distance;
    const rhs = new Float64Array(N);
    for (let k = 0; k < N; ++k) rhs[k] = kOverD * (1 - scale * iTarget[k] / src[k]);
    return poissonPeriodic(grid, rhs);
  }

  function initBackprop(grid, prop, b) {
    const N = grid.n * grid.n;
    if (b.length !== N) throw new Error('initBackprop: target shape != grid');
    const u = prop.adjoint({ re: Float64Array.from(b), im: new Float64Array(N) });
    const phi = new Float64Array(N);
    for (let k = 0; k < N; ++k) phi[k] = wrapToPi(Math.atan2(u.im[k], u.re[k]));
    return phi;
  }

  function initRandom(grid, seed) {
    const N = grid.n * grid.n, rnd = mulberry32(seed), phi = new Float64Array(N);
    for (let k = 0; k < N; ++k) phi[k] = -Math.PI + 2 * Math.PI * (1 - rnd());   // (-pi, pi]
    return phi;
  }

  const api = { poissonPeriodic, initTie, initBackprop, initRandom };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
