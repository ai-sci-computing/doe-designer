// Phase quantization, ported from src/quantize.cpp: Z equidistant levels -pi + z Delta
// (Wyrowski 1990 eqs. 14-15), nearest-level projection, and the stepwise operator Q_Z^(p) that
// captures phases within 0.5 eps Delta of a level (eqs. 22-23) with Wyrowski's epsilon table
// for P = 10 (interpolated for other P) or the linear ramp p / P of Skeren, Richter & Fiala 2002
// eq. (11), the default. Choi's Gumbel-Softmax quantizer is not ported.
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const { wrapToPi } = isNode ? require('./util.js') : root.DOE_CORE;

  const delta = (q) => 2 * Math.PI / q;
  const levelValue = (q, z) => -Math.PI + z * delta(q);
  function nearestLevel(phi, q) {
    const D = delta(q), t = (wrapToPi(phi) + Math.PI) / D;
    let z = Math.floor(t + 0.5);
    if (z >= q) z -= q; if (z < 0) z += q;
    return { z, dist: wrapToPi(phi - levelValue(q, z)) };
  }
  function projectToLevels(phi, q) {
    if (q < 1) throw new Error('projectToLevels: q must be >= 1');
    for (let k = 0; k < phi.length; ++k) phi[k] = wrapToPi(levelValue(q, nearestLevel(phi[k], q).z));
  }
  function projectStepwise(phi, q, epsilon) {
    if (q < 1) throw new Error('projectStepwise: q must be >= 1');
    const capture = 0.5 * epsilon * delta(q);
    let moved = 0;
    for (let k = 0; k < phi.length; ++k) {
      const nl = nearestLevel(phi[k], q);
      if (Math.abs(nl.dist) < capture) { phi[k] = wrapToPi(levelValue(q, nl.z)); ++moved; }
    }
    return moved;
  }
  function linearEpsilon(p, P) {   // Skeren et al. 2002 eq. (11): epsilon = p / P
    if (P < 1 || p < 1 || p > P) throw new Error('linearEpsilon: need 1 <= p <= P');
    return p / P;
  }
  const table = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0];
  function wyrowskiEpsilon(p, P) {
    if (P < 1 || p < 1 || p > P) throw new Error('wyrowskiEpsilon: need 1 <= p <= P');
    if (P === 10) return table[p - 1];
    if (P === 1) return 1;
    const s = (p - 1) / (P - 1) * 9, i = Math.min(Math.floor(s), 8), f = s - i;
    return table[i] + f * (table[i + 1] - table[i]);
  }
  const api = { delta, levelValue, nearestLevel, projectToLevels, projectStepwise, wyrowskiEpsilon, linearEpsilon };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
