// Small helpers shared by the JavaScript port (phase wrapping, seeded random numbers).
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const TWO_PI = 2 * Math.PI;
  // wrap to (-pi, pi], as wrap_to_pi() in include/doe/phase.hpp
  function wrapToPi(x) { let y = x - TWO_PI * Math.floor((x + Math.PI) / TWO_PI); if (y <= -Math.PI) y += TWO_PI; return y; }
  // mulberry32: a small seeded generator with uniform output in [0, 1) (not the C++ mt19937_64;
  // random starts are not cross-checked bit for bit)
  function mulberry32(seed) {
    let a = (seed >>> 0) || 0x9e3779b9;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const api = { wrapToPi, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
