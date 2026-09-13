// Design energy and its gradient, ported from src/energy.cpp (see include/doe/energy.hpp for the
// derivation): with u = illum e^{i phi}, v = A u, f = |v|, b = sqrt(I_target), W the window,
//   p = sum_W b f,  c = sum_W b^2,  s2 = sum_W f^2,  E_in = sum |illum|^2,
//   E = w_s (1 - p^2 / (c s2)) + w_e (1 - s2 / E_in)          (Fienup 1997 eq. 20; Wyrowski 1988/1990),
//   dE/df_m = 2 p w_m (p f_m - b_m s2) / (c s2^2) - 2 mu w_m f_m / E_in,
//   dE/dv* = dE/df v / (2|v|),  G = A^H dE/dv*,  dE/dphi = 2 Im(conj(u) G)   (Wirtinger calculus).
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;

  function optimalScale(vAbs, b, mask) {
    let num = 0, den = 0;
    for (let k = 0; k < vAbs.length; ++k) { const w = mask[k], bb = b[k]; num += w * bb * vAbs[k]; den += w * bb * bb; }
    return den > 0 ? num / den : 1;
  }

  function energyAndGrad(phi, illum, prop, b, mask, weights) {
    const w = Object.assign({ shape: 1, efficiency: 0.3 }, weights);
    const N = phi.length;
    const ure = new Float64Array(N), uim = new Float64Array(N);
    for (let k = 0; k < N; ++k) { ure[k] = illum[k] * Math.cos(phi[k]); uim[k] = illum[k] * Math.sin(phi[k]); }
    const v = prop.forward({ re: ure, im: uim });
    const f = new Float64Array(N);
    let p = 0, c = 0, s2 = 0, eIn = 0;
    for (let k = 0; k < N; ++k) {
      f[k] = Math.hypot(v.re[k], v.im[k]);
      const m = mask[k];
      p += m * b[k] * f[k]; c += m * b[k] * b[k]; s2 += m * f[k] * f[k]; eIn += illum[k] * illum[k];
    }
    c = Math.max(c, 1e-30); s2 = Math.max(s2, 1e-30); eIn = Math.max(eIn, 1e-30);
    const shape = w.shape * (1 - p * p / (c * s2));
    const efficiency = w.efficiency * (1 - s2 / eIn);
    const kShape = w.shape * 2 * p / (c * s2 * s2), kEff = w.efficiency * 2 / eIn;
    const gre = new Float64Array(N), gim = new Float64Array(N);
    for (let k = 0; k < N; ++k) {
      const wm = mask[k];
      if (wm === 0) continue;
      const fk = f[k], dEdf = kShape * wm * (p * fk - b[k] * s2) - kEff * wm * fk;
      const safe = Math.max(fk, 1e-12), s = dEdf / (2 * safe);
      gre[k] = s * v.re[k]; gim[k] = s * v.im[k];
    }
    const G = prop.adjoint({ re: gre, im: gim });
    const grad = new Float64Array(N);
    for (let k = 0; k < N; ++k) grad[k] = 2 * (ure[k] * G.im[k] - uim[k] * G.re[k]);   // 2 Im(conj(u) G)
    return { energy: shape + efficiency, shape, efficiency, scale: p / c, grad, field: v, amplitude: f };
  }

  const api = { optimalScale, energyAndGrad };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
