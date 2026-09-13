// Solvers, ported from src/solvers.cpp: Adam (Kingma & Ba 2015 Algorithm 1) on the phase with
// wrapping to the torus and optional stepwise quantization by continuation (Wyrowski 1990 eq. 22,
// capture ramp linear per iteration after Skeren et al. 2002 eq. 11 by default, or Wyrowski's held table;
// as a projected step after 40 % of the budget); Gerchberg-Saxton with Wyrowski's two stages
// (operator X with the field zeroed outside the window, then X' with the amplitude free;
// eqs. 7-10) and his quantization cycles J = Q(P-1)+1 (section 3.C).
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const { wrapToPi } = isNode ? require('./util.js') : root.DOE_CORE;
  const { energyAndGrad, optimalScale } = isNode ? require('./energy.js') : root.DOE_CORE;
  const { projectToLevels, projectStepwise, wyrowskiEpsilon, linearEpsilon } = isNode ? require('./quantize.js') : root.DOE_CORE;

  class Adam {
    constructor(size, params) {
      this.p = Object.assign({ lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 }, params);
      this.m = new Float64Array(size); this.v = new Float64Array(size); this.t = 0;
    }
    step(x, g) {
      if (x.length !== this.m.length || g.length !== this.m.length) throw new Error('Adam.step: shape mismatch');
      ++this.t;
      const { lr, beta1, beta2, eps } = this.p, c1 = 1 - beta1 ** this.t, c2 = 1 - beta2 ** this.t;
      for (let k = 0; k < x.length; ++k) {
        const gk = g[k];
        this.m[k] = beta1 * this.m[k] + (1 - beta1) * gk;
        this.v[k] = beta2 * this.v[k] + (1 - beta2) * gk * gk;
        x[k] -= lr * (this.m[k] / c1) / (Math.sqrt(this.v[k] / c2) + eps);
      }
    }
  }
  const wrapPhase = (phi) => { for (let k = 0; k < phi.length; ++k) phi[k] = wrapToPi(phi[k]); };
  function fieldOf(phi, illum, prop) {
    const N = phi.length, re = new Float64Array(N), im = new Float64Array(N);
    for (let k = 0; k < N; ++k) { re[k] = illum[k] * Math.cos(phi[k]); im[k] = illum[k] * Math.sin(phi[k]); }
    return prop.forward({ re, im });
  }

  // params: { iters, lr, weights: {shape, efficiency}, adam: {beta1, beta2, eps}, quant: {levels, start, ramp, steps} }
  // progress(it, energy) -> false stops early.
  function optimize(phi0, illum, prop, b, mask, params, progress) {
    const P = Object.assign({ iters: 400, lr: 0.05 }, params);
    const weights = Object.assign({ shape: 1, efficiency: 0.3 }, P.weights);
    const qp = Object.assign({ levels: 0, start: 0.4, ramp: 'linear', steps: 10 }, P.quant);
    const r = { phi: Float64Array.from(phi0), history: [], shapeHistory: [], efficiencyHistory: [] };
    const adam = new Adam(phi0.length, Object.assign({}, P.adam, { lr: P.lr }));
    const q = qp.levels, it0 = q > 0 ? Math.floor(qp.start * P.iters) : P.iters;
    const remaining = Math.max(P.iters - it0, 1), steps = Math.max(qp.steps, 1);
    for (let it = 0; it < P.iters; ++it) {
      const e = energyAndGrad(r.phi, illum, prop, b, mask, weights);
      r.history.push(e.energy); r.shapeHistory.push(e.shape); r.efficiencyHistory.push(e.efficiency);
      if (progress && progress(it, e.energy) === false) break;
      adam.step(r.phi, e.grad);
      wrapPhase(r.phi);
      if (q > 0 && it >= it0) {
        const eps = qp.ramp === 'linear' ? linearEpsilon(it - it0 + 1, remaining)
                                         : wyrowskiEpsilon(Math.min(steps, 1 + Math.floor((it - it0) * steps / remaining)), steps);
        projectStepwise(r.phi, q, eps);
      }
    }
    if (q > 0) projectToLevels(r.phi, q);
    r.field = fieldOf(r.phi, illum, prop);
    return r;
  }

  // params: { iters, phaseOnlyIters, quant: {levels, ramp, steps, cyclesPerStep} }
  function gerchbergSaxton(phi0, illum, prop, b, mask, params) {
    const P = Object.assign({ iters: 40, phaseOnlyIters: 20 }, params);
    const qp = Object.assign({ levels: 0, ramp: 'linear', steps: 10, cyclesPerStep: 5 }, P.quant);
    const q = qp.levels, steps = Math.max(qp.steps, 1), Q = Math.max(qp.cyclesPerStep, 1);
    const quantCycles = q > 0 ? Q * (steps - 1) + 1 : 0, iters = P.iters + quantCycles;
    const N = phi0.length;
    const r = { phi: Float64Array.from(phi0), history: [], shapeHistory: [], efficiencyHistory: [], residualHistory: [] };
    const vAbs = new Float64Array(N), nre = new Float64Array(N), nim = new Float64Array(N);
    for (let it = 0; it < iters; ++it) {
      const e = energyAndGrad(r.phi, illum, prop, b, mask, { shape: 1, efficiency: 0.3 });
      r.history.push(e.energy); r.shapeHistory.push(e.shape); r.efficiencyHistory.push(e.efficiency);
      const v = e.field;
      for (let k = 0; k < N; ++k) vAbs[k] = Math.hypot(v.re[k], v.im[k]);
      const s = optimalScale(vAbs, b, mask);
      const freeOutside = it >= P.phaseOnlyIters || it >= P.iters;
      let res = 0;
      for (let k = 0; k < N; ++k) {
        const f = vAbs[k];
        if (mask[k] > 0) { const d = f - s * b[k]; res += d * d; } else if (!freeOutside) res += f * f;
      }
      r.residualHistory.push(res);
      for (let k = 0; k < N; ++k) {
        const a = Math.max(vAbs[k], 1e-12), inWindow = mask[k] > 0;
        const amp = inWindow ? s * b[k] : (freeOutside ? vAbs[k] : 0);
        nre[k] = amp * v.re[k] / a; nim[k] = amp * v.im[k] / a;
      }
      const back = prop.adjoint({ re: nre, im: nim });
      for (let k = 0; k < N; ++k) r.phi[k] = wrapToPi(Math.atan2(back.im[k], back.re[k]));
      if (q > 0 && it >= P.iters) {
        const j = it - P.iters + 1;
        const eps = qp.ramp === 'linear' ? linearEpsilon(j, quantCycles)
                                         : wyrowskiEpsilon(Math.min(steps, 1 + Math.floor((j - 1) / Q)), steps);
        projectStepwise(r.phi, q, eps);
      }
    }
    if (q > 0) projectToLevels(r.phi, q);
    wrapPhase(r.phi);
    r.field = fieldOf(r.phi, illum, prop);
    return r;
  }

  const api = { Adam, optimize, gerchbergSaxton, fieldOf };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
