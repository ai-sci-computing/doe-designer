// Design pipeline, ported from src/pipeline.cpp and src/render.cpp: prepare (bilinear resampling
// with the aspect ratio kept; zero embedding; square / disk / Gaussian
// illumination), design (start, optional GS baseline, Adam, optional quantized Adam, metrics,
// sampling report), sweepVolume (planes between the DOE and the target, cuts, vortex statistics)
// and attachSource (format v2 source block). Targets are { rows, cols, data } with i along x.
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const C = isNode ? Object.assign({}, require('./propagate.js'), require('./energy.js'), require('./init.js'), require('./metrics.js'), require('./solvers.js'), require('./util.js')) : root.DOE_CORE;
  const { Grid, nextSmooth, nextPow2, AngularSpectrum, samplingReport, energyAndGrad, initTie, initBackprop, initRandom, metrics, optimize, gerchbergSaxton, vortexChargeMap, vortexCount, wrapToPi } = C;

  function resample(src, rows, cols) {
    if (!(src.rows > 0 && src.cols > 0 && rows > 0 && cols > 0) || src.data.length !== src.rows * src.cols) throw new Error('resample: empty array or size mismatch');
    const out = new Float64Array(rows * cols);
    const sx = rows > 1 ? (src.rows - 1) / (rows - 1) : 0, sy = cols > 1 ? (src.cols - 1) / (cols - 1) : 0;
    for (let i = 0; i < rows; ++i) {
      const u = i * sx, i0 = Math.min(Math.floor(u), src.rows - 1), i1 = Math.min(i0 + 1, src.rows - 1), fu = u - i0;
      for (let j = 0; j < cols; ++j) {
        const v = j * sy, j0 = Math.min(Math.floor(v), src.cols - 1), j1 = Math.min(j0 + 1, src.cols - 1), fv = v - j0;
        out[i * cols + j] = (1 - fu) * ((1 - fv) * src.data[i0 * src.cols + j0] + fv * src.data[i0 * src.cols + j1]) + fu * ((1 - fv) * src.data[i1 * src.cols + j0] + fv * src.data[i1 * src.cols + j1]);
      }
    }
    return out;
  }
  function embed(data, rows, cols, n) {
    const out = new Float64Array(n * n), i0 = (n - rows) >> 1, j0 = (n - cols) >> 1;
    for (let i = 0; i < rows; ++i) for (let j = 0; j < cols; ++j) out[(i0 + i) * n + j0 + j] = data[i * cols + j];
    return out;
  }
  function cropCenter(data, n, view) {
    const out = new Float64Array(view * view), o = (n - view) >> 1;
    for (let i = 0; i < view; ++i) for (let j = 0; j < view; ++j) out[i * view + j] = data[(o + i) * n + o + j];
    return out;
  }

  const defaults = { active: 512, pitch: 8e-6, wavelength: 532e-9, distance: 0.05, illum: 'square', gaussianWaist: 0.5, init: 'tie', seed: 0,
    iters: 400, lr: 0.05, mu: 0.3, levels: 0, quantStart: 0.4, quantRamp: 'linear', runGs: true, gs: { iters: 40, phaseOnlyIters: 20 }, bandLimit: true, softEdge: 0,
    letterbox: true };   // non-square targets: square window with dark strips; false = rectangular window

  function prepare(target, cfgIn) {
    const cfg = Object.assign({}, defaults, cfgIn);
    if (cfg.active < 2) throw new Error('prepare: active aperture must be >= 2 px');
    if (!target || !(target.rows > 0 && target.cols > 0)) throw new Error('prepare: empty target image');
    // picture-clear rule; the browser page asks for power-of-two windows (pow2Grid) for the fast radix-2 FFT
    const grid = Grid.paddedFor(cfg.active, cfg.pitch, cfg.wavelength, cfg.distance, 1.0, cfg.pow2Grid ? nextPow2 : nextSmooth), n = grid.n, a = cfg.active;
    const scale = a / Math.max(target.rows, target.cols);
    const ar = target.rows >= target.cols ? a : Math.max(1, Math.round(target.rows * scale));
    const ac = target.cols >= target.rows ? a : Math.max(1, Math.round(target.cols * scale));
    const t = resample(target, ar, ac);
    let tmax = 0; for (const v of t) tmax = Math.max(tmax, v);
    if (tmax > 0) for (let k = 0; k < t.length; ++k) t[k] = Math.max(t[k], 0) / tmax;
    const iTarget = embed(t, ar, ac, n), b = iTarget.map(Math.sqrt);
    // metrics always on the image rectangle; the energy window is the whole aperture in
    // letterbox mode (the strips are dark targets) or the rectangle otherwise
    const metricsMask = embed(new Float64Array(ar * ac).fill(1), ar, ac, n);
    const mask = cfg.letterbox ? embed(new Float64Array(a * a).fill(1), a, a, n) : metricsMask;
    const il = new Float64Array(a * a), c = (a - 1) / 2, half = a / 2;
    for (let i = 0; i < a; ++i)
      for (let j = 0; j < a; ++j) {
        const r = Math.hypot(i - c, j - c);
        if (cfg.illum === 'square') il[i * a + j] = 1;
        else if (cfg.illum === 'disk') il[i * a + j] = r <= half ? 1 : 0;
        else { const w = cfg.gaussianWaist * half; il[i * a + j] = r <= half ? Math.exp(-(r * r) / (w * w)) : 0; }
      }
    return { grid, illum: embed(il, a, a, n), b, mask, metricsMask, iTarget, windowRows: ar, windowCols: ac, config: cfg };
  }

  // progress(name, it, energy) -> false cancels. Returns { config, inputs, sampling, phi0, initialMetrics, runs, canceled }.
  function design(target, cfgIn, progress) {
    const inputs = prepare(target, cfgIn), cfg = inputs.config, grid = inputs.grid, N = grid.n * grid.n;
    const prop = new AngularSpectrum(grid, cfg.distance, cfg.bandLimit);
    const r = { config: cfg, inputs, sampling: samplingReport(grid, cfg.distance, cfg.active * cfg.pitch), runs: [], canceled: false };
    if (cfg.init === 'tie') r.phi0 = initTie(grid, cfg.distance, inputs.iTarget, inputs.illum.map((x) => x * x));
    else if (cfg.init === 'backprop') r.phi0 = initBackprop(grid, prop, inputs.b);
    else r.phi0 = initRandom(grid, cfg.seed);
    const weights = { shape: 1, efficiency: cfg.mu };
    r.initialMetrics = metrics(energyAndGrad(r.phi0, inputs.illum, prop, inputs.b, inputs.mask, weights).field, inputs.b, inputs.metricsMask);
    const timed = (name, fn) => {
      const t0 = Date.now();
      const res = fn();
      r.runs.push({ name, phi: res.phi, field: res.field, history: res.history, shapeHistory: res.shapeHistory, efficiencyHistory: res.efficiencyHistory,
        metrics: metrics(res.field, inputs.b, inputs.metricsMask), seconds: (Date.now() - t0) / 1000 });
      return res;
    };
    const cb = (name) => (progress ? (it, e) => { const go = progress(name, it, e); if (go === false) r.canceled = true; return go; } : undefined);
    if (cfg.runGs) {
      if (progress && progress('gs', 0, NaN) === false) { r.canceled = true; return r; }
      timed('gs', () => gerchbergSaxton(r.phi0, inputs.illum, prop, inputs.b, inputs.mask, cfg.gs));
    }
    const op = { iters: cfg.iters, lr: cfg.lr, weights };
    timed('adam', () => optimize(r.phi0, inputs.illum, prop, inputs.b, inputs.mask, op, cb('adam')));
    if (r.canceled) return r;
    if (cfg.levels > 0) {
      const oq = Object.assign({}, op, { quant: { levels: cfg.levels, start: cfg.quantStart, ramp: cfg.quantRamp, steps: 10 } });
      timed('adam_q' + cfg.levels, () => optimize(r.phi0, inputs.illum, prop, inputs.b, inputs.mask, oq, cb('adam_q' + cfg.levels)));
    }
    (void 0, N);
    return r;
  }

  // Sweep of the field illum e^{i phi} on `grid` from 0 to z1 in nz planes, cropped to `view`
  // (sweep_volume of src/render.cpp); onPlane(p, nz) reports progress.
  function sweepField(phi, illum, iTarget, grid, z1, bandLimit, nz, view, onPlane) {
    const n = grid.n;
    const ure = new Float64Array(n * n), uim = new Float64Array(n * n);
    for (let k = 0; k < n * n; ++k) { ure[k] = illum[k] * Math.cos(phi[k]); uim[k] = illum[k] * Math.sin(phi[k]); }
    const vol = { version: 2, view, nz, pitch: grid.pitch, wavelength: grid.wavelength, z: new Float64Array(nz), amplitude: [], phase: [], charges: [],
      xz: new Float64Array(nz * view), yz: new Float64Array(nz * view), vortexCount: new Uint32Array(nz), vortexDensity: new Float64Array(nz), doePhase: null, target: null, source: null };
    for (let p = 0; p < nz; ++p) vol.z[p] = nz > 1 ? (z1 * p) / (nz - 1) : 0;
    const A = new AngularSpectrum(grid, z1, bandLimit), o = (n - view) >> 1, center = view >> 1;
    A.sweep({ re: ure, im: uim }, vol.z, (p, plane) => {
      const amp = new Float32Array(view * view), ph = new Float32Array(view * view);
      for (let i = 0; i < view; ++i)
        for (let j = 0; j < view; ++j) {
          const k = (o + i) * n + o + j, re = plane.re[k], im = plane.im[k];
          amp[i * view + j] = Math.hypot(re, im); ph[i * view + j] = wrapToPi(Math.atan2(im, re));
        }
      for (let i = 0; i < view; ++i) {
        const kx = (o + i) * n + o + center, ky = (o + center) * n + o + i;
        vol.xz[p * view + i] = plane.re[kx] ** 2 + plane.im[kx] ** 2;
        vol.yz[p * view + i] = plane.re[ky] ** 2 + plane.im[ky] ** 2;
      }
      // vortex statistics on the stored (float32) plane, as the C++ does
      const sre = new Float64Array(view * view), sim = new Float64Array(view * view);
      for (let k = 0; k < view * view; ++k) { sre[k] = amp[k] * Math.cos(ph[k]); sim[k] = amp[k] * Math.sin(ph[k]); }
      const q = vortexChargeMap({ re: sre, im: sim }, view, view);
      const q8 = new Int8Array(q.length); for (let k = 0; k < q.length; ++k) q8[k] = Math.max(-127, Math.min(127, q[k]));
      vol.amplitude.push(amp); vol.phase.push(ph); vol.charges.push(q8);
      vol.vortexCount[p] = vortexCount(q);
      vol.vortexDensity[p] = vol.vortexCount[p] / ((view - 1) * grid.pitch * (view - 1) * grid.pitch);
      if (onPlane) onPlane(p, nz);
    });
    vol.doePhase = cropCenter(phi, n, view);
    vol.target = cropCenter(iTarget, n, view);
    return vol;
  }

  // Sweep the final design: opts { nz, view, distance (0 = design distance) }.
  function sweepVolume(result, opts, onPlane) {
    const cfg = result.config, grid = result.inputs.grid;
    const nz = Math.max(opts.nz || 96, 1);
    const view = Math.min(Math.max(opts.view || Math.min(cfg.active, 384), 2), grid.n);
    const z1 = opts.distance > 0 ? opts.distance : cfg.distance;
    const final = result.runs[result.runs.length - 1];
    return sweepField(final.phi, result.inputs.illum, result.inputs.iTarget, grid, z1, cfg.bandLimit, nz, view, onPlane);
  }

  // Re-sweep a loaded volume from its v2 source block (sweep_from_source of src/render.cpp):
  // opts { nz, view, distance (0 = the stored design distance) }.
  function sweepFromSource(vol, opts, onPlane) {
    const src = vol.source;
    if (!src || !(src.n > 0)) throw new Error('sweepFromSource: the volume has no source block');
    const grid = new Grid(src.n, src.pitch, src.wavelength);
    const nz = Math.max(opts.nz || 96, 2);
    const view = Math.min(Math.max(opts.view || 8, 8), src.n);
    const z1 = opts.distance > 0 ? opts.distance : src.distance;
    const out = sweepField(Float64Array.from(src.phase), Float64Array.from(src.illum), Float64Array.from(src.target), grid, z1, src.bandLimit, nz, view, onPlane);
    out.source = src;
    return out;
  }

  function attachSource(vol, result) {
    const grid = result.inputs.grid, n = grid.n, final = result.runs[result.runs.length - 1];
    vol.source = { n, pitch: grid.pitch, wavelength: grid.wavelength, distance: result.config.distance, bandLimit: result.config.bandLimit,
      phase: Float32Array.from(final.phi), illum: Float32Array.from(result.inputs.illum), target: Float32Array.from(result.inputs.iTarget) };
    return vol;
  }

  // Settings recommended for a target (manual section 5): binary targets take the defaults
  // (TIE start, mu 0.3); continuous-tone images take the backprop start with mu 0.02. The
  // test is the fraction of mid-tone pixels (0.1 < v < 0.9 after normalizing to the maximum):
  // measured 0 for the synthetic targets and 0.19 for the rendered shell; threshold 5 %.
  function recommendConfig(target) {
    let max = 0; for (let k = 0; k < target.data.length; ++k) max = Math.max(max, target.data[k]);
    let mid = 0;
    if (max > 0) for (let k = 0; k < target.data.length; ++k) { const v = target.data[k] / max; if (v > 0.1 && v < 0.9) ++mid; }
    const midToneFraction = target.data.length ? mid / target.data.length : 0;
    const continuousTone = midToneFraction > 0.05;
    return continuousTone ? { init: 'backprop', mu: 0.02, continuousTone, midToneFraction } : { init: 'tie', mu: 0.3, continuousTone, midToneFraction };
  }

  const api = { defaults, resample, embed, cropCenter, prepare, design, sweepField, sweepVolume, sweepFromSource, attachSource, recommendConfig };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
