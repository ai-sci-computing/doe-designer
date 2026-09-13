// Cycle C of the JavaScript port: quantization (Wyrowski 1990 eqs. 14-15, 22-23), Adam (Kingma & Ba
// 2015 Algorithm 1), Gerchberg-Saxton with Wyrowski's two stages, the gradient solver with stepwise
// quantization, the pipeline (prepare / design / sweep) and the .doev writer. The pipeline is
// cross-checked against the C++ binary: doe_design writes phases (.npy), metrics (report.json) and
// the volume (.doev); the port must reproduce them from the same prepared target.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { Grid, AngularSpectrum } = require('../core/propagate.js');
const { energyAndGrad } = require('../core/energy.js');
const { initRandom } = require('../core/init.js');
const { metrics } = require('../core/metrics.js');
const { levelValue, projectToLevels, projectStepwise, wyrowskiEpsilon, linearEpsilon } = require('../core/quantize.js');
const { Adam, optimize, gerchbergSaxton } = require('../core/solvers.js');
const { prepare, design, sweepVolume, attachSource, resample } = require('../core/pipeline.js');
const { parseDoev, writeDoev } = require('../doev.js');

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} |${a} - ${b}| > ${tol}`);
const wrap = (x) => { let y = x - 2 * Math.PI * Math.floor((x + Math.PI) / (2 * Math.PI)); if (y <= -Math.PI) y += 2 * Math.PI; return y; };
function onLevels(phi, q, tol = 1e-9) {
  for (const x of phi) { let best = 10; for (let z = 0; z < q; ++z) best = Math.min(best, Math.abs(wrap(x - (-Math.PI + z * 2 * Math.PI / q)))); if (best > tol) return false; }
  return true;
}
function problem() {   // cross target, as in test_quantized_solvers.cpp
  const grid = Grid.padded(48, 8e-6, 532e-9), n = grid.n, lo = (n - 48) / 2, N = n * n;
  const illum = new Float64Array(N), b = new Float64Array(N), mask = new Float64Array(N);
  for (let i = lo; i < lo + 48; ++i)
    for (let j = lo; j < lo + 48; ++j) {
      illum[i * n + j] = 1; mask[i * n + j] = 1;
      const bar = Math.abs(grid.x(i)) < 4 * grid.pitch || Math.abs(grid.x(j)) < 4 * grid.pitch;
      b[i * n + j] = (bar && Math.abs(grid.x(i)) < 16 * grid.pitch && Math.abs(grid.x(j)) < 16 * grid.pitch) ? 1 : 0;
    }
  return { grid, n, N, illum, b, mask, prop: new AngularSpectrum(grid, 0.02, true) };
}
function readNpy(file) {   // NumPy .npy v1.0, little-endian float64, C order
  const buf = fs.readFileSync(file);
  assert.equal(buf.toString('latin1', 1, 6), 'NUMPY');
  const hlen = buf.readUInt16LE(8), header = buf.toString('latin1', 10, 10 + hlen);
  assert.ok(header.includes("'<f8'") && header.includes('False'));
  const shape = header.match(/\(\s*(\d+)\s*,\s*(\d+)\s*\)/).slice(1).map(Number);
  const data = new Float64Array(buf.buffer.slice(buf.byteOffset + 10 + hlen, buf.byteOffset + 10 + hlen + 8 * shape[0] * shape[1]));
  return { rows: shape[0], cols: shape[1], data };
}

test('quantize: level values, nearest projection, Wyrowski epsilon table, stepwise capture', () => {
  close(levelValue(4, 0), -Math.PI, 1e-15); close(levelValue(4, 1), -Math.PI / 2, 1e-15); close(levelValue(4, 3), Math.PI / 2, 1e-15);
  const phi = Float64Array.from([0.1, -0.1, 1.5, 3.1, -3.1, 2.4, Math.PI]);
  const r = phi.slice(); projectToLevels(r, 4);
  // nearest levels of {-pi, -pi/2, 0, pi/2}; the level -pi is reported as +pi (range (-pi, pi]);
  // 2.4 is 0.74 from pi and 0.83 from pi/2
  const PI = +Math.PI.toFixed(9), H = +(Math.PI / 2).toFixed(9);
  assert.deepEqual(Array.from(r).map((x) => +x.toFixed(9)), [0, 0, H, PI, PI, PI, PI]);
  assert.ok(onLevels(r, 4));
  assert.deepEqual([1, 2, 5, 10].map((p) => wyrowskiEpsilon(p, 10)), [0.3, 0.5, 0.75, 1.0]);
  assert.equal(wyrowskiEpsilon(1, 1), 1);
  close(wyrowskiEpsilon(2, 3), 0.775, 1e-12);     // s = 4.5 -> between table[4] = 0.75 and table[5] = 0.8 at f = 0.5
  assert.throws(() => wyrowskiEpsilon(0, 10));
  // stepwise: capture |phi - level| < 0.5 eps Delta (Delta = pi/2 for 4 levels)
  const s = Float64Array.from([0.1, 0.5, -0.1, 1.2]);
  const moved = projectStepwise(s, 4, 0.3);        // capture 0.5 * 0.3 * pi/2 = 0.236
  assert.equal(moved, 2);
  assert.deepEqual(Array.from(s).map((x) => +x.toFixed(9)), [0, 0.5, 0, 1.2]);
});

test('Adam: constant gradient moves each coordinate by lr per step; hand-computed sequence; minimizes a quadratic', () => {
  const a = new Adam(3, { lr: 0.1 });
  const x = Float64Array.from([1, 2, 3]), g = Float64Array.from([5, -0.01, 100]);
  a.step(x, g);
  for (let k = 0; k < 3; ++k) close(x[k], [1, 2, 3][k] - 0.1 * Math.sign(g[k]), 1e-6);   // bias correction exact at t = 1
  // Algorithm 1 written out for a varying gradient
  const b1 = 0.9, b2 = 0.999, eps = 1e-8, lr = 0.05;
  const gs = [1.0, -0.5, 2.0];
  let m = 0, v = 0, theta = 0.3;
  const ad = new Adam(1, { lr });
  const y = Float64Array.from([0.3]);
  gs.forEach((gt, i) => {
    m = b1 * m + (1 - b1) * gt; v = b2 * v + (1 - b2) * gt * gt;
    const mh = m / (1 - b1 ** (i + 1)), vh = v / (1 - b2 ** (i + 1));
    theta -= lr * mh / (Math.sqrt(vh) + eps);
    ad.step(y, Float64Array.from([gt]));
    close(y[0], theta, 1e-15);
  });
  const q = new Adam(2, { lr: 0.1 }), z = Float64Array.from([3, -2]);
  for (let it = 0; it < 500; ++it) q.step(z, Float64Array.from([2 * (z[0] - 1), 2 * (z[1] + 0.5)]));
  close(z[0], 1, 1e-3); close(z[1], -0.5, 1e-3);
});

test('optimize: energy falls, term histories sum, phases stay on the torus, progress can stop early', () => {
  const P = problem();
  const phi0 = initRandom(P.grid, 7);
  const r = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 60, lr: 0.05, weights: { shape: 1, efficiency: 0.3 } });
  assert.equal(r.history.length, 60);
  assert.ok(r.history[59] < 0.7 * r.history[0]);
  for (let t = 0; t < 60; ++t) close(r.shapeHistory[t] + r.efficiencyHistory[t], r.history[t], 1e-12);
  for (const x of r.phi) assert.ok(x <= Math.PI && x > -Math.PI);
  assert.equal(r.field.re.length, P.N);
  let seen = 0;
  const s = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 50 }, (it) => { ++seen; return it < 9; });
  assert.equal(seen, 10); assert.equal(s.history.length, 10);
});

test('optimize + Wyrowski stepwise quantization: continuous before the stage, on 4 levels at the end, T9 loss < 3 dB', () => {
  const P = problem();
  const phi0 = initRandom(P.grid, 4);
  const cont = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 200 });
  const mc = metrics(cont.field, P.b, P.mask);
  const q = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 200, quant: { levels: 4, start: 0.4 } });
  assert.ok(onLevels(q.phi, 4));
  const mq = metrics(q.field, P.b, P.mask);
  assert.ok(mc.psnrDb - mq.psnrDb < 3, `loss ${mc.psnrDb - mq.psnrDb} dB`);
  assert.ok(mq.ncc > 0.7);
});

test('quantize: linear capture ramp p/P (Skeren et al. 2002 eq. 11)', () => {
  assert.equal(linearEpsilon(3, 12), 0.25);
  assert.equal(linearEpsilon(12, 12), 1);
  assert.equal(linearEpsilon(1, 240), 1 / 240);
  assert.throws(() => linearEpsilon(0, 5));
  assert.throws(() => linearEpsilon(6, 5));
});

test('optimize: the linear ramp is the default, the table ramp is a different schedule, both end on 4 levels', () => {
  const P = problem();
  const phi0 = initRandom(P.grid, 3);
  const lin = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 60, quant: { levels: 4, start: 0.4, ramp: 'linear' } });
  const def = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 60, quant: { levels: 4, start: 0.4 } });
  const tab = optimize(phi0, P.illum, P.prop, P.b, P.mask, { iters: 60, quant: { levels: 4, start: 0.4, ramp: 'table' } });
  for (const r of [lin, def, tab]) assert.ok(onLevels(r.phi, 4));
  assert.deepEqual(Array.from(def.phi), Array.from(lin.phi));
  let diff = 0; for (let k = 0; k < lin.phi.length; ++k) diff = Math.max(diff, Math.abs(wrap(lin.phi[k] - tab.phi[k])));
  assert.ok(diff > 1e-6);
});

test('prepare: the padded window follows the picture-clear rule (480 px at 256 px / 50 mm, 720 at 512), or the next power of two with pow2Grid', () => {
  const t = { rows: 64, cols: 64, data: new Float64Array(64 * 64).fill(1) };
  assert.equal(prepare(t, { active: 256, pitch: 8e-6, wavelength: 532e-9, distance: 0.05 }).grid.n, 480);
  assert.equal(prepare(t, { active: 256, pitch: 8e-6, wavelength: 532e-9, distance: 0.05, pow2Grid: true }).grid.n, 512);
  assert.equal(prepare(t, { active: 512, pitch: 8e-6, wavelength: 532e-9, distance: 0.05 }).grid.n, 720);
  assert.equal(prepare(t, { active: 512, pitch: 8e-6, wavelength: 532e-9, distance: 0.05, pow2Grid: true }).grid.n, 1024);
});

test('gerchbergSaxton: self-consistent target is a fixed point; residual never increases within a stage; quantization cycles J = Q(P-1)+1 land on levels', () => {
  const P = problem();
  P.prop = new AngularSpectrum(P.grid, 0.02, false);   // unitary propagator: the error-reduction theorem applies exactly (as in test_solvers.cpp)
  const phi0 = initRandom(P.grid, 11);
  const r = gerchbergSaxton(phi0, P.illum, P.prop, P.b, P.mask, { iters: 40, phaseOnlyIters: 20 });
  assert.equal(r.history.length, 40); assert.equal(r.residualHistory.length, 40);
  for (let t = 1; t < 20; ++t) assert.ok(r.residualHistory[t] <= r.residualHistory[t - 1] * (1 + 1e-12), `stage 1 at ${t}`);
  for (let t = 21; t < 40; ++t) assert.ok(r.residualHistory[t] <= r.residualHistory[t - 1] * (1 + 1e-12), `stage 2 at ${t}`);
  for (const x of r.phi) assert.ok(x <= Math.PI && x > -Math.PI);
  // fixed point: propagate a phase, take |v| on the window as the target -> GS keeps the phase
  const e = energyAndGrad(phi0, P.illum, P.prop, P.b, P.mask);
  const bSelf = new Float64Array(P.N); for (let k = 0; k < P.N; ++k) bSelf[k] = P.mask[k] * e.amplitude[k];
  const f = gerchbergSaxton(phi0, P.illum, P.prop, bSelf, P.mask, { iters: 5, phaseOnlyIters: 0 });
  for (let t = 0; t < 5; ++t) assert.ok(f.residualHistory[t] < 1e-18);
  const qg = gerchbergSaxton(phi0, P.illum, P.prop, P.b, P.mask, { iters: 10, phaseOnlyIters: 5, quant: { levels: 4, steps: 10, cyclesPerStep: 5 } });
  assert.equal(qg.history.length, 10 + 5 * 9 + 1);
  assert.ok(onLevels(qg.phi, 4));
});

test('prepare: aspect ratio kept, window = resampled rectangle, illumination shapes; resample keeps constants and endpoints', () => {
  const t = { rows: 100, cols: 60, data: new Float64Array(6000) };
  for (let i = 40; i < 60; ++i) for (let j = 0; j < 60; ++j) t.data[i * 60 + j] = 1;
  const inp = prepare(t, { active: 64, pitch: 8e-6, wavelength: 532e-9, illum: 'square', letterbox: false });
  const n = inp.grid.n, c = n >> 1, o = (n - 64) >> 1;   // picture-clear rule: 280 px at 64 px / 50 mm
  assert.equal(n, 280);
  let maskSum = 0, illumE = 0, mSum = 0; for (let k = 0; k < n * n; ++k) { maskSum += inp.mask[k]; illumE += inp.illum[k] ** 2; mSum += inp.metricsMask[k]; }
  assert.equal(maskSum, 64 * 38); assert.equal(illumE, 64 * 64); assert.equal(mSum, 64 * 38);
  // letterbox (the default): the energy window is the square, the strips are dark targets, metrics on the rectangle
  const lb = prepare(t, { active: 64, pitch: 8e-6, wavelength: 532e-9 });
  let lbMask = 0, lbMetrics = 0, stripB = 0;
  const jb = (n - 38) >> 1;
  for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) {
    lbMask += lb.mask[i * n + j]; lbMetrics += lb.metricsMask[i * n + j];
    const sq = i >= o && i < o + 64 && j >= o && j < o + 64, rect = i >= o && i < o + 64 && j >= jb && j < jb + 38;
    if (sq && !rect) stripB += lb.b[i * n + j];
  }
  assert.equal(lbMask, 64 * 64); assert.equal(lbMetrics, 64 * 38); assert.equal(stripB, 0);
  assert.equal(lb.config.letterbox, true);
  const i0 = o, j0 = (n - 38) >> 1;
  assert.equal(inp.mask[i0 * n + j0], 1); assert.equal(inp.mask[i0 * n + j0 - 1], 0); assert.equal(inp.illum[i0 * n + j0 - 1], 1);
  assert.equal(inp.b[c * n + c], 1); assert.equal(inp.iTarget[c * n + c], 1);
  const disk = prepare(t, { active: 64, pitch: 8e-6, wavelength: 532e-9, illum: 'disk' });
  let de = 0; for (const x of disk.illum) de += x * x;
  close(de, Math.PI * 32 * 32, 0.03 * Math.PI * 32 * 32);
  const g = prepare(t, { active: 64, pitch: 8e-6, wavelength: 532e-9, illum: 'gaussian' });
  assert.ok(g.illum[c * n + c] > 0.99 && g.illum[(c + 31) * n + c] < 0.5 && g.illum[(c + 33) * n + c] === 0);
  const r = resample({ rows: 3, cols: 3, data: Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]) }, 5, 5);
  close(r[0], 0, 0); close(r[24], 8, 1e-12); close(r[12], 4, 1e-12); close(r[1], 0.5, 1e-12);
});

test('pipeline reproduces doe_design: GS, Adam and 4-level phases, metrics, planes, cuts and vortex charges', (t) => {
  const repo = path.resolve(__dirname, '..', '..');
  const exe = process.env.DOE_DESIGN || path.join(repo, 'build', 'apps', 'doe_design');
  if (!fs.existsSync(exe)) { t.skip('doe_design is not built'); return; }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'doe-js-'));
  execFileSync(exe, ['--target', path.join(repo, 'images', 'targets', 'logo.png'), '--active', '32', '--iters', '8', '--levels', '4',
    '--distance', '0.02', '--nz', '4', '--view-size', '24', '--threads', '1', '--out', dir], { stdio: 'ignore' });
  const buf = fs.readFileSync(path.join(dir, 'volume.doev'));
  const ref = parseDoev(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const report = JSON.parse(fs.readFileSync(path.join(dir, 'report.json'), 'utf8'));
  // the prepared target (already resampled and embedded by the C++), cropped back to the aperture
  const n = ref.source.n, a = 32, o = (n - a) / 2;
  const target = { rows: a, cols: a, data: new Float64Array(a * a) };
  for (let i = 0; i < a; ++i) for (let j = 0; j < a; ++j) target.data[i * a + j] = ref.source.target[(o + i) * n + o + j];
  const cfg = { active: 32, pitch: 8e-6, wavelength: 532e-9, distance: 0.02, illum: 'square', init: 'tie', iters: 8, lr: 0.05, mu: 0.3, levels: 4, quantStart: 0.4, runGs: true, gs: { iters: 40, phaseOnlyIters: 20 }, bandLimit: true };
  const res = design(target, cfg);
  assert.deepEqual(res.runs.map((r) => r.name), ['gs', 'adam', 'adam_q4']);
  const crop = (phi) => { const out = new Float64Array(a * a); for (let i = 0; i < a; ++i) for (let j = 0; j < a; ++j) out[i * a + j] = phi[(o + i) * n + o + j]; return out; };
  for (const [name, tol] of [['gs', 1e-6], ['adam', 1e-8]]) {
    const cpp = readNpy(path.join(dir, `phase_${name}.npy`)), js = crop(res.runs.find((r) => r.name === name).phi);
    let worst = 0; for (let k = 0; k < a * a; ++k) worst = Math.max(worst, Math.abs(wrap(js[k] - cpp.data[k])));
    assert.ok(worst < tol, `${name}: max phase difference ${worst}`);
  }
  const q = readNpy(path.join(dir, 'phase_adam_q4.npy')), jq = crop(res.runs[2].phi);
  let same = 0; for (let k = 0; k < a * a; ++k) if (Math.abs(wrap(jq[k] - q.data[k])) < 1e-9) ++same;
  assert.ok(same >= 0.99 * a * a, `adam_q4: ${same} of ${a * a} pixels on the same level`);
  for (const run of res.runs) {
    const m = report.runs[run.name].metrics;
    close(run.metrics.efficiency, m.efficiency, 1e-6, run.name + ' efficiency');
    close(run.metrics.ncc, m.ncc, 1e-6, run.name + ' ncc');
    close(run.metrics.amplitudeRmse, m.amplitude_rmse, 1e-6, run.name + ' rmse');
    close(run.metrics.psnrDb, m.psnr_db, 1e-4, run.name + ' psnr');
    assert.equal(run.metrics.vortexCount, m.vortex_count, run.name + ' vortex count');
    assert.equal(run.metrics.vortexCountBright, m.vortex_count_bright, run.name + ' bright vortex count');
    close(run.metrics.vortexDensityBright * 1e-6 / (8e-6 * 8e-6), m.vortex_density_bright_per_mm2, 1e-6 * Math.max(1, m.vortex_density_bright_per_mm2), run.name + ' bright density per mm2');
  }
  assert.equal(res.initialMetrics.vortexCountBright, report.initial.vortex_count_bright);
  close(res.initialMetrics.ncc, report.initial.ncc, 1e-6);
  assert.equal(res.sampling.warnings.length, report.sampling.warnings.length);
  // sweep of the final design: planes, cuts, charges as in the file
  const vol = sweepVolume(res, { nz: 4, view: 24 });
  attachSource(vol, res);
  assert.equal(vol.nz, 4); assert.equal(vol.view, 24);
  for (let p = 0; p < 4; ++p) {
    close(vol.z[p], ref.z[p], 1e-12);
    let amax = 0; for (const x of ref.amplitude[p]) amax = Math.max(amax, x);
    for (let k = 0; k < 24 * 24; ++k) {
      close(vol.amplitude[p][k], ref.amplitude[p][k], 1e-5 * amax, `plane ${p} amplitude ${k}`);
      if (ref.amplitude[p][k] > 1e-3 * amax) close(wrap(vol.phase[p][k] - ref.phase[p][k]), 0, 1e-4, `plane ${p} phase ${k}`);
    }
    assert.equal(vol.vortexCount[p], ref.vortexCount[p], `plane ${p} vortex count`);
    close(vol.vortexDensity[p], ref.vortexDensity[p], 1e-6 * ref.vortexDensity[p] + 1e-9);
    let diff = 0; for (let k = 0; k < 23 * 23; ++k) if (vol.charges[p][k] !== ref.charges[p][k]) ++diff;
    assert.ok(diff <= 2, `plane ${p}: ${diff} charges differ`);
    for (let i = 0; i < 24; ++i) { close(vol.xz[p * 24 + i], ref.xz[p * 24 + i], 1e-9 * amax * amax + 1e-12); close(vol.yz[p * 24 + i], ref.yz[p * 24 + i], 1e-9 * amax * amax + 1e-12); }
  }
  for (let k = 0; k < 24 * 24; ++k) { close(vol.target[k], ref.target[k], 1e-9); close(wrap(vol.doePhase[k] - ref.doePhase[k]), 0, 1e-6); }
  assert.equal(vol.source.n, ref.source.n); close(vol.source.distance, 0.02, 0);
  for (let k = 0; k < n * n; k += 7) { close(vol.source.illum[k], ref.source.illum[k], 1e-6); close(vol.source.target[k], ref.source.target[k], 1e-6); }
  fs.rmSync(dir, { recursive: true, force: true });
});

test('design: metrics are evaluated on the image rectangle in letterbox mode (efficiency = light in the rectangle)', () => {
  const t = { rows: 24, cols: 12, data: new Float64Array(288) }; for (let k = 60; k < 200; ++k) t.data[k] = 1;
  const res = design(t, { active: 24, pitch: 8e-6, wavelength: 532e-9, distance: 0.004, iters: 3, runGs: false });
  const n = res.inputs.grid.n, f = res.runs[0].field;
  let total = 0, rect = 0;
  for (let k = 0; k < n * n; ++k) { const I = f.re[k] ** 2 + f.im[k] ** 2; total += I; rect += res.inputs.metricsMask[k] * I; }
  close(res.runs[0].metrics.efficiency, rect / total, 1e-12);
  let mask = 0; for (const v of res.inputs.mask) mask += v;
  assert.equal(mask, 24 * 24);
});

test('writeDoev / parseDoev round trip of a swept volume', () => {
  const t = { rows: 16, cols: 16, data: new Float64Array(256) }; for (let k = 100; k < 140; ++k) t.data[k] = 1;
  const res = design(t, { active: 16, pitch: 8e-6, wavelength: 532e-9, distance: 0.005, iters: 3, runGs: false });
  const vol = sweepVolume(res, { nz: 3, view: 12 }); attachSource(vol, res);
  const back = parseDoev(writeDoev(vol));
  assert.equal(back.version, 2); assert.equal(back.view, 12); assert.equal(back.nz, 3);
  assert.deepEqual(Array.from(back.z), Array.from(vol.z));
  assert.deepEqual(Array.from(back.amplitude[1]), Array.from(vol.amplitude[1]));
  assert.deepEqual(Array.from(back.charges[2]), Array.from(vol.charges[2]));
  assert.deepEqual(Array.from(back.vortexCount), Array.from(vol.vortexCount));
  assert.deepEqual(Array.from(back.xz), Array.from(vol.xz));
  assert.deepEqual(Array.from(back.target), Array.from(vol.target));
  assert.equal(back.source.n, 40); assert.deepEqual(Array.from(back.source.phase), Array.from(vol.source.phase));   // 16 px at 5 mm: 37 -> 40 px
});

test('sweepFromSource: re-propagation from the v2 source block equals the sweep of the design', () => {
  const { sweepFromSource } = require('../core/pipeline.js');
  const t = { rows: 16, cols: 16, data: new Float64Array(256) }; for (let k = 100; k < 140; ++k) t.data[k] = 1;
  const res = design(t, { active: 16, pitch: 8e-6, wavelength: 532e-9, distance: 0.005, iters: 3, runGs: false });
  const vol = attachSource(sweepVolume(res, { nz: 3, view: 12 }), res);
  const again = sweepFromSource(parseDoev(writeDoev(vol)), { nz: 3, view: 12 });
  for (let p = 0; p < 3; ++p) for (let k = 0; k < 144; ++k) close(again.amplitude[p][k], vol.amplitude[p][k], 1e-6);
  assert.deepEqual(Array.from(again.vortexCount), Array.from(vol.vortexCount));
  assert.deepEqual(Array.from(again.target), Array.from(vol.target));
  assert.equal(again.source.n, 40);
  // other plane count, crop and distance: the last plane sits at the new distance
  const far = sweepFromSource(vol, { nz: 5, view: 8, distance: 0.01 });
  assert.equal(far.nz, 5); assert.equal(far.view, 8); close(far.z[4], 0.01, 1e-15);
  assert.throws(() => sweepFromSource({ source: null }, { nz: 2, view: 8 }), /source/);
});

test('recommendConfig: continuous-tone images get the backprop recipe of the manual, binary targets the defaults', () => {
  const { recommendConfig } = require('../core/pipeline.js');
  // a binary target: only 0 and 1 (mid-tone fraction 0)
  const bin = { rows: 20, cols: 20, data: new Float64Array(400) }; for (let k = 100; k < 260; ++k) bin.data[k] = 1;
  const rb = recommendConfig(bin);
  assert.equal(rb.init, 'tie'); assert.equal(rb.mu, 0.3); assert.equal(rb.continuousTone, false);
  assert.ok(rb.midToneFraction < 0.05);
  // a shaded image: a smooth ramp (every value between 0 and 1)
  const ramp = { rows: 20, cols: 20, data: Float64Array.from({ length: 400 }, (_, k) => (k % 20) / 19) };
  const rr = recommendConfig(ramp);
  assert.equal(rr.init, 'backprop'); assert.equal(rr.mu, 0.02); assert.equal(rr.continuousTone, true);
  assert.ok(rr.midToneFraction > 0.5);
  // antialiased edges alone (a few percent of mid-tones) do not count as continuous tone
  const aa = { rows: 20, cols: 20, data: new Float64Array(400) }; for (let k = 100; k < 260; ++k) aa.data[k] = 1;
  for (let k = 90; k < 100; ++k) aa.data[k] = 0.5;   // 2.5 % mid-tones
  assert.equal(recommendConfig(aa).continuousTone, false);
  // the measured fraction of the shell image is 0.19, of the synthetic targets 0: the threshold is 5 %
  assert.equal(recommendConfig({ rows: 10, cols: 10, data: Float64Array.from({ length: 100 }, (_, k) => (k < 6 ? 0.5 : (k % 2))) }).continuousTone, true);
  assert.equal(recommendConfig({ rows: 10, cols: 10, data: Float64Array.from({ length: 100 }, (_, k) => (k < 4 ? 0.5 : (k % 2))) }).continuousTone, false);
  // normalized to the image maximum
  const dim = { rows: 10, cols: 10, data: Float64Array.from({ length: 100 }, (_, k) => 0.2 * (k % 2)) };
  assert.equal(recommendConfig(dim).continuousTone, false);
});
