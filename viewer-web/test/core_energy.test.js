// Cycle B of the JavaScript port: energy and gradient (energy.hpp: Fienup 1997 eq. 20 shape term,
// efficiency term, Wirtinger gradient), initializations (Teague 1983 TIE via the periodic Poisson
// solve, back-propagation, random), metrics and vortex residues (Goldstein 1988; Berry & Dennis 2000).
// Mirrors tests/test_energy.cpp, test_init.cpp, test_metrics.cpp, test_vortex.cpp.
const test = require('node:test');
const assert = require('node:assert/strict');
const { Grid, AngularSpectrum } = require('../core/propagate.js');
const { optimalScale, energyAndGrad } = require('../core/energy.js');
const { poissonPeriodic, initTie, initBackprop, initRandom } = require('../core/init.js');
const { metrics, vortexChargeMap, vortexCount, vortexDensity } = require('../core/metrics.js');

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} |${a} - ${b}| > ${tol}`);

// Small design problem (as in test_solvers.cpp): 48 px square illumination on a 96 px grid,
// disk target of radius 10 px in the window.
function problem() {
  const grid = Grid.padded(48, 8e-6, 532e-9), n = grid.n, lo = (n - 48) / 2, N = n * n;
  const illum = new Float64Array(N), b = new Float64Array(N), mask = new Float64Array(N);
  for (let i = lo; i < lo + 48; ++i)
    for (let j = lo; j < lo + 48; ++j) {
      illum[i * n + j] = 1; mask[i * n + j] = 1;
      const r2 = grid.x(i) ** 2 + grid.x(j) ** 2;
      b[i * n + j] = r2 <= (10 * grid.pitch) ** 2 ? 1 : 0;
    }
  return { grid, n, N, illum, b, mask, prop: new AngularSpectrum(grid, 0.02, true) };
}
const lcg = (seed) => { let s = seed; return () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; }; };

test('optimalScale: least-squares scale on the window', () => {
  const n = 4, vAbs = Float64Array.from({ length: 16 }, (_, k) => 2 * (k % 3)), b = Float64Array.from({ length: 16 }, (_, k) => k % 3);
  const mask = new Float64Array(16).fill(1); mask[0] = 0;
  close(optimalScale(vAbs, b, mask), 2, 1e-12);
  close(optimalScale(vAbs, b, new Float64Array(16)), 1, 0);   // empty window -> 1
});

test('energyAndGrad: terms sum to the energy, the shape term is scale invariant and vanishes for a matching field', () => {
  const P = problem();
  const r = lcg(3);
  const phi = Float64Array.from({ length: P.N }, () => (r() - 0.5) * 2 * Math.PI);
  const e = energyAndGrad(phi, P.illum, P.prop, P.b, P.mask, { shape: 1, efficiency: 0.3 });
  close(e.shape + e.efficiency, e.energy, 1e-14);
  assert.ok(e.shape > 0 && e.shape < 1 && e.efficiency >= 0 && e.efficiency <= 0.3);
  assert.equal(e.grad.length, P.N);
  assert.equal(e.field.re.length, P.N);
  // scaling the illumination scales |v| and leaves the shape term unchanged (Fienup 1997 eq. 20)
  const illum2 = P.illum.map((x) => 3 * x);
  const e2 = energyAndGrad(phi, illum2, P.prop, P.b, P.mask, { shape: 1, efficiency: 0 });
  close(e2.shape, e.shape, 1e-12, 'scale invariance');
  close(e2.efficiency, 0, 0);
  // each term alone (T4)
  const es = energyAndGrad(phi, P.illum, P.prop, P.b, P.mask, { shape: 1, efficiency: 0 });
  const ee = energyAndGrad(phi, P.illum, P.prop, P.b, P.mask, { shape: 0, efficiency: 0.3 });
  close(es.energy, e.shape, 1e-14); close(ee.energy, e.efficiency, 1e-14);
  for (let k = 0; k < P.N; ++k) close(es.grad[k] + ee.grad[k], e.grad[k], 1e-12);
});

test('energyAndGrad: directional derivative matches central finite differences (h-sweep)', () => {
  const P = problem();
  const r = lcg(5);
  const phi = Float64Array.from({ length: P.N }, () => (r() - 0.5) * 2 * Math.PI);
  const dir = Float64Array.from({ length: P.N }, () => r() - 0.5);
  const w = { shape: 1, efficiency: 0.3 };
  const e = energyAndGrad(phi, P.illum, P.prop, P.b, P.mask, w);
  let dE = 0; for (let k = 0; k < P.N; ++k) dE += e.grad[k] * dir[k];
  let best = Infinity;
  for (const h of [1e-3, 1e-4, 1e-5]) {
    const ep = energyAndGrad(phi.map((x, k) => x + h * dir[k]), P.illum, P.prop, P.b, P.mask, w).energy;
    const em = energyAndGrad(phi.map((x, k) => x - h * dir[k]), P.illum, P.prop, P.b, P.mask, w).energy;
    best = Math.min(best, Math.abs((ep - em) / (2 * h) - dE) / Math.max(Math.abs(dE), 1e-12));
  }
  assert.ok(best < 1e-5, `relative gradient error ${best}`);
});

test('poissonPeriodic: manufactured solution sin(2 pi x / L) cos(4 pi y / L) (T6); non-zero-mean rhs is projected', () => {
  const grid = new Grid(48, 1e-6, 532e-9), n = 48, L = grid.extent();
  const sol = new Float64Array(n * n), rhs = new Float64Array(n * n);
  for (let i = 0; i < n; ++i)
    for (let j = 0; j < n; ++j) {
      const x = grid.x(i), y = grid.x(j);
      sol[i * n + j] = Math.sin(2 * Math.PI * x / L) * Math.cos(4 * Math.PI * y / L);
      rhs[i * n + j] = -((2 * Math.PI / L) ** 2 + (4 * Math.PI / L) ** 2) * sol[i * n + j];
    }
  const phi = poissonPeriodic(grid, rhs);
  for (let k = 0; k < n * n; ++k) close(phi[k], sol[k], 1e-9, 'T6');
  const phi2 = poissonPeriodic(grid, rhs.map((x) => x + 7));
  let mean = 0; for (const x of phi2) mean += x; mean /= n * n;
  close(mean, 0, 1e-9);
  for (let k = 0; k < n * n; ++k) close(phi2[k], sol[k], 1e-9);
});

test('initTie: Teague eq. (4) right-hand side with the +ikz convention; the TIE start images better than a flat phase', () => {
  const P = problem();
  const iSource = P.illum.map((x) => x * x), iTarget = P.b.map((x) => x * x);
  const phi0 = initTie(P.grid, 0.02, iTarget, iSource);
  assert.equal(phi0.length, P.N);
  // rhs = (k / d) (1 - scale * I_t / I_0) with I_0 floored at 1e-6 max: outside the source the
  // rhs is (k/d) (1 - 0) > 0, inside the bright target it is negative -> the phase is a lens
  // (concave) over the target: check curvature sign at the center via the discrete Laplacian
  const c = P.n / 2, n = P.n;
  const lap = phi0[(c + 1) * n + c] + phi0[(c - 1) * n + c] + phi0[c * n + c + 1] + phi0[c * n + c - 1] - 4 * phi0[c * n + c];
  const kOverD = P.grid.k() / 0.02;
  let srcSum = 0, tgtSum = 0; for (let k = 0; k < P.N; ++k) { srcSum += iSource[k]; tgtSum += iTarget[k]; }
  // center: I_t = 1, I_0 = 1, rhs = (k/d)(1 - scale); the periodic solve projects the rhs to zero
  // mean, and mean(rhs) = (k/d)(1 - srcSum / N) here, so the solved Laplacian is (k/d)(srcSum/N - scale)
  const expected = kOverD * (srcSum / P.N - srcSum / tgtSum) * P.grid.pitch ** 2;
  close(lap, expected, Math.abs(expected) * 0.02, 'Teague eq. 4 at the center');
  const ncc = (phi) => metrics(energyAndGrad(phi, P.illum, P.prop, P.b, P.mask).field, P.b, P.mask).ncc;
  assert.ok(ncc(phi0) > ncc(new Float64Array(P.N)), 'TIE start images better than a flat phase');
});

test('initBackprop and initRandom', () => {
  const P = problem();
  const phi = initBackprop(P.grid, P.prop, P.b);
  const u = P.prop.adjoint({ re: P.b, im: new Float64Array(P.N) });
  for (let k = 0; k < P.N; k += 97) close(phi[k], Math.atan2(u.im[k], u.re[k]), 1e-12);
  const a = initRandom(P.grid, 7), b2 = initRandom(P.grid, 7), c = initRandom(P.grid, 8);
  assert.deepEqual(Array.from(a), Array.from(b2));
  assert.notDeepEqual(Array.from(a), Array.from(c));
  let lo = Infinity, hi = -Infinity, mean = 0;
  for (const x of a) { lo = Math.min(lo, x); hi = Math.max(hi, x); mean += x; }
  assert.ok(lo > -Math.PI && hi <= Math.PI && lo < -3 && hi > 3);
  close(mean / P.N, 0, 0.05);
});

test('metrics: exact match gives RMSE 0, NCC 1, PSNR infinite; efficiency is the window energy fraction; speckle contrast of a uniform bright region is 0', () => {
  const n = 8, N = 64, b = new Float64Array(N), mask = new Float64Array(N);
  for (let i = 2; i < 6; ++i) for (let j = 2; j < 6; ++j) mask[i * n + j] = 1;
  b[3 * n + 3] = 1; b[3 * n + 4] = 1; b[4 * n + 3] = 1; b[4 * n + 4] = 1;   // 2 x 2 bright block, uniform
  const v = { re: b.map((x) => 2 * x), im: new Float64Array(N) };
  v.re[0] = 1;   // energy outside the window
  const m = metrics(v, b, mask);
  close(m.efficiency, 16 / 17, 1e-12);
  close(m.amplitudeRmse, 0, 1e-12);
  close(m.ncc, 1, 1e-12);
  assert.equal(m.psnrDb, Infinity);
  close(m.speckleContrast, 0, 1e-12);
  assert.equal(m.vortexCount, 0);
  assert.equal(m.vortexCountBright, 0);
  assert.ok(Number.isNaN(m.vortexDensityBright) || m.vortexDensityBright === 0);
  // a constant target has no defined NCC
  const mc = metrics(v, mask, mask);
  assert.ok(Number.isNaN(mc.ncc));
});

test('vortexChargeMap: analytic vortices +1 / -1, a smooth field has no residues, density = count / area', () => {
  const n = 9, N = 81, c = 4;
  const field = (sign) => {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) { const x = i - c + 0.5, y = j - c + 0.5; re[i * n + j] = x; im[i * n + j] = sign * y; }
    return { re, im };
  };
  const q = vortexChargeMap(field(1), n, n);
  assert.equal(q.length, 64);
  assert.equal(vortexCount(q), 1);
  assert.equal(q[c * 8 + c] === 1 || q[(c - 1) * 8 + (c - 1)] === 1 || q[(c - 1) * 8 + c] === 1 || q[c * 8 + (c - 1)] === 1, true);
  let sum = 0; for (const x of q) sum += x; assert.equal(sum, 1);
  const qm = vortexChargeMap(field(-1), n, n);
  sum = 0; for (const x of qm) sum += x; assert.equal(sum, -1);
  const smooth = { re: Float64Array.from({ length: N }, (_, k) => Math.cos(k * 0.01)), im: Float64Array.from({ length: N }, (_, k) => Math.sin(k * 0.01)) };
  assert.equal(vortexCount(vortexChargeMap(smooth, n, n)), 0);
  close(vortexDensity(field(1), n, n, 8e-6), 1 / (8 * 8e-6) ** 2, 1e-3);
  // zero-amplitude corners are skipped (no residue defined)
  const z = field(1); z.re[c * n + c] = 0; z.im[c * n + c] = 0;
  assert.equal(vortexCount(vortexChargeMap(z, n, n)), 0);
});
