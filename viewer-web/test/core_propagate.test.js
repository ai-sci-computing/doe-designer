// Cycle A of the JavaScript port: FFT (ortho scaling, mixed radix for 7-smooth sizes), the padded
// grid (next_smooth, fftfreq order), the band-limited angular-spectrum transfer function
// (Goodman 2017 eq. 3-78; Matsushima & Shimobaba 2009 eq. 13) and the propagator's adjoint.
// Mirrors tests/test_fft.cpp, test_grid.cpp, test_next_smooth.cpp and test_propagate.cpp.
const test = require('node:test');
const assert = require('node:assert/strict');
const { FFT1D, fft2, ifft2 } = require('../core/fft.js');
const { nextSmooth, nextPow2, Grid, transferFunction, AngularSpectrum, samplingReport } = require('../core/propagate.js');

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} |${a} - ${b}| > ${tol}`);

function naiveDft(re, im, inverse) {
  const n = re.length, outRe = new Float64Array(n), outIm = new Float64Array(n), s = inverse ? 1 : -1;
  for (let k = 0; k < n; ++k)
    for (let t = 0; t < n; ++t) {
      const a = s * 2 * Math.PI * k * t / n;
      outRe[k] += re[t] * Math.cos(a) - im[t] * Math.sin(a);
      outIm[k] += re[t] * Math.sin(a) + im[t] * Math.cos(a);
    }
  return { re: outRe, im: outIm };
}

test('FFT1D matches a naive DFT for power-of-two and 7-smooth lengths, forward and inverse', () => {
  for (const n of [1, 2, 4, 8, 32, 6, 60, 200, 210, 128]) {
    const re = Float64Array.from({ length: n }, (_, k) => Math.sin(0.7 * k) + 0.3 * k), im = Float64Array.from({ length: n }, (_, k) => Math.cos(1.3 * k));
    for (const inverse of [false, true]) {
      const f = new FFT1D(n), a = re.slice(), b = im.slice();
      f.transform(a, b, 0, inverse);   // unscaled, like FFTW
      const ref = naiveDft(re, im, inverse);
      for (let k = 0; k < n; ++k) { close(a[k], ref.re[k], 1e-9 * n, `n=${n} re[${k}]`); close(b[k], ref.im[k], 1e-9 * n, `n=${n} im[${k}]`); }
    }
  }
});

test('fft2: ortho scaling (Parseval), delta -> constant, round trip', () => {
  const n = 12, re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let k = 0; k < n * n; ++k) { re[k] = Math.sin(k * 0.37); im[k] = Math.cos(k * 0.11) * 0.5; }
  const e0 = re.reduce((s, x, k) => s + x * x + im[k] * im[k], 0);
  const R = re.slice(), I = im.slice();
  fft2(R, I, n);
  const e1 = R.reduce((s, x, k) => s + x * x + I[k] * I[k], 0);
  close(e1, e0, 1e-9, 'Parseval');
  ifft2(R, I, n);
  for (let k = 0; k < n * n; ++k) { close(R[k], re[k], 1e-12); close(I[k], im[k], 1e-12); }
  // delta at the origin -> constant 1/n everywhere (ortho)
  const dRe = new Float64Array(n * n), dIm = new Float64Array(n * n); dRe[0] = 1;
  fft2(dRe, dIm, n);
  for (let k = 0; k < n * n; ++k) { close(dRe[k], 1 / n, 1e-12); close(dIm[k], 0, 1e-12); }
});

test('nextSmooth: smallest 7-smooth number >= n (brute force to 2000)', () => {
  const smooth = (x) => { for (const p of [2, 3, 5, 7]) while (x % p === 0) x /= p; return x === 1; };
  for (let n = 1; n <= 2000; ++n) {
    let m = n; while (!smooth(m)) ++m;
    assert.equal(nextSmooth(n), m, `n=${n}`);
  }
  assert.equal(nextSmooth(1000), 1000);
  assert.equal(nextSmooth(1025), 1029);   // 3 * 7^3
});

test('Grid: padded size, fftfreq order, extent, max diffraction angle', () => {
  const g = Grid.padded(100, 8e-6, 532e-9);
  assert.equal(g.n, 200);
  close(g.f(0), 0, 0);
  close(g.f(1), 1 / (200 * 8e-6), 1e-6);
  close(g.f(100), -100 / (200 * 8e-6), 1e-6);     // Nyquist bin is negative for even n (numpy)
  close(g.f(199), -1 / (200 * 8e-6), 1e-6);
  close(g.extent(), 200 * 8e-6, 1e-15);
  close(g.maxDiffractionAngle(), Math.asin(532e-9 / (2 * 8e-6)), 1e-15);
  close(g.k(), 2 * Math.PI / 532e-9, 1e-3);
  assert.equal(Grid.padded(32, 8e-6, 532e-9).n, 64);
});

test('transferFunction: Goodman eq. 3-78 bin by bin, evanescent waves dropped, Matsushima band limit', () => {
  const g = new Grid(64, 0.3e-6, 532e-9), d = 0.001;    // pitch below lambda / sqrt(2): evanescent bins exist
  const tf = transferFunction(g, d, false);
  let evan = 0;
  for (let i = 0; i < 64; ++i)
    for (let j = 0; j < 64; ++j) {
      const fx = g.f(i), fy = g.f(j), arg = 1 - (532e-9 * fx) ** 2 - (532e-9 * fy) ** 2, k = i * 64 + j;
      if (arg <= 0) { ++evan; close(tf.re[k], 0, 0); close(tf.im[k], 0, 0); continue; }
      const ph = 2 * Math.PI / 532e-9 * Math.sqrt(arg) * d;
      close(tf.re[k], Math.cos(ph), 1e-9); close(tf.im[k], Math.sin(ph), 1e-9);
    }
  assert.ok(evan > 0);
  assert.equal(tf.evanescentCount, evan);
  // band limit: f_lim = 1 / (lambda sqrt((2 d / S)^2 + 1)), rect window; zero distance -> no limit
  const g2 = new Grid(128, 8e-6, 532e-9), d2 = 0.05;
  const tb = transferFunction(g2, d2, true);
  const flim = 1 / (532e-9 * Math.sqrt((2 * d2 / g2.extent()) ** 2 + 1));
  close(tb.fLimit, flim, 1e-6);
  for (let i = 0; i < 128; ++i)
    for (let j = 0; j < 128; ++j) {
      const k = i * 128 + j, inside = Math.abs(g2.f(i)) <= flim && Math.abs(g2.f(j)) <= flim;
      const mag = Math.hypot(tb.re[k], tb.im[k]);
      if (inside) close(mag, 1, 1e-12); else close(mag, 0, 0);
    }
  assert.ok(tb.passbandFraction < 1 && tb.passbandFraction > 0.05);   // 0.093 at these settings
  assert.equal(transferFunction(g2, 0, true).fLimit, Infinity);
});

test('AngularSpectrum: forward is an isometry in the passband, adjoint satisfies <Au, v> = <u, A^H v>, zero distance is the identity', () => {
  const g = new Grid(32, 8e-6, 532e-9), A = new AngularSpectrum(g, 0.01, true);
  const n = 32, N = n * n;
  const rnd = (seed) => { let s = seed; return () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 - 0.5; }; };
  const r1 = rnd(1), r2 = rnd(2);
  const u = { re: Float64Array.from({ length: N }, r1), im: Float64Array.from({ length: N }, r1) };
  const v = { re: Float64Array.from({ length: N }, r2), im: Float64Array.from({ length: N }, r2) };
  const Au = A.forward(u), Ahv = A.adjoint(v);
  let lhsRe = 0, lhsIm = 0, rhsRe = 0, rhsIm = 0;   // <a, b> = sum conj(a) b
  for (let k = 0; k < N; ++k) {
    lhsRe += Au.re[k] * v.re[k] + Au.im[k] * v.im[k]; lhsIm += Au.re[k] * v.im[k] - Au.im[k] * v.re[k];
    rhsRe += u.re[k] * Ahv.re[k] + u.im[k] * Ahv.im[k]; rhsIm += u.re[k] * Ahv.im[k] - u.im[k] * Ahv.re[k];
  }
  close(lhsRe, rhsRe, 1e-10); close(lhsIm, rhsIm, 1e-10);
  // a smooth (band-limited) input keeps its energy: a wide Gaussian
  const c = n / 2, gRe = new Float64Array(N), gIm = new Float64Array(N);
  for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) gRe[i * n + j] = Math.exp(-((i - c) ** 2 + (j - c) ** 2) / 50);
  const out = A.forward({ re: gRe, im: gIm });
  const e0 = gRe.reduce((s, x) => s + x * x, 0), e1 = out.re.reduce((s, x, k) => s + x * x + out.im[k] * out.im[k], 0);
  close(e1 / e0, 1, 1e-6, 'isometry');
  const I = new AngularSpectrum(g, 0, true).forward(u);
  for (let k = 0; k < N; ++k) { close(I.re[k], u.re[k], 1e-12); close(I.im[k], u.im[k], 1e-12); }
});

test('AngularSpectrum.sweep visits planes with the plane-specific transfer function; plane at the design distance equals forward()', () => {
  const g = new Grid(16, 8e-6, 532e-9), A = new AngularSpectrum(g, 0.004, true), N = 256;
  const u = { re: Float64Array.from({ length: N }, (_, k) => (k % 17) / 17), im: new Float64Array(N) };
  const seen = [];
  A.sweep(u, [0, 0.002, 0.004], (p, plane) => seen.push({ p, re: plane.re.slice(), im: plane.im.slice() }));
  assert.deepEqual(seen.map((s) => s.p), [0, 1, 2]);
  for (let k = 0; k < N; ++k) close(seen[0].re[k], u.re[k], 1e-12);
  const f = A.forward(u);
  for (let k = 0; k < N; ++k) { close(seen[2].re[k], f.re[k], 1e-12); close(seen[2].im[k], f.im[k], 1e-12); }
});

test('samplingReport: wrapped light must miss the picture, D + d tan(theta) <= S (mirrors test_propagate.cpp)', () => {
  const g = Grid.padded(512, 8e-6, 532e-9), D = 512 * 8e-6;   // 1024 px, 8.19 mm window
  assert.ok(samplingReport(g, 0.05, D).wrapMissesPicture);
  assert.ok(samplingReport(g, 0.12, D).wrapMissesPicture);
  const far = samplingReport(g, 0.13, D);
  assert.ok(!far.wrapMissesPicture);
  assert.ok(far.warnings.some((w) => w.includes('picture')));
});

test('Grid.paddedFor: plate + d tan(theta), 7-smooth like the C++ or the next power of two for the browser', () => {
  const p = 8e-6, lam = 532e-9;
  assert.equal(Grid.paddedFor(512, p, lam, 0.05).n, 720);
  assert.equal(Grid.paddedFor(256, p, lam, 0.05).n, 480);
  assert.equal(Grid.paddedFor(512, p, lam, 0.10).n, 945);
  assert.equal(Grid.paddedFor(32, p, lam, 0.02).n, 120);
  assert.equal(Grid.paddedFor(512, p, lam, 0.05, 2.0).n, 1024);   // a minimum factor still applies
  assert.equal(Grid.paddedFor(512, p, lam, 0.05, 1.0, nextPow2).n, 1024);
  assert.equal(Grid.paddedFor(256, p, lam, 0.05, 1.0, nextPow2).n, 512);
  assert.equal(Grid.paddedFor(32, p, lam, 0.02, 1.0, nextPow2).n, 128);
  assert.equal(nextPow2(720), 1024); assert.equal(nextPow2(1024), 1024); assert.equal(nextPow2(1), 1);
  for (const [N, d] of [[512, 0.05], [256, 0.05], [512, 0.1], [128, 0.3]])
    for (const round of [nextSmooth, nextPow2]) {
      const g = Grid.paddedFor(N, p, lam, d, 1.0, round);
      assert.ok(g.n >= N);
      assert.ok(samplingReport(g, d, N * p).wrapMissesPicture);
    }
});
