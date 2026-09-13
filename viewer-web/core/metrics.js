// Figures of merit and vortex residues, ported from src/metrics.cpp and src/vortex.cpp:
// efficiency (Wyrowski & Bryngdahl 1988), amplitude RMSE after least-squares scaling,
// speckle contrast sigma/mean on the bright pixels (Goodman 2007), NCC and PSNR of the
// intensities, and the plaquette residues of the phase (Goldstein, Zebker & Werner 1988;
// Fried & Vaughn 1992; Berry & Dennis 2000 eq. 2.4 for the charge sign).
(function () {
  // Module factory. As a classic script it also registers its own source so that the page can
  // build a Web Worker from it (a worker script cannot be loaded from a file:// URL).
  function factory(root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const { optimalScale } = isNode ? require('./energy.js') : root.DOE_CORE;
  const { wrapToPi } = isNode ? require('./util.js') : root.DOE_CORE;

  // Residues of an rows x cols field: (rows-1) x (cols-1) Int32Array, loop (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1).
  function vortexChargeMap(u, rows, cols) {
    if (rows < 2 || cols < 2) throw new Error('vortexChargeMap: field too small');
    const q = new Int32Array((rows - 1) * (cols - 1));
    const ph = new Float64Array(rows * cols), nz = new Uint8Array(rows * cols);
    for (let k = 0; k < rows * cols; ++k) { ph[k] = Math.atan2(u.im[k], u.re[k]); nz[k] = (u.re[k] !== 0 || u.im[k] !== 0) ? 1 : 0; }
    for (let i = 0; i + 1 < rows; ++i)
      for (let j = 0; j + 1 < cols; ++j) {
        const a = i * cols + j, b = (i + 1) * cols + j, c = (i + 1) * cols + j + 1, d = i * cols + j + 1;
        if (!(nz[a] && nz[b] && nz[c] && nz[d])) continue;
        const s = wrapToPi(ph[b] - ph[a]) + wrapToPi(ph[c] - ph[b]) + wrapToPi(ph[d] - ph[c]) + wrapToPi(ph[a] - ph[d]);
        q[i * (cols - 1) + j] = Math.round(s / (2 * Math.PI));
      }
    return q;
  }
  function vortexCount(q) { let n = 0; for (let k = 0; k < q.length; ++k) if (q[k] !== 0) ++n; return n; }
  function vortexDensity(u, rows, cols, pitch) { return vortexCount(vortexChargeMap(u, rows, cols)) / ((rows - 1) * pitch * (cols - 1) * pitch); }

  function metrics(v, b, mask) {
    const N = v.re.length, n = Math.round(Math.sqrt(N));
    const m = { efficiency: 0, amplitudeRmse: 0, speckleContrast: NaN, ncc: NaN, psnrDb: NaN, vortexCount: 0, vortexCountBright: 0, vortexDensityBright: NaN };
    const vAbs = new Float64Array(N);
    let total = 0, signal = 0, bMax = 0;
    for (let k = 0; k < N; ++k) {
      const I = v.re[k] * v.re[k] + v.im[k] * v.im[k];
      vAbs[k] = Math.sqrt(I); total += I; signal += mask[k] * I; bMax = Math.max(bMax, b[k]);
    }
    m.efficiency = signal / (total + 1e-30);
    const s = optimalScale(vAbs, b, mask);
    let err = 0, peakT = 0, nW = 0, sumA = 0, sumT = 0;
    const a = [], t = [], bright = [];
    for (let k = 0; k < N; ++k) {
      if (!(mask[k] > 0)) continue;
      const f = vAbs[k], tb = s * b[k];
      err += (f - tb) * (f - tb); peakT = Math.max(peakT, tb); nW += 1;
      a.push(f * f); t.push(tb * tb); sumA += f * f; sumT += tb * tb;
      if (b[k] > 0.5 * bMax) bright.push(f * f);
    }
    if (nW === 0) return m;
    m.amplitudeRmse = Math.sqrt(err / nW) / (peakT + 1e-12);
    if (bright.length) {
      let mean = 0; for (const x of bright) mean += x; mean /= bright.length;
      let variance = 0; for (const x of bright) variance += (x - mean) * (x - mean); variance /= bright.length;
      m.speckleContrast = mean > 0 ? Math.sqrt(variance) / mean : NaN;
    }
    const meanA = sumA / nW, meanT = sumT / nW;
    let cov = 0, varA = 0, varT = 0, mse = 0;
    for (let k = 0; k < a.length; ++k) { cov += (a[k] - meanA) * (t[k] - meanT); varA += (a[k] - meanA) ** 2; varT += (t[k] - meanT) ** 2; mse += (a[k] - t[k]) ** 2; }
    mse /= nW;
    const tiny = 1e-24 * (meanA * meanA + meanT * meanT) * nW;
    m.ncc = (varA > tiny && varT > tiny) ? cov / Math.sqrt(varA * varT) : NaN;
    const peak = peakT * peakT;
    m.psnrDb = mse > 0 ? 10 * Math.log10(peak * peak / mse) : Infinity;
    // vortices inside the window / inside the bright features (all four corners)
    const q = vortexChargeMap(v, n, n), c1 = n - 1;
    let brightPlaquettes = 0;
    const isB = (i, j) => b[i * n + j] > 0.5 * bMax, inW = (i, j) => mask[i * n + j] > 0;
    for (let i = 0; i < c1; ++i)
      for (let j = 0; j < c1; ++j) {
        const win = inW(i, j) && inW(i + 1, j) && inW(i + 1, j + 1) && inW(i, j + 1);
        const bri = isB(i, j) && isB(i + 1, j) && isB(i + 1, j + 1) && isB(i, j + 1);
        if (bri) ++brightPlaquettes;
        if (q[i * c1 + j] !== 0) { if (win) ++m.vortexCount; if (bri) ++m.vortexCountBright; }
      }
    m.vortexDensityBright = brightPlaquettes > 0 ? m.vortexCountBright / brightPlaquettes : NaN;   // per plaquette; x 1/pitch^2 for per m^2
    return m;
  }

  const api = { metrics, vortexChargeMap, vortexCount, vortexDensity };
  if (isNode) module.exports = api; else { root.DOE_CORE = root.DOE_CORE || {}; Object.assign(root.DOE_CORE, api); }
  }
  const root = typeof globalThis !== 'undefined' ? globalThis : this;
  factory(root);
  if (!(typeof module !== 'undefined' && module.exports)) (root.DOE_MODULE_SOURCES = root.DOE_MODULE_SOURCES || []).push(factory.toString());
})();
