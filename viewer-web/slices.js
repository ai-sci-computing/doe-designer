// Textures of the web viewer, the same mappings as src/viewer_core.cpp (slice_texture,
// entry_face_texture, exit_face_texture, wall_texture) and the playback of SliceAnimator.
// Every function returns RGBA bytes for a texture of the given width: pixel (i = x, j = y) sits
// at (j * width + i) * 4, so x runs along the texture width. Volume arrays are [i, j] with
// i the slow index (see doev.js), hence the index i * view + j when reading them.
(function (root) {
  const maps = (typeof module !== 'undefined' && module.exports) ? require('./colormaps.js') : root.DOE_COLORMAPS;
  const twilight = maps.twilight, viridis = maps.viridis;

  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  function putMap(out, k, map, t) {
    const idx = 3 * Math.min(255, Math.max(0, Math.round(clamp01(t) * 255)));
    out[k] = map[idx]; out[k + 1] = map[idx + 1]; out[k + 2] = map[idx + 2]; out[k + 3] = 255;
  }
  const q8 = (x) => Math.round(clamp01(x) * 255);
  // hsv_phasor of src/image.cpp: hue = phase (0 red, 2 pi / 3 green, -2 pi / 3 blue), value = amplitude
  function putHsv(out, k, value, ph) {
    let h = ph % (2 * Math.PI);
    if (h < 0) h += 2 * Math.PI;
    h = h / (2 * Math.PI) * 6;
    const sector = Math.floor(h) % 6, f = h - Math.floor(h);
    let r, g, b;
    switch (sector) {
      case 0: r = 1; g = f; b = 0; break;
      case 1: r = 1 - f; g = 1; b = 0; break;
      case 2: r = 0; g = 1; b = f; break;
      case 3: r = 0; g = 1 - f; b = 1; break;
      case 4: r = f; g = 0; b = 1; break;
      default: r = 1; g = 0; b = 1 - f; break;
    }
    out[k] = q8(r * value); out[k + 1] = q8(g * value); out[k + 2] = q8(b * value); out[k + 3] = 255;
  }

  // Slice p in one of the modes: 'intensity', 'log', 'hsv', 'vortices', 'real'.
  function sliceRGBA(v, p, style) {
    if (!(p >= 0 && p < v.nz)) throw new RangeError('sliceRGBA: plane index ' + p);
    const st = Object.assign({ mode: 'hsv', floorDb: 40, gamma: 1, timePhase: 0 }, style);
    const n = v.view, amp = v.amplitude[p], ph = v.phase[p];
    let amax = 0;
    for (let k = 0; k < amp.length; ++k) if (amp[k] > amax) amax = amp[k];
    const invGamma = 1 / Math.max(st.gamma, 1e-6);
    const out = new Uint8ClampedArray(n * n * 4);
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j) {
        const a = amax > 0 ? amp[i * n + j] / amax : 0;
        const k = (j * n + i) * 4;
        switch (st.mode) {
          case 'intensity': putMap(out, k, viridis, Math.pow(a * a, invGamma)); break;
          case 'log': { const db = a > 0 ? 20 * Math.log10(a) : -1e300; putMap(out, k, viridis, 1 + db / st.floorDb); break; }
          case 'hsv': putHsv(out, k, a, ph[i * n + j] + st.timePhase); break;
          case 'vortices': { const g = Math.round(a * 200); out[k] = g; out[k + 1] = g; out[k + 2] = g; out[k + 3] = 255; break; }
          case 'real': {
            const re = Math.max(-1, Math.min(1, a * Math.cos(ph[i * n + j] + st.timePhase)));
            const m = Math.abs(re), w = q8(1 - m);
            if (re >= 0) { out[k] = 255; out[k + 1] = w; out[k + 2] = w; } else { out[k] = w; out[k + 1] = w; out[k + 2] = 255; }
            out[k + 3] = 255; break;
          }
          default: throw new Error('sliceRGBA: unknown mode ' + st.mode);
        }
      }
    if (st.mode === 'vortices') {
      const q = v.charges[p], m = n - 1;
      for (let i = 0; i < m; ++i)
        for (let j = 0; j < m; ++j) {
          const c = q[i * m + j];
          if (c === 0) continue;
          const k = (j * n + i) * 4;
          if (c > 0) { out[k] = 255; out[k + 1] = 40; out[k + 2] = 40; } else { out[k] = 40; out[k + 1] = 90; out[k + 2] = 255; }
        }
    }
    return out;
  }

  // Entry face: DOE phase on the twilight map, or (with a source block) the illuminated DOE
  // field as HSV with value = illumination amplitude.
  function entryFaceRGBA(v, which) {
    const n = v.view;
    if (which === 'illuminated' && v.source && v.source.n > 0) {
      const N = v.source.n, i0 = (N - n) >> 1, j0 = (N - n) >> 1;
      let amax = 0;
      for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) amax = Math.max(amax, v.source.illum[(i0 + i) * N + j0 + j]);
      const out = new Uint8ClampedArray(n * n * 4);
      for (let i = 0; i < n; ++i)
        for (let j = 0; j < n; ++j) {
          const s = (i0 + i) * N + j0 + j;
          putHsv(out, (j * n + i) * 4, amax > 0 ? clamp01(v.source.illum[s] / amax) : 0, v.source.phase[s]);
        }
      return out;
    }
    if (!v.doePhase || v.doePhase.length !== n * n) throw new Error('entryFaceRGBA: no DOE phase in the volume');
    const out = new Uint8ClampedArray(n * n * 4);
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j) {
        let ph = v.doePhase[i * n + j];
        ph = ph - 2 * Math.PI * Math.floor((ph + Math.PI) / (2 * Math.PI));  // wrap to (-pi, pi]
        putMap(out, (j * n + i) * 4, twilight, (ph + Math.PI) / (2 * Math.PI));
      }
    return out;
  }

  function intensityRGBA(I, width, height, logScale, floorDb) {
    let imax = 0;
    for (let k = 0; k < I.length; ++k) if (I[k] > imax) imax = I[k];
    const out = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width; ++i)
      for (let j = 0; j < height; ++j) {
        const rel = imax > 0 ? Math.max(I[i * height + j], 0) / imax : 0;
        let t;
        if (logScale) { const db = rel > 0 ? 10 * Math.log10(rel) : -1e300; t = 1 + db / floorDb; } else t = rel;
        putMap(out, (j * width + i) * 4, viridis, t);
      }
    return out;
  }

  // Exit face: 'target' (from the file) or 'reconstruction' (intensity of the last plane), viridis, linear.
  function exitFaceRGBA(v, which) {
    const n = v.view;
    if (which === 'target') {
      if (!v.target || v.target.length !== n * n) throw new Error('exitFaceRGBA: no target in the volume');
      return intensityRGBA(v.target, n, n, false, 40);
    }
    const amp = v.amplitude[v.nz - 1];
    const I = new Float64Array(n * n);
    for (let k = 0; k < I.length; ++k) I[k] = amp[k] * amp[k];
    return intensityRGBA(I, n, n, false, 40);
  }

  // Wall: the xz or yz cut (nz x view in the file) as a log-intensity image, width = view
  // (x or y), height = nz (z), row 0 = DOE plane.
  function wallRGBA(v, which, floorDb) {
    const cut = which === 'xz' ? v.xz : v.yz, n = v.view, nz = v.nz;
    const t = new Float64Array(n * nz);           // [i = x, j = plane]
    for (let p = 0; p < nz; ++p) for (let i = 0; i < n; ++i) t[i * nz + p] = cut[p * n + i];
    return { rgba: intensityRGBA(t, n, nz, true, floorDb), width: n, height: nz };
  }

  // Playback through the planes (SliceAnimator of src/viewer_core.cpp).
  class SliceAnimator {
    constructor(planes) { this.planes = planes; this.pos = 0; this.dir = 1; this.playing = false; this.bounce = false; this.speed = 8; }
    plane() { return Math.min(this.planes - 1, Math.max(0, Math.floor(this.pos + 1e-9))); }
    position() { return this.pos; }
    step(dt) {
      if (!this.playing || this.planes < 2) return;
      this.pos += this.dir * this.speed * dt;
      const last = this.planes - 1;
      if (this.bounce) {
        while (this.pos > last || this.pos < 0) {
          if (this.pos > last) { this.pos = 2 * last - this.pos; this.dir = -1; }
          if (this.pos < 0) { this.pos = -this.pos; this.dir = 1; }
        }
      } else {
        this.pos = this.pos % this.planes;
        if (this.pos < 0) this.pos += this.planes;
      }
    }
    seek(plane) { this.pos = Math.min(Math.max(plane, 0), Math.max(this.planes - 1, 0)); this.dir = 1; }
  }

  const api = { sliceRGBA, entryFaceRGBA, exitFaceRGBA, wallRGBA, SliceAnimator };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DOE_SLICES = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
