// Reader of the .doev volume file written by write_doev() in src/render.cpp
// (little-endian, no compression; v2 adds the full-resolution source block).
//
// Layout, in order: "DOEV", u32 version (1 or 2), u32 view, u32 nz, f64 pitch, f64 wavelength,
// f64 z[nz]; per plane f32 amplitude[view^2], f32 phase[view^2], i8 charges[(view-1)^2];
// f64 xz[nz*view], f64 yz[nz*view], u32 vortexCount[nz], f64 vortexDensity[nz];
// u32 hasFaces, then f64 doePhase[view^2], f64 target[view^2];
// v2: u32 hasSource, then u32 n, f64 pitch, f64 wavelength, f64 distance, u32 bandLimit,
// f32 phase[n^2], f32 illum[n^2], f32 target[n^2].
//
// Arrays keep the file's [i, j] order (i along x, row-major with i as the slow index), i.e. the
// element (i, j) of a view x view array is at index i * view + j. Classic script: defines
// window.DOEV in a page and module.exports under Node.
(function (root) {
  function parseDoev(buffer) {
    if (buffer instanceof ArrayBuffer === false) buffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const dv = new DataView(buffer);
    let off = 0;
    const need = (bytes) => { if (off + bytes > buffer.byteLength) throw new Error('parseDoev: truncated file'); };
    const u32 = () => { need(4); const v = dv.getUint32(off, true); off += 4; return v; };
    const f64 = () => { need(8); const v = dv.getFloat64(off, true); off += 8; return v; };
    const typed = (Ctor, count) => {
      const bytes = count * Ctor.BYTES_PER_ELEMENT;
      need(bytes);
      // copy into an aligned buffer (the offset inside the file is not necessarily aligned)
      const out = new Ctor(new Uint8Array(buffer, off, bytes).slice().buffer);
      off += bytes;
      return out;
    };
    need(4);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    off = 4;
    if (magic !== 'DOEV') throw new Error('parseDoev: not a .doev file');
    const version = u32();
    if (version !== 1 && version !== 2) throw new Error('parseDoev: unsupported version ' + version);
    const view = u32(), nz = u32(), pitch = f64(), wavelength = f64();
    const z = typed(Float64Array, nz);
    const amplitude = [], phase = [], charges = [];
    for (let p = 0; p < nz; ++p) {
      amplitude.push(typed(Float32Array, view * view));
      phase.push(typed(Float32Array, view * view));
      charges.push(typed(Int8Array, (view - 1) * (view - 1)));
    }
    const xz = typed(Float64Array, nz * view), yz = typed(Float64Array, nz * view);
    const vortexCount = typed(Uint32Array, nz), vortexDensity = typed(Float64Array, nz);
    let doePhase = null, target = null;
    if (u32() === 1) { doePhase = typed(Float64Array, view * view); target = typed(Float64Array, view * view); }
    let source = null;
    if (version >= 2 && u32() === 1) {
      const n = u32();
      source = { n, pitch: f64(), wavelength: f64(), distance: f64(), bandLimit: u32() === 1 };
      source.phase = typed(Float32Array, n * n);
      source.illum = typed(Float32Array, n * n);
      source.target = typed(Float32Array, n * n);
    }
    return { version, view, nz, pitch, wavelength, z, amplitude, phase, charges, xz, yz, vortexCount, vortexDensity, doePhase, target, source };
  }
  // Writer, the inverse of parseDoev (format version 2). Returns an ArrayBuffer.
  function writeDoev(v) {
    const parts = [];
    const push = (arr) => parts.push(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    const u32 = (...x) => push(Uint32Array.from(x)), f64 = (...x) => push(Float64Array.from(x));
    push(Uint8Array.from([0x44, 0x4f, 0x45, 0x56]));   // "DOEV"
    u32(2, v.view, v.nz); f64(v.pitch, v.wavelength);
    push(Float64Array.from(v.z));
    for (let p = 0; p < v.nz; ++p) { push(Float32Array.from(v.amplitude[p])); push(Float32Array.from(v.phase[p])); push(Int8Array.from(v.charges[p])); }
    push(Float64Array.from(v.xz)); push(Float64Array.from(v.yz)); push(Uint32Array.from(v.vortexCount)); push(Float64Array.from(v.vortexDensity));
    const faces = v.doePhase && v.target ? 1 : 0;
    u32(faces);
    if (faces) { push(Float64Array.from(v.doePhase)); push(Float64Array.from(v.target)); }
    u32(v.source ? 1 : 0);
    if (v.source) {
      u32(v.source.n); f64(v.source.pitch, v.source.wavelength, v.source.distance); u32(v.source.bandLimit ? 1 : 0);
      push(Float32Array.from(v.source.phase)); push(Float32Array.from(v.source.illum)); push(Float32Array.from(v.source.target));
    }
    let total = 0; for (const p of parts) total += p.byteLength;
    const out = new Uint8Array(total); let off = 0;
    for (const p of parts) { out.set(p, off); off += p.byteLength; }
    return out.buffer;
  }
  const api = { parseDoev, writeDoev };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DOEV = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
