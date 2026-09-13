// .doev parser: the layout of write_doev() in src/render.cpp (v2 adds the
// full-resolution source block). Two checks: a buffer built here with known values, and a
// file written by the real doe_design binary (cross-check of the layout, skipped when the
// binary is not built; ctest passes its path in DOE_DESIGN).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { parseDoev } = require('../doev.js');

// Little-endian writer mirroring put()/put_vec() of render.cpp.
function writer() {
  const parts = [];
  const push = (kind, values) => {
    const arr = kind === 'u32' ? Uint32Array.from(values) : kind === 'f64' ? Float64Array.from(values)
      : kind === 'f32' ? Float32Array.from(values) : Int8Array.from(values);
    parts.push(new Uint8Array(arr.buffer));
  };
  return {
    magic: (s) => parts.push(Uint8Array.from(Buffer.from(s, 'ascii'))),
    u32: (v) => push('u32', [v]), f64: (v) => push('f64', [v]),
    u32s: (v) => push('u32', v), f64s: (v) => push('f64', v), f32s: (v) => push('f32', v), i8s: (v) => push('i8', v),
    buffer: () => { const b = Buffer.concat(parts.map((p) => Buffer.from(p))); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
  };
}

function tinyFile({ version = 2, faces = true, source = true } = {}) {
  const view = 4, nz = 3, n = 8;
  const w = writer();
  w.magic('DOEV'); w.u32(version); w.u32(view); w.u32(nz); w.f64(8e-6); w.f64(532e-9);
  w.f64s([0, 0.01, 0.02]);
  for (let p = 0; p < nz; ++p) {
    w.f32s(Array.from({ length: view * view }, (_, k) => (p + 1) * 0.1 + k));          // amplitude
    w.f32s(Array.from({ length: view * view }, (_, k) => -Math.PI + (k / (view * view)) * 2 * Math.PI)); // phase
    const q = new Array((view - 1) * (view - 1)).fill(0); q[0] = 1; q[7] = -1; if (p === 2) q[4] = 1;
    w.i8s(q);
  }
  w.f64s(Array.from({ length: nz * view }, (_, k) => 100 + k));  // xz
  w.f64s(Array.from({ length: nz * view }, (_, k) => 200 + k));  // yz
  w.u32s([2, 2, 3]);
  w.f64s([2e6, 2e6, 3e6]);
  w.u32(faces ? 1 : 0);
  if (faces) { w.f64s(Array.from({ length: view * view }, (_, k) => k * 0.1)); w.f64s(Array.from({ length: view * view }, (_, k) => 1 - k * 0.05)); }
  if (version >= 2) {
    w.u32(source ? 1 : 0);
    if (source) {
      w.u32(n); w.f64(8e-6); w.f64(532e-9); w.f64(0.02); w.u32(1);
      w.f32s(Array.from({ length: n * n }, (_, k) => k * 0.01));
      w.f32s(Array.from({ length: n * n }, (_, k) => (k % 2)));
      w.f32s(Array.from({ length: n * n }, (_, k) => (k % 3) * 0.5));
    }
  }
  return w.buffer();
}

test('parseDoev reads a v2 file with faces and source block', () => {
  const v = parseDoev(tinyFile());
  assert.equal(v.version, 2);
  assert.equal(v.view, 4);
  assert.equal(v.nz, 3);
  assert.equal(v.pitch, 8e-6);
  assert.equal(v.wavelength, 532e-9);
  assert.deepEqual(Array.from(v.z), [0, 0.01, 0.02]);
  assert.equal(v.amplitude.length, 3);
  assert.equal(v.amplitude[1].length, 16);
  assert.ok(Math.abs(v.amplitude[1][3] - 3.2) < 1e-6);
  assert.ok(Math.abs(v.phase[0][0] + Math.PI) < 1e-6);
  assert.equal(v.charges[2].length, 9);
  assert.deepEqual(Array.from(v.charges[2]), [1, 0, 0, 0, 1, 0, 0, -1, 0]);
  assert.equal(v.xz.length, 12);
  assert.equal(v.xz[5], 105);
  assert.equal(v.yz[0], 200);
  assert.deepEqual(Array.from(v.vortexCount), [2, 2, 3]);
  assert.deepEqual(Array.from(v.vortexDensity), [2e6, 2e6, 3e6]);
  assert.ok(Math.abs(v.doePhase[15] - 1.5) < 1e-12);
  assert.ok(Math.abs(v.target[0] - 1) < 1e-12);
  assert.ok(v.source);
  assert.equal(v.source.n, 8);
  assert.equal(v.source.distance, 0.02);
  assert.equal(v.source.bandLimit, true);
  assert.equal(v.source.phase.length, 64);
  assert.ok(Math.abs(v.source.illum[1] - 1) < 1e-12);
  assert.ok(Math.abs(v.source.target[2] - 1) < 1e-6);
});

test('parseDoev reads a v1 file (no source block) and a v2 file without faces or source', () => {
  const v1 = parseDoev(tinyFile({ version: 1 }));
  assert.equal(v1.version, 1);
  assert.equal(v1.source, null);
  assert.ok(v1.doePhase);
  const bare = parseDoev(tinyFile({ faces: false, source: false }));
  assert.equal(bare.doePhase, null);
  assert.equal(bare.target, null);
  assert.equal(bare.source, null);
});

test('parseDoev rejects a wrong magic, an unknown version and a truncated file', () => {
  const good = tinyFile();
  const bad = new Uint8Array(good.slice(0)); bad[0] = 0x58;  // "XOEV"
  assert.throws(() => parseDoev(bad.buffer), /not a \.doev file/);
  const v9 = new Uint8Array(good.slice(0)); new DataView(v9.buffer).setUint32(4, 9, true);
  assert.throws(() => parseDoev(v9.buffer), /version/);
  assert.throws(() => parseDoev(good.slice(0, good.byteLength - 100)), /truncated/);
});

test('parseDoev agrees with the file written by doe_design (layout cross-check)', (t) => {
  const repo = path.resolve(__dirname, '..', '..');
  const exe = process.env.DOE_DESIGN || path.join(repo, 'build', 'apps', 'doe_design');
  if (!fs.existsSync(exe)) { t.skip('doe_design is not built'); return; }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'doev-'));
  execFileSync(exe, ['--target', path.join(repo, 'images', 'targets', 'disk.png'), '--active', '32', '--iters', '3', '--no-gs',
    '--distance', '0.02', '--nz', '5', '--view-size', '24', '--out', dir], { stdio: 'ignore' });
  const buf = fs.readFileSync(path.join(dir, 'volume.doev'));
  const v = parseDoev(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  assert.equal(v.version, 2);
  assert.equal(v.view, 24);
  assert.equal(v.nz, 5);
  assert.equal(v.z[0], 0);
  assert.ok(Math.abs(v.z[4] - 0.02) < 1e-12);
  assert.equal(v.source.n, 120);              // padded grid of a 32 px aperture at 20 mm (picture-clear rule: 116 -> 120)
  assert.ok(Math.abs(v.source.distance - 0.02) < 1e-12);
  for (let p = 0; p < v.nz; ++p) {
    assert.equal(v.amplitude[p].length, 24 * 24);
    let nonzero = 0;
    for (const q of v.charges[p]) { assert.ok(q === -1 || q === 0 || q === 1); if (q !== 0) ++nonzero; }
    assert.equal(nonzero, v.vortexCount[p]);   // the count column matches the charge maps
    for (const a of v.amplitude[p]) assert.ok(Number.isFinite(a) && a >= 0);
  }
  // plane 0 is the illuminated DOE: unit amplitude inside the aperture (the 24 px crop is inside the 32 px aperture)
  assert.ok(Math.abs(v.amplitude[0][0] - 1) < 1e-6);
  assert.ok(Math.abs(v.amplitude[0][24 * 12 + 12] - 1) < 1e-6);
  // faces: the target face is the disk, bright in the center and dark in the corner
  assert.ok(v.target[24 * 12 + 12] > 0.9);
  assert.ok(v.target[0] < 0.1);
  fs.rmSync(dir, { recursive: true, force: true });
});
