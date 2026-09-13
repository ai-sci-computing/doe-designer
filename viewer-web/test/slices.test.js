// Slice, face and wall mappings of the web viewer, mirrored from tests/test_viewer_core.cpp
// (same fixture, same expected pixels as the native viewer's slice_texture / entry_face_texture /
// exit_face_texture / wall_texture) and the playback logic of SliceAnimator.
const test = require('node:test');
const assert = require('node:assert/strict');
const { sliceRGBA, entryFaceRGBA, exitFaceRGBA, wallRGBA, SliceAnimator } = require('../slices.js');
const { twilight, viridis } = require('../colormaps.js');

function tinyVolume() {
  const view = 4, nz = 3;
  const v = { version: 2, view, nz, pitch: 8e-6, wavelength: 532e-9, z: Float64Array.from([0, 0.01, 0.02]),
    amplitude: [], phase: [], charges: [], xz: new Float64Array(nz * view), yz: new Float64Array(nz * view),
    vortexCount: Uint32Array.from([2, 2, 2]), vortexDensity: Float64Array.from([1, 1, 1]), doePhase: null, target: null, source: null };
  for (let p = 0; p < nz; ++p) {
    const amp = new Float32Array(view * view), ph = new Float32Array(view * view);
    amp[1 * view + 1] = 1.0;            // (i = 1, j = 1): one bright pixel, phase 0 -> red
    amp[2 * view + 2] = 0.1;            // (2, 2): a dim one (-20 dB), phase 2 pi / 3 -> green
    ph[2 * view + 2] = 2 * Math.PI / 3;
    const q = new Int8Array((view - 1) * (view - 1)); q[0] = 1; q[2 * 3 + 1] = -1;  // plaquettes (0,0) = +1, (2,1) = -1
    v.amplitude.push(amp); v.phase.push(ph); v.charges.push(q);
  }
  return v;
}
// pixel (i = x, j = y) of a view x view RGBA image: row j, column i
const px = (img, w, i, j) => Array.from(img.subarray((j * w + i) * 4, (j * w + i) * 4 + 3));
const lut = (map, t) => { const k = Math.min(255, Math.max(0, Math.round(t * 255))); return [map[3 * k], map[3 * k + 1], map[3 * k + 2]]; };

test('sliceRGBA: intensity, log intensity, HSV, real part and vortex modes match the native viewer', () => {
  const v = tinyVolume();
  const lin = sliceRGBA(v, 1, { mode: 'intensity', floorDb: 40, gamma: 1, timePhase: 0 });
  assert.equal(lin.length, 4 * 4 * 4);
  assert.deepEqual(px(lin, 4, 1, 1), lut(viridis, 1));   // max color
  assert.deepEqual(px(lin, 4, 0, 0), lut(viridis, 0));   // zero
  assert.ok(px(lin, 4, 2, 2)[0] < 80);                   // 1 % intensity: nearly the zero color
  assert.equal(lin[3], 255);                             // opaque
  const lg = sliceRGBA(v, 1, { mode: 'log', floorDb: 40, gamma: 1, timePhase: 0 });
  assert.deepEqual(px(lg, 4, 2, 2), lut(viridis, 0.5));  // -20 dB of a 40 dB floor: half way up
  const hs = sliceRGBA(v, 1, { mode: 'hsv', floorDb: 40, gamma: 1, timePhase: 0 });
  assert.deepEqual(px(hs, 4, 1, 1), [255, 0, 0]);
  assert.deepEqual(px(hs, 4, 0, 0), [0, 0, 0]);
  const hs2 = sliceRGBA(v, 1, { mode: 'hsv', floorDb: 40, gamma: 1, timePhase: 2 * Math.PI / 3 });
  assert.deepEqual(px(hs2, 4, 1, 1), [0, 255, 0]);       // hue rotated by 2 pi / 3: green
  const re = sliceRGBA(v, 1, { mode: 'real', floorDb: 40, gamma: 1, timePhase: 0 });
  assert.deepEqual(px(re, 4, 1, 1), [255, 0, 0]);        // Re(1 e^{i0}) = +1 -> red
  assert.deepEqual(px(re, 4, 0, 0), [255, 255, 255]);    // zero -> white
  const re2 = sliceRGBA(v, 1, { mode: 'real', floorDb: 40, gamma: 1, timePhase: Math.PI });
  assert.deepEqual(px(re2, 4, 1, 1), [0, 0, 255]);       // -1 -> blue
  const re3 = sliceRGBA(v, 1, { mode: 'real', floorDb: 40, gamma: 1, timePhase: Math.PI / 2 });
  assert.ok(px(re3, 4, 1, 1)[0] > 250 && px(re3, 4, 1, 1)[2] > 250);  // Re = 0 -> white
  const vx = sliceRGBA(v, 1, { mode: 'vortices', floorDb: 40, gamma: 1, timePhase: 0 });
  assert.deepEqual(px(vx, 4, 0, 0), [255, 40, 40]);      // +1 residue at plaquette (0,0)
  assert.deepEqual(px(vx, 4, 2, 1), [40, 90, 255]);      // -1 residue at plaquette (2,1)
  assert.deepEqual(px(vx, 4, 1, 1), [200, 200, 200]);    // gray intensity elsewhere
  assert.throws(() => sliceRGBA(v, 7, { mode: 'hsv' }), /plane/);
});

test('sliceRGBA: gamma changes the intensity modes only', () => {
  const v = tinyVolume();
  const g = sliceRGBA(v, 1, { mode: 'intensity', floorDb: 40, gamma: 2, timePhase: 0 });
  assert.deepEqual(px(g, 4, 2, 2), lut(viridis, 0.1));   // (0.01)^(1/2) = 0.1
});

test('faces: entry face is the DOE phase on the twilight map (or the illuminated field as HSV), exit face target / reconstruction', () => {
  const v = tinyVolume();
  assert.throws(() => entryFaceRGBA(v, 'phase'), /DOE phase/);
  v.doePhase = new Float64Array(16); v.doePhase[1 * 4 + 1] = Math.PI - 1e-9;
  const e = entryFaceRGBA(v, 'phase');
  assert.deepEqual(px(e, 4, 0, 0), lut(twilight, 0.5));  // phase 0 -> middle of the cyclic map
  assert.deepEqual(px(e, 4, 1, 1), lut(twilight, 1));    // phase pi -> end of the map
  // with a source block, the illuminated face is HSV: hue = phase, value = illumination
  v.source = { n: 8, pitch: 8e-6, wavelength: 532e-9, distance: 0.02, bandLimit: true,
    phase: new Float32Array(64), illum: new Float32Array(64), target: new Float32Array(64) };
  v.source.illum[3 * 8 + 3] = 1.0;                       // crop offset (8 - 4) / 2 = 2: view pixel (1, 1)
  v.source.phase[3 * 8 + 3] = 2 * Math.PI / 3;           // green
  const il = entryFaceRGBA(v, 'illuminated');
  assert.deepEqual(px(il, 4, 1, 1), [0, 255, 0]);
  assert.deepEqual(px(il, 4, 0, 0), [0, 0, 0]);
  assert.throws(() => exitFaceRGBA(v, 'target'), /target/);
  v.target = new Float64Array(16); v.target[2 * 4 + 2] = 0.5;
  const t = exitFaceRGBA(v, 'target');
  assert.deepEqual(px(t, 4, 2, 2), lut(viridis, 1));     // normalized to its own maximum
  assert.deepEqual(px(t, 4, 0, 0), lut(viridis, 0));
  const r = exitFaceRGBA(v, 'reconstruction');            // last plane's intensity
  assert.deepEqual(px(r, 4, 1, 1), lut(viridis, 1));
  assert.deepEqual(px(r, 4, 2, 2), lut(viridis, 0.01));
});

test('walls: xz / yz cuts as log-intensity images, width = view (x), height = nz (z), row 0 = DOE plane', () => {
  const v = tinyVolume();
  v.xz[0 * 4 + 1] = 1.0;      // plane 0, x = 1: the maximum
  v.xz[2 * 4 + 3] = 0.01;     // plane 2, x = 3: -20 dB
  const w = wallRGBA(v, 'xz', 40);
  assert.equal(w.rgba.length, 4 * 3 * 4);
  assert.equal(w.width, 4); assert.equal(w.height, 3);
  assert.deepEqual(px(w.rgba, 4, 1, 0), lut(viridis, 1));
  assert.deepEqual(px(w.rgba, 4, 3, 2), lut(viridis, 0.5));
  assert.deepEqual(px(w.rgba, 4, 0, 0), lut(viridis, 0));
  const y = wallRGBA(v, 'yz', 40);
  assert.equal(y.width, 4); assert.equal(y.height, 3);
});

test('SliceAnimator: plays at the given speed, wraps, pauses, seeks, bounces', () => {
  const a = new SliceAnimator(10);
  assert.equal(a.plane(), 0);
  assert.equal(a.playing, false);
  a.step(1.0);
  assert.equal(a.plane(), 0);           // paused
  a.playing = true; a.speed = 4.0;      // planes per second
  a.step(0.5);
  assert.equal(a.plane(), 2);
  a.step(2.0);                          // 8 more -> 10 -> wraps to 0
  assert.equal(a.plane(), 0);
  a.seek(7);
  assert.equal(a.plane(), 7);
  assert.ok(Math.abs(a.position() - 7) < 1e-12);
  a.seek(-3); assert.equal(a.plane(), 0);
  a.seek(99); assert.equal(a.plane(), 9);
  a.bounce = true; a.seek(9); a.step(0.5);   // +2 with bounce -> 7
  assert.equal(a.plane(), 7);
});
