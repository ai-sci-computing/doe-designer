// The page cannot load worker.js from a file:// URL (browsers block worker scripts from the
// file origin), so the worker is built from the sources of the core scripts that the page has
// already loaded: each core script registers its factory's source in DOE_MODULE_SOURCES when it
// runs as a classic script (no CommonJS `module`), and worker_main.js exposes doeWorkerMain.
// This test simulates both sides in vm contexts without `module`.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const files = ['fft.js', 'util.js', 'propagate.js', 'energy.js', 'init.js', 'metrics.js', 'quantize.js', 'solvers.js', 'pipeline.js'].map((f) => path.join(__dirname, '..', 'core', f));

test('core scripts loaded as classic scripts register their sources and populate DOE_CORE', () => {
  const page = { console };
  page.globalThis = page;
  vm.createContext(page);
  for (const f of files) vm.runInContext(fs.readFileSync(f, 'utf8'), page, { filename: f });
  assert.equal(page.DOE_MODULE_SOURCES.length, files.length);
  for (const src of page.DOE_MODULE_SOURCES) assert.ok(typeof src === 'string' && src.startsWith('function'));
  assert.equal(typeof page.DOE_CORE.design, 'function');
  assert.equal(typeof page.DOE_CORE.fft2, 'function');
  const wm = fs.readFileSync(path.join(__dirname, '..', 'worker_main.js'), 'utf8');
  vm.runInContext(wm, page, { filename: 'worker_main.js' });
  assert.equal(typeof page.doeWorkerMain, 'function');
  assert.ok(page.DOE_WORKER_MAIN_SOURCE.startsWith('function'));
});

test('a worker built from the registered sources designs a tiny target and posts progress and a volume', async () => {
  const page = { console }; page.globalThis = page; vm.createContext(page);
  for (const f of files) vm.runInContext(fs.readFileSync(f, 'utf8'), page, { filename: f });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'worker_main.js'), 'utf8'), page, { filename: 'worker_main.js' });
  // the blob the page would create
  const blob = page.DOE_MODULE_SOURCES.map((s) => `(${s})(self);`).join('\n') + `\n(${page.DOE_WORKER_MAIN_SOURCE})(self, self.DOE_CORE);`;
  const posted = [];
  const worker = { console, postMessage: (m) => posted.push(m) };
  worker.self = worker; worker.globalThis = worker;
  vm.createContext(worker);
  vm.runInContext(blob, worker, { filename: 'blob-worker.js' });
  assert.equal(typeof worker.onmessage, 'function');
  const target = { rows: 16, cols: 12, data: new Float64Array(192) }; for (let k = 40; k < 100; ++k) target.data[k] = 1;
  worker.onmessage({ data: { type: 'design', target, cfg: { active: 16, pitch: 8e-6, wavelength: 532e-9, distance: 0.005, iters: 5, runGs: true, levels: 2 }, sweep: { nz: 3, view: 12 } } });
  const done = posted.find((m) => m.type === 'done');
  assert.ok(done, 'done message');
  assert.ok(posted.some((m) => m.type === 'progress' && m.phase === 'adam'));
  assert.ok(posted.some((m) => m.type === 'progress' && m.phase === 'sweep'));
  // (objects from the worker context have their own prototypes: compare values, not structures)
  assert.equal(done.summary.runs.map((r) => r.name).join(','), 'gs,adam,adam_q2');
  assert.equal(done.volume.nz, 3); assert.equal(done.volume.view, 12); assert.equal(done.volume.source.n, 40);   // 16 px at 5 mm: picture-clear rule gives 40
  assert.equal(done.summary.window.rows, 16); assert.equal(done.summary.window.cols, 12);
  posted.length = 0;
  worker.onmessage({ data: { type: 'resweep', volume: done.volume, sweep: { nz: 4, view: 8, distance: 0.01 } } });
  const again = posted.find((m) => m.type === 'done');
  assert.equal(again.volume.nz, 4); assert.ok(Math.abs(again.volume.z[3] - 0.01) < 1e-15);
  posted.length = 0;
  worker.onmessage({ data: { type: 'design', target: { rows: 0, cols: 0, data: new Float64Array(0) }, cfg: { active: 16 }, sweep: { nz: 2, view: 8 } } });
  assert.ok(posted.some((m) => m.type === 'error'), 'errors are reported, not thrown');
});
