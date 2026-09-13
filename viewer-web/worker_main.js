// Worker side of the web viewer. doeWorkerMain(self, C) installs the message handler in a Web
// Worker whose global already carries the core (DOE_CORE); the page builds that worker from the
// sources of the core scripts it has loaded (see viewer.js), because a worker script cannot be
// loaded from a file:// URL. Messages in:
//   { type: 'design', target: {rows, cols, data}, cfg, sweep: {nz, view} }
//   { type: 'resweep', volume, sweep: {nz, view, distance} }     (volume with a source block)
// Messages out:
//   { type: 'progress', phase: 'gs' | 'adam' | 'adam_qN' | 'sweep', it, total, energy }
//   { type: 'done', volume, summary }   (typed arrays transferred)
//   { type: 'error', message }
// Canceling terminates the worker (the page creates a new one for the next run).
function doeWorkerMain(self, C) {
  function transferables(vol) {
    const list = [vol.z.buffer, vol.xz.buffer, vol.yz.buffer, vol.vortexCount.buffer, vol.vortexDensity.buffer];
    for (const a of vol.amplitude) list.push(a.buffer);
    for (const a of vol.phase) list.push(a.buffer);
    for (const a of vol.charges) list.push(a.buffer);
    if (vol.doePhase) list.push(vol.doePhase.buffer, vol.target.buffer);
    if (vol.source) list.push(vol.source.phase.buffer, vol.source.illum.buffer, vol.source.target.buffer);
    return list;
  }
  self.onmessage = (ev) => {
    const msg = ev.data;
    try {
      if (msg.type === 'design') {
        let last = 0;
        const total = msg.cfg.iters || 400;
        const res = C.design(msg.target, msg.cfg, (name, it, e) => {
          const now = Date.now();
          if (now - last > 80 || it === total - 1) { last = now; self.postMessage({ type: 'progress', phase: name, it, total, energy: e }); }
          return true;
        });
        self.postMessage({ type: 'progress', phase: 'sweep', it: 0, total: msg.sweep.nz });
        const vol = C.sweepVolume(res, msg.sweep, (p, nz) => { if (p % 4 === 0) self.postMessage({ type: 'progress', phase: 'sweep', it: p, total: nz }); });
        C.attachSource(vol, res);
        const summary = {
          config: res.config, sampling: res.sampling, initialMetrics: res.initialMetrics, initName: res.config.init,
          window: { rows: res.inputs.windowRows, cols: res.inputs.windowCols }, padded: res.inputs.grid.n,
          runs: res.runs.map((r) => ({ name: r.name, metrics: r.metrics, seconds: r.seconds, history: r.history, shapeHistory: r.shapeHistory, efficiencyHistory: r.efficiencyHistory })),
        };
        self.postMessage({ type: 'done', volume: vol, summary }, transferables(vol));
      } else if (msg.type === 'resweep') {
        const vol = C.sweepFromSource(msg.volume, msg.sweep, (p, nz) => { if (p % 4 === 0) self.postMessage({ type: 'progress', phase: 'sweep', it: p, total: nz }); });
        self.postMessage({ type: 'done', volume: vol, summary: null }, transferables(vol));
      }
    } catch (e) {
      self.postMessage({ type: 'error', message: e && e.message ? e.message : String(e) });
    }
  };
}
if (typeof globalThis !== 'undefined') globalThis.DOE_WORKER_MAIN_SOURCE = doeWorkerMain.toString();
