// The web viewer page: Three.js scene of the box between the DOE plane and the target plane,
// with the moving slice, the two faces and the two log-intensity walls, driven by the same
// mappings as the native viewer (slices.js) from a .doev file (doev.js). Display only: the
// native doe_view re-propagates and re-designs; this page shows what a file contains.
(function () {
  const { parseDoev, writeDoev } = window.DOEV;
  const { sliceRGBA, entryFaceRGBA, exitFaceRGBA, wallRGBA, SliceAnimator } = window.DOE_SLICES;
  const $ = (id) => document.getElementById(id);

  // ---- scene -------------------------------------------------------------------------------
  const canvas = $('gl');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161a);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  const orbit = { theta: 0.9, phi: 1.15, radius: 3.2, target: new THREE.Vector3(0, 0, 0) };
  function placeCamera() {
    const s = Math.sin(orbit.phi);
    camera.position.set(orbit.target.x + orbit.radius * s * Math.sin(orbit.theta),
      orbit.target.y + orbit.radius * Math.cos(orbit.phi),
      orbit.target.z + orbit.radius * s * Math.cos(orbit.theta));
    camera.lookAt(orbit.target);
  }
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    orbit.theta -= (e.clientX - drag.x) * 0.008;
    orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, orbit.phi - (e.clientY - drag.y) * 0.008));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); orbit.radius = Math.min(20, Math.max(0.3, orbit.radius * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    }
  }

  // ---- volume, textures, meshes -----------------------------------------------------------
  let vol = null, anim = null, group = null;
  const tex = {};                    // name -> DataTexture
  const mesh = {};                   // name -> Mesh
  const st = { mode: 'hsv', floorDb: 40, gamma: 1, timePhase: 0 };
  let timePhase = 0, lastPlane = -1, dirty = true, wallsDirty = true, facesDirty = true;

  function dataTexture(rgba, w, h) {
    const t = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.magFilter = THREE.NearestFilter; t.minFilter = THREE.LinearFilter; t.colorSpace = THREE.SRGBColorSpace;
    t.needsUpdate = true;
    return t;
  }
  function updateTexture(t, rgba) { t.image.data.set(rgba); t.needsUpdate = true; }

  function depth() { return parseFloat($('zc').value); }   // box depth relative to its width (width = 1)

  function buildScene() {
    if (group) { scene.remove(group); group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); }
    for (const k of Object.keys(tex)) { tex[k].dispose(); delete tex[k]; }
    group = new THREE.Group();
    const n = vol.view, d = depth();
    tex.slice = dataTexture(sliceRGBA(vol, 0, st), n, n);
    tex.entry = dataTexture(entryFaceRGBA(vol, faceChoice('entry')), n, n);
    tex.exit = dataTexture(exitFaceRGBA(vol, faceChoice('exit')), n, n);
    const wx = wallRGBA(vol, 'xz', st.floorDb), wy = wallRGBA(vol, 'yz', st.floorDb);
    tex.xz = dataTexture(wx.rgba, wx.width, wx.height);
    tex.yz = dataTexture(wy.rgba, wy.width, wy.height);
    const face = (t, opacity) => new THREE.MeshBasicMaterial({ map: t, side: THREE.DoubleSide, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    const square = new THREE.PlaneGeometry(1, 1);
    // entry face at z = 0 (DOE plane), exit face at z = d (target plane); the box is centered at z = d / 2
    mesh.entry = new THREE.Mesh(square, face(tex.entry, 1)); mesh.entry.position.z = -d / 2;
    mesh.exit = new THREE.Mesh(square, face(tex.exit, 1)); mesh.exit.position.z = d / 2;
    mesh.slice = new THREE.Mesh(square, face(tex.slice, 1)); mesh.slice.position.z = -d / 2;
    // walls: xz cut on the bottom (y = -1/2), yz cut on the left (x = -1/2); texture row 0 = DOE plane
    const wallOp = parseFloat($('wallOp').value);
    const gx = new THREE.PlaneGeometry(1, d);
    // PlaneGeometry(1, d) lies in xy with u along +x and v along +y (texture row 0 at y = -d/2).
    // Euler order XYZ applies Rz first, then Ry, then Rx.
    // Bottom wall: Rx(pi/2) maps (x, y) -> (x, 0, y): u stays along x, row 0 lands on the entry face (z = -d/2).
    mesh.xz = new THREE.Mesh(gx, face(tex.xz, wallOp)); mesh.xz.rotation.x = Math.PI / 2; mesh.xz.position.y = -0.5;
    // Left wall: Rz(pi/2) then Ry(pi/2) maps (x, y) -> (0, x, y): u along the box's y, row 0 on the entry face.
    const gy = new THREE.PlaneGeometry(1, d);
    mesh.yz = new THREE.Mesh(gy, face(tex.yz, wallOp)); mesh.yz.rotation.z = Math.PI / 2; mesh.yz.rotation.y = Math.PI / 2; mesh.yz.position.x = -0.5;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, d)), new THREE.LineBasicMaterial({ color: 0x5a616c }));
    for (const m of [mesh.entry, mesh.exit, mesh.xz, mesh.yz, mesh.slice, edges]) group.add(m);
    scene.add(group);
    lastPlane = -1; dirty = true;
    applyToggles();
  }

  // 'none' hides a face; the texture then keeps a valid choice (phase / reconstruction)
  function faceChoice(which) {
    const v = $(which).value;
    if (v !== 'none') return v;
    return which === 'entry' ? 'phase' : 'reconstruction';
  }
  function applyToggles() {
    if (!group) return;
    const w = $('walls').checked, op = parseFloat($('wallOp').value);
    for (const m of [mesh.xz, mesh.yz]) { m.visible = w; m.material.opacity = op; m.material.transparent = op < 1; m.material.depthWrite = op >= 1; m.material.needsUpdate = true; }
    mesh.slice.visible = $('slicecb').checked;
    mesh.entry.visible = $('entry').value !== 'none';
    mesh.exit.visible = $('exit').value !== 'none';
  }

  // ---- panel --------------------------------------------------------------------------------
  function fmt(x, unit, scale, digits) { return (x * scale).toFixed(digits) + ' ' + unit; }
  function showInfo() {
    const z1 = vol.z[vol.nz - 1];
    const src = vol.source ? `\nsource block: ${vol.source.n} × ${vol.source.n} px, design distance ${fmt(vol.source.distance, 'mm', 1e3, 2)}` : '\nformat v1 (no source block)';
    $('info').textContent = `${vol.view} × ${vol.view} px crop, ${vol.nz} planes, 0 to ${fmt(z1, 'mm', 1e3, 2)}\npitch ${fmt(vol.pitch, 'µm', 1e6, 1)}, wavelength ${fmt(vol.wavelength, 'nm', 1e9, 0)}, crop ${fmt(vol.view * vol.pitch, 'mm', 1e3, 2)}` + src;
    $('plane').max = vol.nz - 1;
    $('drop').classList.add('hidden');
    drawDensity();
  }
  function drawDensity() {
    const c = $('density'), g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    const n = vol.nz, dens = vol.vortexDensity;
    let dmax = 0; for (const v of dens) dmax = Math.max(dmax, v);
    const l = 34, r = 6, t = 12, b = 16, w = c.width - l - r, h = c.height - t - b;
    g.strokeStyle = '#3a3f48'; g.strokeRect(l, t, w, h);
    g.fillStyle = '#8b929c'; g.font = '10px system-ui'; g.textAlign = 'left';
    g.fillText('vortices / mm² vs z', l + 2, t - 3);
    g.textAlign = 'right'; g.fillText((dmax * 1e-6).toFixed(0), l - 3, t + 9); g.fillText('0', l - 3, t + h);
    g.strokeStyle = '#7fb0ff'; g.lineWidth = 1.2; g.beginPath();
    for (let p = 0; p < n; ++p) { const x = l + (n > 1 ? p / (n - 1) : 0) * w, y = t + h - (dmax > 0 ? dens[p] / dmax : 0) * h; if (p === 0) g.moveTo(x, y); else g.lineTo(x, y); }
    g.stroke();
    if (anim) { const p = anim.plane(); const x = l + (n > 1 ? p / (n - 1) : 0) * w; g.strokeStyle = '#ffb060'; g.beginPath(); g.moveTo(x, t); g.lineTo(x, t + h); g.stroke(); }
  }
  function bind(id, fn, digits) {
    const el = $(id), out = $(id + 'V');
    const show = () => { if (out) out.textContent = digits === undefined ? el.value : parseFloat(el.value).toFixed(digits); };
    el.addEventListener('input', () => { show(); fn(parseFloat(el.value)); });
    show();
  }
  bind('plane', (v) => { if (anim) { anim.seek(v); dirty = true; } });
  bind('speed', (v) => { if (anim) anim.speed = v; });
  bind('floor', (v) => { st.floorDb = v; dirty = true; wallsDirty = true; });
  bind('gamma', (v) => { st.gamma = v; dirty = true; }, 2);
  bind('omega', () => {}, 1);
  bind('zc', () => { if (vol) buildScene(); }, 2);
  bind('wallOp', applyToggles, 2);
  $('mode').addEventListener('change', () => { st.mode = $('mode').value; dirty = true; });
  $('walls').addEventListener('change', applyToggles);
  $('slicecb').addEventListener('change', applyToggles);
  $('entry').addEventListener('change', () => { facesDirty = true; applyToggles(); });
  $('exit').addEventListener('change', () => { facesDirty = true; applyToggles(); });
  $('bounce').addEventListener('change', () => { if (anim) anim.bounce = $('bounce').checked; });
  $('play').addEventListener('click', () => { if (!anim) return; anim.playing = !anim.playing; $('play').textContent = anim.playing ? 'Pause' : 'Play'; });
  $('anim').addEventListener('change', () => { if (!$('anim').checked) { timePhase = 0; st.timePhase = 0; dirty = true; } });

  // ---- loading ------------------------------------------------------------------------------
  function load(buffer, name) {
    try {
      vol = parseDoev(buffer);
    } catch (e) { $('info').textContent = 'cannot read ' + name + ': ' + e.message; return; }
    afterLoad(name);
  }
  function afterLoad(name) {
    anim = new SliceAnimator(vol.nz);
    anim.speed = parseFloat($('speed').value); anim.bounce = $('bounce').checked;
    $('play').textContent = 'Play';
    $('entry').querySelector('[value=illuminated]').disabled = !vol.source;
    $('exit').querySelector('[value=target]').disabled = !vol.target;
    if (!vol.target) $('exit').value = 'reconstruction';
    showInfo();
    buildScene();
    if (initialPlane !== null) { anim.seek(initialPlane); dirty = true; }
    document.title = name + ' – doe-designer web viewer';
    $('resweep').disabled = running || !vol.source;
    $('saveDoev').disabled = false; $('savePhase').disabled = !vol.doePhase;
    if (vol.source) { $('pDistance').value = (vol.source.distance * 1e3).toFixed(2); $('pNz').value = vol.nz; $('pView').value = vol.view; }
  }
  $('file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) f.arrayBuffer().then((b) => load(b, f.name)); });
  const view = $('view');
  view.addEventListener('dragover', (e) => { e.preventDefault(); });
  view.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) f.arrayBuffer().then((b) => load(b, f.name)); });
  // URL parameters: file=URL (fetched; needs an http server), and optional theta, phi, radius
  // (camera), plane, mode, walls=0/1, slice=0/1, zc, for reproducible views and screenshots.
  const params = new URLSearchParams(location.search);
  const num = (k) => (params.has(k) ? parseFloat(params.get(k)) : null);
  if (num('theta') !== null) orbit.theta = num('theta');
  if (num('phi') !== null) orbit.phi = num('phi');
  if (num('radius') !== null) orbit.radius = num('radius');
  if (params.has('mode')) { $('mode').value = params.get('mode'); st.mode = $('mode').value; }
  if (params.has('walls')) $('walls').checked = params.get('walls') !== '0';
  if (params.has('slice')) $('slicecb').checked = params.get('slice') !== '0';
  if (num('zc') !== null) { $('zc').value = num('zc'); $('zcV').textContent = num('zc').toFixed(2); }
  if (num('wallOp') !== null) { $('wallOp').value = num('wallOp'); $('wallOpV').textContent = num('wallOp').toFixed(2); }
  const initialPlane = num('plane');
  // design parameters from the URL (active, iters, levels, distance in mm, mu, init, nz, view, gs=0/1),
  // image=URL loads a target (http only), run=1 starts the design once the image is loaded
  for (const [key, id] of [['active', 'pActive'], ['iters', 'pIters'], ['levels', 'pLevels'], ['distance', 'pDistance'], ['mu', 'pMu'], ['init', 'pInit'], ['nz', 'pNz'], ['view', 'pView'], ['illum', 'pIllum'], ['lr', 'pLr']])
    if (params.has(key)) $(id).value = params.get(key);
  if (params.has('gs')) $('pGs').checked = params.get('gs') !== '0';
  if (params.has('letterbox')) $('pLetterbox').checked = params.get('letterbox') !== '0';
  if (params.has('entry')) $('entry').value = params.get('entry');
  if (params.has('exit')) $('exit').value = params.get('exit');
  let autorun = params.get('run') === '1';
  if (params.has('image')) loadImageFile(params.get('image'), params.get('image').split('/').pop());
  const param = params.get('file');
  if (param) {
    $('info').textContent = 'loading ' + param + ' …';
    fetch(param).then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); }).then((b) => load(b, param.split('/').pop()))
      .catch((e) => { $('info').textContent = 'cannot fetch ' + param + ': ' + e.message; });
  }

  // ---- design in a Web Worker ----------------------------------------------------------------
  let worker = null, target = null, summary = null, running = false;
  const val = (id) => parseFloat($(id).value);
  function readConfig() {
    return { pow2Grid: true, wavelength: val('pWavelength') * 1e-9, pitch: val('pPitch') * 1e-6, active: parseInt($('pActive').value, 10), distance: val('pDistance') * 1e-3,
      illum: $('pIllum').value, init: $('pInit').value, iters: Math.max(1, Math.round(val('pIters'))), lr: val('pLr'), mu: val('pMu'),
      levels: parseInt($('pLevels').value, 10), runGs: $('pGs').checked, bandLimit: $('pBand').checked, letterbox: $('pLetterbox').checked };
  }
  function sweepOpts() { const a = parseInt($('pActive').value, 10); return { nz: Math.max(2, Math.round(val('pNz'))), view: Math.min(Math.max(8, Math.round(val('pView'))), a) }; }
  function setStatus(text, frac) { $('status').textContent = text; $('progress').style.width = (Math.max(0, Math.min(1, frac || 0)) * 100).toFixed(1) + '%'; }
  function setRunning(on) {
    running = on;
    $('run').disabled = on || !target; $('cancel').disabled = !on; $('resweep').disabled = on || !(vol && vol.source);
    $('image').disabled = on; $('file').disabled = on;
  }
  // image -> luminance (Rec. 601) in the [i = x, j = y] layout of the C++ (to_gray)
  function imageToTarget(img) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, w, h).data, data = new Float64Array(w * h);
    for (let y = 0; y < h; ++y) for (let x = 0; x < w; ++x) { const k = (y * w + x) * 4; data[x * h + y] = (0.299 * d[k] + 0.587 * d[k + 1] + 0.114 * d[k + 2]) / 255; }
    return { rows: w, cols: h, data };
  }
  function loadImageFile(fileOrUrl, name) {
    const img = new Image();
    img.onload = () => {
      target = imageToTarget(img);
      // start and mu from the image (manual section 5); the fields stay editable
      const rec = window.DOE_CORE.recommendConfig(target);
      $('pInit').value = rec.init; $('pMu').value = rec.mu;
      $('imageInfo').textContent = `${name}: ${target.rows} × ${target.cols} px, ${(rec.midToneFraction * 100).toFixed(0)} % mid-tones: ` +
        (rec.continuousTone ? 'continuous-tone image, start set to backprop and µ to 0.02' : 'binary target, start TIE and µ 0.3');
      if (fileOrUrl instanceof Blob) URL.revokeObjectURL(img.src);
      setRunning(false);
      if (autorun) { autorun = false; runDesign(); }
    };
    img.onerror = () => { $('imageInfo').textContent = 'cannot read ' + name; };
    img.src = fileOrUrl instanceof Blob ? URL.createObjectURL(fileOrUrl) : fileOrUrl;
  }
  // The worker is built from the sources of the core scripts this page has loaded (a Blob URL),
  // because browsers refuse to load a worker script from a file:// page.
  let workerUrl = null;
  function workerBlobUrl() {
    if (workerUrl) return workerUrl;
    const src = (window.DOE_MODULE_SOURCES || []).map((s) => `(${s})(self);`).join('\n') + `\n(${window.DOE_WORKER_MAIN_SOURCE})(self, self.DOE_CORE);`;
    workerUrl = URL.createObjectURL(new Blob([src], { type: 'application/javascript' }));
    return workerUrl;
  }
  function newWorker() {
    if (worker) worker.terminate();
    worker = new Worker(workerBlobUrl());
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress') {
        const frac = m.total > 0 ? m.it / m.total : 0;
        const e = Number.isFinite(m.energy) ? `, energy ${m.energy.toFixed(4)}` : '';
        setStatus(`${m.phase}: ${m.it + 1} / ${m.total}${e}`, m.phase === 'sweep' ? 0.9 + 0.1 * frac : 0.9 * frac);
      } else if (m.type === 'done') {
        setRunning(false);
        setStatus('done', 1);
        if (m.summary) { summary = m.summary; showSummary(); }
        vol = m.volume;
        afterLoad(m.summary ? 'design' : 'design (re-propagated)');
      } else if (m.type === 'error') { setRunning(false); setStatus('error: ' + m.message, 0); }
    };
    worker.onerror = (e) => { setRunning(false); setStatus('worker error: ' + (e.message || e), 0); };
    return worker;
  }
  function runDesign() {
    if (!target || running) return;
    const cfg = readConfig(), sw = sweepOpts();
    summary = null;
    setRunning(true); setStatus('starting …', 0);
    try { newWorker().postMessage({ type: 'design', target, cfg, sweep: sw }); }
    catch (e) { setRunning(false); setStatus('cannot start the worker: ' + (e.message || e), 0); }
  }
  function resweep() {
    if (!vol || !vol.source || running) return;
    const sw = sweepOpts(); sw.view = Math.min(sw.view, vol.source.n); sw.distance = val('pDistance') * 1e-3;
    setRunning(true); setStatus('re-propagating …', 0.9);
    try { newWorker().postMessage({ type: 'resweep', volume: vol, sweep: sw }); }
    catch (e) { setRunning(false); setStatus('cannot start the worker: ' + (e.message || e), 0); }
  }
  function fmtMetric(x, digits) { return Number.isFinite(x) ? x.toFixed(digits) : (Number.isNaN(x) ? '–' : '∞'); }
  function showSummary() {
    const s = summary, per = 1e-6 / (s.config.pitch * s.config.pitch);   // vortices per plaquette -> per mm²
    const rows = [{ name: `initial (${s.initName})`, m: s.initialMetrics, it: '-', t: '-' }].concat(s.runs.map((r) => ({ name: r.name, m: r.metrics, it: r.history.length, t: r.seconds.toFixed(1) })));
    let html = '<table style="border-collapse:collapse; width:100%; font-size:11px"><tr><th style="text-align:left">run</th><th>eff.</th><th>RMSE</th><th>NCC</th><th>PSNR</th><th>vort/mm²</th><th>it</th><th>s</th></tr>';
    for (const r of rows) html += `<tr><td>${r.name}</td><td align="right">${fmtMetric(r.m.efficiency, 3)}</td><td align="right">${fmtMetric(r.m.amplitudeRmse, 3)}</td><td align="right">${fmtMetric(r.m.ncc, 3)}</td><td align="right">${fmtMetric(r.m.psnrDb, 1)}</td><td align="right">${fmtMetric(r.m.vortexDensityBright * per, 0)}</td><td align="right">${r.it}</td><td align="right">${r.t}</td></tr>`;
    html += '</table>';
    $('metrics').innerHTML = html;
    const sp = s.sampling;
    $('sampling').textContent = `padded grid ${s.padded} px, window ${s.window.rows} × ${s.window.cols} px\nspot ${sp.spotSizePx.toFixed(2)} px, Fresnel number ${sp.fresnelNumber.toFixed(0)}, max angle ${(sp.maxAngleRad * 180 / Math.PI).toFixed(2)}°` + (sp.warnings.length ? '\nwarning: ' + sp.warnings.join('; ') : '');
    drawConvergence();
    $('saveDoev').disabled = false; $('savePhase').disabled = false;
  }
  function drawConvergence() {
    const c = $('convergence'), g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    if (!summary) return;
    const colors = ['#7fb0ff', '#ff8a80', '#8ce99a', '#d0a8ff'];
    const l = 36, r = 6, t = 12, b = 14, w = c.width - l - r, h = c.height - t - b;
    let lo = Infinity, hi = -Infinity, longest = 1;
    for (const run of summary.runs) for (const arr of [run.shapeHistory, run.efficiencyHistory]) { longest = Math.max(longest, arr.length); for (const v of arr) if (v > 0) { lo = Math.min(lo, Math.log10(v)); hi = Math.max(hi, Math.log10(v)); } }
    if (!Number.isFinite(lo)) return;
    if (hi - lo < 1e-9) hi = lo + 1;
    g.strokeStyle = '#3a3f48'; g.strokeRect(l, t, w, h);
    g.fillStyle = '#8b929c'; g.font = '10px system-ui'; g.textAlign = 'left'; g.fillText('shape (solid), efficiency (dashed) vs iteration', l + 2, t - 3);
    g.textAlign = 'right'; g.fillText(Math.pow(10, hi).toPrecision(2), l - 3, t + 9); g.fillText(Math.pow(10, lo).toPrecision(2), l - 3, t + h);
    summary.runs.forEach((run, ri) => {
      g.strokeStyle = colors[ri % colors.length]; g.lineWidth = 1.2;
      [[run.shapeHistory, []], [run.efficiencyHistory, [4, 3]]].forEach(([arr, dash]) => {
        g.setLineDash(dash); g.beginPath(); let started = false;
        for (let k = 0; k < arr.length; ++k) { if (!(arr[k] > 0)) continue; const x = l + (longest > 1 ? k / (longest - 1) : 0) * w, y = t + h - (Math.log10(arr[k]) - lo) / (hi - lo) * h; if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y); }
        g.stroke();
      });
      g.setLineDash([]); g.fillStyle = colors[ri % colors.length]; g.textAlign = 'left'; g.fillText(run.name, l + 6 + 60 * ri, t + 12);
    });
  }
  function download(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
  $('saveDoev').addEventListener('click', () => { if (vol) download(new Blob([writeDoev(vol)], { type: 'application/octet-stream' }), 'volume.doev'); });
  $('savePhase').addEventListener('click', () => {
    if (!vol) return;
    const n = vol.view, rgba = entryFaceRGBA(vol, 'phase'), c = document.createElement('canvas'); c.width = n; c.height = n;
    const g = c.getContext('2d'), img = g.createImageData(n, n);
    for (let y = 0; y < n; ++y) for (let x = 0; x < n; ++x) { const s = (y * n + x) * 4, d = ((n - 1 - y) * n + x) * 4; img.data[d] = rgba[s]; img.data[d + 1] = rgba[s + 1]; img.data[d + 2] = rgba[s + 2]; img.data[d + 3] = 255; }
    g.putImageData(img, 0, 0);
    c.toBlob((blob) => download(blob, 'phase.png'), 'image/png');
  });
  $('run').addEventListener('click', runDesign);
  $('cancel').addEventListener('click', () => { if (worker) { worker.terminate(); worker = null; } setRunning(false); setStatus('canceled', 0); });
  $('resweep').addEventListener('click', resweep);
  $('image').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) loadImageFile(f, f.name); });
  $('pActive').addEventListener('change', () => { $('pView').value = Math.min(parseInt($('pView').value, 10), parseInt($('pActive').value, 10)); });

  // ---- frame loop ---------------------------------------------------------------------------
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    resize();
    if (vol && anim) {
      anim.step(dt);
      const p = anim.plane();
      if ($('anim').checked) { timePhase = (timePhase + parseFloat($('omega').value) * dt) % (2 * Math.PI); st.timePhase = timePhase; if (st.mode === 'hsv' || st.mode === 'real') dirty = true; }
      if (p !== lastPlane || dirty) {
        updateTexture(tex.slice, sliceRGBA(vol, p, st));
        const d = depth(), z0 = vol.z[0], z1 = vol.z[vol.nz - 1];
        mesh.slice.position.z = -d / 2 + (z1 > z0 ? (vol.z[p] - z0) / (z1 - z0) : 0) * d;
        $('planeV').textContent = p; $('plane').value = p;
        $('zV').textContent = fmt(vol.z[p], 'mm', 1e3, 3);
        $('vortexV').textContent = `vortices in this plane: ${vol.vortexCount[p]} (${(vol.vortexDensity[p] * 1e-6).toFixed(0)} / mm²)`;
        drawDensity();
        lastPlane = p; dirty = false;
      }
      if (wallsDirty) { const wx = wallRGBA(vol, 'xz', st.floorDb), wy = wallRGBA(vol, 'yz', st.floorDb); updateTexture(tex.xz, wx.rgba); updateTexture(tex.yz, wy.rgba); wallsDirty = false; }
      if (facesDirty) { updateTexture(tex.entry, entryFaceRGBA(vol, faceChoice('entry'))); updateTexture(tex.exit, exitFaceRGBA(vol, faceChoice('exit'))); facesDirty = false; }
    }
    placeCamera();
    renderer.render(scene, camera);
    $('hud').textContent = vol ? '' : '';
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
