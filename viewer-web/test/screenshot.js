#!/usr/bin/env node
// Development helper: drive headless Chrome over the DevTools protocol to open the web viewer,
// wait for a condition (e.g. a finished design) and save a screenshot. Serve the repository
// first (python3 -m http.server 8000) so that ?image= / ?file= URLs resolve. Node 22+ (built-in
// WebSocket), Google Chrome on macOS; adjust the path for other systems.
//   node viewer-web/test/screenshot.js "http://localhost:8000/viewer-web/index.html?image=../images/targets/cross_ring.png&run=1&active=128&iters=60" out.png "document.getElementById('status').textContent === 'done'" 120000
// A sixth argument "SELECTOR=/abs/path" sets a file input after the page has loaded (the file
// picker path, which also works for a file:// page where ?image= cannot):
//   node viewer-web/test/screenshot.js "file:///.../viewer-web/index.html?run=1&active=128" out.png "..." 120000 1500,1000 "#image=/abs/images/targets/logo.png"
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const [url, out, cond, timeoutArg, sizeArg, fileInputArg] = process.argv.slice(2);
const timeout = parseInt(timeoutArg || '60000', 10), size = sizeArg || '1500,1000';
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars', '--remote-debugging-port=' + (process.env.CDP_PORT || 9333),
   '--window-size=' + size, '--user-data-dir=/tmp/cdp-profile-' + process.pid, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  let targets;
  for (let i = 0; i < 50; ++i) { try { targets = await (await fetch('http://127.0.0.1:' + (process.env.CDP_PORT || 9333) + '/json')).json(); if (targets.length) break; } catch (e) { /* not up yet */ } await sleep(200); }
  const page = targets.find((t) => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => { ws.onopen = r; });
  let id = 0; const pending = new Map();
  const send = (method, params) => new Promise((resolve) => { const i = ++id; pending.set(i, resolve); ws.send(JSON.stringify({ id: i, method, params: params || {} })); });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result || m); pending.delete(m.id); return; }
    if (m.method === 'Runtime.consoleAPICalled') console.log('console.' + m.params.type + ':', m.params.args.map((a) => a.value ?? a.description).join(' '));
    if (m.method === 'Runtime.exceptionThrown') console.log('EXCEPTION:', m.params.exceptionDetails.text, m.params.exceptionDetails.exception && m.params.exceptionDetails.exception.description);
  };
  await send('Runtime.enable'); await send('Page.enable');
  await send('DOM.enable');
  await send('Page.navigate', { url });
  if (fileInputArg) {
    const [selector, file] = fileInputArg.split('=');
    // wait until the page's scripts have run (the core is loaded and the input exists)
    for (let i = 0; i < 200; ++i) {
      const r = await send('Runtime.evaluate', { expression: `!!(window.DOE_CORE && window.DOE_CORE.design && document.querySelector(${JSON.stringify(selector)}))`, returnByValue: true });
      if (r.result && r.result.value === true) break;
      await sleep(250);
    }
    const doc = await send('DOM.getDocument', { depth: 1 });
    const node = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector });
    await send('DOM.setFileInputFiles', { nodeId: node.nodeId, files: [file] });
    console.log('set', selector, 'to', file);
  }
  const t0 = Date.now(); let ok = false;
  while (Date.now() - t0 < timeout) {
    const r = await send('Runtime.evaluate', { expression: '(function(){ try { return !!(' + cond + '); } catch (e) { return false; } })()', returnByValue: true });
    if (r.result && r.result.value === true) { ok = true; break; }
    await sleep(250);
  }
  await sleep(600);   // let the last frame render
  const status = await send('Runtime.evaluate', { expression: "document.getElementById('status') ? document.getElementById('status').textContent : ''", returnByValue: true });
  console.log('condition', ok ? 'met' : 'TIMED OUT', 'after', ((Date.now() - t0) / 1000).toFixed(1), 's; status:', status.result && status.result.value);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  ws.close(); chrome.kill();
  process.exit(ok ? 0 : 1);
})().catch((e) => { console.error(e); chrome.kill(); process.exit(2); });
