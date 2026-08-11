#!/usr/bin/env node
import { createRequire } from 'node:module';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((value, index, all) =>
  value.startsWith('--') ? [value.slice(2), all[index + 1]] : null).filter(Boolean));
for (const required of ['bct', 'current-runner', 'current-weights', 'candidate-runner', 'candidate-weights', 'input']) {
  if (!args[required]) throw new Error(`missing --${required}`);
}

const require = createRequire(pathToFileURL(path.join(path.resolve(args.bct), 'package.json')));
const { chromium } = require('@playwright/test');
const routes = new Map([
  ['/gate.js', path.join(here, 'validate_webgpu_candidate_page.js')],
  ['/current.js', path.resolve(args['current-runner'])],
  ['/current.safetensors', path.resolve(args['current-weights'])],
  ['/candidate.js', path.resolve(args['candidate-runner'])],
  ['/candidate.safetensors', path.resolve(args['candidate-weights'])],
  ['/input.nii.gz', path.resolve(args.input)],
]);
const pageHtml = `<!doctype html><script type="module">
import { runCandidateGate } from '/gate.js'; window.runCandidateGate = runCandidateGate;
</script>`;
const server = http.createServer((request, response) => {
  const url = request.url.split('?')[0];
  if (url === '/') { response.setHeader('content-type', 'text/html'); response.end(pageHtml); return; }
  const file = routes.get(url);
  if (!file) { response.statusCode = 404; response.end(); return; }
  fs.readFile(file, (error, data) => {
    if (error) { response.statusCode = 500; response.end(error.message); return; }
    response.setHeader('content-type', url.endsWith('.js') ? 'text/javascript' : 'application/octet-stream');
    response.end(data);
  });
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;

const executablePath = process.env.BC_CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({
  executablePath,
  args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage();
  page.on('pageerror', error => console.error('[pageerror]', error.message));
  await page.goto(`http://127.0.0.1:${port}/`);
  const config = {
    normalization: args.normalization || 'minmax',
    warm: args.warm === undefined ? true : Number(args.warm) !== 0,
  };
  const result = await page.evaluate(async gateConfig =>
    window.runCandidateGate(gateConfig), config);
  console.log(JSON.stringify(result, null, 2));
  const maxDiff = Number(args['max-diff'] || 0);
  process.exitCode = result.differing <= maxDiff ? 0 : 1;
} finally {
  await browser.close();
  server.close();
}
