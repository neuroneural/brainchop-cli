export async function runCandidateGate(config) {
  const NVOX = 256 ** 3;

  async function inputData() {
    const compressed = await (await fetch('/input.nii.gz')).arrayBuffer();
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
    const nii = new Uint8Array(await new Response(stream).arrayBuffer());
    const raw = nii.subarray(352, 352 + NVOX);
    const output = new Float32Array(NVOX);
    let lo = 255, hi = 0;
    for (const value of raw) { if (value < lo) lo = value; if (value > hi) hi = value; }

    if (config.normalization === 'quantile') {
      const sample = [];
      const stride = Math.max(1, Math.floor(raw.length / 100000));
      for (let i = 0; i < raw.length && sample.length < 100000; i += stride) sample.push(raw[i]);
      sample.sort((a, b) => a - b);
      lo = sample[Math.floor(sample.length * 0.05)];
      hi = sample[Math.ceil(sample.length * 0.95) - 1];
    }

    const scale = hi === lo ? 1 : hi - lo;
    for (let i = 0; i < raw.length; i++) output[i] = (raw[i] - lo) / scale;
    return output;
  }

  async function device() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const requiredLimits = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    };
    return adapter.requestDevice({ requiredLimits, requiredFeatures: ['shader-f16'] });
  }

  async function one(moduleUrl, weightsUrl, input) {
    const runner = await import(moduleUrl);
    const setup = runner.setupNet || runner.default?.setupNet;
    if (typeof setup !== 'function') throw new Error(`${moduleUrl} does not export setupNet`);
    const weights = new Uint8Array(await (await fetch(weightsUrl)).arrayBuffer());
    const gpu = await device();
    const allocations = [];
    let liveAllocation = 0, peakLiveAllocation = 0;
    const createBuffer = gpu.createBuffer.bind(gpu);
    gpu.createBuffer = (descriptor) => {
      const size = Number(descriptor.size);
      allocations.push(size);
      liveAllocation += size;
      peakLiveAllocation = Math.max(peakLiveAllocation, liveAllocation);
      const buffer = createBuffer(descriptor);
      const destroy = buffer.destroy.bind(buffer);
      let destroyed = false;
      try {
        buffer.destroy = () => {
          if (!destroyed) { liveAllocation -= size; destroyed = true; }
          return destroy();
        };
      } catch {
        // A non-extensible browser GPUBuffer makes this a conservative peak.
      }
      return buffer;
    };
    gpu.pushErrorScope('validation');
    gpu.pushErrorScope('out-of-memory');
    const setupStart = performance.now();
    const execute = await setup(gpu, weights, () => {});
    const setupMs = performance.now() - setupStart;
    const firstStart = performance.now();
    const output = (await execute(input))[0];
    await gpu.queue.onSubmittedWorkDone();
    const firstMs = performance.now() - firstStart;
    let warmMs = null;
    if (config.warm) {
      const warmStart = performance.now();
      await execute(input);
      await gpu.queue.onSubmittedWorkDone();
      warmMs = performance.now() - warmStart;
    }
    const oom = await gpu.popErrorScope();
    const validation = await gpu.popErrorScope();
    if (oom || validation) throw new Error((oom || validation).message);
    gpu.destroy();
    return {
      output,
      elapsedMs: setupMs + firstMs,
      setupMs,
      firstMs,
      warmMs,
      allocationCount: allocations.length,
      cumulativeAllocationMiB: allocations.reduce((a, b) => a + b, 0) / 2**20,
      peakLiveAllocationMiB: peakLiveAllocation / 2**20,
      largestBufferMiB: Math.max(...allocations) / 2**20,
    };
  }

  const input = await inputData();
  const current = await one('/current.js', '/current.safetensors', input);
  const candidate = await one('/candidate.js', '/candidate.safetensors', input);
  let differing = 0, currentSum = 0, candidateSum = 0;
  for (let i = 0; i < current.output.length; i++) {
    currentSum += current.output[i];
    candidateSum += candidate.output[i];
    if (current.output[i] !== candidate.output[i]) differing++;
  }
  delete current.output;
  delete candidate.output;
  return { differing, currentSum, candidateSum, current, candidate };
}
