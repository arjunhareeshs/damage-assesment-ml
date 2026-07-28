// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
/* global GPUShaderStage, GPUBufferUsage, GPUMapMode */
//
// WebGPU-accelerated logistic regression for the interactive labeler.
//
// The in-browser damage model trains on per-building MOSAIKS features and
// predicts for every building. This module runs the hot loops — full-batch
// gradient-descent training and dense inference — as WebGPU compute shaders.
// Multiclass is handled one-vs-rest (a binary model per class, arg-max at the
// end), which also keeps the math robust to any label set.
//
// A one-time self-test validates the pipeline against a known result; if
// WebGPU is unavailable, unsupported, or the self-test fails, getGpu() returns
// null and the caller falls back to the CPU implementation.

// ── Shaders ─────────────────────────────────────────────────────────────────
// Training: forward+error, gradient (+ bias reduction), weight update. Features
// are standardized in-shader with the training mean/std (passed as buffers).
const TRAIN_WGSL = /* wgsl */ `
struct Params { n: u32, d: u32, m: u32, lr: f32, lambda: f32 };
@group(0) @binding(0) var<storage, read>        Xtr:  array<f32>;
@group(0) @binding(1) var<storage, read>        Ytr:  array<f32>;
@group(0) @binding(2) var<storage, read>        meanv: array<f32>;
@group(0) @binding(3) var<storage, read>        stdv:  array<f32>;
@group(0) @binding(4) var<storage, read_write>  w:    array<f32>;
@group(0) @binding(5) var<storage, read_write>  grad: array<f32>;
@group(0) @binding(6) var<storage, read_write>  errv: array<f32>;
@group(0) @binding(7) var<storage, read_write>  scal: array<f32>;   // [b, db]
@group(0) @binding(8) var<uniform>              P: Params;

@compute @workgroup_size(64)
fn forwardErr(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  var z = scal[0];
  for (var j: u32 = 0u; j < P.d; j = j + 1u) {
    let xs = (Xtr[i * P.d + j] - meanv[j]) / stdv[j];
    z = z + w[j] * xs;
  }
  let p = 1.0 / (1.0 + exp(-z));
  errv[i] = p - Ytr[i];
}

@compute @workgroup_size(64)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d) { return; }
  var s = 0.0;
  for (var i: u32 = 0u; i < P.n; i = i + 1u) {
    let xs = (Xtr[i * P.d + j] - meanv[j]) / stdv[j];
    s = s + errv[i] * xs;
  }
  grad[j] = s / f32(P.n) + P.lambda * w[j];
  if (j == 0u) {
    var sb = 0.0;
    for (var i: u32 = 0u; i < P.n; i = i + 1u) { sb = sb + errv[i]; }
    scal[1] = sb / f32(P.n);
  }
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= P.d) { return; }
  w[j] = w[j] - P.lr * grad[j];
  if (j == 0u) { scal[0] = scal[0] - P.lr * scal[1]; }
}
`;

const PREDICT_WGSL = /* wgsl */ `
struct Params { n: u32, d: u32, m: u32, lr: f32, lambda: f32 };
@group(0) @binding(0) var<storage, read>        Xq:    array<f32>;
@group(0) @binding(1) var<storage, read>        meanv: array<f32>;
@group(0) @binding(2) var<storage, read>        stdv:  array<f32>;
@group(0) @binding(3) var<storage, read>        w:     array<f32>;
@group(0) @binding(4) var<storage, read>        scal:  array<f32>;
@group(0) @binding(5) var<storage, read_write>  outp:  array<f32>;
@group(0) @binding(6) var<uniform>              P: Params;

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.m) { return; }
  var z = scal[0];
  for (var j: u32 = 0u; j < P.d; j = j + 1u) {
    let xs = (Xq[i * P.d + j] - meanv[j]) / stdv[j];
    z = z + w[j] * xs;
  }
  outp[i] = 1.0 / (1.0 + exp(-z));
}
`;

// ── Device acquisition (cached) ─────────────────────────────────────────────
let _gpuState; // undefined = not tried; null = unavailable; object = ready

async function initGpu() {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const ctx = buildContext(device);
    // Validate the whole pipeline once against a known-separable problem.
    const ok = await selfTest(ctx);
    return ok ? ctx : null;
  } catch (e) {
    console.warn("WebGPU init failed; using CPU logistic regression.", e);
    return null;
  }
}

// Returns a ready GPU context, or null if WebGPU can't be used. Cached.
export async function getGpu() {
  if (_gpuState === undefined) _gpuState = await initGpu();
  return _gpuState;
}

export function gpuBackendName() {
  return _gpuState ? "WebGPU" : "CPU";
}

function buildContext(device) {
  const trainModule = device.createShaderModule({ code: TRAIN_WGSL });
  const predModule = device.createShaderModule({ code: PREDICT_WGSL });

  const sStorage = "storage";
  const sRead = "read-only-storage";
  const trainLayout = device.createBindGroupLayout({
    entries: [
      [0, sRead],
      [1, sRead],
      [2, sRead],
      [3, sRead],
      [4, sStorage],
      [5, sStorage],
      [6, sStorage],
      [7, sStorage],
    ].map(([binding, type]) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    })).concat([
      {
        binding: 8,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ]),
  });
  const predLayout = device.createBindGroupLayout({
    entries: [
      [0, sRead],
      [1, sRead],
      [2, sRead],
      [3, sRead],
      [4, sRead],
      [5, sStorage],
    ].map(([binding, type]) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    })).concat([
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ]),
  });

  const trainPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [trainLayout],
  });
  const predPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [predLayout],
  });

  const mkTrain = (entryPoint) =>
    device.createComputePipeline({
      layout: trainPipelineLayout,
      compute: { module: trainModule, entryPoint },
    });

  return {
    device,
    trainLayout,
    predLayout,
    forwardErr: mkTrain("forwardErr"),
    gradient: mkTrain("gradient"),
    update: mkTrain("update"),
    predict: device.createComputePipeline({
      layout: predPipelineLayout,
      compute: { module: predModule, entryPoint: "predict" },
    }),
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function mkBuffer(device, data, usage) {
  const buf = device.createBuffer({ size: data.byteLength, usage });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function paramsBuffer(device, n, d, m, lr, lambda) {
  const ab = new ArrayBuffer(32); // 16-byte aligned uniform
  const dv = new DataView(ab);
  dv.setUint32(0, n, true);
  dv.setUint32(4, d, true);
  dv.setUint32(8, m, true);
  dv.setFloat32(12, lr, true);
  dv.setFloat32(16, lambda, true);
  return mkBuffer(
    device,
    new Uint8Array(ab),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  );
}

function standardizeStats(Xtr, n, d) {
  const mean = new Float32Array(d);
  const std = new Float32Array(d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) mean[j] += Xtr[i * d + j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      const diff = Xtr[i * d + j] - mean[j];
      std[j] += diff * diff;
    }
  for (let j = 0; j < d; j++) {
    std[j] = Math.sqrt(std[j] / n);
    if (std[j] === 0) std[j] = 1; // avoid divide-by-zero on constant features
  }
  return { mean, std };
}

function flatten(rows, n, d) {
  const out = new Float32Array(n * d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) out[i * d + j] = rows[i][j];
  return out;
}

// ── One-vs-rest predict on the GPU ──────────────────────────────────────────
// entries: [{features:number[], label:number}]; classes: sorted unique labels;
// queryMatrix: number[][]. Returns Int array of predicted class labels.
export async function gpuOvrPredict(
  ctx,
  entries,
  classes,
  queryMatrix,
  opts = { learningRate: 0.1, numSteps: 500, lambda: 0.01 }
) {
  const { device } = ctx;
  const n = entries.length;
  const d = entries[0].features.length;
  const m = queryMatrix.length;
  const U = GPUBufferUsage;

  const XtrFlat = flatten(entries.map((e) => e.features), n, d);
  const XqFlat = flatten(queryMatrix, m, d);
  const { mean, std } = standardizeStats(XtrFlat, n, d);

  const Xtr = mkBuffer(device, XtrFlat, U.STORAGE | U.COPY_DST);
  const Xq = mkBuffer(device, XqFlat, U.STORAGE | U.COPY_DST);
  const meanB = mkBuffer(device, mean, U.STORAGE | U.COPY_DST);
  const stdB = mkBuffer(device, std, U.STORAGE | U.COPY_DST);
  const Ytr = device.createBuffer({ size: n * 4, usage: U.STORAGE | U.COPY_DST });
  const w = device.createBuffer({ size: d * 4, usage: U.STORAGE | U.COPY_DST });
  const grad = device.createBuffer({ size: d * 4, usage: U.STORAGE });
  const errv = device.createBuffer({ size: n * 4, usage: U.STORAGE });
  const scal = device.createBuffer({ size: 8, usage: U.STORAGE | U.COPY_DST });
  const outp = device.createBuffer({
    size: m * 4,
    usage: U.STORAGE | U.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: m * 4,
    usage: U.MAP_READ | U.COPY_DST,
  });
  const params = paramsBuffer(
    device,
    n,
    d,
    m,
    opts.learningRate,
    opts.lambda
  );

  const trainBind = device.createBindGroup({
    layout: ctx.trainLayout,
    entries: [
      [0, Xtr],
      [1, Ytr],
      [2, meanB],
      [3, stdB],
      [4, w],
      [5, grad],
      [6, errv],
      [7, scal],
      [8, params],
    ].map(([binding, buffer]) => ({ binding, resource: { buffer } })),
  });
  const predBind = device.createBindGroup({
    layout: ctx.predLayout,
    entries: [
      [0, Xq],
      [1, meanB],
      [2, stdB],
      [3, w],
      [4, scal],
      [5, outp],
      [6, params],
    ].map(([binding, buffer]) => ({ binding, resource: { buffer } })),
  });

  const wgN = Math.ceil(n / 64);
  const wgD = Math.ceil(d / 64);
  const wgM = Math.ceil(m / 64);
  const zerosD = new Float32Array(d);
  const zeros2 = new Float32Array(2);

  const probas = [];
  for (const c of classes) {
    // Per-class targets + fresh weights.
    const yc = new Float32Array(n);
    for (let i = 0; i < n; i++) yc[i] = entries[i].label === c ? 1 : 0;
    device.queue.writeBuffer(Ytr, 0, yc);
    device.queue.writeBuffer(w, 0, zerosD);
    device.queue.writeBuffer(scal, 0, zeros2);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    // Training: dispatches within one pass run in order with barriers between.
    for (let step = 0; step < opts.numSteps; step++) {
      pass.setPipeline(ctx.forwardErr);
      pass.setBindGroup(0, trainBind);
      pass.dispatchWorkgroups(wgN);
      pass.setPipeline(ctx.gradient);
      pass.dispatchWorkgroups(wgD);
      pass.setPipeline(ctx.update);
      pass.dispatchWorkgroups(wgD);
    }
    // Inference for the whole query set.
    pass.setPipeline(ctx.predict);
    pass.setBindGroup(0, predBind);
    pass.dispatchWorkgroups(wgM);
    pass.end();
    enc.copyBufferToBuffer(outp, 0, staging, 0, m * 4);
    device.queue.submit([enc.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    probas.push(new Float32Array(staging.getMappedRange()).slice());
    staging.unmap();
  }

  // Arg-max across the per-class probabilities.
  const preds = new Array(m);
  for (let i = 0; i < m; i++) {
    let best = 0;
    let bestP = -Infinity;
    for (let k = 0; k < classes.length; k++) {
      if (probas[k][i] > bestP) {
        bestP = probas[k][i];
        best = k;
      }
    }
    preds[i] = classes[best];
  }

  [Xtr, Xq, meanB, stdB, Ytr, w, grad, errv, scal, outp, staging, params].forEach(
    (b) => b.destroy()
  );
  return preds;
}

// ── Self-test: train on a trivially separable 2-class problem ───────────────
async function selfTest(ctx) {
  try {
    const entries = [
      { features: [-1, -1], label: 0 },
      { features: [-0.9, -1.1], label: 0 },
      { features: [-1.1, -0.9], label: 0 },
      { features: [1, 1], label: 1 },
      { features: [0.9, 1.1], label: 1 },
      { features: [1.1, 0.9], label: 1 },
    ];
    const query = [
      [-1, -1],
      [1, 1],
    ];
    const preds = await gpuOvrPredict(ctx, entries, [0, 1], query, {
      learningRate: 0.5,
      numSteps: 200,
      lambda: 0.0,
    });
    return preds[0] === 0 && preds[1] === 1;
  } catch (e) {
    console.warn("WebGPU self-test failed; using CPU.", e);
    return false;
  }
}
