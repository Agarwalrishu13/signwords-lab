// Smoke test (no browser needed): builds the real model, trains it on synthetic
// "signs" and checks it learns. Run:  npm install && npm test
import * as tf from "@tensorflow/tfjs";
import assert from "node:assert/strict";

globalThis.tf = tf;
await tf.setBackend("cpu");

const { CONFIG } = await import("../js/config.js");
const { trainModel, predictProbs, augment, previewArchitecture } = await import("../js/core/model.js");
const { frameFeatures, SequenceBuffer, flattenFrames } = await import("../js/core/features.js");

const { SEQ_LEN, FEAT, SLOT } = CONFIG;

// --- features -----------------------------------------------------------------
const fakeHand = (dx = 0) =>
  Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + dx + i * 0.01, y: 0.5 + i * 0.02, z: 0 }));
const { features, present } = frameFeatures({ Right: fakeHand(), Left: null });
assert.equal(features.length, FEAT);
assert.equal(present, 1);
assert.ok(features.slice(SLOT).every((v) => v === 0), "missing hand must be all zeros");
const buf = new SequenceBuffer();
for (let i = 0; i < SEQ_LEN + 5; i++) buf.push(features, present);
assert.ok(buf.ready && buf.frames.length === SEQ_LEN);
assert.equal(buf.flatten().length, SEQ_LEN * FEAT);
assert.equal(buf.handRatio(), 1);

// --- architecture ---------------------------------------------------------------
const arch = previewArchitecture(4);
console.log("layers:", arch.layers.map((l) => `${l.name}${l.shape}`).join(" → "), "|", arch.total, "params");
assert.equal(arch.layers[arch.layers.length - 1].shape, "[·,4]");

// --- synthetic dataset: 3 words with different hand shapes/motions + idle ----------
function makeClip(cls) {
  const frames = [];
  for (let f = 0; f < SEQ_LEN; f++) {
    if (cls === 0) { frames.push(new Float32Array(FEAT)); continue; } // idle = no hands
    const dx = cls === 1 ? 0 : cls === 2 ? (f / SEQ_LEN) * 0.4 - 0.2 : 0.15 * Math.sin(f / 3);
    const spread = cls === 3 ? 0.03 : 0.012;
    const hand = Array.from({ length: 21 }, (_, i) => ({
      x: 0.5 + dx + i * spread + (Math.random() - 0.5) * 0.004,
      y: 0.5 + i * 0.015 + (Math.random() - 0.5) * 0.004,
      z: 0,
    }));
    frames.push(frameFeatures({ Right: hand, Left: null }).features);
  }
  return flattenFrames(frames);
}
const labels = [CONFIG.IDLE, "one", "two", "three"];
const samples = [];
for (let c = 0; c < labels.length; c++) for (let i = 0; i < 24; i++) samples.push({ label: labels[c], data: makeClip(c) });

// --- augmentation keeps shape and leaves absent hands at zero --------------------
const aug = augment(samples[30].data);
assert.equal(aug.length, SEQ_LEN * FEAT);
assert.ok(aug.slice(SLOT, FEAT).every((v) => v === 0), "augment must not invent a second hand");

// --- real training ----------------------------------------------------------------
const epochsSeen = [];
const res = await trainModel({
  samples, labels, epochs: 12, batchSize: 16, lr: 0.003, augmentCopies: 2,
  ctl: { stop: false },
  onEpoch: (e) => epochsSeen.push(e.epoch),
  onLog: (m) => console.log("  log:", m),
});
console.log("epochs:", res.epochsRun, "val acc:", res.valAcc.toFixed(2));
assert.equal(epochsSeen.length, 12);
assert.ok(res.history.loss[11] < res.history.loss[0], "loss should go down");
assert.ok(res.valAcc >= 0.8, `validation accuracy too low: ${res.valAcc}`);

// --- inference ----------------------------------------------------------------------
const probs = predictProbs(res.model, makeClip(2));
console.log("probs for 'two':", probs.map((p) => p.toFixed(2)).join(" "));
assert.equal(probs.indexOf(Math.max(...probs)), 2);

// --- stop control -------------------------------------------------------------------
const ctl = { stop: false };
const early = await trainModel({
  samples, labels, epochs: 30, batchSize: 16, lr: 0.003, augmentCopies: 0, ctl,
  onEpoch: (e) => { if (e.epoch === 2) ctl.stop = true; },
});
assert.ok(early.epochsRun < 30, "stop flag should end training early");

console.log("\nAll smoke tests passed.");
