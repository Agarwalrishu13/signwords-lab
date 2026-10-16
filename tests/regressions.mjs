import assert from "node:assert/strict";
import test from "node:test";
import "fake-indexeddb/auto";
import * as tf from "@tensorflow/tfjs";
import { CONFIG } from "../js/config.js";
import { Sampler } from "../js/core/sampler.js";
import { SequenceBuffer, frameFeatures } from "../js/core/features.js";
import { splitSamples, classMetrics } from "../js/core/evaluation.js";
import * as store from "../js/core/dataset.js";
import { Recognizer } from "../js/core/recognizer.js";
import { trainModel } from "../js/core/model.js";

globalThis.tf = tf;
await tf.setBackend("cpu");
const vector = () => new Float32Array(CONFIG.SEQ_LEN * CONFIG.FEAT);
const hand = Array.from({ length: 21 }, (_, i) => ({ x: 0.4 + i * 0.01, y: 0.3 + i * 0.02, z: 0 }));
const samplerApp = () => ({ cam: { ready: true, paused: false }, buffer: new SequenceBuffer(), hands: { Right: hand }, handsAt: performance.now(), frameId: 1 });

test("capture cancellation settles immediately and prevents overlapping captures", async () => {
  const app = samplerApp(), sampler = new Sampler(app);
  sampler.start();
  const controller = new AbortController();
  const capture = sampler.captureClip({ signal: controller.signal });
  await assert.rejects(sampler.captureClip(), /already recording/);
  controller.abort();
  await assert.rejects(capture, { name: "AbortError" });
  assert.equal(sampler.capture, null);
  sampler.stop();
  await assert.rejects(sampler.captureClip(), /Start the camera/);
});

test("capture detects repeated frames and masks stale hands", async () => {
  const app = samplerApp(), sampler = new Sampler(app);
  sampler.start();
  const capture = sampler.captureClip();
  const started = sampler.capture.started;
  app.handsAt = started;
  for (let i = 1; i <= CONFIG.SEQ_LEN; i++) sampler._tick(started + i * CONFIG.FRAME_MS);
  const clip = await capture;
  assert.equal(clip.freshRatio, 1 / CONFIG.SEQ_LEN);
  assert.equal(clip.presence.at(-1), 0);
  assert.ok(clip.frames.at(-1).every((n) => n === 0));
  sampler.stop();
});

test("fresh camera frames make a complete clip; delayed sampling rejects it", async () => {
  const app = samplerApp(), sampler = new Sampler(app);
  sampler.start();
  const capture = sampler.captureClip();
  const started = sampler.capture.started;
  for (let i = 1; i <= CONFIG.SEQ_LEN; i++) {
    app.handsAt = started + i * CONFIG.FRAME_MS;
    app.frameId++;
    sampler._tick(app.handsAt);
  }
  const clip = await capture;
  assert.equal(clip.freshRatio, 1);
  assert.equal(clip.duration, 1200);
  const delayed = sampler.captureClip();
  sampler._tick(sampler.capture.started + CONFIG.CAPTURE_TIMEOUT_MS + 1);
  await assert.rejects(delayed, /fell behind/);
  sampler.stop();
});

test("degenerate and invalid landmarks become missing hands", () => {
  const degenerate = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  assert.equal(frameFeatures({ Right: degenerate }).present, 0);
  assert.equal(frameFeatures({ Right: [{ x: NaN }] }).present, 0);
});

test("session holdout has no session overlap and a reproducible split", () => {
  const samples = [];
  for (const session of ["one", "two", "three"]) for (const label of [CONFIG.IDLE, "hello"]) {
    for (let i = 0; i < 8; i++) samples.push({ label, session, data: vector() });
  }
  const a = splitSamples(samples, [CONFIG.IDLE, "hello"]);
  const b = splitSamples(samples, [CONFIG.IDLE, "hello"]);
  assert.equal(a.mode, "session");
  assert.deepEqual(a, b);
  const trainSessions = new Set(a.train.map((s) => s.session));
  assert.ok(a.val.every((s) => !trainSessions.has(s.session)));
  assert.ok([0, 1].every((y) => a.train.some((s) => s.y === y) && a.val.some((s) => s.y === y)));
  const fallback = splitSamples(samples.map((s) => ({ ...s, session: "one" })), [CONFIG.IDLE, "hello"]);
  assert.equal(fallback.mode, "clip");
  assert.match(fallback.note, /may overstate/);
});

test("precision and recall handle a class with no predictions", () => {
  const metrics = classMetrics([[8, 2, 0], [1, 4, 0], [0, 3, 0]]);
  assert.equal(metrics[0].precision, 8 / 9);
  assert.equal(metrics[0].recall, 0.8);
  assert.equal(metrics[2].precision, 0);
  assert.equal(metrics[2].f1, 0);
});

test("old browser data migrates without losing clips and vocabulary commits with dataset changes", async () => {
  const legacy = await new Promise((resolve, reject) => {
    const req = indexedDB.open(CONFIG.DB_NAME, 1);
    req.onupgradeneeded = () => {
      const samples = req.result.createObjectStore("samples", { keyPath: "id", autoIncrement: true });
      samples.createIndex("label", "label");
      samples.add({ label: "Hello", data: vector(), t: 123 });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  legacy.close();
  globalThis.localStorage = {
    getItem: (key) => key === CONFIG.WORDS_KEY ? '["Hello"]' : null,
    setItem: () => { throw new Error("LocalStorage unavailable"); },
  };
  assert.deepEqual(await store.loadVocabulary(), ["Hello"]);
  assert.equal((await store.getAllSamples()).length, 1);
  const revision = await store.saveWords(["Hello", "Water"]);
  const snapshot = await store.getSnapshot();
  assert.deepEqual(snapshot.words, ["Hello", "Water"]);
  assert.equal(snapshot.revision, revision);
  assert.equal(snapshot.samples.length, 1);
  await store.clearAll();
  const cleared = await store.getSnapshot();
  assert.deepEqual(cleared.words, []);
  assert.equal(cleared.samples.length, 0);
  delete globalThis.localStorage;
});

test("imports validate before writing, deduplicate, preserve sessions and canonicalize words", async () => {
  await store.clearAll();
  const obj = { app: "signwords-lab", version: 1, seqLen: CONFIG.SEQ_LEN, feat: CONFIG.FEAT,
    words: ["Hello"], samples: [{ label: "Hello", data: Array.from(vector()) }, { label: CONFIG.IDLE, data: Array.from(vector()) }] };
  const first = await store.importDataset(obj, ["hello"]);
  assert.equal(first.imported, 2);
  assert.deepEqual(first.words, ["hello"]);
  const revision = await store.getRevision();
  const second = await store.importDataset(obj, ["hello"]);
  assert.equal(second.imported, 0);
  assert.equal(second.skipped, 2);
  assert.equal(await store.getRevision(), revision);
  const malformed = structuredClone(obj);
  malformed.samples[1].data[0] = "not a number";
  await assert.rejects(store.importDataset(malformed), /invalid or out-of-range/);
  assert.equal((await store.getAllSamples()).length, 2);
  const orphan = structuredClone(obj);
  orphan.samples[1].label = "missing";
  await assert.rejects(store.importDataset(orphan), /missing from the vocabulary/);
  const session = "test-session";
  await store.addSample("hello", vector(), { session });
  const exported = await store.exportDataset(["hello"]);
  assert.equal(exported.version, 2);
  assert.equal(exported.samples.at(-1).session, session);
  assert.equal((await store.importDataset(exported, ["hello"])).imported, 0);
  assert.equal((await store.getAllSamples()).length, 3);
  await store.clearAll();
});

test("identical idle clips in one import retain their counts", async () => {
  const obj = { app: "signwords-lab", version: 1, seqLen: CONFIG.SEQ_LEN, feat: CONFIG.FEAT,
    words: [], samples: Array.from({ length: 8 }, () => ({ label: CONFIG.IDLE, data: Array.from(vector()) })) };
  assert.equal((await store.importDataset(obj)).imported, 8);
  assert.equal((await store.importDataset(obj)).imported, 0);
  assert.equal((await store.countByLabel())[CONFIG.IDLE], 8);
  await store.clearAll();
});

test("ambiguous predictions do not emit; repeated words require sustained idle", async () => {
  let probs = [0.05, 0.9, 0.05];
  const words = [];
  const r = new Recognizer({ onWord: (word) => words.push(word), onPrediction: () => {} });
  r.setModel({ predict: () => tf.tensor2d([probs]) }, [CONFIG.IDLE, "hello", "water"]);
  const b = new SequenceBuffer();
  for (let i = 0; i < CONFIG.SEQ_LEN; i++) b.push(new Float32Array(CONFIG.FEAT), 1);
  for (let i = 0; i < 3; i++) await r.update(b, i * 100);
  assert.deepEqual(words, ["hello"]);
  for (let i = 0; i < 5; i++) await r.update(b, 1000 + i * 100);
  assert.deepEqual(words, ["hello"]);
  probs = [0.55, 0.4, 0.05];
  await r.update(b, 2000); await r.update(b, 2100);
  probs = [0.05, 0.9, 0.05];
  for (let i = 0; i < 3; i++) await r.update(b, 2200 + i * 100);
  assert.equal(words.length, 1);
  probs = [0.9, 0.05, 0.05];
  await r.update(b, 3000); await r.update(b, 3100);
  probs = [0.05, 0.9, 0.05];
  for (let i = 0; i < 3; i++) await r.update(b, 3200 + i * 100);
  assert.deepEqual(words, ["hello", "hello"]);
  r.reset();
  r.threshold = 0.5;
  probs = [0.01, 0.51, 0.48];
  for (let i = 0; i < 4; i++) await r.update(b, 4000 + i * 100);
  assert.equal(words.length, 2);
  // No-hand windows bypass inference entirely.
  r.model = { predict: () => { throw new Error("Should not run"); } };
  b.presence.fill(0);
  await r.update(b, 5000);
});

test("pending predictions are discarded after stopping detection", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  let updates = 0;
  const baseline = tf.memory().numTensors;
  const r = new Recognizer({ onWord: () => {}, onPrediction: () => updates++ });
  r.setModel({ predict: () => {
    const t = tf.tensor2d([[0.1, 0.9]]), read = t.data.bind(t);
    t.data = async () => { await gate; return read(); };
    return t;
  } }, [CONFIG.IDLE, "hello"]);
  const b = new SequenceBuffer();
  for (let i = 0; i < CONFIG.SEQ_LEN; i++) b.push(new Float32Array(CONFIG.FEAT), 1);
  const pending = r.update(b);
  r.reset();
  release();
  await pending;
  assert.equal(updates, 0);
  assert.equal(tf.memory().numTensors, baseline);
});

test("stopping before the first completed epoch releases the unfinished model", async () => {
  const baseline = tf.memory().numTensors;
  const ctl = { stop: true, model: null };
  const samples = [CONFIG.IDLE, "hello"].flatMap((label) => [
    { label, data: vector() }, { label, data: vector() },
  ]);
  await assert.rejects(trainModel({ samples, labels: [CONFIG.IDLE, "hello"], epochs: 5,
    batchSize: 1, lr: 0.001, augmentCopies: 0, ctl }), { name: "AbortError" });
  assert.equal(ctl.model, null);
  assert.equal(tf.memory().numTensors, baseline);
});
