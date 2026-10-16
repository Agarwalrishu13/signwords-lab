// The neural network: architecture, augmentation, training, evaluation, persistence.
// TensorFlow.js is loaded as the global `tf` (see index.html).
import { CONFIG } from "../config.js";

const { SEQ_LEN, FEAT, SLOT } = CONFIG;

// ---------------------------------------------------------------------------
// Architecture: a small 1D convolutional network over the time axis.
//   input [24 frames, 130 features]
//   -> Conv1D(64) -> MaxPool -> Conv1D(96) -> MaxPool -> Flatten
//   -> Dropout -> Dense(64) -> Dense(num_classes, softmax)
// ---------------------------------------------------------------------------
export function buildModel(numClasses, lr = 0.001) {
  const m = tf.sequential({ name: "sign_word_net" });
  m.add(tf.layers.conv1d({ inputShape: [SEQ_LEN, FEAT], filters: 64, kernelSize: 3, padding: "same", activation: "relu", name: "conv_1" }));
  m.add(tf.layers.maxPooling1d({ poolSize: 2, name: "pool_1" }));
  m.add(tf.layers.conv1d({ filters: 96, kernelSize: 3, padding: "same", activation: "relu", name: "conv_2" }));
  m.add(tf.layers.maxPooling1d({ poolSize: 2, name: "pool_2" }));
  m.add(tf.layers.flatten({ name: "flatten" }));
  m.add(tf.layers.dropout({ rate: 0.3, name: "dropout" }));
  m.add(tf.layers.dense({ units: 64, activation: "relu", name: "dense_1" }));
  m.add(tf.layers.dense({ units: numClasses, activation: "softmax", name: "output" }));
  m.compile({ optimizer: tf.train.adam(lr), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  return m;
}

export function describeModel(model) {
  const layers = model.layers.map((l) => ({
    name: l.name,
    type: l.getClassName(),
    shape: JSON.stringify(l.outputShape).replace(/null/g, "·"),
    params: l.countParams(),
  }));
  return { layers, total: layers.reduce((a, l) => a + l.params, 0) };
}

/** Build a throw-away model just to show its structure in the UI. */
export function previewArchitecture(numClasses) {
  const m = buildModel(Math.max(2, numClasses));
  const d = describeModel(m);
  m.dispose();
  return d;
}

// ---------------------------------------------------------------------------
// Data augmentation (training split only): time shift, per-frame jitter and a
// whole-clip position offset, so the model tolerates where/when you sign.
// ---------------------------------------------------------------------------
function gauss() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function augment(src) {
  const out = new Float32Array(src.length);
  const shift = Math.floor(Math.random() * 5) - 2;
  const noise = 0.005 + Math.random() * 0.015;
  const dx = (Math.random() - 0.5) * 0.16;
  const dy = (Math.random() - 0.5) * 0.16;
  for (let f = 0; f < SEQ_LEN; f++) {
    const sf = Math.min(SEQ_LEN - 1, Math.max(0, f + shift));
    for (let s = 0; s < 2; s++) {
      const so = sf * FEAT + s * SLOT;
      const o = f * FEAT + s * SLOT;
      const present = Math.abs(src[so + 3]) + Math.abs(src[so + 4]) > 0;
      if (!present) continue; // keep absent hands as exact zeros
      for (let k = 0; k < SLOT; k++) out[o + k] = src[so + k] + gauss() * noise;
      out[o + 63] += dx;
      out[o + 64] += dy;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function toX(items) {
  const buf = new Float32Array(items.length * SEQ_LEN * FEAT);
  items.forEach((it, i) => buf.set(it.x, i * SEQ_LEN * FEAT));
  return tf.tensor3d(buf, [items.length, SEQ_LEN, FEAT]);
}
function toY(items, nClasses) {
  return tf.tidy(() => tf.cast(tf.oneHot(tf.tensor1d(items.map((i) => i.y), "int32"), nClasses), "float32"));
}

/**
 * samples: [{label, data}], labels: ordered class names.
 * ctl: {stop:boolean} — set ctl.stop = true to end training after the current batch.
 */
export async function trainModel({ samples, labels, epochs, batchSize, lr, augmentCopies, ctl, onEpoch, onBatch, onLog }) {
  const log = onLog || (() => {});
  const idx = new Map(labels.map((l, i) => [l, i]));
  const byClass = labels.map(() => []);
  for (const s of samples) {
    const i = idx.get(s.label);
    if (i !== undefined) byClass[i].push(s.data instanceof Float32Array ? s.data : new Float32Array(s.data));
  }

  const train = [];
  const val = [];
  byClass.forEach((arr, ci) => {
    shuffle(arr);
    const nVal = arr.length >= 5 ? Math.max(1, Math.round(arr.length * 0.2)) : 0;
    arr.forEach((x, i) => (i < nVal ? val : train).push({ x, y: ci }));
  });
  const trainAug = [...train];
  for (let k = 0; k < augmentCopies; k++) for (const t of train) trainAug.push({ x: augment(t.x), y: t.y });
  shuffle(trainAug);
  const hasVal = val.length > 0;
  log(`Split: ${train.length} train clips → ${trainAug.length} after augmentation, ${val.length} validation clips`);

  const xs = toX(trainAug);
  const ys = toY(trainAug, labels.length);
  const xv = hasVal ? toX(val) : null;
  const yv = hasVal ? toY(val, labels.length) : null;

  const model = buildModel(labels.length, lr);
  const arch = describeModel(model);
  log(`Model built: ${arch.layers.length} layers, ${arch.total.toLocaleString()} trainable parameters`);
  ctl.model = model;

  const nBatches = Math.ceil(trainAug.length / batchSize);
  const history = { loss: [], acc: [], valLoss: [], valAcc: [] };
  const t0 = performance.now();

  try {
    await model.fit(xs, ys, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: hasVal ? [xv, yv] : undefined,
      callbacks: {
        onBatchEnd: async (b, logs) => {
          if (ctl.stop) model.stopTraining = true;
          if (onBatch) onBatch({ batch: b + 1, nBatches, logs });
          await tf.nextFrame();
        },
        onEpochEnd: async (e, logs) => {
          const acc = logs.acc ?? logs.accuracy;
          const vAcc = logs.val_acc ?? logs.val_accuracy;
          history.loss.push(logs.loss);
          history.acc.push(acc);
          history.valLoss.push(logs.val_loss ?? null);
          history.valAcc.push(vAcc ?? null);
          if (onEpoch) onEpoch({ epoch: e + 1, epochs, history, elapsed: performance.now() - t0 });
          await tf.nextFrame();
        },
      },
    });

    // Evaluation: confusion matrix on the held-out clips.
    const confusion = labels.map(() => labels.map(() => 0));
    let valAcc = history.acc[history.acc.length - 1] ?? 0;
    if (hasVal) {
      const pred = tf.tidy(() => tf.argMax(model.predict(xv), -1).dataSync());
      let correct = 0;
      val.forEach((v, i) => {
        confusion[v.y][pred[i]]++;
        if (pred[i] === v.y) correct++;
      });
      valAcc = correct / val.length;
    }
    return { model, labels, history, valAcc, hasVal, confusion, epochsRun: history.loss.length };
  } finally {
    xs.dispose();
    ys.dispose();
    if (xv) xv.dispose();
    if (yv) yv.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inference + persistence
// ---------------------------------------------------------------------------
export function predictProbs(model, flat) {
  const x = tf.tensor3d(flat, [1, SEQ_LEN, FEAT]);
  const y = model.predict(x);
  const p = Array.from(y.dataSync());
  x.dispose();
  y.dispose();
  return p;
}

export async function saveModel(model, meta) {
  await model.save(CONFIG.MODEL_URL);
  localStorage.setItem(CONFIG.META_KEY, JSON.stringify(meta));
}

export async function loadSavedModel() {
  try {
    const raw = localStorage.getItem(CONFIG.META_KEY);
    if (!raw) return null;
    const meta = JSON.parse(raw);
    const model = await tf.loadLayersModel(CONFIG.MODEL_URL);
    return { model, meta };
  } catch {
    return null;
  }
}
