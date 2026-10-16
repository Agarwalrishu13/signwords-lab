// Turns a stream of per-window predictions into discrete "word detected" events.
import { CONFIG } from "../config.js";
import { predictProbs } from "./model.js";

export class Recognizer {
  constructor({ onWord, onPrediction }) {
    this.onWord = onWord;
    this.onPrediction = onPrediction;
    this.model = null;
    this.labels = [];
    this.threshold = 0.8;
    this.reset();
  }

  setModel(model, labels) {
    this.model = model;
    this.labels = labels;
    this.reset();
  }

  reset() {
    this.candidate = null;
    this.streak = 0;
    this.armed = true;
    this.lastLabel = null;
    this.lastEmit = 0;
  }

  /** Call every PREDICT_MS with the live SequenceBuffer. */
  update(buffer, now = performance.now()) {
    if (!this.model || !buffer.ready) return;

    let probs = predictProbs(this.model, buffer.flatten());
    let bi = probs.indexOf(Math.max(...probs));
    let label = this.labels[bi];
    let conf = probs[bi];

    // No hands in (almost) the whole window: definitely idle, skip the model's opinion.
    if (buffer.handRatio() < CONFIG.NO_HAND_RATIO) {
      const idleIdx = this.labels.indexOf(CONFIG.IDLE);
      probs = probs.map((_, i) => (i === idleIdx ? 1 : 0));
      label = CONFIG.IDLE;
      conf = 1;
    }

    this.onPrediction({ probs, label, conf, labels: this.labels });

    if (label === CONFIG.IDLE) {
      this.armed = true; // back to rest: the same word may be signed again
      this.candidate = null;
      this.streak = 0;
      return;
    }

    if (conf >= this.threshold) {
      if (this.candidate === label) this.streak++;
      else {
        this.candidate = label;
        this.streak = 1;
      }
    } else {
      this.candidate = null;
      this.streak = 0;
    }

    const canEmit = this.armed || label !== this.lastLabel;
    if (this.streak >= CONFIG.CONFIRM_COUNT && canEmit && now - this.lastEmit >= CONFIG.COOLDOWN_MS) {
      this.armed = false;
      this.lastLabel = label;
      this.lastEmit = now;
      this.streak = 0;
      this.onWord(label, conf);
    }
  }
}
