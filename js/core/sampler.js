// Samples the latest hand state at a fixed rate into the rolling buffer and,
// when asked, captures one fixed-length clip for training data.
import { CONFIG } from "../config.js";
import { frameFeatures } from "./features.js";

export class Sampler {
  constructor(app) {
    this.app = app;
    this.capture = null;
    this.timer = null;
  }

  start() {
    if (this.timer) return;
    this.timer = setInterval(() => this._tick(), CONFIG.FRAME_MS);
  }

  _tick() {
    const { features, present } = frameFeatures(this.app.hands);
    this.app.buffer.push(features, present);
    const c = this.capture;
    if (c) {
      c.frames.push(features);
      c.presence.push(present > 0 ? 1 : 0);
      if (c.frames.length >= CONFIG.SEQ_LEN) {
        this.capture = null;
        c.resolve({ frames: c.frames, presence: c.presence });
      }
    }
  }

  /** Resolves with {frames, presence} after SEQ_LEN more frames have been sampled. */
  captureClip() {
    return new Promise((resolve) => {
      this.capture = { frames: [], presence: [], resolve };
    });
  }
}
