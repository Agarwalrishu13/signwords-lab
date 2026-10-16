// Turns MediaPipe hand landmarks into the numeric features the model learns from.
import { CONFIG } from "../config.js";

const { SLOT, FEAT, SEQ_LEN } = CONFIG;
export const SLOTS = ["Right", "Left"];

/**
 * One frame -> Float32Array(FEAT).
 * Per hand: 21 landmarks translated to the wrist and scaled by palm size
 * (so hand size / distance doesn't matter), plus the wrist's position in the
 * image (so hand *movement* is visible across frames). Missing hand = zeros.
 */
export function frameFeatures(hands) {
  const out = new Float32Array(FEAT);
  let present = 0;
  SLOTS.forEach((name, s) => {
    const lm = hands[name];
    if (!lm) return;
    present++;
    const o = s * SLOT;
    const w = lm[0];
    const scale = Math.hypot(lm[9].x - w.x, lm[9].y - w.y, lm[9].z - w.z) || 1;
    for (let i = 0; i < 21; i++) {
      out[o + i * 3] = (lm[i].x - w.x) / scale;
      out[o + i * 3 + 1] = (lm[i].y - w.y) / scale;
      out[o + i * 3 + 2] = (lm[i].z - w.z) / scale;
    }
    out[o + 63] = (w.x - 0.5) * 2;
    out[o + 64] = (w.y - 0.5) * 2;
  });
  return { features: out, present };
}

/** Flatten an array of frame vectors into one clip vector (SEQ_LEN * FEAT). */
export function flattenFrames(frames) {
  const out = new Float32Array(SEQ_LEN * FEAT);
  for (let i = 0; i < Math.min(frames.length, SEQ_LEN); i++) out.set(frames[i], i * FEAT);
  return out;
}

/** Rolling window of the most recent SEQ_LEN frames. */
export class SequenceBuffer {
  constructor() {
    this.frames = [];
    this.presence = [];
  }
  push(features, present) {
    this.frames.push(features);
    this.presence.push(present > 0 ? 1 : 0);
    if (this.frames.length > SEQ_LEN) {
      this.frames.shift();
      this.presence.shift();
    }
  }
  get ready() {
    return this.frames.length === SEQ_LEN;
  }
  get fill() {
    return this.frames.length / SEQ_LEN;
  }
  handRatio() {
    if (!this.presence.length) return 0;
    return this.presence.reduce((a, b) => a + b, 0) / this.presence.length;
  }
  flatten() {
    return flattenFrames(this.frames);
  }
  clear() {
    this.frames = [];
    this.presence = [];
  }
}
