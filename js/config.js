// Central configuration. Change values here, not inside the modules.
export const CONFIG = {
  // --- Sequence / feature layout ---------------------------------------
  SEQ_LEN: 24,          // frames per clip (24 frames x 50 ms = 1.2 s)
  FRAME_MS: 50,         // sampling interval for the sequence buffer
  SLOT: 65,             // features per hand: 21 landmarks x 3 + wrist x,y
  FEAT: 130,            // two hands (Right + Left) x SLOT

  // --- Labels -----------------------------------------------------------
  IDLE: "__idle__",     // reserved class: "not signing"
  IDLE_NAME: "No sign (idle)",

  // --- Data collection --------------------------------------------------
  TARGET_SAMPLES: 30,   // recommended clips per word
  MIN_SAMPLES: 8,       // hard minimum to allow training
  MIN_HAND_RATIO: 0.4,  // clip is rejected if hands are visible in fewer frames

  // --- Live detection ---------------------------------------------------
  PREDICT_MS: 150,      // how often the model runs while detecting
  CONFIRM_COUNT: 3,     // consecutive confident predictions needed to emit a word
  COOLDOWN_MS: 1000,    // minimum time between two emitted words
  NO_HAND_RATIO: 0.15,  // below this the window is treated as "idle"

  // --- Storage ----------------------------------------------------------
  DB_NAME: "signwords-lab",
  MODEL_URL: "indexeddb://signwords-model",
  META_KEY: "signwords-model-meta",
  WORDS_KEY: "signwords-words",

  // --- External assets (loaded from CDN) --------------------------------
  MP_VERSION: "0.10.14",
  HAND_MODEL_URL:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
