# SignWords Lab

Train your own sign-language **word** recogniser in the browser, then use it to
detect signs, build a sentence and speak each word aloud. Nothing is uploaded:
hand tracking, training and detection all run locally.

The interface is designed for laptop screens, with a navigation rail, a large
camera workspace and an open vocabulary list. Native HTML and CSS share a cobalt
and neutral palette that follows your device's light or dark appearance. Use the
arrow keys, Home and End to navigate the three workflow tabs. Manrope and Phosphor
icons are served locally from `assets/`, with their licences included. The hand
photograph is decorative and disappears when the camera starts. No frontend build
step is required.

## Run it

```bash
python3 serve.py        # opens http://localhost:8000
```

Use Chrome or Edge. The camera and ES modules need `http://localhost`, so don't
open `index.html` by double-clicking it. An internet connection is needed once
per session to load TensorFlow.js, MediaPipe and the hand model from CDNs.

## How it works (the ML pipeline)

```
webcam ──► MediaPipe Hands ──► features ──► 24-frame clip ──► Conv1D network ──► word
           (21 landmarks       (wrist-relative,  (1.2 s window,   (trained here,    │
            × 2 hands)          scale-normalised   sampled every   in TensorFlow.js)▼
                                + wrist position)   50 ms)                    text + speech
```

1. **Collect** – add your own words and record ~30 clips of each. A clip is a 1.2 s
   sequence, so motion is learned, not just hand shape. A special *No sign (idle)*
   class teaches the model when you are not signing.
2. **Train** – a small 1D convolutional network (~80k parameters) is trained on
   your clips with data augmentation. You get live loss/accuracy curves, a training
   log and a confusion matrix on held-out clips. The model is saved in your browser.
3. **Detect & speak** – a sliding 1.2 s window is classified every 150 ms. A word is
   accepted after 3 confident predictions in a row, added to the sentence **followed
   by a space**, and spoken with text-to-speech. Return your hands to rest between
   repeats of the same word.

## Project structure

```
signwords-lab/
├── index.html            page shell (3 tabs: Collect · Train · Detect)
├── serve.py              local dev server
├── css/styles.css
├── assets/               self-hosted typeface, icons and their licences
├── js/
│   ├── main.js           entry point, shared `app` state, camera wiring
│   ├── config.js         all tunables (sequence length, thresholds, storage keys…)
│   ├── core/
│   │   ├── camera.js     webcam + MediaPipe hand tracking
│   │   ├── features.js   landmarks → feature vectors, rolling sequence buffer
│   │   ├── sampler.js    fixed-rate sampling + clip capture
│   │   ├── dataset.js    IndexedDB storage, import/export of clips
│   │   ├── model.js      architecture, augmentation, training, evaluation, save/load
│   │   ├── recognizer.js prediction stream → "word detected" events
│   │   └── speech.js     text-to-speech and UI chimes
│   └── ui/
│       ├── tabs.js  dataPanel.js  trainPanel.js  livePanel.js
│       ├── chart.js  (learning curves)   toast.js
└── tests/smoke_train.mjs  trains the real model on synthetic data in Node
```

## Tips for good accuracy

- 30+ clips per word, with small variations in speed, position and distance.
- Record 10+ *idle* clips (hands resting, hands leaving the frame).
- Keep good, even lighting and a plain background.
- Words that differ only slightly in motion need more clips.
- If two words get confused, check the confusion matrix and record more of both.
- Changing the word list makes the saved model outdated, so retrain.
- Use **Export dataset** to back up your clips or move them to another computer.

## Tests (optional)

```bash
npm install
npm test     # builds the model, trains it on synthetic signs, checks it learns
```
