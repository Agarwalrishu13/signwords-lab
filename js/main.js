// Entry point: creates the shared `app` object and wires every module together.
import { CONFIG } from "./config.js";
import { Camera, cameraErrorMessage } from "./core/camera.js";
import { SequenceBuffer } from "./core/features.js";
import { Sampler } from "./core/sampler.js";
import { Speech } from "./core/speech.js";
import { loadSavedModel } from "./core/model.js";
import * as store from "./core/dataset.js";
import { initTabs } from "./ui/tabs.js";
import { initDataPanel } from "./ui/dataPanel.js";
import { initTrainPanel } from "./ui/trainPanel.js";
import { initLivePanel } from "./ui/livePanel.js";
import { toast } from "./ui/toast.js";

const $ = (id) => document.getElementById(id);
const bus = new EventTarget();

const app = {
  cam: new Camera($("video"), $("canvas")),
  buffer: new SequenceBuffer(),
  hands: { Right: null, Left: null },
  speech: new Speech(),
  words: store.loadWords(),
  counts: {},
  model: null,
  meta: null,
  on: (name, fn) => bus.addEventListener(name, fn),
  emit: (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail })),
  modelCurrent() {
    return !!this.model && !!this.meta && JSON.stringify(this.meta.labels) === JSON.stringify([CONFIG.IDLE, ...this.words]);
  },
};
app.sampler = new Sampler(app);

function refreshSteps() {
  const total = Object.values(app.counts).reduce((a, b) => a + b, 0);
  $("stepDataInfo").textContent = `${app.words.length} word${app.words.length === 1 ? "" : "s"} · ${total} clips`;
  if (!app.model) $("stepTrainInfo").textContent = "not trained";
  else if (app.modelCurrent()) $("stepTrainInfo").textContent = `trained · ${Math.round(app.meta.valAcc * 100)}% acc`;
  else $("stepTrainInfo").textContent = "retrain needed";
}

// camera -> shared hand state + header badges
app.cam.onFrame = (hands, fps) => {
  app.hands = hands;
  $("fpsBadge").textContent = `${fps} fps`;
  const n = (hands.Right ? 1 : 0) + (hands.Left ? 1 : 0);
  $("handBadge").textContent = n ? `${n} hand${n > 1 ? "s" : ""}` : "no hands";
  $("handBadge").classList.toggle("on", n > 0);
  $("bufferBar").style.width = `${app.buffer.fill * 100}%`;
};

$("startBtn").onclick = async () => {
  $("startBtn").disabled = true;
  const stage = $("video").closest(".stage");
  stage.classList.add("is-starting");
  stage.setAttribute("aria-busy", "true");
  try {
    await app.cam.start((msg) => ($("camStatus").textContent = msg));
    $("camStatus").textContent = "Camera is on. Show your hands.";
    stage.classList.add("is-ready");
    $("startBtn").textContent = "Camera running";
    app.emit("camera");
  } catch (err) {
    console.error(err);
    $("startBtn").disabled = false;
    $("camStatus").textContent = cameraErrorMessage(err);
  } finally {
    stage.classList.remove("is-starting");
    stage.setAttribute("aria-busy", "false");
  }
};

(async function boot() {
  initTabs(app);
  try {
    app.counts = await store.countByLabel();
  } catch (e) {
    console.error(e);
    toast("Couldn't open the local database.");
  }
  initDataPanel(app);
  initTrainPanel(app);
  initLivePanel(app);
  app.sampler.start();

  app.on("dataChanged", refreshSteps);
  app.on("modelChanged", refreshSteps);
  refreshSteps();

  // Restore a previously trained model, if there is one.
  if (typeof tf !== "undefined") {
    const saved = await loadSavedModel();
    if (saved) {
      app.model = saved.model;
      app.meta = saved.meta;
      app.emit("modelChanged");
    }
  }
})();
