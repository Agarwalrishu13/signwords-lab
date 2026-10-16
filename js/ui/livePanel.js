// Step 3: run the trained model on the camera, build the sentence, speak each word.
import { CONFIG } from "../config.js";
import { Recognizer } from "../core/recognizer.js";
import { toast } from "./toast.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nameOf = (l) => (l === CONFIG.IDLE ? CONFIG.IDLE_NAME : l);

export function initLivePanel(app) {
  let running = false;
  const sentence = []; // detected words; each is followed by a space when rendered

  const recognizer = new Recognizer({
    onWord(word) {
      sentence.push(word);
      renderSentence();
      app.speech.chime("word");
      app.speech.speak(word); // spoken right after being detected (when "Speak each word" is on)
    },
    onPrediction({ probs, label, conf, labels }) {
      $("liveWord").textContent = label === CONFIG.IDLE ? "-" : label;
      $("liveConf").textContent = label === CONFIG.IDLE ? "waiting for a sign…" : `${Math.round(conf * 100)}% sure`;
      const rows = labels
        .map((l, i) => ({ l, p: probs[i] }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 6);
      $("probList").innerHTML = rows
        .map((r) => `<div class="prob"><span>${esc(nameOf(r.l))}</span><div class="bar"><div style="width:${(r.p * 100).toFixed(0)}%"></div></div><em>${Math.round(r.p * 100)}%</em></div>`)
        .join("");
    },
  });

  function renderSentence() {
    const el = $("transcript");
    if (!sentence.length) {
      el.innerHTML = `<span class="muted">Words you sign appear here, each followed by a space.</span>`;
      return;
    }
    el.textContent = sentence.map((w) => w + " ").join("");
  }

  function refreshStatus() {
    const st = $("modelStatus");
    if (!app.model) st.textContent = "no model yet. Train one first.";
    else if (!app.modelCurrent()) st.textContent = "words changed since training. Retrain.";
    else st.textContent = `${app.meta.labels.length - 1} words · ${Math.round(app.meta.valAcc * 100)}% val accuracy`;
    if (app.modelCurrent()) recognizer.setModel(app.model, app.meta.labels);
  }

  function setRunning(on) {
    running = on;
    $("liveToggle").textContent = on ? "Stop detecting" : "Start detecting";
    $("liveToggle").classList.toggle("recording", on);
    $("stepLiveInfo").textContent = on ? "detecting…" : "stopped";
    if (!on) {
      $("liveWord").textContent = "-";
      $("liveConf").textContent = "stopped";
      $("probList").innerHTML = "";
    } else recognizer.reset();
  }

  $("liveToggle").onclick = () => {
    if (running) return setRunning(false);
    if (!app.cam.ready) return toast("Start the camera first.");
    if (!app.model) return toast("Train a model first (step 2).");
    if (!app.modelCurrent()) return toast("Your words changed. Retrain the model first.");
    recognizer.setModel(app.model, app.meta.labels);
    setRunning(true);
  };

  setInterval(() => {
    if (running) recognizer.update(app.buffer);
  }, CONFIG.PREDICT_MS);

  // ---- sentence buttons -----------------------------------------------------
  $("speakAllBtn").onclick = () => {
    if (!sentence.length) return;
    app.speech.stop();
    app.speech.speak(sentence.join(" "), { force: true });
  };
  $("undoBtn").onclick = () => {
    sentence.pop();
    renderSentence();
  };
  $("clearTextBtn").onclick = () => {
    sentence.length = 0;
    renderSentence();
  };
  $("copyBtn").onclick = async () => {
    try {
      await navigator.clipboard.writeText(sentence.map((w) => w + " ").join(""));
      toast("Copied.");
    } catch {
      toast("Couldn't copy.");
    }
  };

  // ---- settings ---------------------------------------------------------------
  $("threshold").oninput = (e) => {
    recognizer.threshold = e.target.value / 100;
    $("thresholdVal").textContent = e.target.value + "%";
  };
  recognizer.threshold = $("threshold").value / 100;

  app.speech.init($("voiceSelect"));
  $("voiceSelect").onchange = (e) => app.speech.setVoice(e.target.value);
  $("rate").oninput = (e) => {
    app.speech.rate = parseFloat(e.target.value);
    $("rateVal").textContent = parseFloat(e.target.value).toFixed(1) + "×";
  };
  $("volume").oninput = (e) => (app.speech.volume = parseFloat(e.target.value));
  $("autoSpeak").onchange = (e) => (app.speech.enabled = e.target.checked);
  $("chimeOn").onchange = (e) => (app.speech.chimeOn = e.target.checked);
  $("testVoiceBtn").onclick = () => app.speech.speak("Hello, this is your sign language voice.", { force: true });

  app.on("modelChanged", refreshStatus);
  app.on("dataChanged", () => {
    refreshStatus();
    if (running && !app.modelCurrent()) setRunning(false);
  });
  app.on("tab", (e) => e.detail === "live" && refreshStatus());
  refreshStatus();
}
