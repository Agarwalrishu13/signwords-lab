// Step 2: model architecture, training controls, live learning curves, log, confusion matrix.
import { CONFIG } from "../config.js";
import * as store from "../core/dataset.js";
import { previewArchitecture, trainModel, saveModel } from "../core/model.js";
import { LineChart } from "./chart.js";
import { toast } from "./toast.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nameOf = (l) => (l === CONFIG.IDLE ? CONFIG.IDLE_NAME : l);
const fmtTime = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
};

export function initTrainPanel(app) {
  const lossChart = new LineChart($("chartLoss"), {
    title: "Loss",
    series: [
      { key: "loss", label: "train", color: "--chart-train" },
      { key: "valLoss", label: "validation", color: "--chart-validation" },
    ],
  });
  const accChart = new LineChart($("chartAcc"), {
    title: "Accuracy",
    yMin: 0,
    yMax: 1,
    series: [
      { key: "acc", label: "train", color: "--chart-train" },
      { key: "valAcc", label: "validation", color: "--chart-validation" },
    ],
  });
  let ctl = null;

  function log(msg, cls = "") {
    const d = document.createElement("div");
    d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (cls) d.className = cls;
    $("log").appendChild(d);
    $("log").scrollTop = $("log").scrollHeight;
  }

  const classes = () => [CONFIG.IDLE, ...app.words];

  function refresh() {
    const total = Object.values(app.counts).reduce((a, b) => a + b, 0);
    $("sumClasses").textContent = classes().length;
    $("sumSamples").textContent = total;

    const low = app.words.filter((w) => (app.counts[w] || 0) < CONFIG.MIN_SAMPLES);
    let warn = "";
    if (!app.words.length) warn = "Add at least one word on the Collect tab.";
    else if (low.length) warn = `Needs at least ${CONFIG.MIN_SAMPLES} clips: ${low.join(", ")}.`;
    else if (app.words.some((w) => (app.counts[w] || 0) < CONFIG.TARGET_SAMPLES))
      warn = `Tip: ${CONFIG.TARGET_SAMPLES}+ clips per word gives noticeably better accuracy.`;
    $("sumWarn").textContent = warn;

    if (typeof tf !== "undefined") {
      try {
        const a = previewArchitecture(classes().length);
        $("archTotal").textContent = `${a.total.toLocaleString()} parameters`;
        $("archList").innerHTML =
          `<div class="layer"><b>input</b><span>Input</span><span>[${CONFIG.SEQ_LEN}, ${CONFIG.FEAT}]</span><span>2 hands × 65 features × ${CONFIG.SEQ_LEN} frames</span></div>` +
          a.layers
            .map((l) => `<div class="layer"><b>${esc(l.name)}</b><span>${esc(l.type)}</span><span>${esc(l.shape)}</span><span>${l.params.toLocaleString()} params</span></div>`)
            .join("");
      } catch (e) {
        $("archList").textContent = "Couldn't build preview: " + e.message;
      }
    } else {
      $("archList").textContent = "TensorFlow.js failed to load (check your internet connection).";
    }

    if (!app.model) $("modelBadge").textContent = "none";
    else if (app.modelCurrent()) $("modelBadge").textContent = `${Math.round(app.meta.valAcc * 100)}% val acc`;
    else $("modelBadge").textContent = "outdated";
  }

  function renderConfusion(res) {
    if (!res.hasVal) {
      $("confusion").innerHTML = `<p class="hint">Not enough clips for a held-out validation set (need 5+ per class).</p>`;
      return;
    }
    const L = res.labels;
    let html = "<table><tr><th></th>" + L.map((l) => `<th>${esc(nameOf(l))}</th>`).join("") + "</tr>";
    res.confusion.forEach((row, i) => {
      const total = row.reduce((a, b) => a + b, 0);
      html += `<tr><th>${esc(nameOf(L[i]))}</th>` + row
        .map((v, j) => {
          const a = total ? v / total : 0;
          const bg = v ? `color-mix(in srgb, var(${i === j ? "--accent" : "--bad"}) ${Math.round(10 + 20 * a)}%, var(--card))` : "transparent";
          return `<td style="background:${bg}">${v}</td>`;
        })
        .join("") + "</tr>";
    });
    $("confusion").innerHTML = html + "</table>";
  }

  function setRunning(on) {
    $("trainBtn").disabled = on;
    $("stopBtn").disabled = !on;
    for (const id of ["epochs", "batch", "lr", "augment"]) $(id).disabled = on;
  }

  $("stopBtn").onclick = () => {
    if (ctl) {
      ctl.stop = true;
      $("stopBtn").disabled = true;
      log("Stop requested…");
    }
  };

  $("trainBtn").onclick = async () => {
    if (typeof tf === "undefined") return toast("TensorFlow.js didn't load. Check your connection.");
    const words = app.words;
    if (!words.length) return toast("Add at least one word first.");
    const low = words.filter((w) => (app.counts[w] || 0) < CONFIG.MIN_SAMPLES);
    if (low.length) return toast(`Record at least ${CONFIG.MIN_SAMPLES} clips for: ${low.join(", ")}`);

    setRunning(true);
    $("log").innerHTML = "";
    $("trainBar").style.width = "0";
    lossChart.set({});
    accChart.set({});
    ["mEpoch", "mLoss", "mAcc", "mValAcc", "mEta"].forEach((id) => ($(id).textContent = "-"));
    $("confusion").innerHTML = `<p class="hint">Training…</p>`;

    const epochs = Math.max(5, Math.min(300, parseInt($("epochs").value, 10) || 40));
    const batchSize = parseInt($("batch").value, 10);
    const lr = parseFloat($("lr").value);
    const augmentCopies = parseInt($("augment").value, 10);
    ctl = { stop: false, model: null };

    try {
      await tf.ready();
      log(`TensorFlow.js ${tf.version.tfjs} · backend: ${tf.getBackend()}`);
      const labels = classes();
      let samples = await store.getAllSamples();
      log(`Loaded ${samples.length} clips from the database for ${labels.length} classes`);

      // If the user recorded no idle clips, synthesise "no hands" clips so the model has a rest class.
      const idleCount = samples.filter((s) => s.label === CONFIG.IDLE).length;
      if (idleCount < 10) {
        const need = 16 - idleCount;
        for (let i = 0; i < need; i++) samples.push({ label: CONFIG.IDLE, data: new Float32Array(CONFIG.SEQ_LEN * CONFIG.FEAT) });
        log(`Added ${need} synthetic "no hands" clips for the idle class`);
      }

      let epochStart = performance.now();
      const result = await trainModel({
        samples, labels, epochs, batchSize, lr, augmentCopies, ctl,
        onLog: log,
        onBatch: ({ batch, nBatches }) => {
          const done = (ctl._epochDone || 0) + batch / nBatches;
          $("trainBar").style.width = `${Math.min(100, (done / epochs) * 100)}%`;
        },
        onEpoch: ({ epoch, history, elapsed }) => {
          ctl._epochDone = epoch;
          const i = history.loss.length - 1;
          lossChart.set(history);
          accChart.set(history);
          $("mEpoch").textContent = `${epoch}/${epochs}`;
          $("mLoss").textContent = history.loss[i].toFixed(4);
          $("mAcc").textContent = `${(history.acc[i] * 100).toFixed(1)}%`;
          $("mValAcc").textContent = history.valAcc[i] != null ? `${(history.valAcc[i] * 100).toFixed(1)}%` : "-";
          $("mEta").textContent = fmtTime((elapsed / epoch) * (epochs - epoch));
          log(
            `Epoch ${String(epoch).padStart(String(epochs).length, " ")}/${epochs}  loss ${history.loss[i].toFixed(4)}  acc ${(history.acc[i] * 100).toFixed(1)}%` +
              (history.valLoss[i] != null ? `  val_loss ${history.valLoss[i].toFixed(4)}  val_acc ${(history.valAcc[i] * 100).toFixed(1)}%` : "")
          );
          epochStart = performance.now();
        },
      });

      $("trainBar").style.width = "100%";
      const stopped = ctl.stop;
      log(`${stopped ? "Stopped early" : "Finished"} after ${result.epochsRun} epochs. ${result.hasVal ? "Validation" : "Training"} accuracy: ${(result.valAcc * 100).toFixed(1)}%`, "good");
      renderConfusion(result);

      const meta = { labels: result.labels, valAcc: result.valAcc, trainedAt: Date.now(), epochs: result.epochsRun, seqLen: CONFIG.SEQ_LEN };
      await saveModel(result.model, meta);
      log("Model saved to browser storage (IndexedDB). It will be here next time you open the app.", "good");

      const old = app.model;
      app.model = result.model;
      app.meta = meta;
      if (old && old !== result.model) old.dispose();
      app.emit("modelChanged");
      toast(`Model ready · ${(result.valAcc * 100).toFixed(0)}% accuracy`);
    } catch (err) {
      console.error(err);
      log("Training failed: " + (err.message || err), "bad");
      toast("Training failed. See the log.");
    } finally {
      ctl = null;
      setRunning(false);
      refresh();
    }
  };

  app.on("dataChanged", refresh);
  app.on("modelChanged", refresh);
  app.on("tab", (e) => {
    if (e.detail === "train") {
      refresh();
      lossChart.draw();
      accChart.draw();
    }
  });
  refresh();
}
