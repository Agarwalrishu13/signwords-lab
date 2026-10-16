// Step 1: manage the vocabulary and record training clips.
import { CONFIG } from "../config.js";
import * as store from "../core/dataset.js";
import { flattenFrames } from "../core/features.js";
import { toast } from "./toast.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function initDataPanel(app) {
  let selected = CONFIG.IDLE;
  let busy = false;
  let cancel = false;

  const labels = () => [CONFIG.IDLE, ...app.words];
  const nameOf = (l) => (l === CONFIG.IDLE ? CONFIG.IDLE_NAME : l);
  const count = (l) => app.counts[l] || 0;

  function overlay(big, small = "", go = false) {
    const el = $("overlayMsg");
    if (!big) {
      el.className = "overlay-msg";
      el.innerHTML = "";
      return;
    }
    el.className = "overlay-msg show" + (go ? " go" : "");
    el.innerHTML = `<div>${esc(big)}</div>${small ? `<small>${esc(small)}</small>` : ""}`;
  }

  function render() {
    if (!labels().includes(selected)) selected = CONFIG.IDLE;
    $("wordList").innerHTML = labels()
      .map((l) => {
        const n = count(l);
        const pct = Math.min(100, (n / CONFIG.TARGET_SAMPLES) * 100);
        const state = n >= CONFIG.TARGET_SAMPLES ? "good" : n >= CONFIG.MIN_SAMPLES ? "ok" : "low";
        return `<div class="wcard ${l === selected ? "sel" : ""} ${state}" data-label="${esc(l)}">
          <button class="word-select" data-act="select" aria-pressed="${l === selected}"><b>${esc(nameOf(l))}</b><span class="cnt">${n}/${CONFIG.TARGET_SAMPLES} clips</span></button>
          <div class="bar"><div style="width:${pct}%"></div></div>
          <div class="wact">
            <button data-act="clear" ${busy ? "disabled" : ""}>Clear clips</button>
            ${l !== CONFIG.IDLE ? `<button data-act="del" ${busy ? "disabled" : ""}>Delete</button>` : ""}
            ${l === selected ? `<span class="word-selection-state">Selected</span>` : ""}
          </div></div>`;
      })
      .join("");
    $("selLabel").textContent = nameOf(selected);
    $("vocabularyEmpty").hidden = app.words.length > 0;
    const total = Object.values(app.counts).reduce((a, b) => a + b, 0);
    $("dataTotal").textContent = `${total} clips`;
    const ready = app.cam.ready && !busy;
    $("rec1Btn").disabled = !ready;
    $("rec10Btn").disabled = !ready;
    $("recCancelBtn").hidden = !busy;
    $("addWordBtn").disabled = busy;
  }

  // ---- vocabulary -------------------------------------------------------
  function addWord() {
    const w = $("wordInput").value.trim().replace(/\s+/g, " ");
    if (!w) return;
    if (w === CONFIG.IDLE || w.toLowerCase() === CONFIG.IDLE_NAME.toLowerCase()) return toast("That name is reserved.");
    if (app.words.some((x) => x.toLowerCase() === w.toLowerCase())) return toast("That word already exists.");
    app.words.push(w);
    store.saveWords(app.words);
    selected = w;
    $("wordInput").value = "";
    render();
    app.emit("dataChanged");
  }
  $("addWordBtn").onclick = addWord;
  $("wordInput").addEventListener("keydown", (e) => e.key === "Enter" && addWord());

  $("wordList").addEventListener("click", async (e) => {
    const card = e.target.closest(".wcard");
    if (!card) return;
    const label = card.dataset.label;
    const act = e.target.dataset.act;
    if (act === "clear" || act === "del") {
      if (busy) return;
      const msg =
        act === "del" ? `Delete “${nameOf(label)}” and all its clips?` : `Delete all clips for “${nameOf(label)}”?`;
      if (!confirm(msg)) return;
      await store.deleteLabel(label);
      delete app.counts[label];
      if (act === "del") {
        app.words = app.words.filter((w) => w !== label);
        store.saveWords(app.words);
      }
      render();
      app.emit("dataChanged");
      return;
    }
    selected = label;
    render();
  });

  // ---- recording --------------------------------------------------------
  async function recordClips(n) {
    if (!app.cam.ready) return toast("Start the camera first.");
    if (busy) return;
    busy = true;
    cancel = false;
    render();
    const label = selected;
    let saved = 0;
    let skipped = 0;
    try {
      for (let i = 0; i < n && !cancel; i++) {
        const sub = n > 1 ? `Clip ${i + 1} of ${n}` : "";
        for (let s = i === 0 ? 3 : 1; s >= 1 && !cancel; s--) {
          overlay(String(s), sub);
          app.speech.chime("tick");
          await sleep(800);
        }
        if (cancel) break;
        overlay("GO!", `Sign “${nameOf(label)}”`, true);
        app.speech.chime("go");
        const clip = await app.sampler.captureClip();
        overlay("");
        const ratio = clip.presence.reduce((a, b) => a + b, 0) / clip.presence.length;
        if (label !== CONFIG.IDLE && ratio < CONFIG.MIN_HAND_RATIO) {
          skipped++;
          app.speech.chime("error");
          toast("Hands weren't visible long enough. Clip skipped.");
          continue;
        }
        await store.addSample(label, flattenFrames(clip.frames));
        app.counts[label] = count(label) + 1;
        saved++;
        render();
        app.emit("dataChanged");
      }
    } finally {
      overlay("");
      busy = false;
      render();
      toast(`${saved} clip${saved === 1 ? "" : "s"} saved for “${nameOf(label)}”` + (skipped ? ` (${skipped} skipped)` : ""));
    }
  }
  $("rec1Btn").onclick = () => recordClips(1);
  $("rec10Btn").onclick = () => recordClips(10);
  $("recCancelBtn").onclick = () => (cancel = true);

  // ---- dataset import / export -------------------------------------------
  $("exportBtn").onclick = async () => {
    const data = await store.exportDataset(app.words);
    const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "signwords-dataset.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("importBtn").onclick = () => $("importFile").click();
  $("importFile").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      const words = await store.importDataset(JSON.parse(await file.text()));
      for (const w of words) if (!app.words.some((x) => x.toLowerCase() === w.toLowerCase())) app.words.push(w);
      store.saveWords(app.words);
      app.counts = await store.countByLabel();
      render();
      app.emit("dataChanged");
      toast("Dataset imported.");
    } catch (err) {
      toast(err.message || "Couldn't read that file.");
    }
  };
  $("clearAllBtn").onclick = async () => {
    if (busy || !confirm("Delete ALL words and clips? This can't be undone.")) return;
    await store.clearAll();
    app.words = [];
    store.saveWords([]);
    app.counts = {};
    render();
    app.emit("dataChanged");
  };

  app.on("camera", render);
  app.on("dataChanged", render);
  render();
}
