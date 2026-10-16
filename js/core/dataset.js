// Dataset storage. Clips live in IndexedDB (too big for localStorage);
// the vocabulary (list of words) lives in localStorage.
import { CONFIG } from "../config.js";

let dbPromise = null;
function db() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(CONFIG.DB_NAME, 1);
      req.onupgradeneeded = () => {
        const s = req.result.createObjectStore("samples", { keyPath: "id", autoIncrement: true });
        s.createIndex("label", "label");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}
const wrap = (r) =>
  new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
async function store(mode) {
  const d = await db();
  return d.transaction("samples", mode).objectStore("samples");
}

// ---- vocabulary --------------------------------------------------------
export function loadWords() {
  try {
    const w = JSON.parse(localStorage.getItem(CONFIG.WORDS_KEY) || "[]");
    return Array.isArray(w) ? w.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}
export function saveWords(words) {
  try {
    localStorage.setItem(CONFIG.WORDS_KEY, JSON.stringify(words));
  } catch {}
}

// ---- samples -----------------------------------------------------------
export async function addSample(label, data) {
  const s = await store("readwrite");
  await wrap(s.add({ label, data, t: Date.now() }));
}
export async function getAllSamples() {
  const s = await store("readonly");
  return wrap(s.getAll());
}
export async function countByLabel() {
  const all = await getAllSamples();
  const counts = {};
  for (const s of all) counts[s.label] = (counts[s.label] || 0) + 1;
  return counts;
}
export async function deleteLabel(label) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction("samples", "readwrite");
    const cur = t.objectStore("samples").index("label").openCursor(IDBKeyRange.only(label));
    cur.onsuccess = () => {
      const c = cur.result;
      if (c) {
        c.delete();
        c.continue();
      }
    };
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}
export async function clearAll() {
  const s = await store("readwrite");
  await wrap(s.clear());
}

// ---- import / export ---------------------------------------------------
export async function exportDataset(words) {
  const all = await getAllSamples();
  return {
    app: "signwords-lab",
    version: 1,
    seqLen: CONFIG.SEQ_LEN,
    feat: CONFIG.FEAT,
    words,
    samples: all.map((s) => ({
      label: s.label,
      data: Array.from(s.data, (v) => Math.round(v * 1e4) / 1e4),
    })),
  };
}

/** Merges an exported dataset into the DB. Returns the imported word list. */
export async function importDataset(obj) {
  if (!obj || obj.app !== "signwords-lab" || obj.seqLen !== CONFIG.SEQ_LEN || obj.feat !== CONFIG.FEAT) {
    throw new Error("This file isn't a compatible SignWords Lab export.");
  }
  const expected = CONFIG.SEQ_LEN * CONFIG.FEAT;
  const d = await db();
  await new Promise((resolve, reject) => {
    const t = d.transaction("samples", "readwrite");
    const s = t.objectStore("samples");
    for (const sample of obj.samples || []) {
      if (typeof sample.label !== "string" || !Array.isArray(sample.data) || sample.data.length !== expected) continue;
      s.add({ label: sample.label, data: new Float32Array(sample.data), t: Date.now() });
    }
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
  return (obj.words || []).filter((w) => typeof w === "string");
}
