// Reproducible dataset splits and class metrics, independent of TensorFlow.
export function random(seed = 42) {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
export function shuffle(items, rng = Math.random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}
export function splitSamples(samples, labels, seed = 42) {
  const index = new Map(labels.map((l, i) => [l, i]));
  const items = samples.map((s) => {
    if (!index.has(s.label)) throw new Error("Dataset contains an unknown label. Export and review your dataset.");
    return { x: s.data, y: index.get(s.label), session: s.session || "legacy" };
  });
  const byClass = labels.map((_, i) => items.filter((s) => s.y === i));
  if (byClass.some((c) => c.length < 2)) throw new Error("Record at least two real clips for every class, including idle.");
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.session)) groups.set(item.session, []);
    groups.get(item.session).push(item);
  }
  const rng = random(seed);
  if (groups.size > 1) {
    const remaining = byClass.map((c) => c.length);
    const valCounts = labels.map(() => 0);
    const heldOut = new Set();
    let nVal = 0;
    for (const session of shuffle([...groups.keys()].sort(), rng)) {
      const group = groups.get(session);
      const counts = labels.map((_, i) => group.filter((s) => s.y === i).length);
      if (counts.some((n, i) => remaining[i] - n < 1)) continue;
      if (nVal >= items.length * 0.2 && !counts.some((n, i) => n && !valCounts[i])) continue;
      heldOut.add(session);
      nVal += group.length;
      counts.forEach((n, i) => { remaining[i] -= n; valCounts[i] += n; });
    }
    if (valCounts.every((n) => n > 0)) return {
      train: items.filter((s) => !heldOut.has(s.session)),
      val: items.filter((s) => heldOut.has(s.session)),
      mode: "session", note: "Validation holds out entire recording sessions. It is also used for early stopping; a separate test session is still needed for final performance claims.",
    };
  }
  const train = [], val = [];
  for (const group of byClass) {
    const shuffled = shuffle([...group], rng);
    const n = Math.max(1, Math.round(group.length * 0.2));
    val.push(...shuffled.slice(0, n));
    train.push(...shuffled.slice(n));
  }
  return { train, val, mode: "clip", note: "Clip split only: recording sessions overlap, so this score may overstate performance in new conditions. Record every class in a second session for a session holdout." };
}
export function classMetrics(confusion) {
  return confusion.map((row, i) => {
    const support = row.reduce((a, b) => a + b, 0);
    const predicted = confusion.reduce((sum, r) => sum + r[i], 0);
    const precision = predicted ? row[i] / predicted : 0;
    const recall = support ? row[i] / support : 0;
    return { support, precision, recall, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0 };
  });
}
