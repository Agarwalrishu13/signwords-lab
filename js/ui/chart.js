// Tiny dependency-free line chart for the training curves.
export class LineChart {
  /** opts: {title, yMin?, yMax?, series:[{key,label,color}]} */
  constructor(canvas, opts) {
    this.c = canvas;
    this.o = opts;
    this.data = {};
    this.draw();
    addEventListener("resize", () => this.draw());
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
    document.fonts.ready.then(() => this.draw());
  }

  set(data) {
    this.data = data;
    this.draw();
  }

  draw() {
    const c = this.c;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const pad = { l: 40, r: 10, t: 28, b: 22 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const { series, title } = this.o;
    const styles = getComputedStyle(c);
    const color = (token) => token.startsWith("--") ? styles.getPropertyValue(token).trim() : token;
    const muted = color("--muted");

    g.fillStyle = muted;
    g.font = "11px Manrope, sans-serif";
    g.textBaseline = "alphabetic";
    g.fillText(title, pad.l, 16);

    const n = Math.max(0, ...series.map((s) => (this.data[s.key] || []).length));
    if (!n) {
      g.textAlign = "center";
      g.fillText("waiting for training…", w / 2, h / 2);
      g.textAlign = "left";
      return;
    }

    const vals = series.flatMap((s) => (this.data[s.key] || []).filter((v) => v != null && isFinite(v)));
    let lo = this.o.yMin ?? Math.min(0, ...vals);
    let hi = this.o.yMax ?? Math.max(...vals);
    if (hi - lo < 1e-6) hi = lo + 1;
    const X = (i) => pad.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const Y = (v) => pad.t + ph - ((v - lo) / (hi - lo)) * ph;

    g.strokeStyle = color("--line");
    g.lineWidth = 1;
    g.fillStyle = muted;
    g.textAlign = "right";
    for (let k = 0; k <= 4; k++) {
      const v = lo + ((hi - lo) * k) / 4;
      const y = Y(v);
      g.beginPath();
      g.moveTo(pad.l, y);
      g.lineTo(w - pad.r, y);
      g.stroke();
      g.fillText(v.toFixed(2), pad.l - 6, y + 4);
    }
    g.textAlign = "center";
    g.fillText("epoch " + n, w - pad.r - 24, h - 6);
    g.textAlign = "left";

    let lx = w - pad.r;
    for (const s of [...series].reverse()) {
      const arr = this.data[s.key] || [];
      g.strokeStyle = color(s.color);
      g.lineWidth = 2;
      g.beginPath();
      let started = false;
      arr.forEach((v, i) => {
        if (v == null || !isFinite(v)) return;
        if (!started) {
          g.moveTo(X(i), Y(v));
          started = true;
        } else g.lineTo(X(i), Y(v));
      });
      g.stroke();
      const tw = g.measureText(s.label).width;
      lx -= tw + 22;
      g.fillStyle = color(s.color);
      g.fillRect(lx, 8, 10, 3);
      g.fillStyle = muted;
      g.fillText(s.label, lx + 14, 14);
    }
  }
}
