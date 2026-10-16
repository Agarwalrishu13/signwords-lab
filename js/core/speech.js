// Audio: text-to-speech for detected words, plus short UI chimes (WebAudio).
export class Speech {
  constructor() {
    this.voices = [];
    this.voice = null;
    this.rate = 1;
    this.volume = 1;
    this.enabled = true; // speak each detected word
    this.chimeOn = true; // beeps for countdown / detection
    this.ctx = null;
  }

  supported() {
    return "speechSynthesis" in window;
  }

  init(select) {
    const load = () => {
      this.voices = this.supported() ? speechSynthesis.getVoices() : [];
      const current = select.value;
      select.innerHTML = "";
      if (!this.voices.length) {
        const o = document.createElement("option");
        o.textContent = this.supported() ? "Default voice" : "Speech not supported";
        select.appendChild(o);
        return;
      }
      const pref = (navigator.language || "en").slice(0, 2);
      const sorted = [...this.voices].sort(
        (a, b) => (b.lang.startsWith(pref) - a.lang.startsWith(pref)) || a.name.localeCompare(b.name)
      );
      for (const v of sorted) {
        const o = document.createElement("option");
        o.value = v.voiceURI;
        o.textContent = `${v.name} (${v.lang})`;
        select.appendChild(o);
      }
      if (current) select.value = current;
      this.setVoice(select.value);
    };
    load();
    if (this.supported()) speechSynthesis.onvoiceschanged = load;
  }

  setVoice(uri) {
    this.voice = this.voices.find((v) => v.voiceURI === uri) || null;
  }

  speak(text, { force = false } = {}) {
    if (!this.supported() || !text) return;
    if (!this.enabled && !force) return;
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) {
      u.voice = this.voice;
      u.lang = this.voice.lang;
    }
    u.rate = this.rate;
    u.volume = this.volume;
    speechSynthesis.speak(u);
  }

  stop() {
    if (this.supported()) speechSynthesis.cancel();
  }

  chime(kind) {
    if (!this.chimeOn) return;
    try {
      this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
      if (this.ctx.state === "suspended") this.ctx.resume();
      const notes = { tick: [660], go: [880, 1175], word: [784, 1047], error: [220] }[kind] || [660];
      let t = this.ctx.currentTime;
      for (const f of notes) {
        const o = this.ctx.createOscillator();
        const g = this.ctx.createGain();
        o.type = "sine";
        o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.18 * this.volume + 0.0001, t + 0.01);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
        o.connect(g).connect(this.ctx.destination);
        o.start(t);
        o.stop(t + 0.16);
        t += 0.12;
      }
    } catch {}
  }
}
