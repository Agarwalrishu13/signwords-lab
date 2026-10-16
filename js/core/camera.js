// Webcam + MediaPipe hand tracking (up to two hands).
import { CONFIG } from "../config.js";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const COLORS = { Right: "#4cc9a0", Left: "#f0b24a" };

const recoverableCameraErrors = new Set(["NotReadableError", "TrackStartError", "OverconstrainedError", "ConstraintNotSatisfiedError", "NotFoundError", "AbortError"]);

export function cameraErrorMessage(error) {
  if (["NotReadableError", "TrackStartError", "AbortError"].includes(error?.name)) {
    return "Could not open the camera. Close Camera, Teams, Zoom, and other browser tabs using it, then click Start camera again. If it still fails, check Windows Settings > Privacy & security > Camera and allow desktop apps to access it.";
  }
  if (["NotAllowedError", "PermissionDeniedError"].includes(error?.name)) {
    return "Camera access is blocked. Allow camera access in this site's browser permissions and in Windows Settings > Privacy & security > Camera, then click Start camera again.";
  }
  if (["NotFoundError", "DevicesNotFoundError"].includes(error?.name)) {
    return "No available camera was found. Connect or enable your webcam, check its privacy switch, then click Start camera again.";
  }
  if (["OverconstrainedError", "ConstraintNotSatisfiedError"].includes(error?.name)) {
    return "The camera could not provide a supported video stream. Select an available webcam in your browser's camera settings, then try again.";
  }
  if (error?.name === "SecurityError") {
    return "Camera access requires a supported browser on localhost or HTTPS. Run python serve.py and open the printed localhost address in Chrome or Edge.";
  }
  return "Could not start camera tracking: " + (error?.message || String(error)) + ". Check your connection for the hand-tracking download, then try again.";
}

export async function requestCameraStream(mediaDevices, onStatus = () => {}) {
  if (!mediaDevices?.getUserMedia) {
    throw new DOMException("Camera access is unavailable in this browser context.", "SecurityError");
  }
  const request = (video) => mediaDevices.getUserMedia({ video, audio: false });
  let lastError;
  for (const video of [
    { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: { ideal: "user" } },
    true,
  ]) {
    try {
      return await request(video);
    } catch (error) {
      if (!recoverableCameraErrors.has(error.name)) throw error;
      lastError = error;
      onStatus("Trying a compatible camera setting…");
    }
  }
  let devices = [];
  try {
    devices = await mediaDevices.enumerateDevices();
  } catch {
    throw lastError;
  }
  for (const device of devices.filter((device) => device.kind === "videoinput" && device.deviceId)) {
    try {
      onStatus("Trying another available camera…");
      return await request({ deviceId: { exact: device.deviceId } });
    } catch (error) {
      if (!recoverableCameraErrors.has(error.name)) throw error;
      lastError = error;
    }
  }
  throw lastError;
}

export class Camera {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.landmarker = null;
    this.ready = false;
    this.onFrame = () => {};
    this.fps = 0;
    this._frames = 0;
    this._t0 = performance.now();
    this._lastVideoTime = -1;
  }

  async start(onStatus = () => {}) {
    if (this.ready) return;
    let stream;
    try {
      onStatus("Requesting camera…");
      stream = await requestCameraStream(navigator.mediaDevices, onStatus);
      this.video.srcObject = stream;
      await this.video.play();
      await this._loadHandTracker(onStatus);
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      this._frames = 0;
      this._t0 = performance.now();
      this._lastVideoTime = -1;
      this.ready = true;
      requestAnimationFrame(() => this._loop());
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      this.video.srcObject = null;
      this.ready = false;
      this.landmarker?.close();
      this.landmarker = null;
      throw error;
    }
  }

  async _loadHandTracker(onStatus) {
    onStatus("Loading hand-tracking model…");
    const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${CONFIG.MP_VERSION}`;
    const { HandLandmarker, FilesetResolver } = await import(base);
    const vision = await FilesetResolver.forVisionTasks(`${base}/wasm`);
    const make = (delegate) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: CONFIG.HAND_MODEL_URL, delegate },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
    try {
      this.landmarker = await make("GPU");
    } catch {
      this.landmarker = await make("CPU");
    }

  }

  _loop() {
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this._lastVideoTime) {
      this._lastVideoTime = v.currentTime;
      const res = this.landmarker.detectForVideo(v, performance.now());
      const hands = { Right: null, Left: null };
      const lms = res.landmarks || [];
      const sides = res.handedness || res.handednesses || [];
      lms.forEach((lm, i) => {
        let name = sides[i] && sides[i][0] ? sides[i][0].categoryName : "Right";
        if (hands[name]) name = name === "Right" ? "Left" : "Right";
        if (!hands[name]) hands[name] = lm;
      });
      this._draw(hands);
      this._frames++;
      const now = performance.now();
      if (now - this._t0 >= 1000) {
        this.fps = Math.round((this._frames * 1000) / (now - this._t0));
        this._frames = 0;
        this._t0 = now;
      }
      this.onFrame(hands, this.fps);
    }
    requestAnimationFrame(() => this._loop());
  }

  _draw(hands) {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = canvas.width;
    const H = canvas.height;
    for (const name of Object.keys(hands)) {
      const lm = hands[name];
      if (!lm) continue;
      ctx.lineWidth = 3;
      ctx.strokeStyle = COLORS[name];
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lm[a].x * W, lm[a].y * H);
        ctx.lineTo(lm[b].x * W, lm[b].y * H);
        ctx.stroke();
      }
      ctx.fillStyle = "#fff";
      for (const p of lm) {
        ctx.beginPath();
        ctx.arc(p.x * W, p.y * H, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
