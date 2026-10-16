import assert from "node:assert/strict";
import test from "node:test";
import { Camera, cameraErrorMessage, requestCameraStream } from "../js/core/camera.js";

test("unsupported preferred settings fall back to an unconstrained camera", async () => {
  const requests = [];
  const stream = {};
  const mediaDevices = {
    async getUserMedia(constraints) {
      requests.push(constraints);
      if (requests.length === 1) throw new DOMException("Could not start video source", "NotReadableError");
      return stream;
    },
  };
  assert.equal(await requestCameraStream(mediaDevices), stream);
  assert.equal(requests[0].video.width.ideal, 960);
  assert.deepEqual(requests[1], { video: true, audio: false });
});

test("a broken default camera falls back to another enumerated webcam", async () => {
  const requests = [];
  const stream = {};
  const mediaDevices = {
    async getUserMedia(constraints) {
      requests.push(constraints);
      if (constraints.video?.deviceId?.exact === "working-camera") return stream;
      throw new DOMException("Could not start video source", "NotReadableError");
    },
    async enumerateDevices() {
      return [{ kind: "audioinput", deviceId: "microphone" }, { kind: "videoinput", deviceId: "broken-camera" }, { kind: "videoinput", deviceId: "working-camera" }];
    },
  };
  assert.equal(await requestCameraStream(mediaDevices), stream);
  assert.equal(requests.length, 4);
  assert.ok(requests.every((request) => request.audio === false));
});

test("denied permission does not trigger retries or camera enumeration", async () => {
  let requests = 0;
  const mediaDevices = {
    async getUserMedia() {
      requests++;
      throw new DOMException("Permission denied", "NotAllowedError");
    },
    async enumerateDevices() { assert.fail("Must not enumerate after permission denial"); },
  };
  await assert.rejects(requestCameraStream(mediaDevices), { name: "NotAllowedError" });
  assert.equal(requests, 1);
});

test("persistent hardware errors remain actionable and missing browser support is identified", async () => {
  const failure = new DOMException("Could not start video source", "NotReadableError");
  const mediaDevices = { async getUserMedia() { throw failure; }, async enumerateDevices() { return []; } };
  await assert.rejects(requestCameraStream(mediaDevices), (error) => error === failure);
  assert.match(cameraErrorMessage(failure), /Close Camera, Teams, Zoom/);
  assert.doesNotMatch(cameraErrorMessage(failure), /localhost/);
  await assert.rejects(requestCameraStream(undefined), { name: "SecurityError" });
});

test("failed initialization releases the stream and tracker, and a subsequent start succeeds", async () => {
  let stopped = 0;
  let closed = 0;
  const stream = { getTracks: () => [{ stop() { stopped++; } }] };
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousAnimationFrame = globalThis.requestAnimationFrame;
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: { async getUserMedia() { return stream; } } } });
  globalThis.requestAnimationFrame = () => 1;
  try {
    const video = { srcObject: null, videoWidth: 640, videoHeight: 480, async play() {} };
    const canvas = { getContext: () => ({}) };
    const camera = new Camera(video, canvas);
    camera._loadHandTracker = async () => {
      camera.landmarker = { close() { closed++; } };
      throw new Error("Model download failed");
    };
    await assert.rejects(camera.start(), /Model download failed/);
    assert.equal(stopped, 1);
    assert.equal(closed, 1);
    assert.equal(video.srcObject, null);
    assert.equal(camera.landmarker, null);
    assert.equal(camera.ready, false);
    camera._loadHandTracker = async () => { camera.landmarker = { close() {} }; };
    await camera.start();
    assert.equal(camera.ready, true);
    assert.equal(video.srcObject, stream);
    assert.equal(canvas.width, 640);
    assert.equal(canvas.height, 480);
    await camera.start();
    assert.equal(stopped, 1);
  } finally {
    if (previousNavigator) Object.defineProperty(globalThis, "navigator", previousNavigator);
    else delete globalThis.navigator;
    if (previousAnimationFrame) globalThis.requestAnimationFrame = previousAnimationFrame;
    else delete globalThis.requestAnimationFrame;
  }
});
