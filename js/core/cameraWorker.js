// Classic worker: MediaPipe's WASM loader can use importScripts here.
let landmarker;
self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    try {
      const { HandLandmarker, FilesetResolver } = await import(`${data.base}/vision_bundle.mjs`);
      const vision = await FilesetResolver.forVisionTasks(`${data.base}/wasm`);
      const make = (delegate) => HandLandmarker.createFromOptions(vision, {
        canvas: new OffscreenCanvas(640, 480),
        baseOptions: { modelAssetPath: data.modelURL, delegate },
        runningMode: "VIDEO", numHands: 2,
        minHandDetectionConfidence: 0.6, minTrackingConfidence: 0.5,
      });
      try { landmarker = await make("GPU"); }
      catch { landmarker = await make("CPU"); }
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message || String(err) });
    }
  } else if (data.type === "frame") {
    try {
      const result = landmarker.detectForVideo(data.bitmap, data.timestamp);
      self.postMessage({ type: "result", result: { landmarks: result.landmarks, handedness: result.handedness }, timestamp: data.timestamp });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message || String(err) });
    } finally {
      data.bitmap.close();
    }
  }
};
