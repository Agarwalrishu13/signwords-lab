"""Optional browser regression: uses Chrome's simulated webcam, never a real one.

Install Python Playwright, then run with the app served on localhost:8000:
  python tests/browser_smoke.py
"""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-tools"))
from playwright.sync_api import sync_playwright, expect


def upload(page, dataset):
    page.locator("#importFile").set_input_files({
        "name": "test-dataset.json", "mimeType": "application/json",
        "buffer": json.dumps(dataset).encode(),
    })


def dataset():
    samples = []
    for session in ["browser-session-a", "browser-session-b"]:
        for label in ["__idle__", "Hello"]:
            for clip in range(8):
                data = [0.0] * (24 * 130)
                if label == "Hello":
                    for f in range(24):
                        for k in range(3, 65):
                            data[f * 130 + k] = 0.3 + k * 0.003 + 0.02 * math.sin(f / 3 + clip / 8)
                samples.append({"label": label, "session": session, "clipId": f"{session}-{label}-{clip}", "data": data})
    return {"app": "signwords-lab", "version": 2, "featureVersion": 1,
            "seqLen": 24, "feat": 130, "words": ["Hello"], "samples": samples}


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True, args=[
        "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        "--enable-unsafe-swiftshader", "--ignore-certificate-errors",
    ])
    context = browser.new_context(permissions=["camera"], viewport={"width": 1280, "height": 900})
    # Use the installed, identical TF.js build rather than depending on its CDN.
    context.route("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
                  lambda route: route.fulfill(path=str(ROOT / "node_modules/@tensorflow/tfjs/dist/tf.min.js"), content_type="text/javascript"))
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("dialog", lambda dialog: dialog.accept())
    try:
        page.goto("http://localhost:8000", wait_until="networkidle")
        expect(page.locator("#sessionInfo")).to_contain_text("session 1")
        expect(page.locator("#rec1Btn")).to_be_disabled()
        page.locator("#wordInput").fill("Hello")
        page.locator("#addWordBtn").click()
        expect(page.locator("#dataTotal")).to_have_text("0 clips")
        expect(page.locator('.wcard[data-label="Hello"]')).to_be_visible()
        page.locator('#wordList [data-label="__idle__"] [data-act="select"]').click()
        page.locator("#startBtn").click()
        expect(page.locator("#startBtn")).to_have_text("Cancel camera start")
        page.locator("#startBtn").click()
        expect(page.locator("#startBtn")).to_have_text("Start camera")
        page.locator("#startBtn").click()
        expect(page.locator("#startBtn")).to_have_text("Stop camera", timeout=120000)
        print("Tracking mode:", page.locator("#camStatus").inner_text(), flush=True)
        page.wait_for_function("parseInt(document.getElementById('fpsBadge').textContent) >= 10", timeout=45000)
        started = time.monotonic()
        page.locator("#rec10Btn").click()
        expect(page.locator("#recCancelBtn")).to_be_hidden(timeout=25000)
        elapsed = time.monotonic() - started
        expect(page.locator("#dataTotal")).to_have_text("10 clips")
        assert elapsed < 19, f"Default batch too slow: {elapsed:.2f}s"
        print(f"10 full clips recorded in {elapsed:.2f}s (old scheduled minimum: 21.6s)", flush=True)

        page.locator("#recordDelay").select_option("0")
        page.locator("#rec10Btn").click()
        page.wait_for_function("parseFloat(document.getElementById('captureBar').style.width) > 0")
        started = time.monotonic()
        page.locator("#recCancelBtn").click()
        expect(page.locator("#recCancelBtn")).to_be_hidden()
        assert time.monotonic() - started < 1
        expect(page.locator("#dataTotal")).to_have_text("10 clips")
        print("Immediate cancellation discarded the partial clip", flush=True)

        page.locator("#recordGap").select_option("0")
        started = time.monotonic()
        page.locator("#rec10Btn").click()
        expect(page.locator("#recCancelBtn")).to_be_hidden(timeout=20000)
        elapsed = time.monotonic() - started
        expect(page.locator("#dataTotal")).to_have_text("20 clips")
        assert elapsed < 14, f"Batch without waits too slow: {elapsed:.2f}s"
        print(f"Batch with no preparation/rest recorded in {elapsed:.2f}s", flush=True)

        page.evaluate("window.testTrack = document.getElementById('video').srcObject.getVideoTracks()[0]")
        page.locator("#startBtn").click()
        assert page.evaluate("window.testTrack.readyState") == "ended"
        assert page.evaluate("document.getElementById('video').srcObject") is None
        page.locator("#clearAllBtn").click()
        expect(page.locator("#dataTotal")).to_have_text("0 clips")
        data = dataset()
        upload(page, data)
        expect(page.locator("#dataTotal")).to_have_text("32 clips")
        upload(page, data)
        expect(page.locator("#toast")).to_contain_text("skipped 32 duplicates")
        expect(page.locator("#dataTotal")).to_have_text("32 clips")
        bad = dataset()
        bad["samples"][0]["data"][0] = "bad"
        upload(page, bad)
        expect(page.locator("#toast")).to_contain_text("invalid")
        expect(page.locator("#dataTotal")).to_have_text("32 clips")
        print("Camera release, import validation, and duplicate import checks passed", flush=True)

        page.locator('[data-tab="train"]').click()
        page.locator("#epochs").fill("5")
        page.locator("#augment").select_option("0")
        page.locator("#trainBtn").click()
        expect(page.locator("#log")).to_contain_text("Model and metadata saved", timeout=120000)
        expect(page.locator("#evaluationNote")).to_contain_text("entire recording sessions")
        expect(page.locator("#confusion")).to_contain_text("Precision")
        meta = page.evaluate("async () => { const m = await import('/js/core/model.js'); const saved = await m.loadSavedModel(); const result = saved.meta; m.disposeModel(saved.model); return result; }")
        assert meta["splitMode"] == "session" and meta["bestEpoch"] >= 1
        page.reload(wait_until="networkidle")
        expect(page.locator("#stepTrainInfo")).to_contain_text("trained", timeout=30000)
        print("Browser training, metrics, model persistence, and restoration passed", flush=True)

        page.locator("#startBtn").click()
        expect(page.locator("#startBtn")).to_have_text("Stop camera", timeout=120000)
        page.wait_for_function("parseInt(document.getElementById('fpsBadge').textContent) >= 10", timeout=45000)
        page.locator('[data-tab="live"]').click()
        page.locator("#liveToggle").click()
        expect(page.locator("#liveConf")).to_contain_text("waiting for a sign", timeout=15000)
        expect(page.locator("#transcript")).to_contain_text("Words you sign")
        page.locator('[data-tab="data"]').click()
        expect(page.locator("#stepLiveInfo")).to_have_text("stopped")
        page.locator("#recordDelay").select_option("0")
        page.locator("#rec1Btn").click()
        expect(page.locator("#recCancelBtn")).to_be_hidden(timeout=5000)
        expect(page.locator("#dataTotal")).to_have_text("33 clips")
        expect(page.locator("#stepTrainInfo")).to_have_text("retrain needed")
        page.reload(wait_until="networkidle")
        expect(page.locator("#stepTrainInfo")).to_have_text("retrain needed", timeout=30000)
        print("No-hand detection and dataset-change invalidation persisted after reload", flush=True)

        output = ROOT / "test-results"
        output.mkdir(exist_ok=True)
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('[data-tab="train"]').click()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile page overflows horizontally"
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert not errors, errors
        print("Desktop/mobile layout and browser console checks passed", flush=True)

        fallback = context.new_page()
        fallback.add_init_script("window.Worker = undefined")
        fallback.goto("http://localhost:8000", wait_until="networkidle")
        fallback.locator("#startBtn").click()
        expect(fallback.locator("#startBtn")).to_have_text("Stop camera", timeout=120000)
        expect(fallback.locator("#camStatus")).to_contain_text("compatible tracking")
        fallback.locator("#startBtn").click()
        fallback.close()
        print("Tracking fallback works when workers are unavailable", flush=True)
    finally:
        browser.close()
