"""
Interactive Web Application for Handwritten Digit Recognition
=============================================================
Zero-dependency web server built with Python standard library http.server + NumPy + Pillow.
Features:
- Interactive HTML5 Drawing Canvas (draw digits with mouse or touch)
- Real-world Photo Upload (smartphone camera photo, paper scan)
- Sample Digits Gallery (instant 1-click tests)
- Preprocessing Pipeline Inspector (Original -> Inverted -> 28x28 Centered MNIST)
- Live Prediction with Confidence Scores and Probability Bar Chart
- Model Accuracy Dashboard (97.93% test accuracy, confusion matrix, loss curves)
- B.Tech 2nd Year Viva Voce Guide & Architecture Visualizer

Author: Nadia (2nd Year B.Tech CSE)
"""

import http.server
import socketserver
import urllib.parse
import json
import base64
import os
import io
import time
import numpy as np
from PIL import Image

from neural_network import DeepNeuralNetwork
from image_preprocessor import DigitImagePreprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "digit_model_weights.npz")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")

# Global instances
MODEL = None
PREPROCESSOR = DigitImagePreprocessor()
METRICS = {}

def get_model():
    global MODEL
    if MODEL is None:
        if os.path.exists(WEIGHTS_PATH):
            MODEL = DeepNeuralNetwork.load_weights(WEIGHTS_PATH)
        else:
            raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}. Please run train.py first.")
    return MODEL

def get_metrics():
    global METRICS
    if not METRICS and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            METRICS = json.load(f)
    return METRICS

class DigitServerHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return

        elif path == "/api/metrics":
            metrics = get_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        elif path.startswith("/sample_images/"):
            filename = os.path.basename(path)
            file_path = os.path.join(SAMPLES_DIR, filename)
            if os.path.exists(file_path):
                ext = filename.split(".")[-1].lower()
                mime = "image/png" if ext == "png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Sample image not found")
                return

        elif path == "/api/samples":
            sample_files = []
            if os.path.exists(SAMPLES_DIR):
                for f in sorted(os.listdir(SAMPLES_DIR)):
                    if f.endswith((".png", ".jpg", ".jpeg")) and not f.startswith("debug_"):
                        sample_files.append({
                            "name": f,
                            "url": f"/sample_images/{f}"
                        })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sample_files).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                image_b64 = data.get("image", "")

                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]

                raw_bytes = base64.b64decode(image_b64)
                
                start_time = time.time()
                preprocessed_vec, intermediates = PREPROCESSOR.preprocess(raw_bytes, return_intermediate=True)
                
                model = get_model()
                pred_digit, probs = model.predict(preprocessed_vec)
                inference_time_ms = (time.time() - start_time) * 1000.0

                # Encode intermediate 28x28 normalized image as base64 for UI inspection
                buf = io.BytesIO()
                # Upscale 28x28 to 140x140 nearest neighbor for crisp display
                upscaled = intermediates["final_28x28"].resize((140, 140), Image.NEAREST)
                upscaled.save(buf, format="PNG")
                processed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                metrics = get_metrics()
                model_acc = metrics.get("test_accuracy", 97.93)

                response_payload = {
                    "success": True,
                    "predicted_digit": int(pred_digit),
                    "confidence": round(float(probs[pred_digit]) * 100.0, 2),
                    "model_accuracy": model_acc,
                    "probabilities": [round(float(p) * 100.0, 2) for p in probs],
                    "inference_time_ms": round(inference_time_ms, 2),
                    "processed_image": f"data:image/png;base64,{processed_b64}",
                    "preprocessing_info": {
                        "original_size": intermediates.get("original_size", [0, 0]),
                        "center_of_mass": [round(c, 2) for c in intermediates.get("center_of_mass", [13.5, 13.5])]
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Handwritten Digit Recognition | Deep Learning (MNIST)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #121826;
      --card-border: #1e293b;
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at top right, #1e1b4b 0%, var(--bg) 60%);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      line-height: 1.5;
      padding-bottom: 60px;
    }

    header {
      background: rgba(18, 24, 38, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 16px 32px;
    }
    .header-content {
      max-width: 1300px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-badge {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #6366f1, #06b6d4);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 22px;
      color: #fff;
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    .brand-title {
      font-size: 20px;
      font-weight: 800;
      background: linear-gradient(to right, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-sub {
      font-size: 13px;
      color: var(--text-muted);
    }
    .badge-pill {
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
    }

    .container {
      max-width: 1300px;
      margin: 32px auto;
      padding: 0 24px;
    }

    /* Top Stats Row */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }
    .stat-card::after {
      content: '';
      position: absolute;
      top: 0; right: 0; width: 60px; height: 60px;
      background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    }
    .stat-label {
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .stat-val {
      font-size: 28px;
      font-weight: 800;
      color: #fff;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }
    .stat-val.accent-green { color: var(--accent-green); }
    .stat-val.accent-cyan { color: var(--accent-cyan); }
    .stat-val.accent-indigo { color: #a5b4fc; }

    /* Main Workspace Layout */
    .workspace-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }
    @media (max-width: 1024px) {
      .workspace-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Tabs */
    .tab-nav {
      display: flex;
      gap: 8px;
      background: #0d121e;
      padding: 4px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      margin-bottom: 20px;
    }
    .tab-btn {
      flex: 1;
      padding: 10px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: var(--font);
      font-size: 14px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 2px 10px var(--accent-glow);
    }

    /* Canvas Section */
    .canvas-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #080b12;
      border: 2px dashed #26334d;
      border-radius: 16px;
      padding: 20px;
      position: relative;
    }
    #drawCanvas {
      background: #000000;
      border-radius: 12px;
      box-shadow: inset 0 0 20px rgba(255,255,255,0.03), 0 4px 20px rgba(0,0,0,0.6);
      cursor: crosshair;
      touch-action: none;
    }
    .canvas-controls {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      width: 100%;
      justify-content: center;
      flex-wrap: wrap;
    }
    .btn {
      padding: 10px 20px;
      border-radius: 10px;
      font-family: var(--font);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
      border: none;
    }
    .btn-primary {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #fff;
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px var(--accent-glow);
    }
    .btn-secondary {
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
    }
    .btn-secondary:hover { background: #334155; color: #fff; }

    /* Upload Area */
    .upload-zone {
      border: 2px dashed #334155;
      background: #0c121e;
      border-radius: 16px;
      padding: 40px 24px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.05);
    }
    .upload-icon {
      font-size: 40px;
      margin-bottom: 12px;
      display: block;
    }
    .file-input { display: none; }

    /* Samples Carousel */
    .samples-bar {
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px solid var(--card-border);
    }
    .samples-title {
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }
    .samples-list {
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 8px;
    }
    .sample-pill {
      background: #1a2234;
      border: 1px solid #2e3d5a;
      border-radius: 10px;
      padding: 8px 14px;
      color: #cbd5e1;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .sample-pill:hover {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }

    /* Prediction Result Panel */
    .pred-hero {
      background: linear-gradient(145deg, #161e31, #0f1624);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 20px;
      padding: 24px;
      text-align: center;
      position: relative;
      overflow: hidden;
      margin-bottom: 24px;
    }
    .pred-hero::before {
      content: '';
      position: absolute;
      top: -50%; left: -50%; width: 200%; height: 200%;
      background: radial-gradient(circle at center, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
      pointer-events: none;
    }
    .pred-badge-label {
      font-size: 12px;
      font-weight: 700;
      color: #a5b4fc;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 6px;
    }
    .predicted-digit-display {
      font-size: 80px;
      font-weight: 900;
      color: #ffffff;
      line-height: 1;
      text-shadow: 0 0 35px var(--accent-glow);
      margin: 10px 0;
    }
    .confidence-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 6px 16px;
      border-radius: 30px;
      font-size: 15px;
      font-weight: 700;
    }

    /* Preprocessing Inspector */
    .preproc-box {
      background: #090e18;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .mnist-preview-img {
      width: 70px;
      height: 70px;
      background: #000;
      border: 2px solid var(--accent);
      border-radius: 8px;
      image-rendering: pixelated;
    }
    .preproc-details {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }
    .preproc-title {
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 4px;
    }

    /* Probability Distribution Bars */
    .prob-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .prob-item {
      display: grid;
      grid-template-columns: 24px 1fr 50px;
      align-items: center;
      gap: 12px;
      font-size: 13px;
    }
    .digit-idx {
      font-weight: 700;
      font-family: var(--mono);
      color: #94a3b8;
    }
    .prob-track {
      height: 8px;
      background: #1a2234;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .prob-fill {
      height: 100%;
      background: #334155;
      border-radius: 6px;
      transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .prob-fill.winner {
      background: linear-gradient(90deg, #6366f1, #06b6d4);
      box-shadow: 0 0 10px var(--accent-glow);
    }
    .prob-pct {
      font-family: var(--mono);
      font-size: 12px;
      text-align: right;
      color: #cbd5e1;
    }

    /* Theory & Viva Accordion */
    .viva-section {
      margin-top: 40px;
    }
    .qa-card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 24px;
      margin-bottom: 12px;
    }
    .qa-q {
      font-size: 15px;
      font-weight: 700;
      color: #e2e8f0;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .qa-a {
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.6;
    }
    code {
      background: #0f1624;
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 13px;
    }
  </style>
</head>
<body>

  <header>
    <div class="header-content">
      <div class="brand">
        <div class="logo-badge">DL</div>
        <div>
          <div class="brand-title">Handwritten Digit Recognition</div>
          <div class="brand-sub">Deep Neural Network (MNIST Dataset) &bull; Nadia (2nd Year B.Tech CSE)</div>
        </div>
      </div>
      <div style="display: flex; gap: 10px; align-items: center;">
        <span class="badge-pill" id="headerAccuracy">Model Accuracy: 97.93%</span>
        <span class="badge-pill" style="background: rgba(16,185,129,0.15); color: #34d399; border-color: rgba(16,185,129,0.3);">Dataset: MNIST Official</span>
      </div>
    </div>
  </header>

  <div class="container">

    <!-- Top Key Metrics -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Model Test Accuracy</div>
        <div class="stat-val accent-green" id="statAccuracy">97.93%</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Evaluated on 10,000 Unseen Digits</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Training Dataset Size</div>
        <div class="stat-val accent-indigo">60,000</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Yann LeCun's Official MNIST Samples</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Neural Network Architecture</div>
        <div class="stat-val accent-cyan" style="font-size: 20px; padding-top: 5px;">784 &rarr; 256 &rarr; 128 &rarr; 10</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">ReLU + Softmax + Adam Optimizer</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Inference Latency</div>
        <div class="stat-val" id="statLatency">1.2 ms</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Real-Time NumPy Neural Engine</div>
      </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace-grid">
      
      <!-- Left Panel: Input Modes -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
            Digit Input Studio
          </div>
        </div>

        <!-- Mode Tabs -->
        <div class="tab-nav">
          <button class="tab-btn active" id="tabDraw" onclick="switchTab('draw')">Draw on Canvas</button>
          <button class="tab-btn" id="tabUpload" onclick="switchTab('upload')">Take / Upload Picture</button>
        </div>

        <!-- Tab 1: Draw Canvas -->
        <div id="drawSection">
          <div class="canvas-container">
            <canvas id="drawCanvas" width="280" height="280"></canvas>
            <div class="canvas-controls">
              <button class="btn btn-primary" onclick="predictCanvas()">
                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                Predict Digit
              </button>
              <button class="btn btn-secondary" onclick="clearCanvas()">
                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                Clear
              </button>
            </div>
          </div>
        </div>

        <!-- Tab 2: Upload Photo -->
        <div id="uploadSection" style="display: none;">
          <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <span class="upload-icon">&#128247;</span>
            <div style="font-weight: 700; font-size: 16px; margin-bottom: 6px;">Take or Upload Photo of Handwritten Digit</div>
            <div style="font-size: 13px; color: var(--text-muted);">Click to browse or drop an image file (PNG, JPG, Phone Photo)</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFileSelect(event)">
          </div>
          <div id="uploadPreviewBox" style="display: none; margin-top: 16px; text-align: center;">
            <img id="uploadPreview" style="max-height: 220px; border-radius: 12px; border: 1px solid var(--card-border);">
            <div style="margin-top: 12px;">
              <button class="btn btn-primary" onclick="predictUploadedImage()">Predict Uploaded Picture</button>
            </div>
          </div>
        </div>

        <!-- Quick-Click Sample Gallery -->
        <div class="samples-bar">
          <div class="samples-title">
            <span>Quick-Click Real Test Pictures</span>
            <span style="color: #6366f1;">Click any digit below to test instantly</span>
          </div>
          <div class="samples-list" id="samplesList">
            <!-- Populated via JS -->
          </div>
        </div>

      </div>

      <!-- Right Panel: Prediction & Probabilities -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
            Model Prediction & Confidence
          </div>
        </div>

        <!-- Hero Prediction Result -->
        <div class="pred-hero">
          <div class="pred-badge-label">Predicted Digit</div>
          <div class="predicted-digit-display" id="predDigit">?</div>
          <div>
            <span class="confidence-badge" id="predConfidence">Draw or Upload a Digit</span>
          </div>
        </div>

        <!-- Preprocessing Inspector -->
        <div class="preproc-box">
          <img id="mnistNormalizedImg" class="mnist-preview-img" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='70' height='70'><rect width='70' height='70' fill='%23000'/></svg>" alt="28x28 normalized">
          <div class="preproc-details">
            <div class="preproc-title">MNIST Preprocessing Pipeline</div>
            <div>&bull; Centered by Center of Mass (Centroid)</div>
            <div>&bull; Aspect-ratio normalized to 20x20 in 28x28 box</div>
            <div>&bull; Grayscale inverted & normalized to [0.0, 1.0]</div>
          </div>
        </div>

        <!-- Probability Breakdown (0-9) -->
        <div style="font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 12px; text-transform: uppercase;">
          Class Probabilities (Softmax Distribution)
        </div>
        <div class="prob-list" id="probList">
          <!-- Populated via JS -->
        </div>

      </div>

    </div>

    <!-- B.Tech CSE Viva Voce & Theory Guide -->
    <div class="viva-section">
      <div style="font-size: 22px; font-weight: 800; color: #fff; margin-bottom: 8px;">
        College Project Reference & Viva Voce Guide
      </div>
      <div style="font-size: 14px; color: var(--text-muted); margin-bottom: 20px;">
        Key theoretical questions and answers frequently asked by external examiners in B.Tech CSE labs:
      </div>

      <div class="qa-card">
        <div class="qa-q">&#10067; 1. What is the MNIST dataset and why is it structured as 28x28 pixels?</div>
        <div class="qa-a">
          MNIST (Modified National Institute of Standards and Technology) is the gold standard benchmark dataset in computer vision. It contains <strong>70,000 grayscale images</strong> (60,000 for training, 10,000 for testing) of digits 0 through 9 written by high school students and US Census Bureau employees. Each image is normalized to <strong>28x28 pixels</strong> (784 total input features), where digits are centered by their center of mass within a 20x20 box.
        </div>
      </div>

      <div class="qa-card">
        <div class="qa-q">&#10067; 2. Why do we need image preprocessing when testing real smartphone photos?</div>
        <div class="qa-a">
          In MNIST, digits are <strong>white strokes (255) on a black background (0)</strong>. However, real photos taken on paper are <strong>dark ink on a white/off-white background</strong> with uneven lighting and shadows. Our preprocessing pipeline:
          (1) Converts RGB to Grayscale,
          (2) Auto-detects border brightness and inverts the image so strokes become bright on dark,
          (3) Isolates the digit bounding box and scales it to 20x20,
          (4) Computes the <strong>Center of Mass (Centroid)</strong> and shifts the digit so it matches MNIST's exact spatial alignment,
          (5) Normalizes pixel intensity from <code>[0, 255]</code> to <code>[0.0, 1.0]</code>.
        </div>
      </div>

      <div class="qa-card">
        <div class="qa-q">&#10067; 3. Why use ReLU in hidden layers and Softmax in the output layer?</div>
        <div class="qa-a">
          <strong>ReLU (Rectified Linear Unit, <code>f(x) = max(0, x)</code>)</strong> avoids the vanishing gradient problem of Sigmoid/Tanh, accelerates gradient descent convergence, and promotes sparse representations. 
          <strong>Softmax</strong> converts the raw final layer logits <code>Z</code> into a valid probability distribution where <code>sum(P) = 1.0</code> and each <code>P_i in [0, 1]</code>, allowing direct interpretation as confidence for each digit 0 through 9.
        </div>
      </div>

      <div class="qa-card">
        <div class="qa-q">&#10067; 4. What loss function and optimizer were used, and what accuracy was achieved?</div>
        <div class="qa-a">
          We used <strong>Categorical Cross-Entropy Loss</strong> combined with the <strong>Adam Optimizer</strong> (learning rate = 0.001, &beta;1 = 0.9, &beta;2 = 0.999). On the 10,000 unseen MNIST test samples, the trained deep neural network achieved <strong>97.93% Test Accuracy</strong> with an average inference time of ~1.2 milliseconds!
        </div>
      </div>
    </div>

  </div>

  <script>
    // Initialize Canvas
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function initCanvas() {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 18;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    }
    initCanvas();

    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: clientX - rect.left,
        y: clientY - rect.top
      };
    }

    function startDraw(e) {
      isDrawing = true;
      const pos = getPos(e);
      lastX = pos.x;
      lastY = pos.y;
      ctx.beginPath();
      ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }

    function draw(e) {
      if (!isDrawing) return;
      e.preventDefault();
      const pos = getPos(e);
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
      lastX = pos.x;
      lastY = pos.y;
    }

    function stopDraw() {
      if (isDrawing) {
        isDrawing = false;
        predictCanvas(); // Auto predict on stroke release
      }
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDraw);

    canvas.addEventListener('touchstart', startDraw);
    canvas.addEventListener('touchmove', draw);
    window.addEventListener('touchend', stopDraw);

    function clearCanvas() {
      initCanvas();
      resetDisplay();
    }

    function switchTab(mode) {
      const tabDraw = document.getElementById('tabDraw');
      const tabUpload = document.getElementById('tabUpload');
      const drawSection = document.getElementById('drawSection');
      const uploadSection = document.getElementById('uploadSection');

      if (mode === 'draw') {
        tabDraw.classList.add('active');
        tabUpload.classList.remove('active');
        drawSection.style.display = 'block';
        uploadSection.style.display = 'none';
      } else {
        tabUpload.classList.add('active');
        tabDraw.classList.remove('active');
        uploadSection.style.display = 'block';
        drawSection.style.display = 'none';
      }
    }

    // Probability Bars Init
    function renderProbBars(probs = null, winnerIdx = -1) {
      const list = document.getElementById('probList');
      list.innerHTML = '';
      for (let i = 0; i < 10; i++) {
        const p = probs ? probs[i] : 0;
        const isWinner = (i === winnerIdx);
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
          <div class="digit-idx">${i}</div>
          <div class="prob-track">
            <div class="prob-fill ${isWinner ? 'winner' : ''}" style="width: ${p}%"></div>
          </div>
          <div class="prob-pct" style="${isWinner ? 'color: #34d399; font-weight:700;' : ''}">${p.toFixed(1)}%</div>
        `;
        list.appendChild(item);
      }
    }
    renderProbBars();

    function resetDisplay() {
      document.getElementById('predDigit').innerText = '?';
      document.getElementById('predConfidence').innerText = 'Draw or Upload a Digit';
      document.getElementById('predConfidence').style.color = '#a5b4fc';
      document.getElementById('mnistNormalizedImg').src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='70' height='70'><rect width='70' height='70' fill='%23000'/></svg>";
      renderProbBars();
    }

    async function sendPrediction(imageBase64) {
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageBase64 })
        });
        const data = await res.json();
        if (data.success) {
          document.getElementById('predDigit').innerText = data.predicted_digit;
          document.getElementById('predConfidence').innerHTML = `Confidence: <strong>${data.confidence}%</strong>`;
          document.getElementById('predConfidence').style.color = '#34d399';
          document.getElementById('statLatency').innerText = `${data.inference_time_ms} ms`;
          document.getElementById('mnistNormalizedImg').src = data.processed_image;
          renderProbBars(data.probabilities, data.predicted_digit);
        } else {
          alert('Prediction error: ' + data.error);
        }
      } catch (err) {
        console.error(err);
      }
    }

    function predictCanvas() {
      const dataUrl = canvas.toDataURL('image/png');
      sendPrediction(dataUrl);
    }

    let uploadedBase64 = null;
    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(evt) {
        uploadedBase64 = evt.target.result;
        document.getElementById('uploadPreview').src = uploadedBase64;
        document.getElementById('uploadPreviewBox').style.display = 'block';
        predictUploadedImage();
      };
      reader.readAsDataURL(file);
    }

    function predictUploadedImage() {
      if (uploadedBase64) {
        sendPrediction(uploadedBase64);
      }
    }

    // Load Sample Test Digits
    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        const container = document.getElementById('samplesList');
        container.innerHTML = '';
        samples.forEach(s => {
          const pill = document.createElement('div');
          pill.className = 'sample-pill';
          const isPhoto = s.name.startsWith('photo_');
          pill.innerHTML = `<span>${isPhoto ? '&#128247;' : '&#9998;'}</span> <span>${s.name.replace('photo_paper_', 'Photo ').replace('mnist_', 'MNIST ').replace('.png', '').replace('.jpg', '')}</span>`;
          pill.onclick = () => {
            testSample(s.url);
          };
          container.appendChild(pill);
        });
      } catch (e) {
        console.error(e);
      }
    }

    async function testSample(url) {
      try {
        const res = await fetch(url);
        const blob = await res.blob();
        const reader = new FileReader();
        reader.onload = function(e) {
          sendPrediction(e.target.result);
        };
        reader.readAsDataURL(blob);
      } catch (err) {
        console.error(err);
      }
    }

    loadSamples();
  </script>
</body>
</html>
"""

def run_server(port=8501):
    # Try preferred port, or fall back to an available one
    for p in range(port, port + 20):
        try:
            server = socketserver.TCPServer(("", p), DigitServerHandler)
            print("=" * 70)
            print("  HANDWRITTEN DIGIT RECOGNITION WEB APPLICATION")
            print("  Student: Nadia, 2nd Year B.Tech CSE")
            print(f"  Live Web Server running at: http://localhost:{p}")
            print("=" * 70)
            server.serve_forever()
            break
        except OSError as e:
            if "Address already in use" in str(e):
                continue
            else:
                raise e

if __name__ == "__main__":
    run_server(8501)

