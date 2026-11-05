"""
Zero-Dependency Email Spam Classifier Web Application
Requires ONLY Python Standard Library (No pip install needed!)
Perfect for B.Tech Viva, College Lab Computers, and Offline Demos.
"""

import http.server
import socketserver
import urllib.parse
import json
import re
import math
import csv

# -------------------------------------------------------------
# 1. PURE PYTHON NAIVE BAYES CLASSIFIER IMPLEMENTATION
# -------------------------------------------------------------
class PurePythonNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes = ['Ham', 'Spam']
        self.word_counts = {'Ham': {}, 'Spam': {}}
        self.class_counts = {'Ham': 0, 'Spam': 0}
        self.total_words = {'Ham': 0, 'Spam': 0}
        self.vocab = set()

    def tokenize(self, text):
        """Tokenize text into lowercase words, stripping non-alphanumeric characters."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())

    def train(self, dataset):
        """Train classifier using Bayes' Theorem."""
        for row in dataset:
            subject = row['Subject']
            message = row['Message']
            label = row['Spam/Ham']
            
            self.class_counts[label] += 1
            words = self.tokenize(f"{subject} {message}")
            
            for w in words:
                self.vocab.add(w)
                self.word_counts[label][w] = self.word_counts[label].get(w, 0) + 1
                self.total_words[label] += 1

    def predict(self, text):
        """
        Calculate Log-Posterior Probability using Bayes Rule:
        log P(C|D) = log P(C) + sum(log P(w|C))
        with Laplace Smoothing:
        P(w|C) = (count(w, C) + alpha) / (total_words_in_C + alpha * |V|)
        """
        words = self.tokenize(text)
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        
        scores = {}
        for c in self.classes:
            # Prior probability log P(C)
            prior = math.log(self.class_counts[c] / total_docs) if total_docs > 0 else math.log(0.5)
            log_likelihood = 0.0
            
            for w in words:
                word_freq = self.word_counts[c].get(w, 0)
                # Laplace smoothed likelihood
                prob_word = (word_freq + self.alpha) / (self.total_words[c] + self.alpha * vocab_size)
                log_likelihood += math.log(prob_word)
            
            scores[c] = prior + log_likelihood

        # Convert log scores to normalized probabilities using Softmax
        max_score = max(scores.values())
        exp_scores = {c: math.exp(score - max_score) for c, score in scores.items()}
        sum_exp = sum(exp_scores.values())
        probabilities = {c: exp_scores[c] / sum_exp for c in self.classes}
        
        predicted_class = max(probabilities, key=probabilities.get)
        return predicted_class, probabilities

# -------------------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------------------
RAW_DATA = [
    {"Subject": "Congratulations! You won a prize", "Message": "You have won a free gift. Claim your prize today!", "Spam/Ham": "Spam"},
    {"Subject": "Urgent account notification", "Message": "Your account needs attention. Verify your details.", "Spam/Ham": "Spam"},
    {"Subject": "Free shopping voucher", "Message": "Get a free shopping voucher worth 5000 rupees. Limited offer!", "Spam/Ham": "Spam"},
    {"Subject": "You are a lucky winner", "Message": "You have been selected for a special reward.", "Spam/Ham": "Spam"},
    {"Subject": "Exclusive cash reward", "Message": "Claim your surprise cash reward now.", "Spam/Ham": "Spam"},
    {"Subject": "Meeting scheduled", "Message": "Our project meeting is scheduled for Monday at 10 AM.", "Spam/Ham": "Ham"},
    {"Subject": "College assignment reminder", "Message": "Please submit your machine learning assignment by Friday.", "Spam/Ham": "Ham"},
    {"Subject": "Monthly electricity bill", "Message": "Your electricity bill is available. Please check your account.", "Spam/Ham": "Ham"},
    {"Subject": "Project discussion", "Message": "Can we discuss the project progress tomorrow?", "Spam/Ham": "Ham"},
    {"Subject": "Birthday invitation", "Message": "You are invited to my birthday party this Saturday.", "Spam/Ham": "Ham"}
]

classifier = PurePythonNaiveBayes(alpha=1.0)
classifier.train(RAW_DATA)

# -------------------------------------------------------------
# 3. HTML + CSS + JS WEB INTERFACE
# -------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Spam Detection System | B.Tech Project</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #f0f2f5; color: #1e293b; padding: 24px; }
        .container { max-width: 900px; margin: 0 auto; }
        header { background: #1e3a8a; color: white; padding: 28px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        header h1 { font-size: 26px; margin-bottom: 6px; }
        header p { color: #93c5fd; font-size: 14px; }
        .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.06); }
        .card h2 { font-size: 18px; margin-bottom: 16px; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
        label { display: block; font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px; }
        input[type="text"], textarea { width: 100%; padding: 12px; border: 1.5px solid #cbd5e1; border-radius: 8px; font-size: 14px; margin-bottom: 16px; }
        input:focus, textarea:focus { outline: none; border-color: #2563eb; ring: 2px solid #bfdbfe; }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
        button { cursor: pointer; font-size: 13px; font-weight: 600; border: none; border-radius: 6px; padding: 8px 14px; transition: all 0.2s; }
        .btn-preset { background: #e2e8f0; color: #334155; }
        .btn-preset:hover { background: #cbd5e1; }
        .btn-primary { background: #2563eb; color: white; padding: 12px 20px; font-size: 15px; width: 100%; border-radius: 8px; }
        .btn-primary:hover { background: #1d4ed8; }
        .result-box { display: none; text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 20px; font-weight: bold; }
        .result-spam { background: #fee2e2; border: 2px solid #ef4444; color: #991b1b; }
        .result-ham { background: #dcfce7; border: 2px solid #22c55e; color: #166534; }
        .badge { display: inline-block; font-size: 20px; margin-bottom: 6px; }
        .meter { background: #e2e8f0; border-radius: 10px; height: 16px; overflow: hidden; margin: 8px 0 16px 0; }
        .meter-fill { height: 100%; transition: width 0.4s ease; }
        .spam-fill { background: #ef4444; }
        .ham-fill { background: #22c55e; }
        .table-wrap { overflow-x: auto; margin-top: 24px; background: white; border-radius: 12px; padding: 20px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #e2e8f0; }
        th { background: #f8fafc; color: #475569; font-weight: 600; }
        .tag-spam { background: #fee2e2; color: #b91c1c; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .tag-ham { background: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛡️ Email Spam Detection System</h1>
            <p>B.Tech 2nd Year Mini Project | Pure Python Multinomial Naive Bayes Algorithm</p>
        </header>

        <div class="grid">
            <!-- Left Column: Input Form -->
            <div class="card">
                <h2>Email Input Form</h2>
                <label>Quick Presets (Click to autofill):</label>
                <div class="btn-group">
                    <button class="btn-preset" onclick="fillPreset(0)">🎁 Won Prize (Spam)</button>
                    <button class="btn-preset" onclick="fillPreset(1)">💳 Voucher (Spam)</button>
                    <button class="btn-preset" onclick="fillPreset(5)">📅 Meeting (Ham)</button>
                    <button class="btn-preset" onclick="fillPreset(6)">📚 Assignment (Ham)</button>
                </div>

                <label for="subject">Email Subject:</label>
                <input type="text" id="subject" placeholder="e.g. You won a cash reward!">

                <label for="message">Email Body / Message:</label>
                <textarea id="message" rows="5" placeholder="e.g. Claim your free gift now."></textarea>

                <button class="btn-primary" onclick="analyzeEmail()">🚀 Classify Email</button>
            </div>

            <!-- Right Column: Results -->
            <div class="card">
                <h2>Prediction Result</h2>
                <div id="resultBox" class="result-box">
                    <div id="resultBadge" class="badge"></div>
                    <div id="resultText"></div>
                </div>

                <div id="metrics" style="display: none;">
                    <div style="display:flex; justify-content:space-between; font-size: 13px; font-weight:600;">
                        <span>Spam Probability:</span>
                        <span id="spamPct">0%</span>
                    </div>
                    <div class="meter"><div id="spamBar" class="meter-fill spam-fill" style="width: 0%;"></div></div>

                    <div style="display:flex; justify-content:space-between; font-size: 13px; font-weight:600;">
                        <span>Ham Probability:</span>
                        <span id="hamPct">0%</span>
                    </div>
                    <div class="meter"><div id="hamBar" class="meter-fill ham-fill" style="width: 0%;"></div></div>
                </div>

                <div id="placeholderText" style="color: #64748b; font-size: 14px; text-align: center; padding: 40px 10px;">
                    Enter an email on the left and click <b>Classify Email</b> to test.
                </div>
            </div>
        </div>

        <!-- Training Dataset Table -->
        <div class="table-wrap">
            <h2 style="font-size: 18px; margin-bottom: 12px;">📊 Dataset Used for Training (10 Curated Emails)</h2>
            <table>
                <thead>
                    <tr><th>#</th><th>Subject</th><th>Message</th><th>Label</th></tr>
                </thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
    </div>

    <script>
        const dataset = """ + json.dumps(RAW_DATA) + """;

        function renderTable() {
            const tbody = document.getElementById("tableBody");
            tbody.innerHTML = dataset.map((d, i) => `
                <tr>
                    <td>${i+1}</td>
                    <td><b>${d.Subject}</b></td>
                    <td>${d.Message}</td>
                    <td><span class="${d['Spam/Ham'] === 'Spam' ? 'tag-spam' : 'tag-ham'}">${d['Spam/Ham']}</span></td>
                </tr>
            `).join("");
        }
        renderTable();

        function fillPreset(index) {
            const item = dataset[index];
            document.getElementById("subject").value = item.Subject;
            document.getElementById("message").value = item.Message;
            analyzeEmail();
        }

        async function analyzeEmail() {
            const subject = document.getElementById("subject").value.trim();
            const message = document.getElementById("message").value.trim();
            if (!subject && !message) {
                alert("Please enter a subject or message!");
                return;
            }

            const response = await fetch("/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ subject, message })
            });
            const data = await response.json();

            document.getElementById("placeholderText").style.display = "none";
            const box = document.getElementById("resultBox");
            const badge = document.getElementById("resultBadge");
            const text = document.getElementById("resultText");
            const metrics = document.getElementById("metrics");

            metrics.style.display = "block";
            box.style.display = "block";

            const spamPct = (data.probabilities.Spam * 100).toFixed(1);
            const hamPct = (data.probabilities.Ham * 100).toFixed(1);

            document.getElementById("spamPct").innerText = spamPct + "%";
            document.getElementById("hamPct").innerText = hamPct + "%";
            document.getElementById("spamBar").style.width = spamPct + "%";
            document.getElementById("hamBar").style.width = hamPct + "%";

            if (data.prediction === "Spam") {
                box.className = "result-box result-spam";
                badge.innerText = "🚨 SPAM DETECTED";
                text.innerText = "This email matches promotional or scam patterns.";
            } else {
                box.className = "result-box result-ham";
                badge.innerText = "🟢 HAM (SAFE EMAIL)";
                text.innerText = "This email looks legitimate and safe.";
            }
        }
    </script>
</body>
</html>
"""

# -------------------------------------------------------------
# 4. HTTP REQUEST HANDLER
# -------------------------------------------------------------
class SpamClassifierHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/classify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            
            combined_text = f"{data.get('subject', '')} {data.get('message', '')}"
            pred, probs = classifier.predict(combined_text)
            
            response_data = {
                "prediction": pred,
                "probabilities": {k: float(v) for k, v in probs.items()}
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Clean logging
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(port=8080):
    print("=" * 65)
    print("🚀 EMAIL SPAM CLASSIFIER - STANDALONE WEB SERVER")
    print("=" * 65)
    print(f"• Dataset: 10 emails (5 Spam, 5 Ham) loaded")
    print(f"• Algorithm: Multinomial Naive Bayes (Laplace Smoothing alpha=1.0)")
    print(f"• Dependencies: ZERO external pip packages needed!")
    print(f"\n🌐 Web app is running at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    # Allow address reuse so restarting server doesn't hit 'Address already in use'
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        httpd = socketserver.TCPServer(("", port), SpamClassifierHandler)
    except OSError:
        port = 8081
        print(f"Port 8080 was busy, switching to http://localhost:{port}")
        httpd = socketserver.TCPServer(("", port), SpamClassifierHandler)

    import webbrowser
    import threading
    threading.Timer(0.8, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")

if __name__ == "__main__":
    run_server()

