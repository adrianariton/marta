from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import uuid
import threading

app = Flask(__name__)

# Stores: {id: {"messages": [...], "event": Event(), "response": None}}
pending_requests = {}
lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SLURM Human-in-the-Loop (Flask)</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 15px; height: 100vh; padding: 15px; box-sizing: border-box; }
        .pane { border: 1px solid #444; background: #2d2d2d; display: flex; flex-direction: column; border-radius: 8px; overflow: hidden; position: relative; }
        .header { background: #3d3d3d; padding: 10px; font-size: 0.8em; color: #aaa; border-bottom: 1px solid #444; }
        .chat { flex-grow: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 8px; }
        .msg { padding: 8px 12px; border-radius: 12px; max-width: 85%; font-size: 0.9em; }
        .user { background: #007bff; align-self: flex-end; }
        .assistant { background: #444; align-self: flex-start; }
        .footer { padding: 10px; background: #252525; border-top: 1px solid #444; }
        textarea { width: 100%; height: 50px; background: #111; color: #fff; border: 1px solid #555; padding: 5px; box-sizing: border-box; }
        button { width: 100%; padding: 8px; background: #28a745; color: white; border: none; cursor: pointer; margin-top: 5px; }
        .empty { display: flex; align-items: center; justify-content: center; color: #555; }
    </style>
    <script>
        // Poll for updates every 2 seconds
        setInterval(() => {
            fetch('/check-updates').then(r => r.json()).then(data => {
                if (data.reload) window.location.reload();
            });
        }, 2000);
    </script>
</head>
<body>
    <div class="grid">
        {% for req_id, data in requests.items() %}
        <div class="pane">
            <div class="header">ID: {{ req_id[:8] }}</div>
            <div class="chat">
                {% for m in data.messages %}
                    <div class="msg {{ m.role }}"><strong>{{ m.role }}:</strong> {{ m.content }}</div>
                {% endfor %}
            </div>
            <div class="footer">
                <form action="/submit" method="post">
                    <input type="hidden" name="id" value="{{ req_id }}">
                    <textarea name="response" required></textarea>
                    <button type="submit">Submit</button>
                </form>
            </div>
        </div>
        {% endfor %}
        {% for i in range(4 - (requests|length)) %}
            <div class="pane empty">Waiting for SLURM...</div>
        {% endfor %}
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    with lock:
        # Show only the first 4 pending requests
        display = dict(list(pending_requests.items())[:4])
    return render_template_string(HTML_TEMPLATE, requests=display)


@app.route("/check-updates")
def check_updates():
    # Simple check for the frontend to know when to refresh
    return jsonify({"reload": len(pending_requests) > 0})


@app.route("/get-input", methods=["POST"])
def handle_slurm_request():
    data = request.json
    req_id = data.get("id") or str(uuid.uuid4())

    event = threading.Event()
    with lock:
        pending_requests[req_id] = {
            "messages": data.get("messages", []),
            "event": event,
            "response": None,
        }

    # This blocks the SLURM request thread until event.set() is called
    got_signal = event.wait(timeout=1200)  # 20 minute timeout

    with lock:
        res_data = pending_requests.pop(req_id, None)

    if not got_signal or res_data is None:
        return jsonify({"response": "[Timeout or Error]"})

    return jsonify({"response": res_data["response"]})


@app.route("/submit", methods=["POST"])
def submit():
    req_id = request.form.get("id")
    user_text = request.form.get("response")

    with lock:
        if req_id in pending_requests:
            pending_requests[req_id]["response"] = user_text
            pending_requests[req_id]["event"].set()

    return redirect(url_for("index"))


if __name__ == "__main__":
    # threaded=True is required to handle multiple SLURM nodes at once
    app.run(host="0.0.0.0", port=6767, threaded=True)
