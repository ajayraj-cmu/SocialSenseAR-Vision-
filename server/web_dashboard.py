"""HTTP Web Dashboard for SocialSenseAR.

Serves a live browser dashboard on port 8766 (separate from WebSocket on 8765).

Features:
- Live stats (FPS, segments, voice state) via Server-Sent Events
- Live JPEG frame stream (MJPEG)
- REST commands (blur, unblur, clear, list)
- No external deps beyond standard library + aiohttp (falls back to http.server)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── HTML page ──────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SocialSenseAR Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; height: 100vh; display: flex; flex-direction: column; }
  header { background: #111; border-bottom: 1px solid #222; padding: 10px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.1rem; color: #00d4ff; letter-spacing: 1px; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%; background: #333; transition: background 0.3s; }
  #status-dot.connected { background: #00e676; box-shadow: 0 0 6px #00e676; }
  #status-dot.disconnected { background: #f44336; }
  #status-text { font-size: 0.8rem; color: #888; }

  main { flex: 1; display: grid; grid-template-columns: 1fr 320px; gap: 0; overflow: hidden; }

  /* Camera feed */
  #feed-panel { background: #000; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
  #feed { max-width: 100%; max-height: 100%; object-fit: contain; }
  #feed-placeholder { color: #333; font-size: 1.2rem; text-align: center; }

  /* Right panel */
  #right-panel { background: #111; border-left: 1px solid #1e1e1e; display: flex; flex-direction: column; overflow: hidden; }

  .panel-section { padding: 14px 16px; border-bottom: 1px solid #1e1e1e; }
  .panel-section h3 { font-size: 0.7rem; letter-spacing: 2px; color: #00d4ff; text-transform: uppercase; margin-bottom: 10px; }

  /* Stats grid */
  #stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-card { background: #181818; border-radius: 6px; padding: 8px 10px; }
  .stat-label { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  .stat-value { font-size: 1.4rem; font-weight: bold; color: #fff; line-height: 1.2; }
  .stat-value.good { color: #00e676; }
  .stat-value.warn { color: #ffab40; }
  .stat-value.bad  { color: #f44336; }

  /* Segments list */
  #segs-list { list-style: none; max-height: 160px; overflow-y: auto; }
  #segs-list li { display: flex; align-items: center; gap: 8px; padding: 5px 0; border-bottom: 1px solid #1a1a1a; font-size: 0.82rem; }
  #segs-list li:last-child { border: none; }
  .seg-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .seg-label { flex: 1; }
  .seg-conf { color: #666; font-size: 0.7rem; }
  .seg-effect { background: #1e3a5f; color: #64b5f6; font-size: 0.65rem; padding: 1px 5px; border-radius: 3px; }

  /* Voice state */
  #voice-status { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; }
  #voice-dot { width: 8px; height: 8px; border-radius: 50%; background: #333; }
  #voice-dot.listening { background: #888; }
  #voice-dot.recording { background: #f44336; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  #voice-transcript { margin-top: 6px; font-size: 0.75rem; color: #aaa; min-height: 16px; }
  #voice-last { margin-top: 4px; font-size: 0.75rem; color: #64b5f6; min-height: 16px; }

  /* Command input */
  #cmd-area { padding: 14px 16px; border-top: 1px solid #1e1e1e; margin-top: auto; }
  #cmd-area h3 { font-size: 0.7rem; letter-spacing: 2px; color: #00d4ff; text-transform: uppercase; margin-bottom: 10px; }
  #cmd-row { display: flex; gap: 8px; }
  #cmd-input { flex: 1; background: #181818; border: 1px solid #333; border-radius: 6px; color: #fff; padding: 8px 12px; font-size: 0.9rem; outline: none; }
  #cmd-input:focus { border-color: #00d4ff; }
  #cmd-send { background: #00d4ff; color: #000; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; cursor: pointer; font-size: 0.85rem; }
  #cmd-send:hover { background: #00b8d4; }
  #quick-btns { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .qbtn { background: #1e1e1e; border: 1px solid #333; color: #ccc; border-radius: 4px; padding: 4px 10px; font-size: 0.75rem; cursor: pointer; }
  .qbtn:hover { background: #2a2a2a; border-color: #555; }
  #cmd-log { margin-top: 8px; max-height: 80px; overflow-y: auto; }
  #cmd-log p { font-size: 0.72rem; color: #666; padding: 2px 0; }
  #cmd-log p.ok { color: #00e676; }
  #cmd-log p.err { color: #f44336; }
</style>
</head>
<body>
<header>
  <div id="status-dot" class="disconnected"></div>
  <h1>SocialSenseAR</h1>
  <span id="status-text">Waiting for Quest client...</span>
</header>
<main>
  <div id="feed-panel">
    <img id="feed" src="/mjpeg" alt="" onerror="this.style.display='none'; document.getElementById('feed-placeholder').style.display='block'" style="display:none">
    <div id="feed-placeholder">No camera feed<br><small>Connect a Quest or run test_client --show</small></div>
  </div>

  <div id="right-panel">
    <div class="panel-section">
      <h3>Stats</h3>
      <div id="stats-grid">
        <div class="stat-card"><div class="stat-label">Frame FPS</div><div class="stat-value" id="s-fps">—</div></div>
        <div class="stat-card"><div class="stat-label">Mask FPS</div><div class="stat-value" id="s-mask-fps">—</div></div>
        <div class="stat-card"><div class="stat-label">Segments</div><div class="stat-value" id="s-segs">—</div></div>
        <div class="stat-card"><div class="stat-label">Pipeline</div><div class="stat-value" id="s-pipeline" style="font-size:0.9rem;margin-top:4px">—</div></div>
      </div>
    </div>

    <div class="panel-section">
      <h3>Detected Objects</h3>
      <ul id="segs-list"><li style="color:#444;font-size:0.8rem">None detected</li></ul>
    </div>

    <div class="panel-section">
      <h3>Voice Agent</h3>
      <div id="voice-status">
        <div id="voice-dot"></div>
        <span id="voice-state-text">Disabled</span>
      </div>
      <div id="voice-transcript"></div>
      <div id="voice-last"></div>
    </div>

    <div id="cmd-area">
      <h3>Commands</h3>
      <div id="cmd-row">
        <input id="cmd-input" type="text" placeholder="blur person  /  clear  /  list" />
        <button id="cmd-send">Send</button>
      </div>
      <div id="quick-btns">
        <button class="qbtn" onclick="sendCmd('blur person')">Blur Person</button>
        <button class="qbtn" onclick="sendCmd('blur face')">Blur Face</button>
        <button class="qbtn" onclick="sendCmd('unblur person')">Unblur</button>
        <button class="qbtn" onclick="sendCmd('clear')">Clear All</button>
        <button class="qbtn" onclick="sendCmd('list')">List</button>
      </div>
      <div id="cmd-log"></div>
    </div>
  </div>
</main>

<script>
const SEG_COLORS = ['#00ffff','#ff00ff','#00ff00','#ffff00','#ff8000','#8000ff','#0080ff','#ff0080'];
let connected = false;

// ── SSE stats stream ──────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/events');
  es.onmessage = e => {
    const d = JSON.parse(e.data);
    applyStats(d);
  };
  es.onerror = () => {
    setTimeout(connectSSE, 2000);
  };
}

function applyStats(d) {
  // Connection status
  const dot = document.getElementById('status-dot');
  const stxt = document.getElementById('status-text');
  if (d.client_connected) {
    dot.className = 'connected';
    stxt.textContent = 'Quest client connected';
  } else {
    dot.className = 'disconnected';
    stxt.textContent = 'Waiting for Quest client...';
  }

  // FPS
  const fps = d.fps || 0;
  const fpsEl = document.getElementById('s-fps');
  fpsEl.textContent = fps > 0 ? fps.toFixed(1) : '—';
  fpsEl.className = 'stat-value ' + (fps > 40 ? 'good' : fps > 20 ? 'warn' : fps > 0 ? 'bad' : '');

  const mfps = d.mask_fps || 0;
  const mfpsEl = document.getElementById('s-mask-fps');
  mfpsEl.textContent = mfps > 0 ? mfps.toFixed(1) : '—';
  mfpsEl.className = 'stat-value ' + (mfps > 15 ? 'good' : mfps > 8 ? 'warn' : mfps > 0 ? 'bad' : '');

  document.getElementById('s-segs').textContent = d.num_segments ?? '—';
  document.getElementById('s-pipeline').textContent = d.pipeline || '—';

  // Segments list
  const list = document.getElementById('segs-list');
  const segs = d.segments || [];
  if (segs.length === 0) {
    list.innerHTML = '<li style="color:#444;font-size:0.8rem">None detected</li>';
  } else {
    list.innerHTML = segs.map((s, i) => {
      const color = SEG_COLORS[i % SEG_COLORS.length];
      const eff = s.effect ? `<span class="seg-effect">${s.effect}</span>` : '';
      return `<li>
        <div class="seg-dot" style="background:${color}"></div>
        <span class="seg-label">${s.label}</span>
        ${eff}
        <span class="seg-conf">${(s.confidence*100).toFixed(0)}%</span>
      </li>`;
    }).join('');
  }

  // Voice
  const vs = d.voice || {};
  const vdot = document.getElementById('voice-dot');
  const vtxt = document.getElementById('voice-state-text');
  if (vs.recording) {
    vdot.className = 'recording';
    vtxt.textContent = 'Recording...';
  } else if (vs.listening) {
    vdot.className = 'listening';
    vtxt.textContent = 'Listening (say "Hey Vibe")';
  } else {
    vdot.className = '';
    vtxt.textContent = vs.enabled ? 'Idle' : 'Disabled';
  }
  document.getElementById('voice-transcript').textContent = vs.partial || '';
  document.getElementById('voice-last').textContent = vs.last_response ? '→ ' + vs.last_response : '';

  // Show feed if client connected
  const feedImg = document.getElementById('feed');
  const feedPh = document.getElementById('feed-placeholder');
  if (d.client_connected && d.has_frame) {
    feedImg.style.display = '';
    feedPh.style.display = 'none';
  }
}

// ── Commands ──────────────────────────────────────────────────────────────────
function sendCmd(cmd) {
  cmd = cmd || document.getElementById('cmd-input').value.trim();
  if (!cmd) return;
  document.getElementById('cmd-input').value = '';

  fetch('/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cmd})
  })
  .then(r => r.json())
  .then(d => logCmd(cmd, d.ok, d.message))
  .catch(e => logCmd(cmd, false, String(e)));
}

function logCmd(cmd, ok, msg) {
  const el = document.getElementById('cmd-log');
  const p = document.createElement('p');
  p.className = ok ? 'ok' : 'err';
  p.textContent = (ok ? '✓ ' : '✗ ') + cmd + (msg ? ' — ' + msg : '');
  el.prepend(p);
  while (el.children.length > 8) el.removeChild(el.lastChild);
}

document.getElementById('cmd-send').onclick = () => sendCmd();
document.getElementById('cmd-input').onkeydown = e => { if (e.key === 'Enter') sendCmd(); };

connectSSE();
</script>
</body>
</html>
"""


# ── Server implementation using http.server (stdlib only) ──────────────────────

class WebDashboardServer:
    """Lightweight HTTP server for the browser dashboard.

    Uses only stdlib (http.server + threading) — no aiohttp needed.
    Runs in a daemon thread so it doesn't block the asyncio event loop.

    API:
      GET  /            → dashboard HTML
      GET  /events      → SSE stream (text/event-stream)
      GET  /mjpeg       → MJPEG stream
      POST /command     → JSON body {cmd: "blur person"}
    """

    def __init__(self, port: int = 8766, on_command: Optional[Callable[[str], str]] = None):
        self.port = port
        self.on_command = on_command or (lambda cmd: "ok")

        # Shared state (updated by the WebSocket server)
        self._lock = threading.Lock()
        self._stats: dict = {
            "client_connected": False,
            "fps": 0.0,
            "mask_fps": 0.0,
            "num_segments": 0,
            "pipeline": "—",
            "segments": [],
            "voice": {},
            "has_frame": False,
        }
        self._latest_jpeg: bytes = b""
        self._sse_clients: list = []   # list of queue.Queue

        self._thread: Optional[threading.Thread] = None
        self._httpd = None

    # ── Public update methods (called from WebSocket server) ──────────────────

    def update_stats(self, stats: dict):
        with self._lock:
            self._stats.update(stats)
        self._push_sse(json.dumps(self._stats))

    def update_frame(self, jpeg_bytes: bytes):
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._stats["has_frame"] = True

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self):
        """Start HTTP server in a daemon thread."""
        import queue
        self._queue_cls = queue.Queue
        self._thread = threading.Thread(target=self._run, daemon=True, name="web-dashboard")
        self._thread.start()
        logger.info(f"Web dashboard: http://localhost:{self.port}/")

    def _run(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence access log

            def do_GET(self):
                if self.path in ('/', '/dashboard', '/index.html'):
                    self._send_html(_HTML)
                elif self.path == '/events':
                    self._sse()
                elif self.path == '/mjpeg':
                    self._mjpeg()
                elif self.path == '/stats':
                    with server_ref._lock:
                        data = json.dumps(server_ref._stats).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/command':
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length)
                    try:
                        obj = json.loads(body)
                        cmd = obj.get('cmd', '').strip()
                        result = server_ref.on_command(cmd) if cmd else "empty"
                        resp = json.dumps({"ok": True, "message": result or ""}).encode()
                    except Exception as e:
                        resp = json.dumps({"ok": False, "message": str(e)}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(resp)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def _send_html(self, html: str):
                data = html.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _sse(self):
                """Server-Sent Events — push stats every second."""
                import queue as q_mod
                q = q_mod.Queue(maxsize=10)
                with server_ref._lock:
                    server_ref._sse_clients.append(q)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                try:
                    # Send current state immediately
                    with server_ref._lock:
                        initial = json.dumps(server_ref._stats)
                    self.wfile.write(f"data: {initial}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=5)
                            self.wfile.write(f"data: {msg}\n\n".encode())
                            self.wfile.flush()
                        except q_mod.Empty:
                            # heartbeat
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with server_ref._lock:
                        if q in server_ref._sse_clients:
                            server_ref._sse_clients.remove(q)

            def _mjpeg(self):
                """MJPEG stream."""
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                last_frame = b""
                try:
                    while True:
                        with server_ref._lock:
                            frame = server_ref._latest_jpeg
                        if frame and frame != last_frame:
                            last_frame = frame
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + frame + b"\r\n"
                            )
                            self.wfile.flush()
                        else:
                            time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        try:
            self._httpd = HTTPServer(('0.0.0.0', self.port), Handler)
            self._httpd.serve_forever()
        except Exception as e:
            logger.error(f"Web dashboard failed to start: {e}")

    def _push_sse(self, message: str):
        with self._lock:
            clients = list(self._sse_clients)
        for q in clients:
            try:
                q.put_nowait(message)
            except Exception:
                pass
