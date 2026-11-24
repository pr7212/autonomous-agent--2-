from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.util import redact, truncate

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Agent activity</title>
<style>
body{font:15px system-ui;margin:0;background:#111827;color:#e5e7eb}main{max-width:1100px;margin:32px auto;padding:0 20px}
header,.card{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:18px;margin:12px 0}
header{display:flex;justify-content:space-between;align-items:center}h1{font-size:21px;margin:0}
button{background:#b91c1c;color:white;border:0;border-radius:6px;padding:10px 14px;font-weight:700;cursor:pointer}
button.secondary{background:#374151}button:disabled{opacity:.5;cursor:not-allowed}
.muted{color:#9ca3af}.row{display:flex;gap:16px;flex-wrap:wrap}.row>div{flex:1;min-width:220px}
.bar{height:10px;background:#374151;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:#34d399;width:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;background:#111827;padding:12px;border-radius:6px}
img{max-width:100%;border-radius:6px;background:#111827}ol{padding-left:22px}li{margin:8px 0}
</style></head>
<body><main>
<header><div><h1>Agent activity</h1><span class="muted">Local dashboard · screenshots remain in memory</span></div>
<div><button id="stop" class="secondary">Stop task</button> <button id="emergency-stop">Emergency stop</button></div></header>
<section class="card"><div class="row"><div><b>Task</b><div id="objective" class="muted">Waiting for task…</div></div>
<div><b>Status</b><div id="status">—</div></div><div><b>Current application</b><div id="app">—</div></div>
<div><b>Monitoring</b><div id="monitor-state">—</div><button id="monitor-toggle" class="secondary">Start monitoring</button></div>
<div><b>Emergency stop</b><div id="stopstate">Ready</div></div></div>
<p><span id="progress-label">0 / 0 verified steps</span></p><div class="bar"><i id="progress"></i></div><ol id="steps"></ol></section>
<section class="card"><b>System resources and network</b><pre id="system">System monitoring is available only during an active task.</pre></section>
<section class="card"><b>Latest approved screen capture</b><p class="muted" id="screen-note">No screenshot available. Enable AGENT_DESKTOP_SCREENSHOTS=1 to show redacted active-window captures.</p>
<img id="screen" alt="Redacted active application screenshot" hidden></section>
<section class="card"><b>Execution timeline</b><ol id="timeline"></ol></section>
<section class="card"><b>Live event log</b><pre id="events">Connecting…</pre></section>
</main><script>
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'});const s=await r.json();
document.getElementById('objective').textContent=s.task?.objective||'Waiting for task…';
document.getElementById('status').textContent=s.task?.status||'—';
document.getElementById('app').textContent=s.current_app||'—';
document.getElementById('stopstate').textContent=s.emergency_stop?'STOP REQUESTED':'Ready';
document.getElementById('monitor-state').textContent=s.monitoring?.enabled?'Monitoring active':'Monitoring paused';
document.getElementById('monitor-toggle').textContent=s.monitoring?.enabled?'Stop monitoring':'Start monitoring';
document.getElementById('monitor-toggle').disabled=!s.monitoring?.active_task;
window.monitorEnabled=!!s.monitoring?.enabled;
document.getElementById('progress-label').textContent=`${s.task?.verified||0} / ${s.task?.total||0} verified steps`;
document.getElementById('progress').style.width=`${s.task?.percent||0}%`;
document.getElementById('steps').innerHTML='';for(const step of (s.task?.steps||[])){const li=document.createElement('li');li.textContent=`${step.status.toUpperCase()} · ${step.description}`;document.getElementById('steps').appendChild(li)}
document.getElementById('events').textContent=(s.events||[]).map(e=>`${e.ts} [${e.kind}] ${e.message}`).join('\\n')||'No events yet';
document.getElementById('timeline').innerHTML='';for(const event of (s.events||[]).slice(-30)){const li=document.createElement('li');li.textContent=`${event.ts} · ${event.kind} · ${event.message}`;document.getElementById('timeline').appendChild(li)}
document.getElementById('system').textContent=s.system?JSON.stringify(s.system,null,2):'Monitoring is paused. Start monitoring during an active task to view telemetry.';
const image=document.getElementById('screen');const note=document.getElementById('screen-note');
if(s.screenshot){image.src='/api/screenshot?t='+Date.now();image.hidden=false;note.hidden=true}else{image.hidden=true;note.hidden=false}
}catch(e){document.getElementById('events').textContent='Dashboard connection error: '+e}}
document.getElementById('stop').onclick=async()=>{if(confirm('Request a cooperative task stop?')){await fetch('/api/stop',{method:'POST'});}};
document.getElementById('emergency-stop').onclick=async()=>{if(confirm('Emergency stop the task at its next safe point?')){await fetch('/api/emergency-stop',{method:'POST'});}};
document.getElementById('monitor-toggle').onclick=async()=>{const r=await fetch('/api/monitoring',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:!window.monitorEnabled})});if(!r.ok){const e=await r.json();alert(e.error||'Could not change monitoring state')}refresh();};
refresh();setInterval(refresh,1200);
</script></body></html>"""


class DashboardServer:
    """A read-only local dashboard with a single same-origin emergency-stop endpoint."""

    def __init__(
        self, history, stop, desktop, secrets=(), host: str = "127.0.0.1",
        port: int = 8765, activity=None,
    ):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("dashboard must bind to loopback")
        self.history = history
        self.stop = stop
        self.desktop = desktop
        self.activity = activity
        self.secrets = [s for s in secrets if s]
        self.host = host
        self.server = None
        self.thread = None

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body = PAGE.encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", body)
                elif self.path.split("?", 1)[0] == "/api/state":
                    self._send_json(owner._state())
                elif self.path.split("?", 1)[0] == "/api/screenshot":
                    image = owner.desktop.latest_screenshot
                    if image is None or (owner.activity and not owner.activity.status()["enabled"]):
                        self._send(404, "text/plain; charset=utf-8", b"No screenshot available")
                    else:
                        self._send(200, "image/png", image)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")

            def do_POST(self):
                expected_origin = f"http://{self.headers.get('Host', '')}"
                if self.headers.get("Origin") != expected_origin:
                    self._send(403, "text/plain; charset=utf-8", b"Forbidden")
                    return
                if self.path in ("/api/stop", "/api/emergency-stop"):
                    if self.path == "/api/stop":
                        owner.stop.request_via_file()
                    else:
                        owner.stop.trigger("dashboard emergency stop")
                    if owner.activity:
                        try:
                            owner.activity.set_monitoring(False)
                        except RuntimeError:
                            pass
                    self._send(200, "application/json; charset=utf-8", b'{"ok":true}')
                    return
                if self.path == "/api/monitoring":
                    if not owner.activity:
                        self._send(404, "application/json; charset=utf-8", b'{"error":"monitoring unavailable"}')
                        return
                    try:
                        size = int(self.headers.get("Content-Length", "0"))
                        if size < 1 or size > 100:
                            raise ValueError("invalid request size")
                        request = json.loads(self.rfile.read(size))
                        if not isinstance(request, dict):
                            raise ValueError("request must be an object")
                        state = owner.activity.set_monitoring(request.get("enabled"))
                    except (ValueError, json.JSONDecodeError) as exc:
                        self._send(400, "application/json; charset=utf-8", json.dumps({"error": str(exc)}).encode())
                    except RuntimeError as exc:
                        self._send(409, "application/json; charset=utf-8", json.dumps({"error": str(exc)}).encode())
                    else:
                        self._send(200, "application/json; charset=utf-8", json.dumps(state).encode())
                    return
                self._send(404, "text/plain; charset=utf-8", b"Not found")

            def _send_json(self, data):
                body = redact(json.dumps(data, ensure_ascii=False), owner.secrets).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)

            def _send(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self._handler = Handler
        self.port = port

    def start(self) -> str:
        if self.server is not None:
            return self.url
        self.server = ThreadingHTTPServer((self.host, self.port), self._handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, name="agent-dashboard", daemon=True)
        self.thread.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def close(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)
        self.server = None
        self.thread = None

    def _state(self) -> dict:
        rows = self.history.db.query("SELECT state_json FROM tasks ORDER BY updated_at DESC LIMIT 1")
        task = None
        if rows:
            stored = json.loads(rows[0]["state_json"])
            steps = stored.get("steps", [])
            total = len(steps)
            verified = sum(1 for s in steps if s.get("status") == "verified")
            task = {
                "id": stored.get("id", ""),
                "objective": truncate(str(stored.get("objective", "")), 1000),
                "status": stored.get("status", ""),
                "verified": verified,
                "total": total,
                "percent": round(100 * verified / total) if total else 0,
                "steps": [
                    {"status": s.get("status", ""), "description": truncate(str(s.get("description", "")), 300)}
                    for s in steps
                ],
            }
        events = []
        if task:
            events = self.history.events(task["id"])[-120:]
            events = [
                {"ts": e["ts"], "kind": e["kind"], "message": truncate(e["message"], 400)}
                for e in events
            ]
        monitoring = self.activity.status() if self.activity else {
            "active_task": False, "enabled": False, "current_app": "",
        }
        system = self.activity.system_snapshot() if self.activity else None
        return {
            "task": task,
            "events": events,
            "current_app": (
                monitoring.get("current_app") if monitoring.get("enabled")
                else (self.desktop.current_app if self.activity is None else "")
            ),
            "monitoring": monitoring,
            "system": system,
            "screenshot": (
                self.desktop.latest_screenshot is not None
                and (self.activity is None or monitoring.get("enabled"))
            ),
            "emergency_stop": self.stop.is_set(),
        }
