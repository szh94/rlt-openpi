"""Dependency-free live dashboard for JSONL training metrics."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


_MAX_RECORDS_PER_REQUEST = 2000
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
_AXIS_KEYS = ("step", "pretrain/step", "total_env_steps", "total_episodes")


class MetricsMarkdownDashboard:
    """Maintain a VS Code-friendly Markdown view of recent metrics."""

    def __init__(self, path: Path, max_points: int = 100) -> None:
        self.path = path
        self.max_points = max_points
        self._latest: dict[str, object] = {}
        self._series: dict[str, deque[float]] = {}
        self._write()

    def update(self, record: dict[str, object]) -> None:
        """Add one metrics record and refresh the Markdown file."""
        self._latest.update(record)
        for key, value in record.items():
            if key == "timestamp" or key in _AXIS_KEYS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            self._series.setdefault(key, deque(maxlen=self.max_points)).append(numeric_value)
        self._write()

    def _write(self) -> None:
        timestamp = self._latest.get("timestamp", "waiting for metrics")
        step = next(
            (self._latest[key] for key in _AXIS_KEYS if key in self._latest),
            "-",
        )
        lines = [
            "# RLT OpenPI live metrics",
            "",
            f"- Last update: `{_markdown_text(timestamp)}`",
            f"- Step: `{_markdown_text(step)}`",
            f"- Trend window: last `{self.max_points}` logged points per metric",
            "",
            "## Latest values",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        if self._latest:
            lines.extend(
                f"| `{_markdown_text(key)}` | `{_markdown_text(value)}` |"
                for key, value in sorted(self._latest.items())
            )
        else:
            lines.append("| status | `Waiting for the first logged metrics...` |")

        lines.extend(["", "## Recent trends", ""])
        if self._series:
            for key, values in sorted(self._series.items()):
                lines.extend(
                    [
                        f"### {_markdown_text(key)}",
                        "",
                        f"Latest: `{values[-1]:.6g}`",
                        "",
                        f"`{_sparkline(values)}`",
                        "",
                    ]
                )
        else:
            lines.append("Waiting for numeric metrics...")

        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_path.replace(self.path)


def _sparkline(values: deque[float]) -> str:
    points = list(values)
    low = min(points)
    span = max(points) - low
    if span == 0:
        return _SPARKLINE_CHARS[len(_SPARKLINE_CHARS) // 2] * len(points)
    last_index = len(_SPARKLINE_CHARS) - 1
    return "".join(
        _SPARKLINE_CHARS[round((value - low) / span * last_index)] for value in points
    )


def _markdown_text(value: object) -> str:
    text = str(value).replace("|", "\\|").replace("`", "'")
    return text if len(text) <= 200 else text[:197] + "..."


class MetricsDashboard:
    """Serve a live metrics page backed by an append-only JSONL file."""

    def __init__(self, metrics_path: Path, host: str, port: int) -> None:
        self.metrics_path = metrics_path
        self.dashboard_path = metrics_path.parent / "dashboard.html"
        self.dashboard_path.write_text(_DASHBOARD_HTML, encoding="utf-8")

        handler = partial(_MetricsRequestHandler, metrics_path=metrics_path)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="metrics-dashboard",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the dashboard server and release its port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


class _MetricsRequestHandler(BaseHTTPRequestHandler):
    """Serve the dashboard and incremental metrics API."""

    def __init__(self, *args: object, metrics_path: Path, **kwargs: object) -> None:
        self.metrics_path = metrics_path
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        if request.path in ("/", "/dashboard.html"):
            self._send_bytes(_DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if request.path == "/api/metrics":
            self._serve_metrics(parse_qs(request.query))
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress per-request HTTP logs in the training terminal."""

    def _serve_metrics(self, query: dict[str, list[str]]) -> None:
        try:
            requested_offset = max(0, int(query.get("offset", ["0"])[0]))
        except ValueError:
            requested_offset = 0

        records: list[object] = []
        reset = False
        next_offset = 0
        has_more = False

        try:
            with self.metrics_path.open("rb") as metrics_file:
                metrics_file.seek(0, 2)
                file_size = metrics_file.tell()
                if requested_offset > file_size:
                    requested_offset = 0
                    reset = True
                metrics_file.seek(requested_offset)

                while len(records) < _MAX_RECORDS_PER_REQUEST:
                    line_offset = metrics_file.tell()
                    line = metrics_file.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        metrics_file.seek(line_offset)
                        break
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                next_offset = metrics_file.tell()
                has_more = next_offset < file_size
        except FileNotFoundError:
            pass

        payload = json.dumps(
            {
                "offset": next_offset,
                "records": records,
                "reset": reset,
                "has_more": has_more,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8")

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RLT OpenPI Metrics</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; padding: 24px; background: #0b1020; color: #e5e7eb; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    #status { color: #93c5fd; font-size: 13px; }
    #charts { display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 14px; margin-top: 20px; }
    .card { background: #111827; border: 1px solid #263247; border-radius: 10px; padding: 14px; }
    .metric-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .metric-name { overflow-wrap: anywhere; color: #cbd5e1; }
    .metric-value { font-variant-numeric: tabular-nums; color: #7dd3fc; }
    svg { display: block; width: 100%; height: 130px; }
    .axis { stroke: #334155; stroke-width: 1; }
    .line { fill: none; stroke: #38bdf8; stroke-width: 2; vector-effect: non-scaling-stroke; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }
    th, td { border-bottom: 1px solid #263247; padding: 8px; text-align: left; overflow-wrap: anywhere; }
    th { color: #94a3b8; }
    .empty { margin-top: 24px; color: #94a3b8; }
  </style>
</head>
<body>
  <header>
    <div><h1>RLT OpenPI live metrics</h1><div id="runInfo"></div></div>
    <div id="status">Connecting...</div>
  </header>
  <div id="empty" class="empty">Waiting for the first logged metrics...</div>
  <main id="charts"></main>
  <table id="latest" hidden><thead><tr><th>Metric</th><th>Latest value</th></tr></thead><tbody></tbody></table>
  <script>
    const MAX_POINTS = 1000;
    let offset = 0;
    let sequence = 0;
    const series = new Map();
    const latest = new Map();
    const ignored = new Set(["timestamp", "step", "pretrain/step", "total_env_steps", "total_episodes"]);

    function resetData() { offset = 0; sequence = 0; series.clear(); latest.clear(); }
    function xValue(record) {
      return record.step ?? record["pretrain/step"] ?? record.total_env_steps ?? record.total_episodes ?? sequence;
    }
    function ingest(record) {
      sequence += 1;
      const x = Number(xValue(record));
      for (const [key, value] of Object.entries(record)) {
        latest.set(key, value);
        if (ignored.has(key) || typeof value !== "number" || !Number.isFinite(value)) continue;
        if (!series.has(key)) series.set(key, []);
        const points = series.get(key);
        points.push([Number.isFinite(x) ? x : sequence, value]);
        if (points.length > MAX_POINTS) points.splice(0, points.length - MAX_POINTS);
      }
    }
    function esc(value) {
      return String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
    }
    function format(value) {
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(6);
      return typeof value === "object" ? JSON.stringify(value) : String(value);
    }
    function polyline(points) {
      if (!points.length) return "";
      const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
      const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
      const xspan = xmax - xmin || 1, yspan = ymax - ymin || 1;
      return points.map(([x,y]) => `${((x-xmin)/xspan*100).toFixed(3)},${(100-(y-ymin)/yspan*100).toFixed(3)}`).join(" ");
    }
    function render() {
      document.getElementById("empty").hidden = series.size > 0;
      const charts = document.getElementById("charts");
      charts.innerHTML = [...series.entries()].sort(([a],[b]) => a.localeCompare(b)).map(([key, points]) => `
        <section class="card"><div class="metric-head"><span class="metric-name">${esc(key)}</span>
        <span class="metric-value">${esc(format(points.at(-1)[1]))}</span></div>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none"><line class="axis" x1="0" y1="100" x2="100" y2="100"/>
        <polyline class="line" points="${polyline(points)}"/></svg></section>`).join("");

      const table = document.getElementById("latest");
      table.hidden = latest.size === 0;
      table.querySelector("tbody").innerHTML = [...latest.entries()].sort(([a],[b]) => a.localeCompare(b))
        .map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(format(value))}</td></tr>`).join("");
      const step = latest.get("step") ?? latest.get("pretrain/step") ?? latest.get("total_env_steps") ?? "-";
      document.getElementById("runInfo").textContent = `step: ${step} | records shown: up to ${MAX_POINTS} per metric`;
    }
    async function poll() {
      try {
        const response = await fetch(`/api/metrics?offset=${offset}`, {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (payload.reset) resetData();
        offset = payload.offset;
        payload.records.forEach(ingest);
        render();
        document.getElementById("status").textContent = `Live | ${new Date().toLocaleTimeString()}`;
        setTimeout(poll, payload.has_more ? 20 : 1000);
      } catch (error) {
        document.getElementById("status").textContent = `Disconnected: ${error}`;
        setTimeout(poll, 3000);
      }
    }
    poll();
  </script>
</body>
</html>
"""
