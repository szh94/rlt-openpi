"""Optional Markdown view for recently logged training metrics."""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path


_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
_AXIS_KEYS = ("step", "pretrain/step", "total_env_steps", "total_episodes")


class MetricsMarkdownDashboard:
    """Maintain a VS Code-friendly Markdown view of recent metrics."""

    def __init__(
        self,
        path: Path,
        max_points: int = 100,
        refresh_interval_seconds: float = 10.0,
    ) -> None:
        self.path = path
        self.max_points = max_points
        self.refresh_interval_seconds = refresh_interval_seconds
        self._latest: dict[str, object] = {}
        self._series: dict[str, deque[float]] = {}
        self._last_write_time = -math.inf
        self._write()
        # The first metrics record should replace the initial waiting page immediately.
        self._last_write_time = -math.inf

    def update(self, record: dict[str, object]) -> None:
        """Add one metrics record and refresh Markdown at the configured interval."""
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
        if time.monotonic() - self._last_write_time >= self.refresh_interval_seconds:
            self._write()

    def flush(self) -> None:
        """Write the latest collected metrics regardless of the refresh interval."""
        if self._latest:
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
            f"- File refresh interval: `{self.refresh_interval_seconds:g}` seconds",
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
        self._last_write_time = time.monotonic()


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
