"""Unified JSONL/wandb logger with an optional live metrics dashboard.

Set ``RLT_METRICS_PORT`` to enable the dashboard. It binds to
``RLT_METRICS_HOST`` (``127.0.0.1`` by default).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from rlt_openpi.utils.metrics_dashboard import MetricsDashboard, MetricsMarkdownDashboard


@dataclass
class LoggerConfig:
    """Logger configuration.

    Args:
        project: wandb project name.
        enabled: Whether wandb logging is active.
        dashboard_host: Host interface for the optional live dashboard.
        dashboard_port: Dashboard port, or zero to disable it.
    """

    project: str = "rlt-openpi"
    enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 0


class Logger:
    """Metric logger writing local JSONL and optionally wandb.

    Args:
        config: Logger configuration.
        run_config: Training config dict to log as wandb run config.
        run_name: Optional wandb run name.
        metrics_path: Optional path for the append-only JSONL metrics file.
    """

    def __init__(
        self,
        config: LoggerConfig,
        run_config: dict[str, Any] | None = None,
        run_name: str | None = None,
        metrics_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self._wandb_run = None
        self._metrics_file: TextIO | None = None
        self._dashboard: MetricsDashboard | None = None
        self._markdown_dashboard: MetricsMarkdownDashboard | None = None

        if metrics_path is not None:
            resolved_metrics_path = Path(metrics_path)
            resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = resolved_metrics_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            print(f"Local metrics log: {resolved_metrics_path}")
            markdown_path = resolved_metrics_path.parent / "metrics_live.md"
            try:
                self._markdown_dashboard = MetricsMarkdownDashboard(markdown_path)
                print(f"Live metrics file: {markdown_path}")
            except OSError as exc:
                print(f"[WARNING] live metrics file failed to initialize: {exc}")

            if config.dashboard_port > 0:
                if config.dashboard_host not in ("127.0.0.1", "localhost"):
                    print(
                        "[WARNING] metrics dashboard is not bound to loopback; "
                        "ensure the port is protected"
                    )
                try:
                    self._dashboard = MetricsDashboard(
                        metrics_path=resolved_metrics_path,
                        host=config.dashboard_host,
                        port=config.dashboard_port,
                    )
                    print(
                        "Live metrics dashboard: "
                        f"http://{config.dashboard_host}:{config.dashboard_port}"
                    )
                    print(
                        "VS Code Remote-SSH: forward remote port "
                        f"{config.dashboard_port} in the Ports panel"
                    )
                except OSError as exc:
                    print(f"[WARNING] metrics dashboard failed to start: {exc}")

        if config.enabled:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=config.project,
                    name=run_name,
                    config=run_config or {},
                )
                print(f"wandb run initialized: {self._wandb_run.url}")
            except Exception:
                print("[WARNING] wandb init failed; continuing without wandb")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Key-value pairs to log.
            step: Optional global step for wandb x-axis.
        """
        if self._metrics_file is not None:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **metrics,
            }
            if step is not None:
                record["step"] = step
            json_record = {
                str(key): _to_json_value(value) for key, value in record.items()
            }
            self._metrics_file.write(
                json.dumps(json_record, ensure_ascii=False) + "\n"
            )
            self._metrics_file.flush()
            if self._markdown_dashboard is not None:
                try:
                    self._markdown_dashboard.update(json_record)
                except OSError as exc:
                    print(f"[WARNING] live metrics file update failed: {exc}")
                    self._markdown_dashboard = None

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        """Flush local metrics and finalize the wandb run."""
        if self._dashboard is not None:
            self._dashboard.close()
            self._dashboard = None
        if self._metrics_file is not None:
            self._metrics_file.flush()
            self._metrics_file.close()
            self._metrics_file = None
        if self._wandb_run is not None:
            self._wandb_run.finish()

    @staticmethod
    def from_train_config(train_config: Any) -> Logger:
        """Create a Logger from a training config dataclass.

        Reads ``wandb_project`` and ``wandb_enabled`` fields if present.
        """
        cfg_dict = asdict(train_config) if hasattr(train_config, "__dataclass_fields__") else {}
        logger_config = LoggerConfig(
            project=getattr(train_config, "wandb_project", "rlt-openpi"),
            enabled=getattr(train_config, "wandb_enabled", True),
            dashboard_host=os.environ.get("RLT_METRICS_HOST", "127.0.0.1"),
            dashboard_port=_dashboard_port_from_env(),
        )
        run_name = getattr(train_config, "run_name", None) or None
        save_dir = Path(getattr(train_config, "save_dir", "checkpoints"))
        metrics_path = save_dir / run_name / "metrics.jsonl" if run_name else None
        return Logger(
            config=logger_config,
            run_config=cfg_dict,
            run_name=run_name,
            metrics_path=metrics_path,
        )


def _to_json_value(value: Any) -> Any:
    """Convert common metric containers and scalar types to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return _to_json_value(item_fn())
        except (TypeError, ValueError, RuntimeError):
            pass

    tolist_fn = getattr(value, "tolist", None)
    if callable(tolist_fn):
        try:
            return _to_json_value(tolist_fn())
        except (TypeError, ValueError, RuntimeError):
            pass

    return str(value)


def _dashboard_port_from_env() -> int:
    """Read and validate the optional dashboard port."""
    raw_port = os.environ.get("RLT_METRICS_PORT", "").strip()
    if not raw_port:
        return 0
    try:
        port = int(raw_port)
    except ValueError:
        print(f"[WARNING] invalid RLT_METRICS_PORT={raw_port!r}; dashboard disabled")
        return 0
    if not 0 <= port <= 65535:
        print(f"[WARNING] RLT_METRICS_PORT must be 0-65535; dashboard disabled")
        return 0
    return port
