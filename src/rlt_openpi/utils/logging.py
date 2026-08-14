"""Unified JSONL/wandb logger with optional live Markdown metrics."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from rlt_openpi.utils.metrics_dashboard import MetricsMarkdownDashboard


@dataclass
class LoggerConfig:
    """Logger configuration.

    Args:
        project: wandb project name.
        enabled: Whether wandb logging is active.
        live_metrics_enabled: Whether to maintain ``metrics_live.md``.
    """

    project: str = "rlt-openpi"
    enabled: bool = True
    live_metrics_enabled: bool = True


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
            if config.live_metrics_enabled:
                markdown_path = resolved_metrics_path.parent / "metrics_live.md"
                try:
                    self._markdown_dashboard = MetricsMarkdownDashboard(markdown_path)
                    print(f"Live metrics file: {markdown_path}")
                except OSError as exc:
                    print(f"[WARNING] live metrics file failed to initialize: {exc}")

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
        if self._markdown_dashboard is not None:
            try:
                self._markdown_dashboard.flush()
            except OSError as exc:
                print(f"[WARNING] final live metrics file update failed: {exc}")
            self._markdown_dashboard = None
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
            live_metrics_enabled=_env_flag("RLT_METRICS_LIVE", default=True),
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


def _env_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    print(f"[WARNING] invalid {name}={raw_value!r}; using default {default}")
    return default
