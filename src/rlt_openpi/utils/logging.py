"""Unified logger wrapping wandb and stdout."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Logger configuration.

    Args:
        project: wandb project name.
        enabled: Whether wandb logging is active.
    """

    project: str = "rlt-openpi"
    enabled: bool = True


class Logger:
    """Metric logger wrapping wandb + stdout.

    Args:
        config: Logger configuration.
        run_config: Training config dict to log as wandb run config.
    """

    def __init__(
        self,
        config: LoggerConfig,
        run_config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = config
        self._wandb_run = None

        if config.enabled:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=config.project,
                    name=run_name,
                    config=run_config or {},
                )
                logger.info("wandb run initialized: %s", self._wandb_run.url)
            except Exception:
                logger.warning("wandb init failed, falling back to stdout only", exc_info=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Key-value pairs to log.
            step: Optional global step for wandb x-axis.
        """
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        """Finalize the wandb run."""
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
        )
        run_name = getattr(train_config, "run_name", None) or None
        return Logger(config=logger_config, run_config=cfg_dict, run_name=run_name)
