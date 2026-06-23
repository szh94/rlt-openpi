"""Non-blocking keyboard listener for human reward signals.

During a real-robot episode the human watches the robot and presses a
single key (no Enter needed) to label the outcome:

    - ``s`` or ``Space`` → **success** (reward +1, episode ends)
    - ``f``              → **failure** (reward  0, episode ends)
    - ``p``              → **progress** (reward +0.5, episode continues)

Uses terminal raw (cbreak) mode for instant keypress detection.
Falls back gracefully when no TTY is available (e.g. headless runs).
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import tty

logger = logging.getLogger(__name__)


class HumanReward:
    """Non-blocking keyboard listener for success/failure/progress signals."""

    def __init__(self, progress_reward: float = 0.5) -> None:
        self._signal: str | None = None
        self._old_settings: list | None = None
        self._raw_mode = False
        self._progress_reward = progress_reward

    def start(self) -> None:
        """Enter raw terminal mode for instant keypress detection."""
        self._signal = None
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw_mode = True
        except (termios.error, OSError):
            self._raw_mode = False
            logger.warning("Raw terminal mode unavailable, falling back to line input (type + Enter)")

    def stop(self) -> None:
        """Restore original terminal settings."""
        if self._raw_mode and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._raw_mode = False
            self._old_settings = None

    @property
    def progress_reward(self) -> float:
        """Reward value for a progress signal."""
        return self._progress_reward

    def check(self) -> str | None:
        """Poll for keypress.  Returns ``'s'``, ``'f'``, ``'p'``, or ``None``.

        Terminal signals (``'s'``, ``'f'``) are latched — once detected they
        are returned on every subsequent call.  The progress signal (``'p'``)
        is **consumed on read**: it is returned once, then cleared so the
        next call returns ``None`` until a new keypress arrives.
        """
        if self._signal is not None:
            return self._signal

        if self._raw_mode:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in ("s", " "):
                    self._signal = "s"
                elif ch == "f":
                    self._signal = "f"
                elif ch == "p":
                    return "p"
        return self._signal
