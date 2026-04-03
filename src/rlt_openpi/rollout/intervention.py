"""Human intervention interface for real-robot rollouts.

This module provides a stub ``InterventionManager`` that always reports
no intervention.  Subclass it for real hardware setups where an operator
can override the RL agent's actions (e.g. via keyboard or joystick).
"""

from __future__ import annotations

from numpy.typing import NDArray


class InterventionManager:
    """Stub intervention manager.

    Args:
        enabled: Whether intervention checking is active.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def check_intervention(self) -> bool:
        """Return True if the human operator is intervening."""
        return False

    def get_human_action(self, action_dim: int, chunk_length: int) -> NDArray | None:
        """Return the human-provided action chunk, or None.

        Args:
            action_dim: Dimension of a single-step action.
            chunk_length: Number of steps in the chunk.

        Returns:
            Action chunk ``[chunk_length, action_dim]`` if intervening,
            otherwise ``None``.
        """
        return None
