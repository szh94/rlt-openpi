"""VR-based human intervention for Franka Panda online RL.

Uses the Oculus VR controller (via DROID's ``VRPolicy``) to let a human
operator take over the robot during RL rollouts.  Grip button toggles
intervention; A/B buttons signal success/failure.

The manager steps the DROID env directly at the single-action level
(required because ``VRPolicy`` computes relative velocity commands from
the current robot state) and returns an ``InterventionResult`` so the
rollout worker skips its own ``env.step()`` call.

Usage::

    python scripts/train_online_rl.py \\
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \\
        --intervention-factory rlt_openpi.envs.franka.intervention.make_vr_intervention \\
        --task-prompt "stack the three blocks on the tray" \\
        ...
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from rlt_openpi.rollout.intervention import InterventionManager, InterventionResult

logger = logging.getLogger(__name__)


class VRInterventionManager(InterventionManager):
    """Oculus VR controller intervention for Franka Panda.

    Reads the VR controller state via DROID's ``VRPolicy``.  When the
    grip button is held (``movement_enabled``), the human controls the
    robot for the duration of one action chunk.

    The manager steps the DROID env directly at each sub-step so that
    ``VRPolicy.forward()`` always sees the real robot state.

    Args:
        droid_env: Raw DROID ``RobotEnv`` instance (shared with the
            ``RobotEnv`` wrapper created by the env factory).
        get_obs_fn: Callable that converts a DROID observation into the
            model-format dict (same function used by the env factory).
        control_hz: Robot control frequency.
    """

    def __init__(
        self,
        droid_env: Any,
        get_obs_fn: Any,
        control_hz: int = 15,
    ) -> None:
        super().__init__(enabled=True)
        self.droid_env = droid_env
        self.get_obs_fn = get_obs_fn
        self.control_hz = control_hz
        self._control_period = 1.0 / control_hz

        import sys
        sys.path.insert(0, "/home/alin/franka_teleop")
        sys.path.insert(0, "/home/alin/franka_teleop/droid/fairo/polymetis/polymetis/python")
        from droid.controllers.oculus_controller import VRPolicy

        self.vr = VRPolicy()
        logger.info("VRInterventionManager initialized (control_hz=%d)", control_hz)

    def check_intervention(self) -> bool:
        """Return True when the VR grip button is held."""
        info = self.vr.get_info()
        if not info["controller_on"]:
            return False
        return info["movement_enabled"]

    def get_human_action(
        self, action_dim: int, chunk_length: int
    ) -> InterventionResult | None:
        """Collect one chunk of VR-teleoperated actions.

        Steps the DROID env at the control frequency for up to
        ``chunk_length`` steps.  Terminates early if the operator
        releases the grip or presses A (success) / B (failure).

        Args:
            action_dim: Dimension of a single-step action.
            chunk_length: Number of steps in the chunk (C).

        Returns:
            ``InterventionResult`` with the collected action chunk and
            env outputs, or ``None`` if the controller disconnected.
        """
        info = self.vr.get_info()
        if not info["controller_on"]:
            return None

        actions: list[np.ndarray] = []
        rewards = np.zeros(chunk_length, dtype=np.float32)
        done = False
        result_info: dict[str, Any] = {}
        steps_executed = 0

        for k in range(chunk_length):
            t_start = time.time()

            # Read the current DROID observation (includes robot_state)
            droid_obs = self.droid_env.get_observation()

            # Compute VR controller action relative to current robot state.
            # Returns [7]: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            vr_action = self.vr.forward(droid_obs)

            # Adapt to the env's action_dim (pad or truncate)
            if len(vr_action) < action_dim:
                vr_action = np.concatenate([
                    vr_action,
                    np.zeros(action_dim - len(vr_action), dtype=vr_action.dtype),
                ])
            elif len(vr_action) > action_dim:
                vr_action = vr_action[:action_dim]

            # Step the robot
            self.droid_env.step(vr_action)
            actions.append(vr_action)
            steps_executed += 1

            # Check VR buttons for episode termination
            vr_info = self.vr.get_info()
            if vr_info["success"]:
                rewards[k] = 1.0
                done = True
                result_info["success"] = True
                logger.info("VR intervention: SUCCESS (A button)")
                break
            if vr_info["failure"]:
                done = True
                result_info["success"] = False
                logger.info("VR intervention: FAILURE (B button)")
                break

            # Stop if operator released the grip mid-chunk
            if not vr_info["movement_enabled"]:
                logger.info("VR intervention: grip released after %d/%d steps", k + 1, chunk_length)
                break

            # Enforce control frequency
            elapsed = time.time() - t_start
            sleep_time = self._control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Zero-pad if we terminated early
        while len(actions) < chunk_length:
            actions.append(np.zeros(action_dim, dtype=np.float32))

        result_info["steps_executed"] = steps_executed

        # Read the final observation in model format
        next_obs = self.get_obs_fn()

        return InterventionResult(
            action_chunk=np.stack(actions, axis=0),
            next_obs=next_obs,
            rewards=rewards,
            done=done,
            info=result_info,
        )


def make_vr_intervention(env: Any, **kwargs: Any) -> VRInterventionManager:
    """Factory function for creating a VRInterventionManager.

    Expects ``env`` to be a ``RobotEnv`` created by
    ``rlt_openpi.envs.franka.env_factory.make_franka_env``, which stashes the
    raw DROID env and observation function as attributes.

    Args:
        env: The ``RobotEnv`` returned by the env factory.
        **kwargs: Forwarded to ``VRInterventionManager``.

    Returns:
        Configured ``VRInterventionManager``.
    """
    if not hasattr(env, "droid_env") or not hasattr(env, "droid_get_obs_fn"):
        raise AttributeError(
            "env is missing 'droid_env' or 'droid_get_obs_fn'. "
            "Make sure you're using rlt_openpi.envs.franka.env_factory.make_franka_env "
            "which attaches these attributes."
        )

    return VRInterventionManager(
        droid_env=env.droid_env,
        get_obs_fn=env.droid_get_obs_fn,
        control_hz=kwargs.pop("control_hz", 15),
        **kwargs,
    )
