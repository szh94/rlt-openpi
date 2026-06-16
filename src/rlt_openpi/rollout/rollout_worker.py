"""Rollout worker for online RL data collection.

Orchestrates environment interaction, VLA embedding extraction, RL token
encoding, actor inference, and replay buffer storage.  Supports both
VLA-only warmup rollouts and full RL episode collection with optional
human intervention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from rlt_openpi.models.actor import Actor
from rlt_openpi.models.rl_token import RLTokenModel
from rlt_openpi.rollout.intervention import InterventionManager, InterventionResult
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.vla.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Statistics for a single collected episode."""

    total_reward: float = 0.0
    num_chunks: int = 0
    num_steps: int = 0
    done: bool = False
    interventions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutWorker:
    """Collects environment rollouts for online RL training.

    During warmup, runs the VLA-only policy and stores transitions.
    During RL training, uses the actor conditioned on z_rl and VLA
    reference actions, with optional human intervention override.

    Args:
        env: Chunk-level environment wrapper.
        vla: Frozen VLA wrapper for embeddings and reference actions.
        rl_token_model: Frozen RL token encoder (Stage 1 output).
        actor: RL actor network.
        replay_buffer: Buffer to store transitions.
        intervention_mgr: Human intervention manager.
        chunk_length: C, number of steps per action chunk.
        action_dim: Dimension of a single-step action.
        device: Torch device for model inference.
    """

    def __init__(
        self,
        env: Any,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
        actor: Actor,
        replay_buffer: ReplayBuffer,
        intervention_mgr: InterventionManager,
        chunk_length: int,
        action_dim: int,
        device: torch.device | str = "cuda",
        max_deviation: float = 0.3,
        deviation_abort_threshold: float = 0.8,
    ) -> None:
        self.env = env
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.intervention_mgr = intervention_mgr
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.max_deviation = max_deviation
        self.deviation_abort_threshold = deviation_abort_threshold

        self._action_chunk_dim = chunk_length * action_dim

        # Stores the VLA observation state from the most recent
        # _extract_rl_state call, used to unnormalize actor actions
        # before env.step.
        self._last_vla_state: Tensor | None = None

    def _obs_to_vla_input(self, obs: dict[str, Any]) -> Any:
        """Prepare observation dict for VLA inference.

        Uses ``VLAWrapper.preprocess_obs`` if available (real VLA), which
        applies the full OpenPI transform chain and returns an
        ``Observation``.  Falls back to simple batch-wrapping for tests
        with a mock VLA.
        """
        if hasattr(self.vla, "preprocess_obs"):
            return self.vla.preprocess_obs(obs)

        # Fallback: simple batch-wrap (for tests with mock VLA)
        batched: dict[str, Any] = {}
        for key, val in obs.items():
            arr = np.asarray(val)
            batched[key] = arr[np.newaxis]  # add batch dim
        return batched

    @torch.no_grad()
    def _extract_rl_state(self, obs: dict[str, Any]) -> tuple[NDArray, NDArray]:
        """Extract RL state x = cat(z_rl, s^p) and VLA reference chunk.

        Both x and a_tilde are in **normalized** space — suitable for
        Actor input and replay buffer storage.  Actions must be
        unnormalized via :meth:`_unnormalize_action_chunk` before
        ``env.step``.

        Stores the VLA observation state in ``self._last_vla_state``
        for later unnormalization.

        Returns:
            x: RL state [state_dim] as numpy array (normalized).
            a_tilde_flat: Flattened VLA reference chunk [action_chunk_dim]
                as numpy (normalized).
        """
        vla_input = self._obs_to_vla_input(obs)

        # Store state for later unnormalization
        if hasattr(vla_input, "state"):
            self._last_vla_state = vla_input.state.to(
                dtype=torch.float32, device=self.device
            )

        # Extract VLA embeddings and encode into z_rl
        z, pad_mask = self.vla.extract_embeddings(vla_input)
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]

        # Get VLA reference action chunk in normalized space
        a_tilde = self.vla.get_rl_chunk_reference_normalized(vla_input, self.chunk_length)  # [1, C, action_dim]
        a_tilde_flat = a_tilde.reshape(1, -1)  # [1, C*d]

        # Proprioceptive state s^p from the preprocessed VLA observation.
        # DroidInputs merges joint_pos + gripper into state, then
        # PadStatesAndActions zero-pads to the VLA's internal width.
        # Slice to action_dim to drop the padding.
        s_p = vla_input.state[:, :self.action_dim].to(dtype=torch.float32, device=self.device)  # [1, d]

        # RL state: x = cat(z_rl, s^p)
        x = torch.cat([z_rl, s_p], dim=-1)  # [1, state_dim]

        return (
            x.squeeze(0).cpu().numpy(),
            a_tilde_flat.squeeze(0).cpu().numpy(),
        )

    @torch.no_grad()
    def _get_warmup_action(self, obs: dict[str, Any]) -> NDArray:
        """Get normalized VLA reference action for warmup.

        Returns the reference in **normalized** space for buffer storage.
        Call :meth:`_unnormalize_action_chunk` before ``env.step``.

        Returns:
            action_chunk: [C, action_dim] numpy array (normalized).
        """
        vla_input = self._obs_to_vla_input(obs)
        if hasattr(vla_input, "state"):
            self._last_vla_state = vla_input.state.to(
                dtype=torch.float32, device=self.device
            )
        a_tilde = self.vla.get_rl_chunk_reference_normalized(vla_input, self.chunk_length)
        return a_tilde.squeeze(0).cpu().numpy()  # [C, action_dim]

    def _unnormalize_action_chunk(self, action_norm: NDArray) -> NDArray:
        """Convert normalized action chunk to robot space for env.step.

        Requires ``_last_vla_state`` to be set via a prior call to
        ``_extract_rl_state`` or ``_get_warmup_action``.

        Args:
            action_norm: [C, action_dim] normalized action chunk.

        Returns:
            action_chunk: [C, action_dim] in robot space.
        """
        if self._last_vla_state is None:
            raise RuntimeError(
                "_last_vla_state is None — call _extract_rl_state or "
                "_get_warmup_action before _unnormalize_action_chunk"
            )
        a_norm_t = torch.as_tensor(action_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_raw_t = self.vla.unnormalize_actions(a_norm_t, self._last_vla_state)
        return a_raw_t.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _get_actor_action(self, x: NDArray, a_tilde_flat: NDArray) -> NDArray:
        """Get action from the RL actor.

        Args:
            x: RL state [state_dim].
            a_tilde_flat: Flattened VLA reference chunk [action_chunk_dim].

        Returns:
            action_chunk: [C, action_dim] numpy array.
        """
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tilde_t = torch.as_tensor(a_tilde_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

        a_flat = self.actor(x_t, a_tilde_t)  # [1, C*d]

        # Safety: abort if raw deviation is abnormally large
        raw_dev = (a_flat - a_tilde_t).abs()
        raw_max_dev = raw_dev.max().item()
        if raw_max_dev > self.deviation_abort_threshold:
            max_idx = raw_dev.argmax().item()
            logger.error(
                "ABORT: Actor raw deviation %.4f exceeds threshold %.4f "
                "(idx=%d: a_tilde=%.4f, a_actor=%.4f). "
                "Stats — a_tilde: mean=%.4f std=%.4f min=%.4f max=%.4f; "
                "a_actor: mean=%.4f std=%.4f min=%.4f max=%.4f. "
                "The model is producing extreme outputs — check training.",
                raw_max_dev,
                self.deviation_abort_threshold,
                max_idx,
                a_tilde_t[0, max_idx].item(),
                a_flat[0, max_idx].item(),
                a_tilde_t.mean().item(), a_tilde_t.std().item(),
                a_tilde_t.min().item(), a_tilde_t.max().item(),
                a_flat.mean().item(), a_flat.std().item(),
                a_flat.min().item(), a_flat.max().item(),
            )
            raise RuntimeError(
                f"Actor raw max deviation {raw_max_dev:.4f} > "
                f"abort threshold {self.deviation_abort_threshold:.4f} "
                f"(idx={max_idx}, a_tilde={a_tilde_t[0, max_idx].item():.4f}, "
                f"a_actor={a_flat[0, max_idx].item():.4f})"
            )

        # Safety: cap deviation from VLA reference
        deviation = a_flat - a_tilde_t
        deviation = torch.clamp(deviation, -self.max_deviation, self.max_deviation)
        a_flat = a_tilde_t + deviation
        a_flat = a_flat.clamp(-1.0, 1.0)

        return a_flat.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)

    def collect_warmup(self, num_chunks: int) -> int:
        """Run VLA-only policy and store transitions in the replay buffer.

        Collects ``num_chunks`` chunk-level transitions across potentially
        multiple episodes (auto-resets on termination).

        All actions stored in the buffer are in **normalized** space.
        Actions sent to ``env.step`` are unnormalized to robot space.

        Args:
            num_chunks: Number of chunk-level transitions to collect.

        Returns:
            Total number of transitions stored.
        """
        stored = 0
        obs = self.env.reset()

        for _ in range(num_chunks):
            # Get normalized VLA reference (for buffer storage)
            action_chunk_norm = self._get_warmup_action(obs)  # [C, action_dim], normalized

            # Build RL state for this observation (sets _last_vla_state)
            x, a_tilde_flat = self._extract_rl_state(obs)
            a_flat = action_chunk_norm.reshape(-1)  # [C*d], normalized

            # Unnormalize for env.step
            action_chunk_raw = self._unnormalize_action_chunk(action_chunk_norm)

            # Step environment (robot space)
            next_obs, rewards, done, _info = self.env.step(action_chunk_raw)

            # Build next RL state
            next_x, _ = self._extract_rl_state(next_obs)

            # Store transition (all normalized)
            self.replay_buffer.add(
                x=x,
                a=a_flat,
                a_tilde=a_tilde_flat,
                rewards=rewards,
                next_x=next_x,
                done=float(done),
            )
            stored += 1

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        return stored

    def collect_episode(self, store_transitions: bool = True) -> EpisodeStats:
        """Collect a single RL episode using the actor policy.

        All actions in the replay buffer are stored in **normalized** space.
        Actions sent to ``env.step`` are unnormalized to robot space.

        Args:
            store_transitions: Whether to add transitions to the replay
                buffer.  Set to ``False`` during evaluation to avoid
                unnecessary buffer writes.

        Returns:
            Episode statistics.
        """
        stats = EpisodeStats()
        obs = self.env.reset()

        while True:
            # Extract RL state and normalized VLA reference (sets _last_vla_state)
            x, a_tilde_flat = self._extract_rl_state(obs)

            # Check for human intervention.
            # If the intervention manager stepped the robot internally
            # (InterventionResult), we use its outputs directly and skip
            # env.step().  Otherwise fall through to the actor.
            intervention: InterventionResult | None = None
            if self.intervention_mgr.check_intervention():
                intervention = self.intervention_mgr.get_human_action(
                    self.action_dim, self.chunk_length
                )

            if intervention is not None:
                # Intervention: action is in robot space.
                action_chunk_raw = intervention.action_chunk
                next_obs = intervention.next_obs
                rewards = intervention.rewards
                done = intervention.done
                info = intervention.info
                stats.interventions += 1

                # Normalize human action for buffer storage
                a_raw_t = torch.as_tensor(
                    action_chunk_raw, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                a_norm_t = self.vla.normalize_actions(a_raw_t, self._last_vla_state)
                a_flat = a_norm_t.squeeze(0).cpu().numpy().reshape(-1)
            else:
                # Actor: normalized in → normalized out
                action_chunk_norm = self._get_actor_action(x, a_tilde_flat)
                a_flat = action_chunk_norm.reshape(-1)  # [C*d], normalized

                # Unnormalize for env.step
                action_chunk_raw = self._unnormalize_action_chunk(action_chunk_norm)
                next_obs, rewards, done, info = self.env.step(action_chunk_raw)

            if store_transitions:
                # Build next RL state (requires VLA forward pass)
                next_x, _ = self._extract_rl_state(next_obs)

                self.replay_buffer.add(
                    x=x,
                    a=a_flat,
                    a_tilde=a_tilde_flat,
                    rewards=rewards,
                    next_x=next_x,
                    done=float(done),
                )

            # Update stats
            stats.total_reward += float(rewards.sum())
            stats.num_chunks += 1
            # Use env-reported steps if available, else fall back to chunk_length
            stats.num_steps += info.get("steps_executed", self.chunk_length)

            if done:
                stats.done = True
                stats.extra = info
                break

            obs = next_obs

        return stats
