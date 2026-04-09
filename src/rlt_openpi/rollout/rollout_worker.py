"""Rollout worker for online RL data collection.

Orchestrates environment interaction, VLA embedding extraction, RL token
encoding, actor inference, and replay buffer storage.  Supports both
VLA-only warmup rollouts and full RL episode collection with optional
human intervention.
"""

from __future__ import annotations

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

        self._action_chunk_dim = chunk_length * action_dim

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

        Returns:
            x: RL state [state_dim] as numpy array.
            a_tilde_flat: Flattened VLA reference chunk [action_chunk_dim] as numpy.
        """
        vla_input = self._obs_to_vla_input(obs)

        # Extract VLA embeddings and encode into z_rl
        z, pad_mask = self.vla.extract_embeddings(vla_input)
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]

        # Get VLA reference action chunk (first C steps)
        a_tilde = self.vla.get_rl_chunk_reference(vla_input, self.chunk_length)  # [1, C, action_dim]
        a_tilde_flat = a_tilde.reshape(1, -1)  # [1, C*d]

        # Proprioceptive state s^p from observation
        state = np.asarray(obs["state"], dtype=np.float32)
        s_p = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, d]

        # RL state: x = cat(z_rl, s^p)
        x = torch.cat([z_rl, s_p], dim=-1)  # [1, state_dim]

        return (
            x.squeeze(0).cpu().numpy(),
            a_tilde_flat.squeeze(0).cpu().numpy(),
        )

    @torch.no_grad()
    def _get_warmup_action(self, obs: dict[str, Any]) -> NDArray:
        """Get action from VLA-only policy for warmup.

        Returns:
            action_chunk: [C, action_dim] numpy array.
        """
        vla_input = self._obs_to_vla_input(obs)
        a_tilde = self.vla.get_rl_chunk_reference(vla_input, self.chunk_length)  # [1, C, action_dim]
        return a_tilde.squeeze(0).cpu().numpy()  # [C, action_dim]

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
        return a_flat.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)

    def collect_warmup(self, num_chunks: int) -> int:
        """Run VLA-only policy and store transitions in the replay buffer.

        Collects ``num_chunks`` chunk-level transitions across potentially
        multiple episodes (auto-resets on termination).

        Args:
            num_chunks: Number of chunk-level transitions to collect.

        Returns:
            Total number of transitions stored.
        """
        stored = 0
        obs = self.env.reset()

        for _ in range(num_chunks):
            # Get VLA reference action (used as both executed and reference)
            action_chunk = self._get_warmup_action(obs)  # [C, action_dim]

            # Build RL state for this observation
            x, a_tilde_flat = self._extract_rl_state(obs)
            a_flat = action_chunk.reshape(-1)  # [C*d]

            # Step environment
            next_obs, rewards, done, _info = self.env.step(action_chunk)

            # Build next RL state
            next_x, _ = self._extract_rl_state(next_obs)

            # Store transition
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

        At each chunk boundary:
        1. Extract z_rl and VLA reference actions
        2. Form RL state x = cat(z_rl, s^p)
        3. Check for human intervention
        4. Run actor (or use human action) to get action chunk
        5. Step environment for C steps
        6. Store transition in replay buffer (if ``store_transitions``)

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
            # Extract RL state and VLA reference
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
                action_chunk = intervention.action_chunk
                next_obs = intervention.next_obs
                rewards = intervention.rewards
                done = intervention.done
                info = intervention.info
                stats.interventions += 1
            else:
                action_chunk = self._get_actor_action(x, a_tilde_flat)
                next_obs, rewards, done, info = self.env.step(action_chunk)

            a_flat = action_chunk.reshape(-1)  # [C*d]

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
