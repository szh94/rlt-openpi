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
from rlt_openpi.envs.intervention import InterventionManager, InterventionResult
from rlt_openpi.envs.envbase.reward import HumanReward
from rlt_openpi.training.replay_buffer import ReplayBuffer
from rlt_openpi.vla.vla_wrapper import VLAWrapper


def _format_array_2f(value: Any) -> str:
    """Format an array with two fixed decimal places."""
    return np.array2string(
        np.asarray(value, dtype=np.float64),
        formatter={"float_kind": lambda x: f"{x:.2f}"},
    )


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
        action_center: NDArray | None = None,
        action_scale: NDArray | None = None,
        max_deviation: float = 3.0,
        deviation_abort_threshold: float = 0.8,
        max_episode_chunks: int = 150,
    ) -> None:
        if env is None:
            raise ValueError(
                "env must be provided — mock env support is commented out; "
                "all observations and steps now come from the real env"
            )
        self.env = env
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.intervention_mgr = intervention_mgr
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.action_center = np.zeros(action_dim, dtype=np.float32) if action_center is None else np.asarray(action_center, dtype=np.float32)
        self.action_scale = np.ones(action_dim, dtype=np.float32) if action_scale is None else np.maximum(np.asarray(action_scale, dtype=np.float32), 1e-6)
        if self.action_center.shape != (action_dim,) or self.action_scale.shape != (action_dim,):
            raise ValueError("action_center and action_scale must have shape [action_dim]")
        self.max_deviation = max_deviation
        self.deviation_abort_threshold = deviation_abort_threshold
        self.max_episode_chunks = max_episode_chunks

        self._action_chunk_dim = chunk_length * action_dim
        self._feedback = HumanReward()

    def _normalize_action(self, action: NDArray) -> NDArray:
        """Convert robot-space actions to the actor/critic action space."""
        return ((np.asarray(action, dtype=np.float32) - self.action_center) / self.action_scale).astype(np.float32)

    def _unnormalize_action(self, action: NDArray) -> NDArray:
        """Convert actor actions back to robot space for env.step()."""
        return (np.asarray(action, dtype=np.float32) * self.action_scale + self.action_center).astype(np.float32)

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
    def _extract_rl_state(self, obs: dict[str, Any]) -> tuple[NDArray, NDArray, NDArray]:
        """Extract RL state x = cat(z_rl, s^p), VLA reference, and action chunk.

        Uses a single VLA forward pass when ``extract_both`` is available
        (JAX path), falling back to two calls for the PyTorch path.

        Returns:
            x: RL state [state_dim] as numpy array.
            a_tilde_flat: Flattened VLA reference chunk [action_chunk_dim] as numpy.
            action_chunk: VLA reference actions [C, action_dim] as numpy.
        """
        vla_input = self._obs_to_vla_input(obs)

        # Single VLA forward pass (JAX) or two calls (PyTorch fallback)
        if hasattr(self.vla, "extract_both"):
            z, pad_mask, actions = self.vla.extract_both(vla_input)
        else:
            z, pad_mask = self.vla.extract_embeddings(vla_input)
            actions = self.vla.sample_reference_actions(vla_input)

        # Encode z_rl from prefix embeddings
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]

        # Get VLA reference action chunk (first C steps)
        a_tilde_raw = actions[:, :self.chunk_length, :]  # [1, C, action_dim]
        action_chunk = a_tilde_raw.squeeze(0).cpu().numpy()  # robot space, for env.step
        a_tilde = self._normalize_action(action_chunk)
        a_tilde_flat = a_tilde.reshape(1, -1)  # normalized [1, C*d]

        # Proprioceptive state s^p from the preprocessed VLA observation.
        # DroidInputs merges joint_pos + gripper into state, then
        # PadStatesAndActions zero-pads to the VLA's internal width.
        # Slice to action_dim to drop the padding.
        s_p = torch.as_tensor(
            np.array(vla_input.state[:, :self.action_dim]),
            dtype=torch.float32, device=self.device,
        )
        # print(f"[DEBUG] s_p (normalized) = {s_p.squeeze(0).cpu().numpy()}")

        # RL state: x = cat(z_rl, s^p)
        x = torch.cat([z_rl, s_p], dim=-1)  # [1, state_dim]

        return (
            x.squeeze(0).cpu().numpy(),
            a_tilde_flat.squeeze(0),
            action_chunk,
        )

    @torch.no_grad()
    def _get_actor_action(self, x: NDArray, a_tilde_flat: NDArray) -> NDArray:
        """Get action from the RL actor.

        Args:
            x: RL state [state_dim].
            a_tilde_flat: Flattened VLA reference chunk [action_chunk_dim].

        Returns:
            action_chunk: [C, action_dim] numpy array.
        """
        # x: [state_dim] -> unsqueeze -> [1, state_dim]
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        # a_tilde_flat: [C*d] -> unsqueeze -> [1, C*d]
        a_tilde_t = torch.as_tensor(a_tilde_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

        # actor forward: input (1, state_dim) & (1, C*d) -> output [1, C*d]
        a_flat = self.actor(x_t, a_tilde_t)  # [1, C*d]

        # Actor output is normalized; only the environment receives raw robot
        # actions.
        a_normalized = a_flat.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)
        return self._unnormalize_action(a_normalized)

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
        print("[DEBUG] collect warmup")
        print(f"[DEBUG] obs.state = {np.asarray(obs['state'])}")

        # print("[DEBUG] obs keys and value shapes:")
        # for k, v in obs.items():
        #     if isinstance(v, dict):
        #         # 嵌套字典
        #         shapes = ", ".join(
        #             f"{ik}: {getattr(iv, 'shape', type(iv).__name__)}"
        #             for ik, iv in v.items()
        #         )
        #         print(f"  {k}: dict{{ {shapes} }}")
        #     elif hasattr(v, "shape"):
        #         print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
        #     else:
        #         print(f"  {k}: type={type(v).__name__}, value={v}")

        for _ in range(num_chunks):
            # Build RL state and get reference actions (single VLA forward pass)
            # print(f"[DEBUG] joint_position = {obs.get('observation/joint_position', obs.get('state'))}")
            x, a_tilde_flat, action_chunk = self._extract_rl_state(obs)
            print(f"[DEBUG] action_chunk = {action_chunk[0]}")
            # print(f"[DEBUG] x last 14 dims = {x[-14:]}")
            a_flat = self._normalize_action(action_chunk).reshape(-1)  # [C*d]

            # Step environment
            next_obs, rewards, done, _info = self.env.step(action_chunk)

            # Build next RL state
            # print("Warmup: Get next rl_state")
            next_x, next_a_tilde, _ = self._extract_rl_state(next_obs)

            # Store transition
            self.replay_buffer.add(
                x=x,
                a=a_flat,
                a_tilde=a_tilde_flat,
                rewards=rewards,
                next_x=next_x,
                next_a_tilde=next_a_tilde,
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
        is_mock_obs = type(getattr(self.env, "obs_source", None)).__name__ == "MockObsSource"

        while True:
            # Extract RL state and VLA reference
            x, a_tilde_flat, reference_chunk = self._extract_rl_state(obs)

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
                if is_mock_obs and stats.num_chunks == 0:
                    print(f"[Mock] obs.state = {_format_array_2f(obs.get('state'))}")
                    print(
                        "[Mock] before actor action[0] = "
                        f"{_format_array_2f(reference_chunk[0])}"
                    )
                else:
                    print(f"Current step = {stats.num_steps}, total_reward = {stats.total_reward:.2f}")

                action_chunk = self._get_actor_action(x, a_tilde_flat)

                if is_mock_obs and stats.num_chunks == 0:
                    print(
                        "[Mock] after actor action[0] = "
                        f"{_format_array_2f(action_chunk[0])}"
                    )

                # Step environment
                next_obs, rewards, done, info = self.env.step(action_chunk)

            # Human interventions and actor outputs are both raw robot-space
            # chunks here; store the normalized representation for TD3.
            a_flat = self._normalize_action(action_chunk).reshape(-1)  # [C*d]

            if store_transitions:
                # Build next RL state (requires VLA forward pass)
                next_x, next_a_tilde, _ = self._extract_rl_state(next_obs)

                self.replay_buffer.add(
                    x=x,
                    a=a_flat,
                    a_tilde=a_tilde_flat,
                    rewards=rewards,
                    next_x=next_x,
                    next_a_tilde=next_a_tilde,
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
