"""Shared checkpoint loading utilities (JAX/Orbax)."""

from __future__ import annotations

import logging
from pathlib import Path

from flax import nnx

from rlt_openpi.models.rl_token import RLTokenModel

log = logging.getLogger(__name__)


def load_rl_token_model(ckpt_path: str) -> RLTokenModel:
    """Load a trained RL token model from a Stage 1 checkpoint.

    The encoder/decoder architecture hypers are read from the saved config,
    so no external config object is needed.

    Args:
        ckpt_path: Path to the Stage 1 checkpoint directory (containing
            ``params/`` subdirectory) or directly to the ``params/`` dir.

    Returns:
        ``RLTokenModel`` with weights restored.
    """
    import orbax.checkpoint as ocp

    ckpt_path = Path(ckpt_path)
    params_dir = ckpt_path / "params" if (ckpt_path / "params").exists() else ckpt_path

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(params_dir))

    # Read architecture hypers from the saved config
    step = int(ckpt.get("step", 0))

    embedding_dim = int(ckpt.get("embedding_dim", 2048))
    encoder_layers = int(ckpt.get("encoder_layers", 2))
    encoder_heads = int(ckpt.get("encoder_heads", 8))
    decoder_layers = int(ckpt.get("decoder_layers", 2))
    decoder_heads = int(ckpt.get("decoder_heads", 8))

    model = RLTokenModel(
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        encoder_heads=encoder_heads,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        rngs=nnx.Rngs(0),
    )

    # Load weights
    model_graphdef, _ = nnx.split(model)
    nnx.update(model, nnx.State.from_pure_dict(model_graphdef, ckpt["model"]))

    log.info("Loaded RL token model from %s (step %d)", ckpt_path, step)
    return model


def load_online_rl_checkpoint(ckpt_path: str):
    """Load a Stage 2 checkpoint and return actor/critic pure state dicts.

    Args:
        ckpt_path: Path to the Stage 2 checkpoint directory.

    Returns:
        Tuple of (actor_state_dict, critic_state_dict, metadata_dict).
    """
    import orbax.checkpoint as ocp

    ckpt_path = Path(ckpt_path)
    params_dir = ckpt_path / "params" if (ckpt_path / "params").exists() else ckpt_path

    checkpointer = ocp.PyTreeCheckpointer()
    ckpt = checkpointer.restore(str(params_dir))

    metadata = {
        "total_env_steps": int(ckpt.get("total_env_steps", 0)),
        "total_updates": int(ckpt.get("total_updates", 0)),
        "total_episodes": int(ckpt.get("total_episodes", 0)),
    }

    return ckpt.get("actor"), ckpt.get("critic"), metadata
