"""Shared checkpoint loading utilities."""

from __future__ import annotations

import logging

import torch

from rlt_openpi.models.rl_token import RLTokenModel

log = logging.getLogger(__name__)


def load_rl_token_model(
    ckpt_path: str,
    device: str = "cuda",
) -> RLTokenModel:
    """Load a trained RL token model from a Stage 1 checkpoint.

    The encoder/decoder architecture hypers are read from the saved config,
    so no external config object is needed.

    Args:
        ckpt_path: Path to the Stage 1 ``.pt`` checkpoint.
        device: Torch device to load onto.

    Returns:
        Frozen ``RLTokenModel`` with weights restored.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_config = ckpt["config"]
    model = RLTokenModel(
        embedding_dim=saved_config.embedding_dim,
        encoder_layers=saved_config.encoder_layers,
        encoder_heads=saved_config.encoder_heads,
        decoder_layers=saved_config.decoder_layers,
        decoder_heads=saved_config.decoder_heads,
    )
    model.load_state_dict(ckpt["model"])
    step = ckpt["step"]
    del ckpt
    model = model.to(device)
    log.info("Loaded RL token model from %s (step %d)", ckpt_path, step)
    return model
