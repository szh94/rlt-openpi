"""Tests for VLA embedding extractor with mock PI0Pytorch."""

from unittest.mock import MagicMock, patch

import torch

from rlt_openpi.vla.embedding_extractor import EmbeddingExtractor


def _make_mock_pi0(B=1, M=20, D=2048, action_horizon=50, action_dim=14):
    """Create a mock PI0Pytorch with the expected interface."""
    pi0 = MagicMock()

    # Make it behave like an nn.Module (parameters iterable)
    pi0.parameters.return_value = iter([])
    pi0.eval.return_value = pi0

    # _preprocess_observation
    images = torch.randn(B, 3, 224, 224)
    img_masks = torch.ones(B, 1, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (B, 10))
    lang_masks = torch.ones(B, 10, dtype=torch.bool)
    state = torch.randn(B, action_dim)
    pi0._preprocess_observation.return_value = (images, img_masks, lang_tokens, lang_masks, state)

    # embed_prefix
    prefix_embs = torch.randn(B, M, D)
    prefix_pad_masks = torch.ones(B, M, dtype=torch.bool)
    prefix_att_masks = torch.ones(B, M, dtype=torch.bool)
    pi0.embed_prefix.return_value = (prefix_embs, prefix_pad_masks, prefix_att_masks)

    # paligemma_with_expert.forward
    prefix_out = torch.randn(B, M, D)
    pi0.paligemma_with_expert.forward.return_value = ([prefix_out, None], None)
    # Mock the language model config and weight access for dtype check
    mock_config = MagicMock()
    mock_config._attn_implementation = "eager"
    mock_layer = MagicMock()
    mock_layer.self_attn.q_proj.weight.dtype = torch.float32
    pi0.paligemma_with_expert.paligemma.language_model.config = mock_config
    pi0.paligemma_with_expert.paligemma.language_model.layers = [mock_layer]

    # _prepare_attention_masks_4d
    pi0._prepare_attention_masks_4d.return_value = torch.ones(B, 1, M, M)

    # sample_actions
    pi0.sample_actions.return_value = torch.randn(B, action_horizon, action_dim)

    return pi0


def test_extract_embeddings_shapes():
    B, M, D = 2, 20, 2048
    pi0 = _make_mock_pi0(B=B, M=M, D=D)

    # Patch nn.Module.__init__ to avoid issues with MagicMock
    with patch.object(EmbeddingExtractor, "__init__", lambda self, model: setattr(self, "pi0", model)):
        extractor = EmbeddingExtractor(pi0)

    obs = {"state": torch.randn(B, 14)}
    z, pad_mask = extractor.extract_embeddings(obs)
    assert z.shape == (B, M, D)
    assert pad_mask.shape == (B, M)
    assert pad_mask.dtype == torch.bool


def test_sample_actions_shape():
    B, action_horizon, action_dim = 1, 50, 14
    pi0 = _make_mock_pi0(B=B, action_horizon=action_horizon, action_dim=action_dim)

    with patch.object(EmbeddingExtractor, "__init__", lambda self, model: setattr(self, "pi0", model)):
        extractor = EmbeddingExtractor(pi0)

    obs = {"state": torch.randn(B, action_dim)}
    device = torch.device("cpu")
    actions = extractor.sample_actions(obs, device)
    assert actions.shape == (B, action_horizon, action_dim)


def test_extract_embeddings_output_is_float32():
    B, M, D = 1, 10, 2048
    pi0 = _make_mock_pi0(B=B, M=M, D=D)

    with patch.object(EmbeddingExtractor, "__init__", lambda self, model: setattr(self, "pi0", model)):
        extractor = EmbeddingExtractor(pi0)

    obs = {"state": torch.randn(B, 14)}
    z, _ = extractor.extract_embeddings(obs)
    assert z.dtype == torch.float32
