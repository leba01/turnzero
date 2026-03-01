"""Tests for turnzero.models.sequential_transformer."""

import pytest
import torch

from turnzero.models.sequential_transformer import (
    SequentialConfig,
    SequentialOTSTransformer,
)
from turnzero.models.transformer import ModelConfig, OTSTransformer

VOCAB = {"species": 100, "item": 50, "ability": 50, "tera_type": 20, "move": 200}


class TestSequentialForward:
    def test_output_shape_with_context(self):
        cfg = SequentialConfig()
        model = SequentialOTSTransformer(VOCAB, cfg)
        B = 4
        ta = torch.zeros(B, 6, 8, dtype=torch.long)
        tb = torch.zeros(B, 6, 8, dtype=torch.long)
        pl = torch.tensor([0, 5, 10, 15])
        pr = torch.tensor([0, 1, 2, 0])
        gn = torch.tensor([0, 1, 2, 1])
        out = model(ta, tb, pl, pr, gn)
        assert out.shape == (B, 90)

    def test_output_shape_without_context(self):
        cfg = SequentialConfig()
        model = SequentialOTSTransformer(VOCAB, cfg)
        B = 4
        ta = torch.zeros(B, 6, 8, dtype=torch.long)
        tb = torch.zeros(B, 6, 8, dtype=torch.long)
        out = model(ta, tb)
        assert out.shape == (B, 90)

    def test_sentinel_matches_no_args(self):
        """Passing explicit sentinels should give same output as no args."""
        cfg = SequentialConfig()
        model = SequentialOTSTransformer(VOCAB, cfg)
        model.eval()
        B = 2
        ta = torch.zeros(B, 6, 8, dtype=torch.long)
        tb = torch.zeros(B, 6, 8, dtype=torch.long)

        with torch.no_grad():
            out_none = model(ta, tb)
            out_sentinel = model(
                ta, tb,
                torch.full((B,), 15, dtype=torch.long),
                torch.zeros(B, dtype=torch.long),
                torch.zeros(B, dtype=torch.long),
            )
        assert torch.allclose(out_none, out_sentinel)


class TestParameterCount:
    def test_param_delta(self):
        cfg_base = ModelConfig()
        cfg_seq = SequentialConfig()
        base = OTSTransformer(VOCAB, cfg_base)
        seq = SequentialOTSTransformer(VOCAB, cfg_seq)

        n_base = sum(p.numel() for p in base.parameters())
        n_seq = sum(p.numel() for p in seq.parameters())
        delta = n_seq - n_base

        # 16*128 + 3*128 + 3*128 = 2816
        assert delta == 2816


class TestConfig:
    def test_sequential_config_inherits(self):
        cfg = SequentialConfig(d_model=256, n_layers=6)
        assert cfg.d_model == 256
        assert cfg.n_layers == 6
        assert isinstance(cfg, ModelConfig)
