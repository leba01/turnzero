"""Sequential OTS Transformer — conditions on BO3 game context.

Inherits OTSTransformer and adds 3 small embeddings (prior_lead2_idx,
prior_result, game_num) that sum into a single context token prepended
to the 12 team tokens. +2,816 params over the base model (~0.24%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from turnzero.models.transformer import ModelConfig, OTSTransformer


@dataclass
class SequentialConfig(ModelConfig):
    """Identical to ModelConfig — inherits all base hyperparameters."""
    pass


class SequentialOTSTransformer(OTSTransformer):
    """OTSTransformer + BO3 context token.

    Adds three embeddings summed into a single prepended context token:
    - prior_lead2_idx: Embedding(16, d_model)  — 0..14 lead pairs + 15 sentinel
    - prior_result:    Embedding(3, d_model)    — 0=sentinel, 1=won, 2=lost
    - game_num:        Embedding(3, d_model)    — 0=sentinel, 1=game2, 2=game3
    """

    def __init__(self, vocab_sizes: dict[str, int], cfg: SequentialConfig) -> None:
        super().__init__(vocab_sizes, cfg)
        d = cfg.d_model
        self.emb_prior_lead = nn.Embedding(16, d)
        self.emb_prior_result = nn.Embedding(3, d)
        self.emb_game_num = nn.Embedding(3, d)

        # Init context embeddings with same std as base
        for emb in [self.emb_prior_lead, self.emb_prior_result, self.emb_game_num]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    @classmethod
    def load_from_checkpoint(
        cls, ckpt_path: str | Path, device: torch.device,
    ) -> "SequentialOTSTransformer":
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
        model_cfg = SequentialConfig(**ckpt["model_config"])
        model = cls(ckpt["vocab_sizes"], model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model

    def forward(
        self,
        team_a: Tensor,
        team_b: Tensor,
        prior_lead2_idx: Tensor | None = None,
        prior_result: Tensor | None = None,
        game_num: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with optional BO3 context.

        Args:
            team_a: (B, 6, 8) LongTensor.
            team_b: (B, 6, 8) LongTensor.
            prior_lead2_idx: (B,) LongTensor, 0..15 (15=sentinel).
            prior_result: (B,) LongTensor, 0..2 (0=sentinel).
            game_num: (B,) LongTensor, 0..2 (0=sentinel).

        Returns:
            (B, 90) logits.
        """
        B = team_a.size(0)
        device = team_a.device

        # Build sentinel tensors when context not provided
        if prior_lead2_idx is None:
            prior_lead2_idx = torch.full((B,), 15, dtype=torch.long, device=device)
        if prior_result is None:
            prior_result = torch.zeros(B, dtype=torch.long, device=device)
        if game_num is None:
            game_num = torch.zeros(B, dtype=torch.long, device=device)

        # Embed teams → (B, 6, d) each
        emb_a = self._embed_team(team_a, side=0)
        emb_b = self._embed_team(team_b, side=1)

        # 12 team tokens
        tokens = torch.cat([emb_a, emb_b], dim=1)  # (B, 12, d)

        # Context token: sum of 3 context embeddings → (B, 1, d)
        ctx = (
            self.emb_prior_lead(prior_lead2_idx)
            + self.emb_prior_result(prior_result)
            + self.emb_game_num(game_num)
        ).unsqueeze(1)

        # Prepend context token → (B, 13, d)
        tokens = torch.cat([ctx, tokens], dim=1)

        # Optionally prepend CLS token → (B, 14, d)
        if self.cfg.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        tokens = self.encoder(tokens)

        # Pool
        if self.cfg.pool == "cls":
            pooled = tokens[:, 0]
        else:
            pooled = tokens.mean(dim=1)

        return self.head(pooled)
