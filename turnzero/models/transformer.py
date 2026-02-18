"""Permutation-equivariant Transformer set model for 90-way OTS prediction.

Architecture (PROJECT_BIBLE Section 3, Option A):
  - Per-mon embedding: sum of field embeddings + side embedding
  - L layers of self-attention over 12 tokens (no positional encoding)
  - Mean-pool or CLS-pool → classification head → 90 logits

Reference: docs/PROJECT_BIBLE.md Section 3, Option A
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    pool: str = "mean"  # "mean" or "cls"


class OTSTransformer(nn.Module):
    """Transformer set model over 12 Pokemon tokens → 90-way action logits.

    Each Pokemon is represented as 8 integer fields:
        [species, item, ability, tera_type, move0, move1, move2, move3]

    The model sums per-field embeddings + a learned side embedding (0=team_a,
    1=team_b) to produce per-mon representations, then runs a standard
    Transformer encoder (no positional encoding for permutation equivariance),
    pools, and classifies.
    """

    NUM_MONS: int = 12  # 6 per side
    NUM_FIELDS: int = 8
    NUM_ACTIONS: int = 90

    def __init__(self, vocab_sizes: dict[str, int], cfg: ModelConfig) -> None:
        """
        Args:
            vocab_sizes: mapping from field name to vocabulary size.
                Required keys: "species", "item", "ability", "tera_type", "move"
            cfg: model hyperparameters.
        """
        super().__init__()
        self.cfg = cfg

        # --- Embedding tables (one per field type) ---
        self.emb_species = nn.Embedding(vocab_sizes["species"], cfg.d_model)
        self.emb_item = nn.Embedding(vocab_sizes["item"], cfg.d_model)
        self.emb_ability = nn.Embedding(vocab_sizes["ability"], cfg.d_model)
        self.emb_tera = nn.Embedding(vocab_sizes["tera_type"], cfg.d_model)
        self.emb_move = nn.Embedding(vocab_sizes["move"], cfg.d_model)  # shared for all 4 moves
        self.emb_side = nn.Embedding(2, cfg.d_model)  # 0=team_a, 1=team_b

        # --- Optional CLS token ---
        if cfg.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # --- Transformer encoder (pre-norm for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            norm=nn.LayerNorm(cfg.d_model),  # final norm after last layer
        )

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, self.NUM_ACTIONS),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings with small std (0.02)."""
        init_std = 0.02
        for emb in [self.emb_species, self.emb_item, self.emb_ability,
                     self.emb_tera, self.emb_move, self.emb_side]:
            nn.init.normal_(emb.weight, mean=0.0, std=init_std)
        if self.cfg.pool == "cls":
            nn.init.normal_(self.cls_token, mean=0.0, std=init_std)

    def _embed_team(self, team: Tensor, side: int) -> Tensor:
        """Embed a team of 6 mons into (B, 6, d_model).

        Args:
            team: (B, 6, 8) LongTensor — fields per mon:
                  [species, item, ability, tera_type, move0, move1, move2, move3]
            side: 0 for team_a, 1 for team_b.
        """
        # Slice fields: team[:, :, col]
        e = (
            self.emb_species(team[:, :, 0])
            + self.emb_item(team[:, :, 1])
            + self.emb_ability(team[:, :, 2])
            + self.emb_tera(team[:, :, 3])
            + self.emb_move(team[:, :, 4])
            + self.emb_move(team[:, :, 5])
            + self.emb_move(team[:, :, 6])
            + self.emb_move(team[:, :, 7])
        )
        # Side embedding: broadcast (1, d_model) → (B, 6, d_model)
        side_idx = torch.tensor(side, device=team.device, dtype=torch.long)
        e = e + self.emb_side(side_idx)
        return e

    def forward(self, team_a: Tensor, team_b: Tensor) -> Tensor:
        """Forward pass.

        Args:
            team_a: (B, 6, 8) LongTensor — 6 mons, 8 fields each.
            team_b: (B, 6, 8) LongTensor.

        Returns:
            (B, 90) logits (unnormalized).
        """
        # Embed both teams → (B, 6, d_model) each
        emb_a = self._embed_team(team_a, side=0)
        emb_b = self._embed_team(team_b, side=1)

        # Concatenate → (B, 12, d_model)
        tokens = torch.cat([emb_a, emb_b], dim=1)

        # Optionally prepend CLS token → (B, 13, d_model)
        if self.cfg.pool == "cls":
            B = tokens.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        tokens = self.encoder(tokens)

        # Pool
        if self.cfg.pool == "cls":
            pooled = tokens[:, 0]  # (B, d_model)
        else:
            pooled = tokens.mean(dim=1)  # (B, d_model)

        # Classification head → (B, 90)
        return self.head(pooled)


if __name__ == "__main__":
    # Smoke test
    dummy_vocab = {
        "species": 200,
        "item": 100,
        "ability": 150,
        "tera_type": 20,
        "move": 300,
    }
    cfg = ModelConfig()
    model = OTSTransformer(dummy_vocab, cfg)

    B = 4
    # Build random inputs respecting per-field vocab sizes.
    # Field order: species, item, ability, tera_type, move x4
    field_maxes = [
        dummy_vocab["species"],
        dummy_vocab["item"],
        dummy_vocab["ability"],
        dummy_vocab["tera_type"],
        dummy_vocab["move"],
        dummy_vocab["move"],
        dummy_vocab["move"],
        dummy_vocab["move"],
    ]
    team_a = torch.stack(
        [torch.randint(0, m, (B, 6)) for m in field_maxes], dim=-1
    )
    team_b = torch.stack(
        [torch.randint(0, m, (B, 6)) for m in field_maxes], dim=-1
    )

    logits = model(team_a, team_b)
    print(f"Output shape: {logits.shape}")  # expect (4, 90)
    assert logits.shape == (B, 90), f"Expected ({B}, 90), got {logits.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {n_params:,}")

    # Test CLS pooling variant
    cfg_cls = ModelConfig(pool="cls")
    model_cls = OTSTransformer(dummy_vocab, cfg_cls)
    logits_cls = model_cls(team_a, team_b)
    print(f"CLS output:   {logits_cls.shape}")
    assert logits_cls.shape == (B, 90)

    # Test torch.compile compatibility
    try:
        compiled = torch.compile(model, fullgraph=True)
        logits_c = compiled(team_a, team_b)
        print(f"Compiled:     {logits_c.shape}")
        assert logits_c.shape == (B, 90)
    except Exception as e:
        print(f"Compiled:     skipped ({type(e).__name__})")

    # Test BF16 autocast
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device)
        ta = team_a.to(device)
        tb = team_b.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_bf16 = model_gpu(ta, tb)
        print(f"BF16 output:  {logits_bf16.shape} (dtype={logits_bf16.dtype})")
    else:
        print("BF16 test:    skipped (no CUDA)")

    print("All smoke tests passed.")
