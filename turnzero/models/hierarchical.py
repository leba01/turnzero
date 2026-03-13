"""Hierarchical Dual Encoder for 90-way OTS prediction.

Architecture:
  Level 1: Per-mon embedding (same as OTSTransformer)
  Level 2: Shared intra-team TransformerEncoder (6 tokens per team)
  Level 3: Bidirectional cross-attention (A attends to B, B attends to A)
  Pool: concat both sides, mean-pool → classification head → 90 logits

Parameter-matched to baseline OTSTransformer (~1.16M params).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HierarchicalConfig:
    d_model: int = 128
    n_intra_layers: int = 2
    n_cross_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: LN → MHA(Q=x, KV=ctx) → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """x attends to context. Both (B, S, d_model)."""
        h = self.norm1(x)
        h, _ = self.cross_attn(query=h, key=context, value=context)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalDualEncoder(nn.Module):
    """Hierarchical dual encoder: intra-team self-attention → cross-team attention.

    Same forward interface as OTSTransformer: forward(team_a, team_b) → (B, 90).
    """

    NUM_MONS_PER_TEAM: int = 6
    NUM_FIELDS: int = 8
    NUM_ACTIONS: int = 90

    def __init__(self, vocab_sizes: dict[str, int], cfg: HierarchicalConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Embedding tables (same as OTSTransformer) ---
        self.emb_species = nn.Embedding(vocab_sizes["species"], cfg.d_model)
        self.emb_item = nn.Embedding(vocab_sizes["item"], cfg.d_model)
        self.emb_ability = nn.Embedding(vocab_sizes["ability"], cfg.d_model)
        self.emb_tera = nn.Embedding(vocab_sizes["tera_type"], cfg.d_model)
        self.emb_move = nn.Embedding(vocab_sizes["move"], cfg.d_model)
        self.emb_side = nn.Embedding(2, cfg.d_model)

        # --- Level 2: Shared intra-team encoder ---
        intra_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.intra_encoder = nn.TransformerEncoder(
            intra_layer,
            num_layers=cfg.n_intra_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        # --- Level 3: Bidirectional cross-attention (separate params per direction) ---
        self.cross_blocks_a = nn.ModuleList([
            CrossAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_cross_layers)
        ])
        self.cross_blocks_b = nn.ModuleList([
            CrossAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_cross_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # --- Classification head (same as OTSTransformer) ---
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, self.NUM_ACTIONS),
        )

        self._init_weights()

    @classmethod
    def load_from_checkpoint(
        cls, ckpt_path: str | Path, device: torch.device,
    ) -> HierarchicalDualEncoder:
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
        model_cfg = HierarchicalConfig(**ckpt["model_config"])
        model = cls(ckpt["vocab_sizes"], model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model

    def _init_weights(self) -> None:
        init_std = 0.02
        for emb in [self.emb_species, self.emb_item, self.emb_ability,
                     self.emb_tera, self.emb_move, self.emb_side]:
            nn.init.normal_(emb.weight, mean=0.0, std=init_std)

    def _embed_team(self, team: Tensor, side: int) -> Tensor:
        """Embed a team of 6 mons into (B, 6, d_model)."""
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
        side_idx = torch.tensor(side, device=team.device, dtype=torch.long)
        e = e + self.emb_side(side_idx)
        return e

    def forward(self, team_a: Tensor, team_b: Tensor) -> Tensor:
        """Forward pass: (B, 6, 8) × 2 → (B, 90) logits."""
        # Level 1: Embed
        emb_a = self._embed_team(team_a, side=0)
        emb_b = self._embed_team(team_b, side=1)

        # Level 2: Intra-team self-attention (shared encoder, independent runs)
        intra_a = self.intra_encoder(emb_a)
        intra_b = self.intra_encoder(emb_b)

        # Level 3: Bidirectional cross-attention (interleaved)
        cross_a, cross_b = intra_a, intra_b
        for block_a, block_b in zip(self.cross_blocks_a, self.cross_blocks_b):
            cross_a_new = block_a(cross_a, cross_b)
            cross_b_new = block_b(cross_b, cross_a)
            cross_a, cross_b = cross_a_new, cross_b_new

        # Final norm + pool
        cross_a = self.final_norm(cross_a)
        cross_b = self.final_norm(cross_b)
        pooled = torch.cat([cross_a, cross_b], dim=1).mean(dim=1)  # (B, d_model)

        return self.head(pooled)


if __name__ == "__main__":
    dummy_vocab = {
        "species": 200,
        "item": 100,
        "ability": 150,
        "tera_type": 20,
        "move": 300,
    }
    cfg = HierarchicalConfig()
    model = HierarchicalDualEncoder(dummy_vocab, cfg)

    B = 4
    field_maxes = [
        dummy_vocab["species"], dummy_vocab["item"], dummy_vocab["ability"],
        dummy_vocab["tera_type"], dummy_vocab["move"], dummy_vocab["move"],
        dummy_vocab["move"], dummy_vocab["move"],
    ]
    team_a = torch.stack(
        [torch.randint(0, m, (B, 6)) for m in field_maxes], dim=-1
    )
    team_b = torch.stack(
        [torch.randint(0, m, (B, 6)) for m in field_maxes], dim=-1
    )

    logits = model(team_a, team_b)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (B, 90), f"Expected ({B}, 90), got {logits.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {n_params:,}")

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
