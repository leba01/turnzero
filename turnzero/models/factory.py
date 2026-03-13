"""Model factory for building architecture variants from config dicts."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from turnzero.models.transformer import ModelConfig, OTSTransformer


def build_model(
    arch: str,
    vocab_sizes: dict[str, int],
    cfg_model: dict[str, Any],
) -> tuple[nn.Module, Any]:
    """Build a model from architecture name and config dict.

    Returns (model, model_cfg) where model_cfg is the dataclass config.
    """
    if arch == "hierarchical":
        from turnzero.models.hierarchical import HierarchicalConfig, HierarchicalDualEncoder
        model_cfg = HierarchicalConfig(
            d_model=cfg_model["d_model"],
            n_intra_layers=cfg_model["n_intra_layers"],
            n_cross_layers=cfg_model["n_cross_layers"],
            n_heads=cfg_model["n_heads"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model["dropout"],
        )
        model = HierarchicalDualEncoder(vocab_sizes, model_cfg)
    elif arch == "opponent_context":
        from turnzero.models.sequential_transformer import OpponentContextConfig, OpponentContextTransformer
        model_cfg = OpponentContextConfig(
            d_model=cfg_model["d_model"],
            n_layers=cfg_model["n_layers"],
            n_heads=cfg_model["n_heads"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model["dropout"],
            pool=cfg_model["pool"],
            n_opp_slots=cfg_model.get("n_opp_slots", 4),
        )
        model = OpponentContextTransformer(vocab_sizes, model_cfg)
    elif arch == "sequential":
        from turnzero.models.sequential_transformer import SequentialConfig, SequentialOTSTransformer
        model_cfg = SequentialConfig(
            d_model=cfg_model["d_model"],
            n_layers=cfg_model["n_layers"],
            n_heads=cfg_model["n_heads"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model["dropout"],
            pool=cfg_model["pool"],
        )
        model = SequentialOTSTransformer(vocab_sizes, model_cfg)
    else:
        model_cfg = ModelConfig(
            d_model=cfg_model["d_model"],
            n_layers=cfg_model["n_layers"],
            n_heads=cfg_model["n_heads"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model["dropout"],
            pool=cfg_model["pool"],
        )
        model = OTSTransformer(vocab_sizes, model_cfg)
    return model, model_cfg


def build_model_from_checkpoint(
    arch: str,
    vocab_sizes: dict[str, int],
    model_config: dict[str, Any],
) -> tuple[nn.Module, Any]:
    """Build a model from checkpoint-saved config (uses **kwargs unpacking).

    Returns (model, model_cfg).
    """
    if arch == "hierarchical":
        from turnzero.models.hierarchical import HierarchicalConfig, HierarchicalDualEncoder
        model_cfg = HierarchicalConfig(**model_config)
        model = HierarchicalDualEncoder(vocab_sizes, model_cfg)
    elif arch == "opponent_context":
        from turnzero.models.sequential_transformer import OpponentContextConfig, OpponentContextTransformer
        model_cfg = OpponentContextConfig(**model_config)
        model = OpponentContextTransformer(vocab_sizes, model_cfg)
    elif arch == "sequential":
        from turnzero.models.sequential_transformer import SequentialConfig, SequentialOTSTransformer
        model_cfg = SequentialConfig(**model_config)
        model = SequentialOTSTransformer(vocab_sizes, model_cfg)
    else:
        model_cfg = ModelConfig(**model_config)
        model = OTSTransformer(vocab_sizes, model_cfg)
    return model, model_cfg


def model_config_to_dict(arch: str, model_cfg: Any) -> dict[str, Any]:
    """Serialize a model config dataclass to a dict for checkpoint saving."""
    if arch == "hierarchical":
        return {
            "d_model": model_cfg.d_model,
            "n_intra_layers": model_cfg.n_intra_layers,
            "n_cross_layers": model_cfg.n_cross_layers,
            "n_heads": model_cfg.n_heads,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
        }
    elif arch == "opponent_context":
        return {
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
            "pool": model_cfg.pool,
            "n_opp_slots": model_cfg.n_opp_slots,
        }
    else:
        return {
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "d_ff": model_cfg.d_ff,
            "dropout": model_cfg.dropout,
            "pool": model_cfg.pool,
        }
