"""Sequential VGC dataset — extends VGCDataset with BO3 context features.

Each example is augmented with (prior_lead2_idx, prior_result, game_num)
from a pre-built BO3 context table. Non-BO3 and Game 1 examples get
all-sentinel values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from turnzero.data.bo3_context import (
    SENTINEL_LEAD,
    _resolve_lead_pair_idx,
    load_context_lookup,
)
from turnzero.data.dataset import Vocab, VGCDataset, build_dataloaders


class SequentialVGCDataset(VGCDataset):
    """VGCDataset augmented with BO3 sequential context."""

    def __init__(
        self,
        jsonl_path: str | Path,
        vocab: Vocab,
        context_path: str | Path | None = None,
        tier1_only: bool = False,
        filter_prior_result: int | None = None,
    ) -> None:
        super().__init__(jsonl_path, vocab, tier1_only=tier1_only)
        self._context: dict[str, dict[str, Any]] = {}
        if context_path is not None:
            self._context = load_context_lookup(context_path)

        # Filter to only examples with a specific prior_result (1=won, 2=lost)
        if filter_prior_result is not None and self._context:
            filtered = []
            for ex in self.examples:
                bid = ex.get("battle_id", "")
                ctx = self._context.get(bid)
                if ctx is None:
                    continue
                side = self._resolve_perspective(ex, ctx)
                if side is None:
                    continue
                if ctx["sides"][side]["prior_result"] == filter_prior_result:
                    filtered.append(ex)
            self.examples = filtered

    def _resolve_perspective(
        self, ex: dict[str, Any], ctx: dict[str, Any],
    ) -> str | None:
        """Determine which side (p1/p2) corresponds to this example's team_a.

        Matches team_a species list against the game's species6 per side.
        """
        team_a_species = frozenset(
            mon["species"] for mon in ex["team_a"]["pokemon"]
        )
        for side in ("p1", "p2"):
            if frozenset(ctx["curr_species6"][side]) == team_a_species:
                return side
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base = super().__getitem__(idx)
        ex = self.examples[idx]
        bid = ex.get("battle_id", "")

        # Defaults: sentinels
        prior_lead2_idx = SENTINEL_LEAD  # 15
        prior_result = 0  # sentinel
        game_num = 0  # sentinel

        ctx = self._context.get(bid)
        if ctx is not None:
            side = self._resolve_perspective(ex, ctx)
            if side is not None:
                side_ctx = ctx["sides"][side]
                # Resolve prior leads to team_a-relative index
                team_a_species = [
                    mon["species"] for mon in ex["team_a"]["pokemon"]
                ]
                prior_lead2_idx = _resolve_lead_pair_idx(
                    frozenset(side_ctx["prior_leads"]), team_a_species,
                )
                prior_result = side_ctx["prior_result"]
                # game_num: 2→1, 3→2 (shift so 0=sentinel)
                raw_gn = ctx["game_num"]
                game_num = max(0, raw_gn - 1)  # game2→1, game3→2

        base["prior_lead2_idx"] = prior_lead2_idx
        base["prior_result"] = prior_result
        base["game_num"] = game_num
        return base


class OpponentContextVGCDataset(SequentialVGCDataset):
    """SequentialVGCDataset + opponent revealed species from prior game."""

    def __init__(
        self,
        jsonl_path: str | Path,
        vocab: Vocab,
        context_path: str | Path | None = None,
        tier1_only: bool = False,
        n_opp_slots: int = 4,
    ) -> None:
        super().__init__(jsonl_path, vocab, context_path=context_path, tier1_only=tier1_only)
        self._n_opp_slots = n_opp_slots

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base = super().__getitem__(idx)
        ex = self.examples[idx]
        bid = ex.get("battle_id", "")

        # Default: all zeros (padding)
        opp_revealed = [0] * self._n_opp_slots

        ctx = self._context.get(bid)
        if ctx is not None:
            side = self._resolve_perspective(ex, ctx)
            if side is not None:
                opp_species = ctx["sides"][side].get("opp_revealed", [])
                for i, sp in enumerate(opp_species[:self._n_opp_slots]):
                    opp_revealed[i] = self.vocab.encode("species", sp)

        base["opp_revealed"] = torch.tensor(opp_revealed, dtype=torch.long)
        return base


def build_sequential_dataloaders(
    split_dir: str | Path,
    batch_size: int = 512,
    num_workers: int = 4,
    vocab_path: str | Path | None = None,
    tier1_only: bool = False,
    context_path: str | Path | None = None,
    filter_prior_result: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Build train/val/test DataLoaders using SequentialVGCDataset.

    Same interface as build_dataloaders but uses SequentialVGCDataset
    when context_path is provided.
    """
    if context_path is None:
        # No context → fall back to base dataloaders
        return build_dataloaders(
            split_dir, batch_size, num_workers, vocab_path, tier1_only,
        )

    split_dir = Path(split_dir)
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"

    # Build or load vocab
    if vocab_path is None:
        from turnzero.data.io_utils import read_jsonl
        train_examples = list(read_jsonl(train_path))
        vocab = Vocab.from_examples(train_examples)
        vocab.save(split_dir / "vocab.json")
    else:
        vocab = Vocab.load(vocab_path)

    fpr = filter_prior_result
    # Datasets
    train_ds = SequentialVGCDataset(
        train_path, vocab, context_path=context_path, tier1_only=tier1_only,
        filter_prior_result=fpr,
    )
    val_ds = SequentialVGCDataset(
        val_path, vocab, context_path=context_path, filter_prior_result=fpr,
    )
    test_ds = SequentialVGCDataset(
        test_path, vocab, context_path=context_path, filter_prior_result=fpr,
    )

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, test_loader, vocab


def build_opponent_context_dataloaders(
    split_dir: str | Path,
    batch_size: int = 512,
    num_workers: int = 4,
    vocab_path: str | Path | None = None,
    tier1_only: bool = False,
    context_path: str | Path | None = None,
    n_opp_slots: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Build train/val/test DataLoaders using OpponentContextVGCDataset."""
    if context_path is None:
        return build_dataloaders(
            split_dir, batch_size, num_workers, vocab_path, tier1_only,
        )

    split_dir = Path(split_dir)
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"

    if vocab_path is None:
        from turnzero.data.io_utils import read_jsonl
        train_examples = list(read_jsonl(train_path))
        vocab = Vocab.from_examples(train_examples)
        vocab.save(split_dir / "vocab.json")
    else:
        vocab = Vocab.load(vocab_path)

    kw = dict(context_path=context_path, n_opp_slots=n_opp_slots)
    train_ds = OpponentContextVGCDataset(
        train_path, vocab, tier1_only=tier1_only, **kw,
    )
    val_ds = OpponentContextVGCDataset(val_path, vocab, **kw)
    test_ds = OpponentContextVGCDataset(test_path, vocab, **kw)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, test_loader, vocab
