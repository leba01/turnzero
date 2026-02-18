"""PyTorch Dataset + DataLoader factory for TurnZero.

Reads assembled JSONL splits, builds vocabulary mappings from training data,
and produces batches of (team_a, team_b, action90_label, lead2_label, ...).

Each mon is encoded as 8 ints: (species, item, ability, tera, move0..3).
Teams are (6, 8) LongTensors.

Reference: docs/WEEK2_PLAN.md Task 1
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from turnzero.data.io_utils import read_jsonl

# ---------------------------------------------------------------------------
# Lead-pair lookup: same ordering as action_space.py
# ---------------------------------------------------------------------------
LEAD_PAIRS: list[tuple[int, int]] = list(combinations(range(6), 2))
LEAD_PAIR_TO_IDX: dict[tuple[int, int], int] = {
    pair: idx for idx, pair in enumerate(LEAD_PAIRS)
}
assert len(LEAD_PAIRS) == 15

# Field names for the 8-int per-mon encoding
_FIELD_TYPES = ("species", "item", "ability", "tera_type", "move")


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
class Vocab:
    """Token-to-index mappings for categorical OTS fields.

    Index 0 is always ``<UNK>`` for every field type.
    Built from training-split examples only; unseen tokens at val/test time
    map to 0.
    """

    UNK = "<UNK>"
    UNK_IDX = 0

    def __init__(self) -> None:
        # field_type -> {token: idx}
        self._tok2idx: dict[str, dict[str, int]] = {}
        for ft in _FIELD_TYPES:
            self._tok2idx[ft] = {self.UNK: self.UNK_IDX}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_examples(cls, examples: list[dict[str, Any]]) -> Vocab:
        """Build vocab from a list of raw MatchExample dicts (train split)."""
        vocab = cls()
        token_sets: dict[str, set[str]] = defaultdict(set)

        for ex in examples:
            for team_key in ("team_a", "team_b"):
                for mon in ex[team_key]["pokemon"]:
                    token_sets["species"].add(mon["species"])
                    token_sets["item"].add(mon["item"])
                    token_sets["ability"].add(mon["ability"])
                    token_sets["tera_type"].add(mon["tera_type"])
                    for m in mon["moves"]:
                        token_sets["move"].add(m)

        # Assign indices 1+ in sorted order for determinism
        for ft in _FIELD_TYPES:
            tokens_sorted = sorted(token_sets[ft] - {cls.UNK, "UNK"})
            for tok in tokens_sorted:
                vocab._tok2idx[ft][tok] = len(vocab._tok2idx[ft])

        return vocab

    # -- encode -------------------------------------------------------------

    def encode(self, field_type: str, token: str) -> int:
        """Return the index for *token* under *field_type*, or UNK_IDX."""
        return self._tok2idx[field_type].get(token, self.UNK_IDX)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write vocab to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._tok2idx, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        """Load vocab from a JSON file."""
        vocab = cls()
        with open(path) as f:
            vocab._tok2idx = json.load(f)
        return vocab

    # -- properties ---------------------------------------------------------

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """Total size (including UNK) for each field type."""
        return {ft: len(mapping) for ft, mapping in self._tok2idx.items()}

    def __repr__(self) -> str:
        sizes = self.vocab_sizes
        return f"Vocab({', '.join(f'{k}={v}' for k, v in sizes.items())})"


# ---------------------------------------------------------------------------
# VGCDataset
# ---------------------------------------------------------------------------
def _encode_team(team_dict: dict[str, Any], vocab: Vocab) -> torch.Tensor:
    """Encode a TeamSheet dict as a (6, 8) LongTensor."""
    rows = []
    for mon in team_dict["pokemon"]:
        row = [
            vocab.encode("species", mon["species"]),
            vocab.encode("item", mon["item"]),
            vocab.encode("ability", mon["ability"]),
            vocab.encode("tera_type", mon["tera_type"]),
            vocab.encode("move", mon["moves"][0]),
            vocab.encode("move", mon["moves"][1]),
            vocab.encode("move", mon["moves"][2]),
            vocab.encode("move", mon["moves"][3]),
        ]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


class VGCDataset(Dataset):
    """PyTorch dataset over assembled JSONL match examples.

    All examples are loaded into memory on init (dataset fits in RAM).
    """

    def __init__(self, jsonl_path: str | Path, vocab: Vocab) -> None:
        self.vocab = vocab
        self.examples: list[dict[str, Any]] = list(read_jsonl(jsonl_path))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]

        team_a = _encode_team(ex["team_a"], self.vocab)
        team_b = _encode_team(ex["team_b"], self.vocab)

        label = ex["label"]
        action90_label = label["action90_id"]

        # lead2_label: index into combinations(range(6), 2)
        lead_pair = tuple(sorted(label["lead2_idx"]))
        lead2_label = LEAD_PAIR_TO_IDX[lead_pair]

        # Metadata flags
        lq = ex["label_quality"]
        bring4_observed = lq["bring4_observed"]

        split_keys = ex.get("split_keys", {})
        is_mirror = split_keys.get("is_mirror", False)

        return {
            "team_a": team_a,       # (6, 8) LongTensor
            "team_b": team_b,       # (6, 8) LongTensor
            "action90_label": action90_label,  # int 0..89
            "lead2_label": lead2_label,        # int 0..14
            "bring4_observed": bring4_observed, # bool
            "is_mirror": is_mirror,             # bool
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_dataloaders(
    split_dir: str | Path,
    batch_size: int = 512,
    num_workers: int = 4,
    vocab_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab]:
    """Build train/val/test DataLoaders from an assembled split directory.

    Parameters
    ----------
    split_dir : path
        Directory containing train.jsonl, val.jsonl, test.jsonl.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        DataLoader worker count.
    vocab_path : path or None
        If None, build vocab from train split and save to split_dir/vocab.json.
        If given, load existing vocab.

    Returns
    -------
    (train_loader, val_loader, test_loader, vocab)
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"

    # Build or load vocab
    if vocab_path is None:
        train_examples = list(read_jsonl(train_path))
        vocab = Vocab.from_examples(train_examples)
        vocab.save(split_dir / "vocab.json")
    else:
        vocab = Vocab.load(vocab_path)
        train_examples = None

    # Datasets
    train_ds = VGCDataset(train_path, vocab)
    val_ds = VGCDataset(val_path, vocab)
    test_ds = VGCDataset(test_path, vocab)

    # If we already loaded the train examples, inject them to avoid re-reading
    if train_examples is not None:
        train_ds.examples = train_examples

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, vocab


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    split_dir = Path("data/assembled/regime_a")
    print(f"Loading data from {split_dir} ...")
    t0 = time.time()
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        split_dir, batch_size=256, num_workers=0,
    )
    t1 = time.time()

    print(f"\nVocab: {vocab}")
    print(f"Vocab sizes: {vocab.vocab_sizes}")
    print(f"Train: {len(train_loader.dataset)} examples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} examples, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} examples, {len(test_loader)} batches")
    print(f"Load time: {t1 - t0:.1f}s")

    print("\n--- First batch shapes ---")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} (first={v[0]})")

    # Sanity checks
    assert batch["team_a"].shape == (256, 6, 8)
    assert batch["team_b"].shape == (256, 6, 8)
    assert batch["action90_label"].max() < 90
    assert batch["lead2_label"].max() < 15
    print("\nAll sanity checks passed.")
