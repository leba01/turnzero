#!/usr/bin/env python3
"""Build retrieval index from training set embeddings (Week 4, Task 1).

Loads ensemble member 1, extracts pooled embeddings for all training
examples, builds a RetrievalIndex with species metadata, saves to
outputs/retrieval/, and runs a quick sanity check.

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/build_retrieval_index.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.dataset import Vocab, VGCDataset
from turnzero.tool.retrieval import (
    RetrievalIndex,
    build_metadata,
    extract_embeddings,
)
from turnzero.uq.ensemble import _load_model_from_ckpt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_A = ROOT / "data" / "assembled" / "regime_a"
ENSEMBLE_DIR = ROOT / "outputs" / "runs"
CKPT_PATH = ENSEMBLE_DIR / "ensemble_001" / "best.pt"
VOCAB_PATH = ENSEMBLE_DIR / "ensemble_001" / "vocab.json"
OUT_RETRIEVAL = ROOT / "outputs" / "retrieval"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Building Retrieval Index (Week 4, Task 1)")
    print("=" * 70)

    # --- Load vocab + train dataset ---
    vocab = Vocab.load(VOCAB_PATH)
    train_ds = VGCDataset(DATA_A / "train.jsonl", vocab)
    print(f"Train set: {len(train_ds):,} examples")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # --- Load ensemble member 1 ---
    print(f"\nLoading model from {CKPT_PATH.parent.name}/{CKPT_PATH.name}")
    model = _load_model_from_ckpt(CKPT_PATH, DEVICE)
    print(f"Device: {DEVICE}")

    # --- Extract embeddings ---
    print("\nExtracting pooled embeddings...")
    emb_dict = extract_embeddings(model, train_loader, DEVICE)
    embeddings = emb_dict["embeddings"]
    print(f"Embeddings: {embeddings.shape} ({embeddings.dtype})")
    print(f"Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Free model from GPU
    del model
    torch.cuda.empty_cache()

    # --- Build metadata ---
    print("\nBuilding metadata from dataset...")
    metadata = build_metadata(train_ds)
    print(f"Metadata entries: {len(metadata):,}")

    # --- Save raw embeddings + labels (for other tools) ---
    OUT_RETRIEVAL.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_RETRIEVAL / "train_embeddings.npz"
    np.savez(
        raw_path,
        embeddings=embeddings,
        action90_true=emb_dict["action90_true"],
        lead2_true=emb_dict["lead2_true"],
        bring4_observed=emb_dict["bring4_observed"],
        is_mirror=emb_dict["is_mirror"],
    )
    print(f"\nRaw embeddings saved: {raw_path}")

    # --- Build + save index ---
    print("\nConstructing RetrievalIndex...")
    index = RetrievalIndex(embeddings, metadata)
    index_path = OUT_RETRIEVAL / "train_index"
    index.save(index_path)

    # --- Sanity checks ---
    print(f"\n{'=' * 70}")
    print("  Sanity Checks")
    print(f"{'=' * 70}")

    # Reload from disk
    loaded = RetrievalIndex.load(index_path)
    print(f"Loaded index: {loaded.embeddings.shape[0]:,} vectors")

    # Query first 3 examples
    for i in range(3):
        neighbors = loaded.query(embeddings[i], k=20)
        summary = loaded.evidence_summary(neighbors)

        print(f"\n--- Query example {i} ---")
        print(f"  Team A: {metadata[i]['species_a']}")
        print(f"  Team B: {metadata[i]['species_b']}")
        print(f"  True action: {metadata[i]['action90_id']}")
        print(f"  Mean similarity: {summary['mean_similarity']:.4f}")
        print(f"  Top-3 actions in neighbors:")
        for af in summary["action_freq"][:3]:
            pct = af["fraction"] * 100
            print(f"    action {af['action90_id']:2d} ({pct:5.1f}%): {af['description']}")
        print(f"  Top lead pairs:")
        for lp in summary["lead_pair_freq"][:3]:
            pct = lp["fraction"] * 100
            print(f"    {lp['lead_pair']:<30s} ({pct:5.1f}%)")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s ({dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
