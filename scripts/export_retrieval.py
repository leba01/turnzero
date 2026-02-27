"""Convert retrieval index to web-friendly format.

Converts float32 embeddings to float16 raw binary for size reduction,
and copies metadata JSON.

Usage:
    uv run python scripts/export_retrieval.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "outputs" / "retrieval"
    out_dir = root / "web" / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load float32 embeddings (already L2-normalized)
    npz_path = index_dir / "train_index.npz"
    data = np.load(npz_path)
    embeddings = data["embeddings"]  # (N, 128) float32
    print(f"Loaded embeddings: {embeddings.shape} {embeddings.dtype}")

    # Convert to float16 raw binary
    emb_f16 = embeddings.astype(np.float16)
    out_bin = out_dir / "retrieval_embeddings.bin"
    emb_f16.tofile(str(out_bin))
    size_mb = out_bin.stat().st_size / 1024 / 1024
    print(f"Saved float16 binary: {out_bin.name} ({size_mb:.1f} MB)")

    # Also save shape metadata for the loader
    shape_meta = {"rows": embeddings.shape[0], "cols": embeddings.shape[1]}
    with open(out_dir / "retrieval_shape.json", "w") as f:
        json.dump(shape_meta, f)

    # Copy metadata JSON
    meta_src = index_dir / "train_index.meta.json"
    meta_dst = out_dir / "retrieval_meta.json"
    shutil.copy2(meta_src, meta_dst)
    meta_size_mb = meta_dst.stat().st_size / 1024 / 1024
    print(f"Copied metadata: {meta_dst.name} ({meta_size_mb:.1f} MB)")

    print(f"\nTotal retrieval assets: {size_mb + meta_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
