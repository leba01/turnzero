"""Retrieval-based evidence: embed training matchups, query by similarity.

Build a retrieval index from the pooled transformer embeddings of training
examples. At query time, find similar matchups and report what experts did.

Architecture note: the model's forward pass is
    _embed_team(a,0) + _embed_team(b,1) → concat → encoder → mean-pool → head
We replicate everything up to mean-pool, stopping before ``self.head``,
to get the 128-dim embedding the model uses internally.

Train set: ~246K examples × 128 × 4 bytes ≈ 126 MB — fits in RAM.
Brute-force cosine similarity via matrix multiply is fast enough.

Reference: docs/PROJECT_BIBLE.md Section 5.1
           docs/WEEK4_PLAN.md Task 1
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from turnzero.action_space import action90_to_lead_back
from turnzero.models.transformer import OTSTransformer

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: OTSTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Extract pooled representations (N, d_model) before the classification head.

    Replicates the model's forward logic (embed → concat → encoder → pool)
    but stops before ``self.head``.

    Parameters
    ----------
    model : OTSTransformer
        Trained model, already on *device* and in eval mode.
    loader : DataLoader
        Must use ``shuffle=False, drop_last=False`` so indices match the
        underlying dataset ordering.
    device : torch.device

    Returns
    -------
    dict with keys:
        "embeddings"      : (N, d_model) float32 ndarray
        "action90_true"   : (N,) int ndarray
        "lead2_true"      : (N,) int ndarray
        "bring4_observed" : (N,) bool ndarray
        "is_mirror"       : (N,) bool ndarray
    """
    model.eval()
    all_emb: list[np.ndarray] = []
    all_action90: list[np.ndarray] = []
    all_lead2: list[np.ndarray] = []
    all_bring4: list[np.ndarray] = []
    all_mirror: list[np.ndarray] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        team_a = batch["team_a"].to(device, non_blocking=True)
        team_b = batch["team_b"].to(device, non_blocking=True)

        with amp_ctx:
            # Replicate forward logic up to pool, skip self.head
            emb_a = model._embed_team(team_a, side=0)
            emb_b = model._embed_team(team_b, side=1)
            tokens = torch.cat([emb_a, emb_b], dim=1)  # (B, 12, d)

            if model.cfg.pool == "cls":
                B = tokens.size(0)
                cls = model.cls_token.expand(B, -1, -1)
                tokens = torch.cat([cls, tokens], dim=1)

            tokens = model.encoder(tokens)

            if model.cfg.pool == "cls":
                pooled = tokens[:, 0]
            else:
                pooled = tokens.mean(dim=1)  # (B, d_model)

        all_emb.append(pooled.float().cpu().numpy())
        all_action90.append(batch["action90_label"].numpy())
        all_lead2.append(batch["lead2_label"].numpy())
        all_bring4.append(batch["bring4_observed"].numpy())
        all_mirror.append(batch["is_mirror"].numpy())

    return {
        "embeddings": np.concatenate(all_emb, axis=0),
        "action90_true": np.concatenate(all_action90, axis=0),
        "lead2_true": np.concatenate(all_lead2, axis=0),
        "bring4_observed": np.concatenate(all_bring4, axis=0),
        "is_mirror": np.concatenate(all_mirror, axis=0),
    }


# ---------------------------------------------------------------------------
# Metadata extraction (species names, action details from raw JSONL dicts)
# ---------------------------------------------------------------------------

def build_metadata(dataset: Any) -> list[dict[str, Any]]:
    """Extract per-example metadata from a VGCDataset.

    Parameters
    ----------
    dataset : VGCDataset
        Must have a ``.examples`` attribute (list of raw MatchExample dicts).

    Returns
    -------
    list of dicts (one per example), each with:
        species_a  : list[str]  — 6 species names for team A
        species_b  : list[str]  — 6 species names for team B
        action90_id: int
        lead2_idx  : list[int]  — [i, j] indices into team A
        back2_idx  : list[int]  — [i, j] indices among remaining 4
    """
    metadata: list[dict[str, Any]] = []
    for ex in dataset.examples:
        species_a = [mon["species"] for mon in ex["team_a"]["pokemon"]]
        species_b = [mon["species"] for mon in ex["team_b"]["pokemon"]]
        label = ex["label"]
        lead, back = action90_to_lead_back(label["action90_id"])
        metadata.append({
            "species_a": species_a,
            "species_b": species_b,
            "action90_id": label["action90_id"],
            "lead2_idx": list(lead),
            "back2_idx": list(back),
        })
    return metadata


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------

class RetrievalIndex:
    """Brute-force cosine similarity index over training matchup embeddings.

    Embeddings are L2-normalized at construction time so that a simple
    matrix multiply gives cosine similarity.

    Parameters
    ----------
    embeddings : (N, d_model) float32 ndarray
        Raw embeddings — will be L2-normalized internally.
    metadata : list[dict]
        Per-example metadata (species_a, species_b, action90_id, etc.).
        Must have the same length as *embeddings*.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPS)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.metadata = metadata
        assert len(self.embeddings) == len(self.metadata), (
            f"Length mismatch: {len(self.embeddings)} embeddings "
            f"vs {len(self.metadata)} metadata entries"
        )

    # -- query --------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """Find the *k* nearest neighbors by cosine similarity.

        Parameters
        ----------
        query_embedding : (d_model,) or (1, d_model) float32
            Query vector (will be L2-normalized internally).
        k : int
            Number of neighbors to return.

        Returns
        -------
        list of dicts sorted by descending similarity, each containing
        ``similarity`` (float) plus all metadata keys for that neighbor.
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm < _EPS:
            return []
        q = q / q_norm

        # Cosine similarity = dot product (embeddings already unit-norm)
        sims = self.embeddings @ q  # (N,)

        k = min(k, len(sims))
        # argpartition is O(N) vs O(N log N) for full sort
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        neighbors: list[dict[str, Any]] = []
        for idx in top_idx:
            entry: dict[str, Any] = {
                "similarity": float(sims[idx]),
                "index": int(idx),
            }
            entry.update(self.metadata[idx])
            neighbors.append(entry)
        return neighbors

    # -- evidence summary ---------------------------------------------------

    def evidence_summary(self, neighbors: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate neighbor metadata into a structured evidence summary.

        Counts action, lead-pair, per-mon lead, and per-mon bring frequencies
        across the retrieved neighbors.

        Parameters
        ----------
        neighbors : list[dict]
            Output from :meth:`query`.

        Returns
        -------
        dict with keys:
            n_neighbors     : int
            mean_similarity : float
            action_freq     : list of {action90_id, count, fraction, description}
            lead_pair_freq  : list of {lead_pair, count, fraction}
            mon_lead_freq   : list of {species, count, fraction}
            mon_bring_freq  : list of {species, count, fraction}
        """
        if not neighbors:
            return {"n_neighbors": 0}

        n = len(neighbors)
        mean_sim = float(np.mean([nb["similarity"] for nb in neighbors]))

        # --- action90 frequency ---
        action_counter: Counter[int] = Counter()
        for nb in neighbors:
            action_counter[nb["action90_id"]] += 1

        action_freq: list[dict[str, Any]] = []
        for aid, count in action_counter.most_common():
            lead, back = action90_to_lead_back(aid)
            # Use species from a neighbor that chose this action
            nb_ex = next(nb for nb in neighbors if nb["action90_id"] == aid)
            sp = nb_ex["species_a"]
            desc = (
                f"Lead {sp[lead[0]]}+{sp[lead[1]]}, "
                f"Back {sp[back[0]]}+{sp[back[1]]}"
            )
            action_freq.append({
                "action90_id": aid,
                "count": count,
                "fraction": count / n,
                "description": desc,
            })

        # --- lead pair frequency (by species name) ---
        lead_counter: Counter[str] = Counter()
        for nb in neighbors:
            lead, _ = action90_to_lead_back(nb["action90_id"])
            sp = nb["species_a"]
            pair_str = f"{sp[lead[0]]}+{sp[lead[1]]}"
            lead_counter[pair_str] += 1

        lead_pair_freq = [
            {"lead_pair": pair, "count": c, "fraction": c / n}
            for pair, c in lead_counter.most_common()
        ]

        # --- per-mon lead frequency ---
        lead_mon_counter: Counter[str] = Counter()
        for nb in neighbors:
            lead, _ = action90_to_lead_back(nb["action90_id"])
            sp = nb["species_a"]
            for idx in lead:
                lead_mon_counter[sp[idx]] += 1

        mon_lead_freq = [
            {"species": sp, "count": c, "fraction": c / n}
            for sp, c in lead_mon_counter.most_common()
        ]

        # --- per-mon bring frequency (lead + back = bring-4) ---
        bring_counter: Counter[str] = Counter()
        for nb in neighbors:
            lead, back = action90_to_lead_back(nb["action90_id"])
            sp = nb["species_a"]
            for idx in (*lead, *back):
                bring_counter[sp[idx]] += 1

        mon_bring_freq = [
            {"species": sp, "count": c, "fraction": c / n}
            for sp, c in bring_counter.most_common()
        ]

        return {
            "n_neighbors": n,
            "mean_similarity": mean_sim,
            "action_freq": action_freq,
            "lead_pair_freq": lead_pair_freq,
            "mon_lead_freq": mon_lead_freq,
            "mon_bring_freq": mon_bring_freq,
        }

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save index to disk (npz for embeddings, JSON sidecar for metadata).

        Creates two files:
            <path>.npz       — embeddings array
            <path>.meta.json — metadata list[dict]
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        npz_path = path.with_suffix(".npz")
        np.savez(npz_path, embeddings=self.embeddings)

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, separators=(",", ":"))

        print(
            f"RetrievalIndex saved: {npz_path} + {meta_path.name} "
            f"({len(self.embeddings)} vectors, {self.embeddings.shape[1]}d)"
        )

    @classmethod
    def load(cls, path: str | Path) -> RetrievalIndex:
        """Load a previously saved index from disk."""
        path = Path(path)

        npz_path = path.with_suffix(".npz")
        data = np.load(npz_path)
        embeddings = data["embeddings"]

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path) as f:
            metadata = json.load(f)

        # Embeddings were already L2-normalized before save;
        # __init__ re-normalizes (idempotent for unit vectors).
        return cls(embeddings, metadata)
