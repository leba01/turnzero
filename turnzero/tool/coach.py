"""Minimal demo tool: top-3 action plans with calibrated confidence.

Given two team sheets (species lists), loads the ensemble + temperature
scaling artifact, runs inference, and outputs the top-3 predicted
expert plans with calibrated probabilities. If confidence is below
the abstention threshold, recommends scouting mode.

Reference: docs/PROJECT_BIBLE.md Section 7.2 (demo CLI),
           docs/WEEK3_PLAN.md Task 6
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from turnzero.action_space import ACTION_TABLE, action90_to_lead_back
from turnzero.data.dataset import Vocab, _encode_team
from turnzero.models.transformer import ModelConfig, OTSTransformer
from turnzero.uq.temperature import TemperatureScaler

_EPS = 1e-12


def _load_ensemble(
    ensemble_dir: str | Path,
    device: torch.device,
) -> list[OTSTransformer]:
    """Load all ensemble member checkpoints from a directory.

    Expects ensemble_dir to contain ensemble_001/best.pt ... ensemble_005/best.pt,
    or accepts a list of directories.
    """
    ensemble_dir = Path(ensemble_dir)
    ckpt_dirs = sorted(ensemble_dir.glob("ensemble_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No ensemble_* directories found in {ensemble_dir}")

    models = []
    for d in ckpt_dirs:
        ckpt_path = d / "best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_cfg = ModelConfig(**ckpt["model_config"])
        model = OTSTransformer(ckpt["vocab_sizes"], model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)

    return models


def _parse_team_string(team_str: str) -> list[str]:
    """Parse a comma-separated species list into a list of 6 species names."""
    species = [s.strip() for s in team_str.split(",") if s.strip()]
    if len(species) != 6:
        raise ValueError(
            f"Expected 6 species separated by commas, got {len(species)}: {species}"
        )
    return species


def _build_team_dict(species_list: list[str]) -> dict:
    """Build a minimal TeamSheet dict from species names only.

    Uses UNK for item/ability/tera/moves since we only have species.
    For a full demo, users would provide the complete OTS.
    """
    pokemon = []
    for sp in species_list:
        pokemon.append({
            "species": sp,
            "item": "UNK",
            "ability": "UNK",
            "tera_type": "UNK",
            "moves": ["UNK", "UNK", "UNK", "UNK"],
        })
    return {"pokemon": pokemon}


def _build_team_dict_from_ots(ots_path: str | Path) -> dict:
    """Load a full OTS from a JSON file.

    Expected format:
    {
        "pokemon": [
            {"species": "...", "item": "...", "ability": "...",
             "tera_type": "...", "moves": ["...", "...", "...", "..."]},
            ...
        ]
    }
    """
    with open(ots_path) as f:
        data = json.load(f)
    if "pokemon" not in data or len(data["pokemon"]) != 6:
        raise ValueError(f"OTS file must have 'pokemon' key with exactly 6 entries")
    return data


def _format_plan(action_id: int, species_a: list[str]) -> str:
    """Format an action90 id as a human-readable plan string."""
    lead_pair, back_pair = action90_to_lead_back(action_id)
    leads = [species_a[i] for i in lead_pair]
    backs = [species_a[i] for i in back_pair]
    remaining = set(range(6)) - set(lead_pair) - set(back_pair)
    bench = [species_a[i] for i in sorted(remaining)]

    return (
        f"Lead: {leads[0]} + {leads[1]}  |  "
        f"Back: {backs[0]} + {backs[1]}  |  "
        f"Bench: {bench[0]} + {bench[1]}"
    )


@torch.no_grad()
def run_demo(
    ensemble_dir: str | Path,
    calib_path: str | Path,
    team_a_str: str | None = None,
    team_b_str: str | None = None,
    team_a_ots: str | Path | None = None,
    team_b_ots: str | Path | None = None,
    vocab_path: str | Path | None = None,
    tau: float = 0.04,
    top_k: int = 3,
) -> dict:
    """Run the demo coach tool.

    Provide teams either as comma-separated species strings (team_a_str/team_b_str)
    or as OTS JSON files (team_a_ots/team_b_ots).

    Args:
        ensemble_dir: path to directory containing ensemble_001..005/ subdirs
        calib_path: path to temperature.json
        team_a_str: comma-separated species list for Team A (your team)
        team_b_str: comma-separated species list for Team B (opponent)
        team_a_ots: path to Team A OTS JSON file (alternative to team_a_str)
        team_b_ots: path to Team B OTS JSON file (alternative to team_b_str)
        vocab_path: path to vocab.json (if None, uses ensemble_001/vocab.json)
        tau: abstention threshold for confidence
        top_k: number of top plans to show

    Returns:
        dict with plans, probabilities, confidence, uncertainty metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_dir = Path(ensemble_dir)

    # Parse teams
    if team_a_ots:
        team_a_dict = _build_team_dict_from_ots(team_a_ots)
    elif team_a_str:
        species_a = _parse_team_string(team_a_str)
        team_a_dict = _build_team_dict(species_a)
    else:
        raise ValueError("Must provide either team_a_str or team_a_ots")

    if team_b_ots:
        team_b_dict = _build_team_dict_from_ots(team_b_ots)
    elif team_b_str:
        species_b = _parse_team_string(team_b_str)
        team_b_dict = _build_team_dict(species_b)
    else:
        raise ValueError("Must provide either team_b_str or team_b_ots")

    species_a = [mon["species"] for mon in team_a_dict["pokemon"]]
    species_b = [mon["species"] for mon in team_b_dict["pokemon"]]

    # Load vocab
    if vocab_path is None:
        ckpt_dirs = sorted(ensemble_dir.glob("ensemble_*"))
        vocab_path = ckpt_dirs[0] / "vocab.json"
    vocab = Vocab.load(vocab_path)

    # Encode teams
    team_a_tensor = _encode_team(team_a_dict, vocab).unsqueeze(0).to(device)
    team_b_tensor = _encode_team(team_b_dict, vocab).unsqueeze(0).to(device)

    # Load temperature
    scaler = TemperatureScaler.load(calib_path)
    T = scaler.T

    # Load ensemble and run inference
    models = _load_ensemble(ensemble_dir, device)
    M = len(models)

    member_probs = []
    for model in models:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(team_a_tensor, team_b_tensor)
        probs = torch.softmax(logits.float() / T, dim=-1).cpu().numpy()[0]
        member_probs.append(probs)
        del model  # free VRAM for next member

    # Clear all models from GPU
    models.clear()
    torch.cuda.empty_cache()

    member_probs = np.stack(member_probs, axis=0)  # (M, 90)
    p_bar = member_probs.mean(axis=0)  # (90,)

    # Uncertainty
    entropy = -np.sum(p_bar * np.log(p_bar + _EPS))
    member_H = np.array([-np.sum(mp * np.log(mp + _EPS)) for mp in member_probs])
    mi = entropy - member_H.mean()
    confidence = float(p_bar.max())

    # Top-k plans
    top_ids = np.argsort(p_bar)[::-1][:top_k]
    plans = []
    for rank, aid in enumerate(top_ids):
        plans.append({
            "rank": rank + 1,
            "action90_id": int(aid),
            "probability": float(p_bar[aid]),
            "description": _format_plan(int(aid), species_a),
        })

    abstain = confidence < tau

    result = {
        "team_a": species_a,
        "team_b": species_b,
        "top_plans": plans,
        "confidence": confidence,
        "entropy": float(entropy),
        "mutual_information": float(mi),
        "temperature": T,
        "n_ensemble_members": M,
        "abstain": abstain,
        "tau": tau,
    }

    # Pretty print
    print("\n" + "=" * 70)
    print("  TurnZero Coach — Turn-Zero Team Preview Advisor")
    print("=" * 70)
    print(f"\n  Your Team (A):  {', '.join(species_a)}")
    print(f"  Opponent (B):   {', '.join(species_b)}")
    print(f"\n  Ensemble: {M} members  |  Temperature: {T:.3f}")
    print(f"  Confidence: {confidence:.4f}  |  Entropy: {entropy:.3f}  |  MI: {mi:.4f}")

    if abstain:
        print(f"\n  {'!' * 60}")
        print(f"  LOW CONFIDENCE (conf={confidence:.4f} < tau={tau:.4f})")
        print(f"  Recommendation: SCOUTING MODE")
        print(f"  The model is uncertain about this matchup.")
        print(f"  Consider reviewing similar matchups manually.")
        print(f"  {'!' * 60}")
    else:
        print(f"\n  Top-{top_k} Recommended Plans:")
        print(f"  {'─' * 60}")
        for plan in plans:
            pct = plan["probability"] * 100
            print(f"  #{plan['rank']}  ({pct:5.2f}%)  {plan['description']}")

    print()
    return result
