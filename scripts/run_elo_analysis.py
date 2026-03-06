#!/usr/bin/env python3
"""Elo-stratified adaptation analysis.

Extracts Elo ratings from |player| lines and stratifies the win-stay/lose-shift
pattern by skill tier. Tests whether stronger players show more/less WSLS.

Outputs:
    outputs/eval/elo_adaptation.json

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_elo_analysis.py
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency

from scripts.run_bo3_adaptation import (
    RAW_PATH,
    OUT_EVAL,
    GameInfo,
    _wilson_ci,
    extract_bo3_linkage,
    parse_game_info,
)


def _extract_ratings(log_text: str) -> dict[str, int] | None:
    """Extract Elo ratings from |player| lines. Returns {side: rating} or None."""
    ratings: dict[str, int] = {}
    for line in log_text.split("\n"):
        if "|player|" not in line:
            continue
        parts = line.split("|")
        # |player|p1|Name|avatar|rating
        if len(parts) >= 6 and parts[2] in ("p1", "p2"):
            try:
                ratings[parts[2]] = int(parts[5])
            except (ValueError, IndexError):
                pass
    if "p1" in ratings and "p2" in ratings:
        return ratings
    return None


@dataclass
class EloRecord:
    set_id: str
    side: str
    rating: int
    prior_result: str       # "won_prior" | "lost_prior"
    lead_changed: bool
    g2_won: bool


def build_elo_records(
    set_to_games: dict[str, dict[int, str]],
    raw_data: dict[str, Any],
) -> list[EloRecord]:
    """Build records with Elo ratings attached."""
    game_cache: dict[str, GameInfo] = {}
    rating_cache: dict[str, dict[str, int]] = {}  # battle_id -> ratings
    parse_ok = parse_fail = 0

    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        if len(gnums) < 2:
            continue
        for gn in gnums:
            bid = games[gn]
            if bid in game_cache:
                continue
            _, log_text = raw_data[bid]
            gi = parse_game_info(bid, sid, gn, log_text)
            if gi is not None:
                game_cache[bid] = gi
                parse_ok += 1
                ratings = _extract_ratings(log_text)
                if ratings:
                    rating_cache[bid] = ratings
            else:
                parse_fail += 1

    print(f"  Parsed: {parse_ok:,} ok, {parse_fail:,} failed")
    print(f"  Ratings found: {len(rating_cache):,} / {parse_ok:,} games")

    records: list[EloRecord] = []
    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        for i in range(len(gnums) - 1):
            g1_bid = games[gnums[i]]
            g2_bid = games[gnums[i + 1]]
            if g1_bid not in game_cache or g2_bid not in game_cache:
                continue
            g1 = game_cache[g1_bid]
            g2 = game_cache[g2_bid]
            if g1.player_names != g2.player_names:
                continue
            if g1.winner_side is None or g2.winner_side is None:
                continue
            if g1_bid not in rating_cache:
                continue

            ratings = rating_cache[g1_bid]
            for side in ("p1", "p2"):
                if side not in ratings:
                    continue
                prior = "won_prior" if g1.winner_side == side else "lost_prior"
                records.append(EloRecord(
                    set_id=sid,
                    side=side,
                    rating=ratings[side],
                    prior_result=prior,
                    lead_changed=g1.leads[side] != g2.leads[side],
                    g2_won=g2.winner_side == side,
                ))

    return records


def _rate_ci(k: int, n: int) -> dict:
    rate = k / n if n > 0 else 0.0
    lo, hi = _wilson_ci(rate, n)
    return {"rate": rate, "n": n, "k": k, "ci_lo": lo, "ci_hi": hi}


def compute_elo_stratified(records: list[EloRecord]) -> dict:
    """Stratify by Elo quartiles and compute adaptation rates + effectiveness."""
    all_ratings = sorted(r.rating for r in records)
    q25 = int(np.percentile(all_ratings, 25))
    q50 = int(np.percentile(all_ratings, 50))
    q75 = int(np.percentile(all_ratings, 75))

    tiers = {
        "Q1_low": (0, q25),
        "Q2": (q25, q50),
        "Q3": (q50, q75),
        "Q4_high": (q75, 99999),
    }

    results = {
        "n_records": len(records),
        "rating_stats": {
            "min": int(all_ratings[0]),
            "q25": q25, "median": q50, "q75": q75,
            "max": int(all_ratings[-1]),
            "mean": float(np.mean(all_ratings)),
        },
        "tiers": {},
    }

    for tier_name, (lo, hi) in tiers.items():
        tier_recs = [r for r in records if lo <= r.rating < hi]
        if not tier_recs:
            continue

        after_loss = [r for r in tier_recs if r.prior_result == "lost_prior"]
        after_win = [r for r in tier_recs if r.prior_result == "won_prior"]

        n_change_loss = sum(1 for r in after_loss if r.lead_changed)
        n_change_win = sum(1 for r in after_win if r.lead_changed)

        # Effectiveness: changers vs stayers win rate (after loss only)
        loss_changed = [r for r in after_loss if r.lead_changed]
        loss_stayed = [r for r in after_loss if not r.lead_changed]
        changed_won = sum(1 for r in loss_changed if r.g2_won)
        stayed_won = sum(1 for r in loss_stayed if r.g2_won)

        results["tiers"][tier_name] = {
            "rating_range": [lo, hi],
            "n": len(tier_recs),
            "change_rate_after_loss": _rate_ci(n_change_loss, len(after_loss)),
            "change_rate_after_win": _rate_ci(n_change_win, len(after_win)),
            "effectiveness_after_loss": {
                "changed_win_rate": _rate_ci(changed_won, len(loss_changed)),
                "stayed_win_rate": _rate_ci(stayed_won, len(loss_stayed)),
            },
        }

    # Test: is change rate after loss correlated with tier?
    # Build contingency table: rows=tiers, cols=[changed, stayed]
    tier_names = list(results["tiers"].keys())
    contingency = []
    for tn in tier_names:
        t = results["tiers"][tn]["change_rate_after_loss"]
        contingency.append([t["k"], t["n"] - t["k"]])

    try:
        chi2, p_val, _, _ = chi2_contingency(contingency)
    except ValueError:
        chi2, p_val = 0.0, 1.0

    results["tier_independence_test"] = {
        "chi2": chi2,
        "p_value": p_val,
        "description": "Tests whether change rate after loss differs across Elo tiers",
    }

    return results


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Elo-Stratified Adaptation Analysis")
    print("=" * 70)

    print(f"\nLoading {RAW_PATH.name}...")
    with open(RAW_PATH) as f:
        raw_data = json.load(f)
    print(f"  {len(raw_data):,} battles loaded")

    print("\nExtracting BO3 set linkage...")
    set_to_games = extract_bo3_linkage(raw_data)
    multi = {sid: g for sid, g in set_to_games.items() if len(g) >= 2}
    print(f"  {len(multi):,} sets with >= 2 games")

    print("\nBuilding Elo records...")
    records = build_elo_records(multi, raw_data)
    print(f"  {len(records):,} records with ratings")

    if not records:
        print("ERROR: No records with Elo ratings found!")
        return

    print("\nComputing Elo-stratified stats...")
    stats = compute_elo_stratified(records)

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_EVAL / "elo_adaptation.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  ELO-STRATIFIED ADAPTATION SUMMARY")
    print(f"{'=' * 70}")

    rs = stats["rating_stats"]
    print(f"\n  Rating distribution: [{rs['min']}, {rs['max']}], "
          f"median={rs['median']}, mean={rs['mean']:.0f}")
    print(f"  Quartiles: Q1<{rs['q25']}, Q2<{rs['median']}, Q3<{rs['q75']}")

    print(f"\n  {'Tier':<12} {'N':>7} {'ChgLoss%':>9} {'ChgWin%':>9} "
          f"{'Chg→Win%':>9} {'Stay→Win%':>10}")
    print("  " + "-" * 60)

    for tn, td in stats["tiers"].items():
        cl = td["change_rate_after_loss"]
        cw = td["change_rate_after_win"]
        eff_c = td["effectiveness_after_loss"]["changed_win_rate"]
        eff_s = td["effectiveness_after_loss"]["stayed_win_rate"]
        print(f"  {tn:<12} {td['n']:>7,} {cl['rate']*100:>8.1f}% {cw['rate']*100:>8.1f}% "
              f"{eff_c['rate']*100:>8.1f}% {eff_s['rate']*100:>9.1f}%")

    test = stats["tier_independence_test"]
    print(f"\n  Tier independence (chi2): {test['chi2']:.1f}, p={test['p_value']:.2e}")
    if test["p_value"] < 0.05:
        print("  → Elo tier significantly affects change rate after loss")
    else:
        print("  → No significant Elo effect on change rate")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
