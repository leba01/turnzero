#!/usr/bin/env python3
"""Adaptation effectiveness analysis: does changing leads after a loss help win game 2?

Builds a 2x2x2 table (prior_result × changed × g2_outcome) and tests whether
changers win G2 at different rates than stayers.

Also analyzes whether losers switch *toward* or *away from* opponent's G1 leads.

Outputs:
    outputs/eval/adaptation_effectiveness.json

Usage:
    cd /home/walter/CS229/turnzero
    PYTHONPATH=. .venv/bin/python scripts/run_adaptation_effectiveness.py
"""

from __future__ import annotations

import json
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


@dataclass
class EffectivenessRecord:
    set_id: str
    side: str
    prior_result: str           # "won_prior" | "lost_prior"
    lead_changed: bool
    g2_won: bool
    # Directional analysis (only for changers after loss)
    switched_toward_opp: bool | None  # True if new leads overlap opponent's g1 leads
    switched_away: bool | None


def build_effectiveness_records(
    set_to_games: dict[str, dict[int, str]],
    raw_data: dict[str, Any],
) -> list[EffectivenessRecord]:
    """Build records tracking whether lead changes correlated with G2 wins."""
    records: list[EffectivenessRecord] = []
    game_cache: dict[str, GameInfo] = {}
    parse_ok = parse_fail = 0

    # Parse all needed games
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
            else:
                parse_fail += 1

    print(f"  Parsed: {parse_ok:,} ok, {parse_fail:,} failed")

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

            for side in ("p1", "p2"):
                opp = "p2" if side == "p1" else "p1"
                prior_result = "won_prior" if g1.winner_side == side else "lost_prior"
                lead_changed = g1.leads[side] != g2.leads[side]
                g2_won = g2.winner_side == side

                # Directional: did changer move toward opponent's G1 leads?
                switched_toward_opp = None
                switched_away = None
                if lead_changed and prior_result == "lost_prior":
                    opp_g1_leads = g1.leads[opp]
                    new_leads = g2.leads[side]
                    old_leads = g1.leads[side]
                    # "toward" = new leads overlap with opponent's g1 leads more than old
                    old_overlap = len(old_leads & opp_g1_leads)
                    new_overlap = len(new_leads & opp_g1_leads)
                    switched_toward_opp = new_overlap > old_overlap
                    switched_away = new_overlap < old_overlap

                records.append(EffectivenessRecord(
                    set_id=sid,
                    side=side,
                    prior_result=prior_result,
                    lead_changed=lead_changed,
                    g2_won=g2_won,
                    switched_toward_opp=switched_toward_opp,
                    switched_away=switched_away,
                ))

    return records


def _rate_ci(k: int, n: int) -> dict:
    rate = k / n if n > 0 else 0.0
    lo, hi = _wilson_ci(rate, n)
    return {"rate": rate, "n": n, "k": k, "ci_lo": lo, "ci_hi": hi}


def compute_effectiveness(records: list[EffectivenessRecord]) -> dict:
    """Compute 2x2 tables and tests."""
    # Split by prior result
    after_loss = [r for r in records if r.prior_result == "lost_prior"]
    after_win = [r for r in records if r.prior_result == "won_prior"]

    def _table(subset: list[EffectivenessRecord]) -> dict:
        changed_won = sum(1 for r in subset if r.lead_changed and r.g2_won)
        changed_lost = sum(1 for r in subset if r.lead_changed and not r.g2_won)
        stayed_won = sum(1 for r in subset if not r.lead_changed and r.g2_won)
        stayed_lost = sum(1 for r in subset if not r.lead_changed and not r.g2_won)

        n_changed = changed_won + changed_lost
        n_stayed = stayed_won + stayed_lost

        contingency = [[changed_won, changed_lost], [stayed_won, stayed_lost]]
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
        except ValueError:
            chi2, p_val = 0.0, 1.0

        return {
            "changed_won": changed_won,
            "changed_lost": changed_lost,
            "stayed_won": stayed_won,
            "stayed_lost": stayed_lost,
            "changed_win_rate": _rate_ci(changed_won, n_changed),
            "stayed_win_rate": _rate_ci(stayed_won, n_stayed),
            "chi2": chi2,
            "p_value": p_val,
            "contingency": contingency,
        }

    loss_table = _table(after_loss)
    win_table = _table(after_win)
    overall_table = _table(records)

    # Directional analysis: among losers who changed, toward vs away
    loss_changers = [r for r in after_loss if r.lead_changed]
    toward = [r for r in loss_changers if r.switched_toward_opp is True]
    away = [r for r in loss_changers if r.switched_away is True]
    neutral = [r for r in loss_changers
               if r.switched_toward_opp is False and r.switched_away is False]

    directional = {
        "n_changers": len(loss_changers),
        "toward_opp": {
            "n": len(toward),
            "win_rate": _rate_ci(sum(1 for r in toward if r.g2_won), len(toward)),
        },
        "away_from_opp": {
            "n": len(away),
            "win_rate": _rate_ci(sum(1 for r in away if r.g2_won), len(away)),
        },
        "neutral": {
            "n": len(neutral),
            "win_rate": _rate_ci(sum(1 for r in neutral if r.g2_won), len(neutral)),
        },
    }

    return {
        "n_records": len(records),
        "after_loss": loss_table,
        "after_win": win_table,
        "overall": overall_table,
        "directional_after_loss": directional,
    }


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  Adaptation Effectiveness Analysis")
    print("=" * 70)

    print(f"\nLoading {RAW_PATH.name}...")
    with open(RAW_PATH) as f:
        raw_data = json.load(f)
    print(f"  {len(raw_data):,} battles loaded")

    print("\nExtracting BO3 set linkage...")
    set_to_games = extract_bo3_linkage(raw_data)
    multi = {sid: g for sid, g in set_to_games.items() if len(g) >= 2}
    print(f"  {len(multi):,} sets with >= 2 games")

    print("\nBuilding effectiveness records...")
    records = build_effectiveness_records(multi, raw_data)
    print(f"  {len(records):,} records (require both g1 and g2 winners)")

    print("\nComputing effectiveness stats...")
    stats = compute_effectiveness(records)

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_EVAL / "adaptation_effectiveness.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print results
    print(f"\n{'=' * 70}")
    print("  ADAPTATION EFFECTIVENESS RESULTS")
    print(f"{'=' * 70}")

    for label, key in [("AFTER LOSS", "after_loss"), ("AFTER WIN", "after_win"),
                        ("OVERALL", "overall")]:
        t = stats[key]
        cwr = t["changed_win_rate"]
        swr = t["stayed_win_rate"]
        print(f"\n  {label}:")
        print(f"    Changed leads → won G2: {cwr['rate']*100:.1f}% "
              f"[{cwr['ci_lo']*100:.1f}, {cwr['ci_hi']*100:.1f}] "
              f"(n={cwr['n']:,})")
        print(f"    Stayed leads  → won G2: {swr['rate']*100:.1f}% "
              f"[{swr['ci_lo']*100:.1f}, {swr['ci_hi']*100:.1f}] "
              f"(n={swr['n']:,})")
        diff = cwr['rate'] - swr['rate']
        print(f"    Difference: {diff*100:+.1f}pp  "
              f"(chi2={t['chi2']:.1f}, p={t['p_value']:.2e})")

    d = stats["directional_after_loss"]
    print(f"\n  DIRECTIONAL (losers who changed, n={d['n_changers']:,}):")
    for direction in ("toward_opp", "away_from_opp", "neutral"):
        dd = d[direction]
        wr = dd["win_rate"]
        label = {"toward_opp": "Toward opponent's leads",
                 "away_from_opp": "Away from opponent's leads",
                 "neutral": "Neither toward nor away"}[direction]
        print(f"    {label}: {wr['rate']*100:.1f}% win rate "
              f"[{wr['ci_lo']*100:.1f}, {wr['ci_hi']*100:.1f}] "
              f"(n={dd['n']:,})")

    # Framing decision
    loss_t = stats["after_loss"]
    diff = loss_t["changed_win_rate"]["rate"] - loss_t["stayed_win_rate"]["rate"]
    p = loss_t["p_value"]
    print(f"\n  FRAMING DECISION:")
    if p > 0.05:
        print(f"    Difference not significant (p={p:.3f}) → adaptation is noise")
    elif diff > 0:
        print(f"    Changers win more ({diff*100:+.1f}pp, p={p:.2e}) "
              f"→ adaptation works but is unpredictable")
    else:
        print(f"    Changers win less ({diff*100:+.1f}pp, p={p:.2e}) "
              f"→ adaptation backfires under pressure")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
