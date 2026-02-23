#!/usr/bin/env python3
"""BO3 within-set adaptation analysis.

Measures how players change leads/bring-4 between games of the same BO3 set.
Links individual games via the |uhtml|bestof| line in Showdown logs.

Outputs:
    outputs/eval/bo3_adaptation.json
    outputs/plots/paper/bo3_adaptation_rates.{png,pdf}

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/run_bo3_adaptation.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

_DPI = 300
_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "data" / "raw" / "logs-gen9vgc2025reggbo3.json"
CLUSTER_JSON = ROOT / "outputs" / "eval" / "cluster_analysis.json"
OUT_EVAL = ROOT / "outputs" / "eval"
OUT_PLOTS = ROOT / "outputs" / "plots" / "paper"


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=_DPI)
    fig.savefig(out.with_suffix(".pdf"), dpi=_DPI)
    plt.close(fig)
    print(f"  Saved: {out.with_suffix('.png')}")


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class GameInfo:
    battle_id: str
    set_id: str
    game_num: int
    player_names: dict[str, str]  # {"p1": "Name", "p2": "Name"}
    winner_side: str | None       # "p1" or "p2" or None (forfeit/tie)
    leads: dict[str, frozenset[str]]    # {"p1": frozenset(species), ...}
    revealed: dict[str, frozenset[str]]
    species6: dict[str, frozenset[str]]  # full 6 species per side
    species_key: dict[str, str]   # "|".join(sorted(species)) per side


@dataclass
class Transition:
    set_id: str
    side: str
    prior_result: str        # "won_prior" | "lost_prior"
    lead_changed: bool
    bring4_changed: bool | None  # None if either game not Tier 1
    is_mirror: bool
    team_entropy: float | None   # from cluster_analysis.json


# ── Extraction ────────────────────────────────────────────────────────


_RE_SET_ID = re.compile(r'/game-(bestof3-[^"]+)')
_RE_GAME_NUM = re.compile(r"Game (\d+)")


def extract_bo3_linkage(
    raw_data: dict[str, Any],
) -> dict[str, dict[int, str]]:
    """Scan logs for |uhtml|bestof| lines and group by set_id.

    Returns {set_id: {game_num: battle_id}}.
    """
    sets: dict[str, dict[int, str]] = {}
    for battle_id, (_, log_text) in raw_data.items():
        for line in log_text.split("\n"):
            if "|uhtml|bestof|" not in line:
                continue
            m_set = _RE_SET_ID.search(line)
            m_game = _RE_GAME_NUM.search(line)
            if m_set and m_game:
                sid = m_set.group(1)
                gnum = int(m_game.group(1))
                sets.setdefault(sid, {})[gnum] = battle_id
            break  # only one bestof line per log
    return sets


def _parse_ident(ident: str) -> tuple[str, str]:
    """Parse 'p1a: Nickname' -> ('p1', 'Nickname')."""
    ident = ident.strip()
    side = ident.split(":")[0][:2]
    nickname = ident.split(": ", 1)[1]
    return side, nickname


def _species_from_details(details: str) -> str:
    return details.split(",")[0].strip()


def _match_to_showteam(
    switch_species: str, showteam_species: list[str], used: set[int] | None = None
) -> int | None:
    skip = used or set()
    for i, st in enumerate(showteam_species):
        if i not in skip and switch_species == st:
            return i
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if switch_species.startswith(st + "-") or st.startswith(switch_species + "-"):
            return i
    sw_base = switch_species.split("-")[0]
    for i, st in enumerate(showteam_species):
        if i in skip:
            continue
        if st.split("-")[0] == sw_base:
            return i
    return None


def parse_game_info(
    battle_id: str, set_id: str, game_num: int, log_text: str
) -> GameInfo | None:
    """Parse a single game log for player names, winner, leads, revealed, species."""
    lines = log_text.split("\n")

    # ── Player names ──
    player_names: dict[str, str] = {}
    for line in lines:
        if "|player|" in line:
            parts = line.split("|")
            # |player|p1|Name|avatar|rating
            if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
                player_names[parts[2]] = parts[3]

    if "p1" not in player_names or "p2" not in player_names:
        return None

    # ── Winner ──
    winner_name = None
    for line in lines:
        if "|win|" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                winner_name = parts[2]
            break

    winner_side = None
    if winner_name:
        for side, name in player_names.items():
            if name == winner_name:
                winner_side = side
                break

    # ── Showteam ──
    from turnzero.data.parser import parse_showteam_line

    showteam: dict[str, list] = {}
    for line in lines:
        if "|showteam|" in line:
            try:
                side, pokemon = parse_showteam_line(line)
                showteam[side] = pokemon
            except (ValueError, IndexError):
                pass

    if "p1" not in showteam or "p2" not in showteam:
        return None

    st_species = {s: [p.species for p in showteam[s]] for s in ("p1", "p2")}

    # ── Leads + revealed (same logic as parser.py) ──
    start_idx = None
    for i, line in enumerate(lines):
        if "|start" in line:
            start_idx = i
            break
    if start_idx is None:
        return None

    nick_to_idx: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    revealed: dict[str, set[str]] = {"p1": set(), "p2": set()}
    leads_done = False

    for line in lines[start_idx + 1 :]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1].strip()
        if tag == "turn":
            leads_done = True
        if tag in ("switch", "drag") and len(parts) >= 5:
            side, nick = _parse_ident(parts[2])
            sw_sp = _species_from_details(parts[3])
            if nick not in nick_to_idx[side]:
                idx = _match_to_showteam(sw_sp, st_species[side])
                if idx is not None:
                    nick_to_idx[side][nick] = idx
                else:
                    continue
            species = st_species[side][nick_to_idx[side][nick]]
            revealed[side].add(species)
            if not leads_done and len(leads[side]) < 2:
                leads[side].append(species)

    for side in ("p1", "p2"):
        if len(leads[side]) != 2:
            return None

    species6 = {s: frozenset(st_species[s]) for s in ("p1", "p2")}
    species_key = {
        s: "|".join(sorted(st_species[s])) for s in ("p1", "p2")
    }

    return GameInfo(
        battle_id=battle_id,
        set_id=set_id,
        game_num=game_num,
        player_names=player_names,
        winner_side=winner_side,
        leads={s: frozenset(leads[s]) for s in ("p1", "p2")},
        revealed={s: frozenset(revealed[s]) for s in ("p1", "p2")},
        species6=species6,
        species_key=species_key,
    )


# ── Transition building ──────────────────────────────────────────────


def build_transitions(
    set_to_games: dict[str, dict[int, str]],
    raw_data: dict[str, Any],
    entropy_index: dict[str, float],
) -> list[Transition]:
    """Build transition records for sequential game pairs within each set."""
    transitions: list[Transition] = []
    parse_ok = 0
    parse_fail = 0

    # Parse all relevant games
    game_cache: dict[str, GameInfo] = {}
    battle_ids_needed: set[str] = set()
    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        if len(gnums) < 2:
            continue
        for gn in gnums:
            battle_ids_needed.add(games[gn])

    print(f"  Parsing {len(battle_ids_needed):,} games from {len(set_to_games):,} sets...")

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

    # Build transitions from sequential pairs
    for sid, games in set_to_games.items():
        gnums = sorted(games.keys())
        for i in range(len(gnums) - 1):
            g1_bid = games[gnums[i]]
            g2_bid = games[gnums[i + 1]]
            if g1_bid not in game_cache or g2_bid not in game_cache:
                continue

            g1 = game_cache[g1_bid]
            g2 = game_cache[g2_bid]

            # Verify same players (names match)
            if g1.player_names != g2.player_names:
                continue

            for side in ("p1", "p2"):
                # Prior result
                if g1.winner_side is None:
                    continue
                prior_result = (
                    "won_prior" if g1.winner_side == side else "lost_prior"
                )

                # Lead change
                lead_changed = g1.leads[side] != g2.leads[side]

                # Bring-4 change (only if both games have 4 revealed)
                bring4_changed = None
                if len(g1.revealed[side]) == 4 and len(g2.revealed[side]) == 4:
                    bring4_changed = g1.revealed[side] != g2.revealed[side]

                # Mirror
                is_mirror = g1.species6["p1"] == g1.species6["p2"]

                # Team entropy
                sk = g1.species_key[side]
                team_entropy = entropy_index.get(sk)

                transitions.append(Transition(
                    set_id=sid,
                    side=side,
                    prior_result=prior_result,
                    lead_changed=lead_changed,
                    bring4_changed=bring4_changed,
                    is_mirror=is_mirror,
                    team_entropy=team_entropy,
                ))

    return transitions


# ── Statistics ────────────────────────────────────────────────────────


def compute_adaptation_stats(transitions: list[Transition]) -> dict:
    """Compute adaptation rates by condition with Wilson CIs and chi-squared."""
    from scipy.stats import chi2_contingency

    n_total = len(transitions)
    n_lead_changed = sum(1 for t in transitions if t.lead_changed)

    # Bring-4 (only transitions with data)
    b4_trans = [t for t in transitions if t.bring4_changed is not None]
    n_b4_changed = sum(1 for t in b4_trans if t.bring4_changed)

    # By prior result
    after_loss = [t for t in transitions if t.prior_result == "lost_prior"]
    after_win = [t for t in transitions if t.prior_result == "won_prior"]
    n_lead_after_loss = sum(1 for t in after_loss if t.lead_changed)
    n_lead_after_win = sum(1 for t in after_win if t.lead_changed)

    # By mirror
    mirrors = [t for t in transitions if t.is_mirror]
    non_mirrors = [t for t in transitions if not t.is_mirror]
    n_lead_mirror = sum(1 for t in mirrors if t.lead_changed)
    n_lead_non_mirror = sum(1 for t in non_mirrors if t.lead_changed)

    # Chi-squared: after_loss vs after_win lead change
    contingency = [
        [n_lead_after_loss, len(after_loss) - n_lead_after_loss],
        [n_lead_after_win, len(after_win) - n_lead_after_win],
    ]
    chi2, p_val, _, _ = chi2_contingency(contingency)

    def _rate_ci(k: int, n: int) -> dict:
        rate = k / n if n > 0 else 0.0
        lo, hi = _wilson_ci(rate, n)
        return {"rate": rate, "n": n, "k": k, "ci_lo": lo, "ci_hi": hi}

    stats: dict[str, Any] = {
        "n_transitions": n_total,
        "lead_change": _rate_ci(n_lead_changed, n_total),
        "bring4_change": _rate_ci(n_b4_changed, len(b4_trans)),
        "after_loss": {
            "lead_change": _rate_ci(n_lead_after_loss, len(after_loss)),
        },
        "after_win": {
            "lead_change": _rate_ci(n_lead_after_win, len(after_win)),
        },
        "mirror": {
            "lead_change": _rate_ci(n_lead_mirror, len(mirrors)),
            "n": len(mirrors),
        },
        "non_mirror": {
            "lead_change": _rate_ci(n_lead_non_mirror, len(non_mirrors)),
            "n": len(non_mirrors),
        },
        "chi_squared": {
            "chi2": chi2,
            "p_value": p_val,
            "contingency": contingency,
        },
    }

    # By entropy tercile (transitions with entropy data)
    ent_trans = [t for t in transitions if t.team_entropy is not None]
    if len(ent_trans) >= 30:
        entropies = sorted(t.team_entropy for t in ent_trans)
        t1 = entropies[len(entropies) // 3]
        t2 = entropies[2 * len(entropies) // 3]
        terciles = {"low": [], "mid": [], "high": []}
        for t in ent_trans:
            if t.team_entropy <= t1:
                terciles["low"].append(t)
            elif t.team_entropy <= t2:
                terciles["mid"].append(t)
            else:
                terciles["high"].append(t)

        stats["entropy_terciles"] = {}
        for label, group in terciles.items():
            n_lc = sum(1 for t in group if t.lead_changed)
            b4_g = [t for t in group if t.bring4_changed is not None]
            n_b4 = sum(1 for t in b4_g if t.bring4_changed)
            stats["entropy_terciles"][label] = {
                "lead_change": _rate_ci(n_lc, len(group)),
                "bring4_change": _rate_ci(n_b4, len(b4_g)),
                "entropy_range": [
                    min(t.team_entropy for t in group),
                    max(t.team_entropy for t in group),
                ],
            }
        stats["entropy_coverage"] = {
            "n_with_entropy": len(ent_trans),
            "n_total": n_total,
            "coverage": len(ent_trans) / n_total if n_total > 0 else 0.0,
        }

    # Adaptation effectiveness: did changing leads correlate with winning game 2?
    # (only for after_loss transitions)
    changed_won = sum(
        1 for t in after_loss if t.lead_changed
        # We don't have g2 winner stored in Transition, skip for now
    )

    return stats


# ── Figure ────────────────────────────────────────────────────────────


def make_figure(stats: dict) -> plt.Figure:
    """Grouped bar chart of lead-change rates by condition."""
    conditions = [
        ("Overall", stats["lead_change"]),
        ("After Win", stats["after_win"]["lead_change"]),
        ("After Loss", stats["after_loss"]["lead_change"]),
        ("Mirror", stats["mirror"]["lead_change"]),
        ("Non-Mirror", stats["non_mirror"]["lead_change"]),
    ]

    has_terciles = "entropy_terciles" in stats
    if has_terciles:
        # Second panel: bring-4 change by entropy tercile
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # ── Panel A: Lead-2 change rate by condition ──
    labels = [c[0] for c in conditions]
    rates = [c[1]["rate"] * 100 for c in conditions]
    ci_lo = [c[1]["ci_lo"] * 100 for c in conditions]
    ci_hi = [c[1]["ci_hi"] * 100 for c in conditions]
    errors = [[r - lo for r, lo in zip(rates, ci_lo)],
              [hi - r for r, hi in zip(rates, ci_hi)]]

    colors = [_COLORS[0], _COLORS[2], _COLORS[3], _COLORS[4], _COLORS[1]]
    x = np.arange(len(labels))
    bars = ax1.bar(x, rates, yerr=errors, capsize=4, color=colors,
                   edgecolor="white", linewidth=0.5, width=0.65)

    # Value labels
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Lead-2 Change Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Lead Adaptation by Condition", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Bring-4 and lead-2 change by entropy tercile ──
    if has_terciles:
        terc = stats["entropy_terciles"]
        terc_labels = ["Low H", "Mid H", "High H"]
        lead_rates = [terc[k]["lead_change"]["rate"] * 100 for k in ("low", "mid", "high")]
        lead_ci_lo = [terc[k]["lead_change"]["ci_lo"] * 100 for k in ("low", "mid", "high")]
        lead_ci_hi = [terc[k]["lead_change"]["ci_hi"] * 100 for k in ("low", "mid", "high")]
        b4_rates = [terc[k]["bring4_change"]["rate"] * 100 for k in ("low", "mid", "high")]
        b4_ci_lo = [terc[k]["bring4_change"]["ci_lo"] * 100 for k in ("low", "mid", "high")]
        b4_ci_hi = [terc[k]["bring4_change"]["ci_hi"] * 100 for k in ("low", "mid", "high")]

        x2 = np.arange(3)
        w = 0.32

        lead_err = [[r - lo for r, lo in zip(lead_rates, lead_ci_lo)],
                     [hi - r for r, hi in zip(lead_rates, lead_ci_hi)]]
        b4_err = [[r - lo for r, lo in zip(b4_rates, b4_ci_lo)],
                   [hi - r for r, hi in zip(b4_rates, b4_ci_hi)]]

        bars1 = ax2.bar(x2 - w / 2, lead_rates, w, yerr=lead_err, capsize=3,
                        color=_COLORS[0], edgecolor="white", label="Lead-2")
        bars2 = ax2.bar(x2 + w / 2, b4_rates, w, yerr=b4_err, capsize=3,
                        color=_COLORS[1], edgecolor="white", label="Bring-4")

        ax2.set_ylabel("Change Rate (%)")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(terc_labels, fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_title("(b) Adaptation by Team Entropy", fontsize=11)
        ax2.legend(frameon=False, fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("  BO3 Adaptation Analysis")
    print("=" * 70)

    # ── Load raw data ──
    print(f"\nLoading {RAW_PATH.name}...")
    t_load = time.time()
    with open(RAW_PATH) as f:
        raw_data = json.load(f)
    print(f"  {len(raw_data):,} battles loaded in {time.time() - t_load:.1f}s")

    # ── Load entropy index ──
    entropy_index: dict[str, float] = {}
    if CLUSTER_JSON.exists():
        print(f"Loading entropy index from {CLUSTER_JSON.name}...")
        with open(CLUSTER_JSON) as f:
            cluster_data = json.load(f)
        for key, metrics in cluster_data.items():
            entropy_index[key] = metrics["mean_entropy"]
        print(f"  {len(entropy_index):,} team entropies loaded")
    else:
        print("  Warning: cluster_analysis.json not found, skipping entropy analysis")

    # ── Extract BO3 linkage ──
    print("\nExtracting BO3 set linkage...")
    t_link = time.time()
    set_to_games = extract_bo3_linkage(raw_data)
    multi_game_sets = {
        sid: games for sid, games in set_to_games.items() if len(games) >= 2
    }
    print(f"  {len(set_to_games):,} total sets found")
    print(f"  {len(multi_game_sets):,} sets with >= 2 linkable games")

    game_count = Counter(len(g) for g in set_to_games.values())
    for n_games, count in sorted(game_count.items()):
        print(f"    {n_games}-game sets: {count:,}")
    print(f"  Linkage time: {time.time() - t_link:.1f}s")

    # ── Build transitions ──
    print("\nBuilding transitions...")
    t_trans = time.time()
    transitions = build_transitions(multi_game_sets, raw_data, entropy_index)
    print(f"  {len(transitions):,} transitions built in {time.time() - t_trans:.1f}s")

    if not transitions:
        print("ERROR: No transitions found!")
        return

    # ── Compute stats ──
    print("\nComputing adaptation statistics...")
    stats = compute_adaptation_stats(transitions)

    # Add metadata
    stats["metadata"] = {
        "source_file": RAW_PATH.name,
        "total_battles": len(raw_data),
        "total_sets": len(set_to_games),
        "multi_game_sets": len(multi_game_sets),
        "game_count_distribution": dict(sorted(game_count.items())),
    }

    # ── Save JSON ──
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    json_path = OUT_EVAL / "bo3_adaptation.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Generate figure ──
    print("\nGenerating figure...")
    fig = make_figure(stats)
    _save_fig(fig, OUT_PLOTS / "bo3_adaptation_rates")

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print("  BO3 ADAPTATION SUMMARY")
    print(f"{'=' * 70}")

    lc = stats["lead_change"]
    b4 = stats["bring4_change"]
    al = stats["after_loss"]["lead_change"]
    aw = stats["after_win"]["lead_change"]
    mi = stats["mirror"]["lead_change"]
    nm = stats["non_mirror"]["lead_change"]
    chi = stats["chi_squared"]

    print(f"\n  Transitions analyzed: {stats['n_transitions']:,}")
    print(f"  From {len(multi_game_sets):,} multi-game sets")
    print(f"\n  Overall lead-2 change rate: {lc['rate']*100:.1f}% "
          f"[{lc['ci_lo']*100:.1f}, {lc['ci_hi']*100:.1f}]")
    print(f"  Overall bring-4 change rate: {b4['rate']*100:.1f}% "
          f"[{b4['ci_lo']*100:.1f}, {b4['ci_hi']*100:.1f}] "
          f"(n={b4['n']:,})")
    print(f"\n  After loss:  {al['rate']*100:.1f}% "
          f"[{al['ci_lo']*100:.1f}, {al['ci_hi']*100:.1f}] (n={al['n']:,})")
    print(f"  After win:   {aw['rate']*100:.1f}% "
          f"[{aw['ci_lo']*100:.1f}, {aw['ci_hi']*100:.1f}] (n={aw['n']:,})")
    print(f"  Chi-squared: {chi['chi2']:.1f}, p = {chi['p_value']:.2e}")
    print(f"\n  Mirror:      {mi['rate']*100:.1f}% (n={stats['mirror']['n']:,})")
    print(f"  Non-mirror:  {nm['rate']*100:.1f}% (n={stats['non_mirror']['n']:,})")

    if "entropy_terciles" in stats:
        print(f"\n  Entropy terciles (coverage: "
              f"{stats['entropy_coverage']['n_with_entropy']:,}"
              f"/{stats['entropy_coverage']['n_total']:,} = "
              f"{stats['entropy_coverage']['coverage']*100:.1f}%):")
        for label in ("low", "mid", "high"):
            t_data = stats["entropy_terciles"][label]
            lc_t = t_data["lead_change"]
            b4_t = t_data["bring4_change"]
            e_range = t_data["entropy_range"]
            print(f"    {label:>4}: lead {lc_t['rate']*100:.1f}%, "
                  f"bring4 {b4_t['rate']*100:.1f}% "
                  f"(H: [{e_range[0]:.2f}, {e_range[1]:.2f}])")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
    main()
