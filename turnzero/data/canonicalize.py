"""Stage 3: Canonicalize parsed MatchExamples.

Normalizes names (CamelCase → display), sorts moves (UNK last) and
pokemon (by canonical key), recomputes stable hashes, and deduplicates
exact (team_a_id, team_b_id, action90_id, format_id) quadruples.

Reference: docs/PROJECT_BIBLE.md Section 2.3
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from turnzero.action_space import lead_back_to_action90
from turnzero.data.io_utils import read_jsonl, write_manifest
from turnzero.schemas import (
    Label,
    MatchExample,
    Pokemon,
    TeamSheet,
)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# CamelCase → display exceptions where the regex gives wrong output.
# Bidirectional: the inverse (display → CamelCase) is derivable by stripping
# spaces/hyphens, but these specific cases need an explicit lookup.
_NAME_EXCEPTIONS: dict[str, str] = {
    # Abilities with prepositions (lowercase "of" embedded)
    "SwordofRuin": "Sword of Ruin",
    "TabletsofRuin": "Tablets of Ruin",
    "VesselofRuin": "Vessel of Ruin",
    "BeadsofRuin": "Beads of Ruin",
    "PowerofAlchemy": "Power of Alchemy",
    # Moves with hyphens lost in CamelCase
    "Uturn": "U-turn",
    "WillOWisp": "Will-O-Wisp",
    "FreezeDry": "Freeze-Dry",
    "XScissor": "X-Scissor",
    "PowerUpPunch": "Power-Up Punch",
    # All-lowercase (Showdown encoding quirk for signature moves)
    "behemothblade": "Behemoth Blade",
    "behemothbash": "Behemoth Bash",
}

# Insert space before uppercase letter that follows a lowercase letter.
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def camel_to_display(name: str) -> str:
    """Convert a CamelCase name to canonical display format.

    Examples::

        "FakeOut"           → "Fake Out"
        "AssaultVest"       → "Assault Vest"
        "SwordofRuin"       → "Sword of Ruin"
        "Uturn"             → "U-turn"
        "behemothblade"     → "Behemoth Blade"
        "UNK"               → "UNK"
        "Urshifu-Rapid-Strike" → "Urshifu-Rapid-Strike"

    The function is idempotent: already-canonical names pass through unchanged.
    """
    if name == "UNK":
        return name
    if name in _NAME_EXCEPTIONS:
        return _NAME_EXCEPTIONS[name]
    return _CAMEL_SPLIT_RE.sub(" ", name)


# ---------------------------------------------------------------------------
# Hashing helpers (canonical-name versions of parser helpers)
# ---------------------------------------------------------------------------

def _compute_hash(text: str) -> str:
    """Truncated sha256 hex digest (16 chars = 64 bits)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _canonical_key(p: Pokemon) -> str:
    """Canonical sort key for a single Pokemon.

    Moves are sorted alphabetically with UNK last.
    """
    ms = ",".join(sorted(p.moves, key=lambda m: (m == "UNK", m)))
    return f"{p.species}|{p.item}|{p.ability}|{p.tera_type}|{ms}"


def _team_id(pokemon: list[Pokemon]) -> str:
    """Order-invariant team hash from canonical builds."""
    keys = sorted(_canonical_key(p) for p in pokemon)
    return _compute_hash("|".join(keys))


def _species_key(pokemon: list[Pokemon]) -> str:
    """Order-invariant hash of species set."""
    return _compute_hash("|".join(sorted(p.species for p in pokemon)))


# ---------------------------------------------------------------------------
# Pokemon / Team / Example canonicalization
# ---------------------------------------------------------------------------

def canonicalize_pokemon(p: Pokemon) -> Pokemon:
    """Normalize names and sort moves for a single Pokemon."""
    return Pokemon(
        species=camel_to_display(p.species),
        item=camel_to_display(p.item),
        ability=camel_to_display(p.ability),
        tera_type=camel_to_display(p.tera_type),
        moves=sorted(
            [camel_to_display(m) for m in p.moves],
            key=lambda m: (m == "UNK", m),
        ),
    )


def canonicalize_example(ex: MatchExample) -> MatchExample:
    """Canonicalize both teams and remap labels for team_a reordering.

    Steps:
      1. Normalize all name fields (CamelCase → display).
      2. Sort moves within each Pokemon (UNK last).
      3. Sort Pokemon by canonical key.
      4. Remap label indices from old team_a order to new canonical order.
      5. Recompute team_id, species_key, and action90_id.
    """
    # --- Canonicalize individual pokemon ---
    canon_a = [canonicalize_pokemon(p) for p in ex.team_a.pokemon]
    canon_b = [canonicalize_pokemon(p) for p in ex.team_b.pokemon]

    # --- Sort team_a, tracking old→new index mapping ---
    indexed_a = list(enumerate(canon_a))
    indexed_a.sort(key=lambda pair: _canonical_key(pair[1]))
    sorted_old_indices_a = [old_idx for old_idx, _ in indexed_a]
    sorted_mons_a = [mon for _, mon in indexed_a]

    old_to_new_a = {old: new for new, old in enumerate(sorted_old_indices_a)}

    # --- Sort team_b (no label remapping needed) ---
    canon_b.sort(key=_canonical_key)

    # --- Build TeamSheets with recomputed hashes ---
    team_a = TeamSheet(
        team_id=_team_id(sorted_mons_a),
        species_key=_species_key(sorted_mons_a),
        format_id=ex.team_a.format_id,
        pokemon=sorted_mons_a,
        reconstruction_quality=ex.team_a.reconstruction_quality,
    )
    team_b = TeamSheet(
        team_id=_team_id(canon_b),
        species_key=_species_key(canon_b),
        format_id=ex.team_b.format_id,
        pokemon=canon_b,
        reconstruction_quality=ex.team_b.reconstruction_quality,
    )

    # --- Remap label indices ---
    new_lead2 = tuple(sorted(old_to_new_a[i] for i in ex.label.lead2_idx))
    new_back2 = tuple(sorted(old_to_new_a[i] for i in ex.label.back2_idx))
    new_action90 = lead_back_to_action90(new_lead2, new_back2)

    label = Label(
        lead2_idx=new_lead2,
        back2_idx=new_back2,
        action90_id=new_action90,
    )

    return MatchExample(
        example_id=ex.example_id,
        match_group_id=ex.match_group_id,
        battle_id=ex.battle_id,
        team_a=team_a,
        team_b=team_b,
        label=label,
        label_quality=ex.label_quality,
        format_id=ex.format_id,
        metadata=ex.metadata,
        split_keys=ex.split_keys,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_key(ex: MatchExample) -> tuple[str, str, int, str]:
    """Exact-duplicate key: (team_a_id, team_b_id, action90_id, format_id)."""
    return (ex.team_a.team_id, ex.team_b.team_id,
            ex.label.action90_id, ex.format_id)


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def run_canonicalize(in_path: str, out_dir: str) -> dict[str, Any]:
    """Canonicalize all examples and write deduped output.

    Args:
        in_path: Path to parsed match_examples.jsonl.
        out_dir: Output directory for canonicalized JSONL + manifest.

    Returns:
        Manifest dict with canonicalization statistics.
    """
    in_path_p = Path(in_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"Reading examples from {in_path_p} ...")
    t0 = time.time()

    seen: set[tuple[str, str, int, str]] = set()
    unique_team_ids: set[str] = set()
    examples_read = 0
    examples_written = 0
    duplicates = 0
    errors = 0
    error_samples: list[dict[str, str]] = []

    jsonl_path = out_dir_p / "match_examples.jsonl"

    with open(jsonl_path, "w") as f:
        for d in read_jsonl(in_path_p):
            examples_read += 1
            try:
                ex = MatchExample.from_dict(d)
                canon = canonicalize_example(ex)

                key = _dedup_key(canon)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)

                unique_team_ids.add(canon.team_a.team_id)
                unique_team_ids.add(canon.team_b.team_id)

                f.write(json.dumps(canon.to_dict(), separators=(",", ":")) + "\n")
                examples_written += 1
            except Exception as e:
                errors += 1
                if len(error_samples) < 20:
                    error_samples.append({
                        "example_id": d.get("example_id", "unknown"),
                        "error": str(e),
                    })

            if examples_read % 10000 == 0:
                print(f"  {examples_read} examples processed ...")

    elapsed = time.time() - t0
    print(f"Canonicalized {examples_read} → {examples_written} examples "
          f"({duplicates} dupes, {errors} errors) in {elapsed:.1f}s")

    manifest: dict[str, Any] = {
        "in_path": str(in_path_p),
        "examples_read": examples_read,
        "examples_written": examples_written,
        "duplicates_removed": duplicates,
        "duplicate_rate": round(duplicates / examples_read, 6) if examples_read else 0,
        "unique_team_ids": len(unique_team_ids),
        "errors": errors,
        "error_rate": round(errors / examples_read, 6) if examples_read else 0,
        "canonicalize_time_seconds": round(elapsed, 1),
        "error_samples": error_samples,
    }
    write_manifest(out_dir_p / "canonicalize_manifest.json", manifest)
    return manifest
