"""Export static data files for the web app.

Generates:
  - vocab.json — token-to-index mappings
  - action_table.json — 90-way action ID → lead/back pairs
  - lexicon.json — reverse role lexicon (token → roles[])
  - pokemon_data.json — species list + item/ability/tera/move lists for autocomplete
  - temperature.json — calibration temperature

Usage:
    uv run python scripts/export_web_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from turnzero.action_space import ACTION_TABLE
from turnzero.data.dataset import Vocab
from turnzero.tool.lexicon import REVERSE_LEXICON, ROLE_LEXICON


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "web" / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- vocab.json ---
    vocab_src = root / "data" / "assembled" / "regime_a" / "vocab.json"
    vocab_dst = out_dir / "vocab.json"
    shutil.copy2(vocab_src, vocab_dst)
    print(f"Copied vocab.json")

    # Verify vocab sizes
    vocab = Vocab.load(vocab_src)
    print(f"  Vocab sizes: {vocab.vocab_sizes}")

    # --- action_table.json ---
    # Each entry: [[lead0, lead1], [back0, back1]]
    table = [
        [[lead[0], lead[1]], [back[0], back[1]]]
        for lead, back in ACTION_TABLE
    ]
    with open(out_dir / "action_table.json", "w") as f:
        json.dump(table, f)
    print(f"Saved action_table.json ({len(table)} actions)")

    # --- lexicon.json ---
    # Reverse lexicon: token → list of role tags
    rev_lex = {token: sorted(roles) for token, roles in REVERSE_LEXICON.items()}
    with open(out_dir / "lexicon.json", "w") as f:
        json.dump(rev_lex, f, indent=2)
    print(f"Saved lexicon.json ({len(rev_lex)} entries)")

    # --- pokemon_data.json ---
    # Species, items, abilities, tera types, moves for autocomplete
    pokemon_data = {
        "species": sorted(k for k in vocab._tok2idx["species"] if k != "<UNK>"),
        "items": sorted(k for k in vocab._tok2idx["item"] if k != "<UNK>"),
        "abilities": sorted(k for k in vocab._tok2idx["ability"] if k != "<UNK>"),
        "tera_types": sorted(k for k in vocab._tok2idx["tera_type"] if k != "<UNK>"),
        "moves": sorted(k for k in vocab._tok2idx["move"] if k != "<UNK>"),
    }
    with open(out_dir / "pokemon_data.json", "w") as f:
        json.dump(pokemon_data, f)
    print(f"Saved pokemon_data.json:")
    for k, v in pokemon_data.items():
        print(f"  {k}: {len(v)}")

    # --- temperature.json ---
    temp_src = root / "outputs" / "calibration" / "run_001" / "temperature.json"
    temp_dst = out_dir / "temperature.json"
    shutil.copy2(temp_src, temp_dst)
    print(f"Copied temperature.json")

    print("\nAll static data exported!")


if __name__ == "__main__":
    main()
