"""Generate golden test vectors for TypeScript unit tests.

Runs the full Python pipeline on 5 fixed matchups and saves:
  - Encoded tensors (team_a, team_b as flat int arrays)
  - Per-model raw logits
  - Ensemble averaged probabilities
  - Marginals (lead_probs, bring_probs, lead_pair_probs)
  - Sensitivity KL values

Usage:
    uv run python scripts/generate_test_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from turnzero.action_space import ACTION_TABLE
from turnzero.data.dataset import Vocab, _encode_team
from turnzero.models.transformer import ModelConfig, OTSTransformer
from turnzero.tool.explain import compute_marginals, feature_sensitivity
from turnzero.uq.temperature import TemperatureScaler

# 5 fixed test matchups — species + full OTS where possible
TEST_MATCHUPS = [
    {
        "name": "standard_vgc",
        "team_a": {
            "pokemon": [
                {"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "tera_type": "Ghost", "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]},
                {"species": "Flutter Mane", "item": "Choice Specs", "ability": "Protosynthesis", "tera_type": "Fairy", "moves": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Mystical Fire"]},
                {"species": "Rillaboom", "item": "Miracle Seed", "ability": "Grassy Surge", "tera_type": "Grass", "moves": ["Grassy Glide", "Wood Hammer", "Fake Out", "U-turn"]},
                {"species": "Urshifu-Rapid-Strike", "item": "Focus Sash", "ability": "Unseen Fist", "tera_type": "Water", "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"]},
                {"species": "Tornadus", "item": "Focus Sash", "ability": "Prankster", "tera_type": "Ghost", "moves": ["Tailwind", "Rain Dance", "Taunt", "Bleakwind Storm"]},
                {"species": "Landorus", "item": "Life Orb", "ability": "Sheer Force", "tera_type": "Poison", "moves": ["Earth Power", "Sludge Bomb", "Protect", "Sandsear Storm"]},
            ]
        },
        "team_b": {
            "pokemon": [
                {"species": "Calyrex-Shadow", "item": "Focus Sash", "ability": "As One (Spectrier)", "tera_type": "Grass", "moves": ["Astral Barrage", "Psyshock", "Nasty Plot", "Protect"]},
                {"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "tera_type": "Water", "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]},
                {"species": "Whimsicott", "item": "Covert Cloak", "ability": "Prankster", "tera_type": "Steel", "moves": ["Tailwind", "Moonblast", "Encore", "Protect"]},
                {"species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "tera_type": "Fire", "moves": ["Grassy Glide", "Wood Hammer", "Fake Out", "U-turn"]},
                {"species": "Chien-Pao", "item": "Life Orb", "ability": "Sword of Ruin", "tera_type": "Ice", "moves": ["Ice Spinner", "Crunch", "Sacred Sword", "Protect"]},
                {"species": "Landorus", "item": "Choice Scarf", "ability": "Sheer Force", "tera_type": "Flying", "moves": ["Earth Power", "Sludge Bomb", "Psychic", "U-turn"]},
            ]
        },
    },
    {
        "name": "species_only",
        "team_a": {
            "pokemon": [
                {"species": "Incineroar", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Flutter Mane", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Rillaboom", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Urshifu-Rapid-Strike", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Tornadus", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Landorus", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
            ]
        },
        "team_b": {
            "pokemon": [
                {"species": "Calyrex-Shadow", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Incineroar", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Whimsicott", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Rillaboom", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Chien-Pao", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Landorus", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
            ]
        },
    },
    {
        "name": "trick_room",
        "team_a": {
            "pokemon": [
                {"species": "Torkoal", "item": "Charcoal", "ability": "Drought", "tera_type": "Fire", "moves": ["Eruption", "Heat Wave", "Earth Power", "Protect"]},
                {"species": "Dusclops", "item": "Eviolite", "ability": "Frisk", "tera_type": "Dark", "moves": ["Trick Room", "Night Shade", "Helping Hand", "Pain Split"]},
                {"species": "Amoonguss", "item": "Rocky Helmet", "ability": "Regenerator", "tera_type": "Water", "moves": ["Spore", "Rage Powder", "Pollen Puff", "Protect"]},
                {"species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive", "tera_type": "Grass", "moves": ["Drain Punch", "Wild Charge", "Ice Punch", "Fake Out"]},
                {"species": "Ursaluna-Bloodmoon", "item": "Life Orb", "ability": "Mind's Eye", "tera_type": "Normal", "moves": ["Blood Moon", "Earth Power", "Hyper Voice", "Protect"]},
                {"species": "Pelipper", "item": "Focus Sash", "ability": "Drizzle", "tera_type": "Steel", "moves": ["Hurricane", "Hydro Pump", "Tailwind", "Protect"]},
            ]
        },
        "team_b": {
            "pokemon": [
                {"species": "Flutter Mane", "item": "Booster Energy", "ability": "Protosynthesis", "tera_type": "Stellar", "moves": ["Moonblast", "Shadow Ball", "Thunderbolt", "Protect"]},
                {"species": "Landorus", "item": "Life Orb", "ability": "Sheer Force", "tera_type": "Poison", "moves": ["Earth Power", "Sludge Bomb", "Protect", "Sandsear Storm"]},
                {"species": "Rillaboom", "item": "Miracle Seed", "ability": "Grassy Surge", "tera_type": "Fire", "moves": ["Grassy Glide", "Wood Hammer", "Fake Out", "U-turn"]},
                {"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "tera_type": "Ghost", "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]},
                {"species": "Urshifu-Rapid-Strike", "item": "Choice Band", "ability": "Unseen Fist", "tera_type": "Water", "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "U-turn"]},
                {"species": "Tornadus", "item": "Focus Sash", "ability": "Prankster", "tera_type": "Ghost", "moves": ["Tailwind", "Rain Dance", "Taunt", "Bleakwind Storm"]},
            ]
        },
    },
    {
        "name": "rain_team",
        "team_a": {
            "pokemon": [
                {"species": "Pelipper", "item": "Focus Sash", "ability": "Drizzle", "tera_type": "Steel", "moves": ["Hurricane", "Hydro Pump", "Tailwind", "Protect"]},
                {"species": "Archaludon", "item": "Assault Vest", "ability": "Stamina", "tera_type": "Fairy", "moves": ["Flash Cannon", "Electro Shot", "Body Press", "Draco Meteor"]},
                {"species": "Basculegion", "item": "Choice Band", "ability": "Swift Swim", "tera_type": "Water", "moves": ["Wave Crash", "Last Respects", "Aqua Jet", "Flip Turn"]},
                {"species": "Rillaboom", "item": "Miracle Seed", "ability": "Grassy Surge", "tera_type": "Grass", "moves": ["Grassy Glide", "Wood Hammer", "Fake Out", "U-turn"]},
                {"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "tera_type": "Ghost", "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]},
                {"species": "Ogerpon-Wellspring", "item": "Wellspring Mask", "ability": "Water Absorb", "tera_type": "Water", "moves": ["Ivy Cudgel", "Horn Leech", "Follow Me", "Spiky Shield"]},
            ]
        },
        "team_b": {
            "pokemon": [
                {"species": "Torkoal", "item": "Charcoal", "ability": "Drought", "tera_type": "Fire", "moves": ["Eruption", "Heat Wave", "Earth Power", "Protect"]},
                {"species": "Dusclops", "item": "Eviolite", "ability": "Frisk", "tera_type": "Dark", "moves": ["Trick Room", "Night Shade", "Helping Hand", "Pain Split"]},
                {"species": "Amoonguss", "item": "Rocky Helmet", "ability": "Regenerator", "tera_type": "Water", "moves": ["Spore", "Rage Powder", "Pollen Puff", "Protect"]},
                {"species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive", "tera_type": "Grass", "moves": ["Drain Punch", "Wild Charge", "Ice Punch", "Fake Out"]},
                {"species": "Ursaluna-Bloodmoon", "item": "Life Orb", "ability": "Mind's Eye", "tera_type": "Normal", "moves": ["Blood Moon", "Earth Power", "Hyper Voice", "Protect"]},
                {"species": "Flutter Mane", "item": "Booster Energy", "ability": "Protosynthesis", "tera_type": "Fairy", "moves": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"]},
            ]
        },
    },
    {
        "name": "unk_fallback",
        "team_a": {
            "pokemon": [
                {"species": "FAKEMON", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "Pikachu", "item": "Light Ball", "ability": "Static", "tera_type": "Electric", "moves": ["Volt Tackle", "Iron Tail", "Quick Attack", "Protect"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
            ]
        },
        "team_b": {
            "pokemon": [
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
                {"species": "UNK", "item": "UNK", "ability": "UNK", "tera_type": "UNK", "moves": ["UNK", "UNK", "UNK", "UNK"]},
            ]
        },
    },
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "web" / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # CPU for reproducibility

    # Load vocab
    vocab = Vocab.load(root / "data" / "assembled" / "regime_a" / "vocab.json")

    # Load temperature
    temp_path = root / "outputs" / "calibration" / "run_001" / "temperature.json"
    with open(temp_path) as f:
        T = json.load(f)["T"]

    # Load ensemble
    ensemble_dir = root / "outputs" / "runs"
    ckpt_dirs = sorted(ensemble_dir.glob("ensemble_*"))
    models: list[OTSTransformer] = []
    for d in ckpt_dirs:
        model = OTSTransformer.load_from_checkpoint(d / "best.pt", device)
        models.append(model)
    print(f"Loaded {len(models)} ensemble members")

    test_vectors = []

    for matchup in TEST_MATCHUPS:
        name = matchup["name"]
        print(f"\nGenerating vectors for: {name}")

        team_a_dict = matchup["team_a"]
        team_b_dict = matchup["team_b"]

        # Encode
        team_a_enc = _encode_team(team_a_dict, vocab)  # (6, 8)
        team_b_enc = _encode_team(team_b_dict, vocab)  # (6, 8)
        team_a_t = team_a_enc.unsqueeze(0).to(device)  # (1, 6, 8)
        team_b_t = team_b_enc.unsqueeze(0).to(device)

        # Per-model logits and probs
        member_logits = []
        member_probs = []
        for model in models:
            with torch.no_grad():
                logits = model(team_a_t, team_b_t)
            logits_np = logits.float().cpu().numpy()[0]
            probs_np = torch.softmax(torch.tensor(logits_np) / T, dim=-1).numpy()
            member_logits.append(logits_np.tolist())
            member_probs.append(probs_np.tolist())

        # Ensemble average
        p_bar = np.mean(member_probs, axis=0)

        # Marginals
        marginals = compute_marginals(p_bar)

        # Sensitivity
        sensitivity = feature_sensitivity(models, team_a_t, team_b_t, T, device)

        vector = {
            "name": name,
            "team_a_encoded": team_a_enc.tolist(),
            "team_b_encoded": team_b_enc.tolist(),
            "member_logits": member_logits,
            "member_probs": member_probs,
            "ensemble_probs": p_bar.tolist(),
            "marginals": {
                "lead_probs": marginals["lead_probs"].tolist(),
                "bring_probs": marginals["bring_probs"].tolist(),
                "lead_pair_probs": marginals["lead_pair_probs"].tolist(),
            },
            "sensitivity": sensitivity,
            "temperature": T,
        }
        test_vectors.append(vector)

    # Save
    out_path = out_dir / "test_vectors.json"
    with open(out_path, "w") as f:
        json.dump(test_vectors, f, indent=2)
    print(f"\nSaved {len(test_vectors)} test vectors to {out_path}")


if __name__ == "__main__":
    main()
