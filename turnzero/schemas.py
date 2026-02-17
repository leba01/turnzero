"""Canonical data schemas for TurnZero pipeline.

Reference: docs/PROJECT_BIBLE.md Section 2.1
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Pokemon:
    """Single Pokemon with OTS fields. Unknown fields use 'UNK'."""

    species: str
    item: str = "UNK"
    ability: str = "UNK"
    tera_type: str = "UNK"
    moves: list[str] = field(default_factory=lambda: ["UNK"] * 4)

    def __post_init__(self) -> None:
        if len(self.moves) != 4:
            raise ValueError(f"moves must have exactly 4 entries, got {len(self.moves)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pokemon:
        return cls(**d)


@dataclass
class ReconstructionQuality:
    """Tracks how many OTS fields are known vs UNK."""

    fields_known: int
    fields_total: int = 42  # 6 mons x 7 fields (species + item + ability + tera + 4 moves... but species always known -> still 42 countable)
    source_method: str = "showteam_direct"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReconstructionQuality:
        return cls(**d)


@dataclass
class TeamSheet:
    """A team of 6 Pokemon with stable hashes for clustering and dedup."""

    team_id: str
    species_key: str
    format_id: str
    pokemon: list[Pokemon]
    reconstruction_quality: ReconstructionQuality | None = None

    def __post_init__(self) -> None:
        if len(self.pokemon) != 6:
            raise ValueError(f"TeamSheet must have exactly 6 pokemon, got {len(self.pokemon)}")

    def to_dict(self) -> dict[str, Any]:
        d = {
            "team_id": self.team_id,
            "species_key": self.species_key,
            "format_id": self.format_id,
            "pokemon": [p.to_dict() for p in self.pokemon],
        }
        if self.reconstruction_quality is not None:
            d["reconstruction_quality"] = self.reconstruction_quality.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamSheet:
        pokemon = [Pokemon.from_dict(p) for p in d["pokemon"]]
        rq = None
        if "reconstruction_quality" in d and d["reconstruction_quality"] is not None:
            rq = ReconstructionQuality.from_dict(d["reconstruction_quality"])
        return cls(
            team_id=d["team_id"],
            species_key=d["species_key"],
            format_id=d["format_id"],
            pokemon=pokemon,
            reconstruction_quality=rq,
        )


@dataclass
class Label:
    """Lead-2 + back-2 selection, encoded as the joint 90-way action."""

    lead2_idx: tuple[int, int]
    back2_idx: tuple[int, int]
    action90_id: int

    def __post_init__(self) -> None:
        if self.lead2_idx[0] >= self.lead2_idx[1]:
            raise ValueError(f"lead2_idx must be sorted ascending, got {self.lead2_idx}")
        if self.back2_idx[0] >= self.back2_idx[1]:
            raise ValueError(f"back2_idx must be sorted ascending, got {self.back2_idx}")
        lead_set = set(self.lead2_idx)
        back_set = set(self.back2_idx)
        if lead_set & back_set:
            raise ValueError(f"lead and back indices must not overlap: {self.lead2_idx}, {self.back2_idx}")
        all_idx = lead_set | back_set
        if any(i < 0 or i > 5 for i in all_idx):
            raise ValueError(f"All indices must be in 0..5, got {all_idx}")
        if not 0 <= self.action90_id < 90:
            raise ValueError(f"action90_id must be in [0, 90), got {self.action90_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead2_idx": list(self.lead2_idx),
            "back2_idx": list(self.back2_idx),
            "action90_id": self.action90_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Label:
        return cls(
            lead2_idx=tuple(d["lead2_idx"]),
            back2_idx=tuple(d["back2_idx"]),
            action90_id=d["action90_id"],
        )


@dataclass
class LabelQuality:
    """Tracks whether bring-4 is fully observed."""

    bring4_observed: bool
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"bring4_observed": self.bring4_observed, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelQuality:
        return cls(bring4_observed=d["bring4_observed"], notes=d.get("notes"))


@dataclass
class MatchExample:
    """One directed example (player-A perspective) of a team preview decision.

    Each raw game produces two MatchExamples by swapping sides.
    """

    example_id: str
    match_group_id: str
    battle_id: str
    team_a: TeamSheet
    team_b: TeamSheet
    label: Label
    label_quality: LabelQuality
    format_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    split_keys: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "example_id": self.example_id,
            "match_group_id": self.match_group_id,
            "battle_id": self.battle_id,
            "team_a": self.team_a.to_dict(),
            "team_b": self.team_b.to_dict(),
            "label": self.label.to_dict(),
            "label_quality": self.label_quality.to_dict(),
            "format_id": self.format_id,
            "metadata": self.metadata,
        }
        if self.split_keys is not None:
            d["split_keys"] = self.split_keys
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchExample:
        return cls(
            example_id=d["example_id"],
            match_group_id=d["match_group_id"],
            battle_id=d["battle_id"],
            team_a=TeamSheet.from_dict(d["team_a"]),
            team_b=TeamSheet.from_dict(d["team_b"]),
            label=Label.from_dict(d["label"]),
            label_quality=LabelQuality.from_dict(d["label_quality"]),
            format_id=d["format_id"],
            metadata=d.get("metadata", {}),
            split_keys=d.get("split_keys"),
        )
