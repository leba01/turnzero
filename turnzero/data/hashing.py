"""Shared hashing utilities for team and Pokemon identifiers."""

from __future__ import annotations

import hashlib


def compute_hash(text: str) -> str:
    """Truncated sha256 hex digest (16 chars = 64 bits)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def species_key(pokemon: list) -> str:
    """Order-invariant hash of species set."""
    return compute_hash("|".join(sorted(p.species for p in pokemon)))
