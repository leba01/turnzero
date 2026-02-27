/**
 * Reverse role lexicon for opponent team annotation.
 *
 * Maps move/item/ability names to tactical role tags (e.g., "Fake Out" -> ["fake_out"]).
 * Used to surface opponent cues in the prediction UI.
 */

import type { TeamSheet, OpponentCue } from "@/types/pokemon";

/** Reverse lexicon: token name -> list of role tags. */
export type ReverseLexicon = Record<string, string[]>;

/** Load the reverse lexicon from the static JSON file. */
export async function loadLexicon(): Promise<ReverseLexicon> {
  const response = await fetch("/data/lexicon.json");
  if (!response.ok) {
    throw new Error(`Failed to load lexicon: ${response.status}`);
  }
  return response.json() as Promise<ReverseLexicon>;
}

/**
 * Annotate each Pokemon on a team with its tactical roles.
 *
 * Checks each mon's moves, item, and ability against the reverse lexicon.
 * Returns one OpponentCue per mon, with roles sorted alphabetically and deduplicated.
 */
export function annotateTeam(
  lexicon: ReverseLexicon,
  team: TeamSheet,
): OpponentCue[] {
  return team.pokemon.map((mon) => {
    const roleSet = new Set<string>();

    // Check all 4 moves.
    for (const move of mon.moves) {
      const roles = lexicon[move];
      if (roles) {
        for (const r of roles) roleSet.add(r);
      }
    }

    // Check item.
    const itemRoles = lexicon[mon.item];
    if (itemRoles) {
      for (const r of itemRoles) roleSet.add(r);
    }

    // Check ability.
    const abilityRoles = lexicon[mon.ability];
    if (abilityRoles) {
      for (const r of abilityRoles) roleSet.add(r);
    }

    return {
      species: mon.species,
      roles: [...roleSet].sort(),
    };
  });
}
