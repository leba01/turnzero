/**
 * Vocabulary encoding for the transformer model.
 *
 * Maps species, item, ability, tera_type, and move tokens to integer indices.
 * Each team of 6 Pokemon is encoded as an Int32Array of shape (6, 8) = 48 values.
 * Field order per mon: [species, item, ability, tera_type, move0, move1, move2, move3].
 */

import type { TeamSheet } from "@/types/pokemon";

/** A vocabulary mapping: field type -> token -> integer index. */
export type VocabMap = Record<string, Record<string, number>>;

/** The 8 field types per Pokemon, in encoding order. */
const FIELD_TYPES = [
  "species",
  "item",
  "ability",
  "tera_type",
  "move",
  "move",
  "move",
  "move",
] as const;

/** Load the vocabulary from the static JSON file. */
export async function loadVocab(): Promise<VocabMap> {
  const response = await fetch("/data/vocab.json");
  if (!response.ok) {
    throw new Error(`Failed to load vocab: ${response.status}`);
  }
  return response.json() as Promise<VocabMap>;
}

/** Encode a single token to its integer index. Returns 0 (UNK) for unknown tokens. */
export function encode(
  vocab: VocabMap,
  fieldType: string,
  token: string,
): number {
  const table = vocab[fieldType];
  if (!table) return 0;
  const idx = table[token];
  return idx !== undefined ? idx : 0;
}

/**
 * Encode a full team of 6 Pokemon into a flat Int32Array of 48 values.
 * Layout: [mon0_species, mon0_item, mon0_ability, mon0_tera, mon0_move0..3,
 *          mon1_species, ..., mon5_move3]
 */
export function encodeTeam(vocab: VocabMap, team: TeamSheet): Int32Array {
  const encoded = new Int32Array(6 * 8);
  for (let i = 0; i < 6; i++) {
    const mon = team.pokemon[i];
    const base = i * 8;
    encoded[base + 0] = encode(vocab, "species", mon.species);
    encoded[base + 1] = encode(vocab, "item", mon.item);
    encoded[base + 2] = encode(vocab, "ability", mon.ability);
    encoded[base + 3] = encode(vocab, "tera_type", mon.tera_type);
    encoded[base + 4] = encode(vocab, "move", mon.moves[0]);
    encoded[base + 5] = encode(vocab, "move", mon.moves[1]);
    encoded[base + 6] = encode(vocab, "move", mon.moves[2]);
    encoded[base + 7] = encode(vocab, "move", mon.moves[3]);
  }
  return encoded;
}
