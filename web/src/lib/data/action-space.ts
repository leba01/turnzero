/**
 * Action space for the 90-way team-preview prediction.
 *
 * Each action selects 2 leads from 6 mons and 2 backs from the remaining 4.
 * C(6,2)=15 lead pairs x C(4,2)=6 back pairs = 90 actions.
 *
 * Pre-computed masks enable vectorized marginal probability computation.
 */

/** A single action: which 2 indices lead, which 2 stay in back. */
export interface ActionEntry {
  lead: [number, number];
  back: [number, number];
}

/** Generate all k-combinations of integers [0, n). */
function combinations(n: number, k: number): number[][] {
  const result: number[][] = [];
  const combo = new Array(k);

  function recurse(start: number, depth: number): void {
    if (depth === k) {
      result.push([...combo]);
      return;
    }
    for (let i = start; i < n; i++) {
      combo[depth] = i;
      recurse(i + 1, depth + 1);
    }
  }

  recurse(0, 0);
  return result;
}

/** All 15 lead pairs (combinations of 6 choose 2), in lexicographic order. */
export const LEAD_PAIRS: [number, number][] = combinations(6, 2).map(
  ([a, b]) => [a, b] as [number, number],
);

/**
 * ACTION_TABLE[a] gives the lead pair and back pair for action index a.
 * Enumerated as: for each of the 15 lead pairs, iterate over the 6 back pairs
 * formed from the remaining 4 mons.
 */
export const ACTION_TABLE: ActionEntry[] = (() => {
  const table: ActionEntry[] = [];
  for (const [l0, l1] of LEAD_PAIRS) {
    const remaining = [0, 1, 2, 3, 4, 5].filter((i) => i !== l0 && i !== l1);
    const backPairs = combinations(remaining.length, 2);
    for (const [bi, bj] of backPairs) {
      table.push({
        lead: [l0, l1],
        back: [remaining[bi], remaining[bj]],
      });
    }
  }
  return table;
})();

/**
 * LEAD_MASK — flat Float64Array of shape (90, 6), row-major.
 * LEAD_MASK[a * 6 + i] = 1 if mon i is in the lead pair of action a.
 */
export const LEAD_MASK: Float64Array = (() => {
  const mask = new Float64Array(90 * 6);
  for (let a = 0; a < 90; a++) {
    const { lead } = ACTION_TABLE[a];
    mask[a * 6 + lead[0]] = 1;
    mask[a * 6 + lead[1]] = 1;
  }
  return mask;
})();

/**
 * BRING_MASK — flat Float64Array of shape (90, 6), row-major.
 * BRING_MASK[a * 6 + i] = 1 if mon i is in the bring-4 (lead + back) of action a.
 */
export const BRING_MASK: Float64Array = (() => {
  const mask = new Float64Array(90 * 6);
  for (let a = 0; a < 90; a++) {
    const { lead, back } = ACTION_TABLE[a];
    mask[a * 6 + lead[0]] = 1;
    mask[a * 6 + lead[1]] = 1;
    mask[a * 6 + back[0]] = 1;
    mask[a * 6 + back[1]] = 1;
  }
  return mask;
})();

/**
 * MARGIN_MATRIX — flat Float64Array of shape (90, 15), row-major.
 * MARGIN_MATRIX[a * 15 + j] = 1 if action a's lead pair is the j-th lead pair
 * (from the 15 combinations of 6 choose 2, lexicographic order).
 */
export const MARGIN_MATRIX: Float64Array = (() => {
  const mat = new Float64Array(90 * 15);
  for (let a = 0; a < 90; a++) {
    const { lead } = ACTION_TABLE[a];
    // Find which of the 15 lead pairs this action belongs to.
    // Since we enumerate 6 back pairs per lead pair, the lead pair index is floor(a / 6).
    const j = Math.floor(a / 6);
    // Verify consistency (the lead pair at index j should match).
    if (LEAD_PAIRS[j][0] !== lead[0] || LEAD_PAIRS[j][1] !== lead[1]) {
      throw new Error(
        `Action table ordering mismatch at a=${a}: expected lead pair ${LEAD_PAIRS[j]}, got ${lead}`,
      );
    }
    mat[a * 15 + j] = 1;
  }
  return mat;
})();
