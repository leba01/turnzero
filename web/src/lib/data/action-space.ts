// 90-way action space: C(6,2)=15 lead pairs × C(4,2)=6 back pairs.
// Pre-computed masks enable vectorized marginal probability computation.

/** A single action: which 2 indices lead, which 2 stay in back. */
export interface ActionEntry {
  lead: [number, number];
  back: [number, number];
}

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

/** ACTION_TABLE[a]: lead + back pair for action a. 15 lead pairs × 6 back pairs = 90 actions. */
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

/** LEAD_MASK[a * 6 + i] = 1 if mon i leads in action a. Shape (90, 6) row-major. */
export const LEAD_MASK: Float64Array = (() => {
  const mask = new Float64Array(90 * 6);
  for (let a = 0; a < 90; a++) {
    const { lead } = ACTION_TABLE[a];
    mask[a * 6 + lead[0]] = 1;
    mask[a * 6 + lead[1]] = 1;
  }
  return mask;
})();

/** BRING_MASK[a * 6 + i] = 1 if mon i is in the bring-4 of action a. Shape (90, 6) row-major. */
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

/** MARGIN_MATRIX[a * 15 + j] = 1 if action a uses the j-th lead pair. Shape (90, 15) row-major. */
export const MARGIN_MATRIX: Float64Array = (() => {
  const mat = new Float64Array(90 * 15);
  for (let a = 0; a < 90; a++) {
    const { lead } = ACTION_TABLE[a];
    const j = Math.floor(a / 6); // 6 back pairs per lead pair
    if (LEAD_PAIRS[j][0] !== lead[0] || LEAD_PAIRS[j][1] !== lead[1]) {
      throw new Error(
        `Action table ordering mismatch at a=${a}: expected lead pair ${LEAD_PAIRS[j]}, got ${lead}`,
      );
    }
    mat[a * 15 + j] = 1;
  }
  return mat;
})();
