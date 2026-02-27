/**
 * Marginal probability computation from the 90-way action distribution.
 *
 * Uses pre-computed masks from action-space.ts for vectorized dot products.
 */

import type { Marginals } from "@/types/pokemon";
import { LEAD_MASK, BRING_MASK, MARGIN_MATRIX } from "@/lib/data/action-space";

/**
 * Compute per-mon lead, bring, and per-pair lead probabilities
 * from the 90-way action probability distribution.
 *
 * @param probs90 - probability vector of length 90
 * @returns Marginals with lead_probs (6), bring_probs (6), lead_pair_probs (15)
 */
export function computeMarginals(probs90: Float32Array): Marginals {
  // lead_probs[i] = sum over a of probs90[a] * LEAD_MASK[a*6+i]
  const leadProbs = new Array<number>(6);
  for (let i = 0; i < 6; i++) {
    let sum = 0;
    for (let a = 0; a < 90; a++) {
      sum += probs90[a] * LEAD_MASK[a * 6 + i];
    }
    leadProbs[i] = sum;
  }

  // bring_probs[i] = sum over a of probs90[a] * BRING_MASK[a*6+i]
  const bringProbs = new Array<number>(6);
  for (let i = 0; i < 6; i++) {
    let sum = 0;
    for (let a = 0; a < 90; a++) {
      sum += probs90[a] * BRING_MASK[a * 6 + i];
    }
    bringProbs[i] = sum;
  }

  // lead_pair_probs[j] = sum over a of probs90[a] * MARGIN_MATRIX[a*15+j]
  const leadPairProbs = new Array<number>(15);
  for (let j = 0; j < 15; j++) {
    let sum = 0;
    for (let a = 0; a < 90; a++) {
      sum += probs90[a] * MARGIN_MATRIX[a * 15 + j];
    }
    leadPairProbs[j] = sum;
  }

  return {
    lead_probs: leadProbs,
    bring_probs: bringProbs,
    lead_pair_probs: leadPairProbs,
  };
}
