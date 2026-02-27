/**
 * Numerical utilities for ensemble inference.
 *
 * All functions operate on typed arrays for performance in the hot path.
 */

const EPS = 1e-12;

/**
 * Numerically stable softmax over a Float32Array of logits.
 * Returns a new Float32Array of the same length with values summing to 1.
 */
export function softmax(logits: Float32Array): Float32Array {
  const n = logits.length;
  const result = new Float32Array(n);

  // Find max for numerical stability.
  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }

  // Compute exp(x - max) and sum.
  let sum = 0;
  for (let i = 0; i < n; i++) {
    result[i] = Math.exp(logits[i] - maxVal);
    sum += result[i];
  }

  // Normalize.
  const invSum = 1 / sum;
  for (let i = 0; i < n; i++) {
    result[i] *= invSum;
  }

  return result;
}

/**
 * Shannon entropy H(p) = -sum(p * log(p + eps)).
 * Returns a non-negative scalar in nats.
 */
export function shannonEntropy(probs: Float32Array): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) {
      h -= p * Math.log(p + EPS);
    }
  }
  return h;
}

/**
 * KL divergence D_KL(p || q) = sum(p * log((p + eps) / (q + eps))).
 * p and q must have the same length.
 */
export function klDivergence(p: Float32Array, q: Float32Array): number {
  if (p.length !== q.length) {
    throw new Error(
      `KL divergence: length mismatch (${p.length} vs ${q.length})`,
    );
  }
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    kl += p[i] * Math.log((p[i] + EPS) / (q[i] + EPS));
  }
  return kl;
}

/**
 * Element-wise average of multiple probability vectors.
 * All input arrays must have the same length.
 */
export function averageProbs(memberProbs: Float32Array[]): Float32Array {
  if (memberProbs.length === 0) {
    throw new Error("averageProbs: empty input");
  }
  const n = memberProbs[0].length;
  const m = memberProbs.length;
  const avg = new Float32Array(n);

  for (let j = 0; j < m; j++) {
    const probs = memberProbs[j];
    for (let i = 0; i < n; i++) {
      avg[i] += probs[i];
    }
  }

  const invM = 1 / m;
  for (let i = 0; i < n; i++) {
    avg[i] *= invM;
  }

  return avg;
}
