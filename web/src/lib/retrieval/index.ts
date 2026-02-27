/**
 * Retrieval engine for TurnZero.
 *
 * Loads 246,762 training-set embeddings (128-dim, float16) and performs
 * brute-force cosine-similarity search to find the most similar historical
 * matchups.  Results are aggregated into frequency tables that the UI
 * renders as "evidence" for the model's prediction.
 */

import type {
  RetrievalNeighbor,
  RetrievalEvidence,
} from "@/types/pokemon";
import { decodeFloat16Array } from "./float16";

// ── Per-example metadata stored alongside the embedding index ──────────

interface RetrievalMeta {
  species_a: string[];
  species_b: string[];
  action90_id: number;
  lead2_idx: number[];
  back2_idx: number[];
}

// ── RetrievalIndex ─────────────────────────────────────────────────────

export class RetrievalIndex {
  /** Flat row-major Float32Array, shape (rows, cols). L2-normalized. */
  readonly embeddings: Float32Array;
  /** Per-row metadata (species, action, lead/back indices). */
  readonly metadata: RetrievalMeta[];
  /** Number of training examples (rows). */
  readonly rows: number;
  /** Embedding dimension (columns, always 128). */
  readonly cols: number;

  private constructor(
    embeddings: Float32Array,
    metadata: RetrievalMeta[],
    rows: number,
    cols: number,
  ) {
    this.embeddings = embeddings;
    this.metadata = metadata;
    this.rows = rows;
    this.cols = cols;
  }

  // ── Loading ────────────────────────────────────────────────────────

  /**
   * Fetch the three data files and build the in-memory index.
   *
   * @param onProgress Optional callback reporting download progress [0, 1].
   *                   Only tracks the large embeddings file.
   * @returns The loaded index, or `null` on failure.
   */
  static async load(
    onProgress?: (pct: number) => void,
  ): Promise<RetrievalIndex | null> {
    try {
      // 1. Shape (tiny JSON — fire first so we know dimensions)
      const shapeRes = await fetch("/data/retrieval_shape.json");
      if (!shapeRes.ok) return null;
      const shape: { rows: number; cols: number } = await shapeRes.json();
      const { rows, cols } = shape;

      // 2. Embeddings binary — stream to report progress
      const embBuffer = await fetchWithProgress(
        "/data/retrieval_embeddings.bin",
        rows * cols * 2, // 2 bytes per float16
        onProgress,
      );
      if (!embBuffer) return null;

      // 3. Decode float16 → float32
      const totalElements = rows * cols;
      const embeddings = decodeFloat16Array(embBuffer, totalElements);

      // 4. L2-normalize each row (idempotent if already unit-norm)
      normalizeRows(embeddings, rows, cols);

      // 5. Metadata
      const metaRes = await fetch("/data/retrieval_meta.json");
      if (!metaRes.ok) return null;
      const metadata: RetrievalMeta[] = await metaRes.json();

      if (metadata.length !== rows) {
        console.error(
          `[RetrievalIndex] metadata length (${metadata.length}) != rows (${rows})`,
        );
        return null;
      }

      return new RetrievalIndex(embeddings, metadata, rows, cols);
    } catch (err) {
      console.error("[RetrievalIndex] load failed:", err);
      return null;
    }
  }

  // ── Query ──────────────────────────────────────────────────────────

  /**
   * Find the `k` nearest neighbors by cosine similarity.
   *
   * Because every stored embedding and the query are L2-normalized,
   * cosine similarity == dot product.
   */
  query(embedding: Float32Array, k: number = 10): RetrievalNeighbor[] {
    const { embeddings, metadata, rows, cols } = this;

    // Normalize query to unit length
    const q = new Float32Array(embedding);
    normalizeVector(q);

    // Compute all dot products (cosine similarities)
    const sims = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      let dot = 0;
      const offset = i * cols;
      for (let j = 0; j < cols; j++) {
        dot += q[j] * embeddings[offset + j];
      }
      sims[i] = dot;
    }

    // Find top-k indices by descending similarity.
    // For small k relative to N, a partial selection would be optimal,
    // but 246K sort is fast enough (~30ms) and simpler.
    const indices = new Uint32Array(rows);
    for (let i = 0; i < rows; i++) indices[i] = i;
    indices.sort((a, b) => sims[b] - sims[a]);

    const topK = Math.min(k, rows);
    const neighbors: RetrievalNeighbor[] = new Array(topK);

    for (let r = 0; r < topK; r++) {
      const idx = indices[r];
      const m = metadata[idx];
      neighbors[r] = {
        similarity: sims[idx],
        species_a: m.species_a,
        species_b: m.species_b,
        action90_id: m.action90_id,
        lead2_idx: m.lead2_idx,
        back2_idx: m.back2_idx,
      };
    }

    return neighbors;
  }

  // ── Evidence summary ───────────────────────────────────────────────

  /**
   * Aggregate a list of neighbors into frequency tables for the UI.
   */
  evidenceSummary(neighbors: RetrievalNeighbor[]): RetrievalEvidence {
    const n = neighbors.length;
    if (n === 0) {
      return {
        n_neighbors: 0,
        mean_similarity: 0,
        lead_pair_freq: [],
        mon_bring_freq: [],
      };
    }

    // Mean similarity
    let simSum = 0;
    for (let i = 0; i < n; i++) simSum += neighbors[i].similarity;

    // Lead-pair frequency: "SpeciesA+SpeciesB" canonical key
    const leadCounts = new Map<string, number>();
    // Per-mon bring frequency (bring-4 = lead2 + back2)
    const monCounts = new Map<string, number>();

    for (const nb of neighbors) {
      // Lead pair — sort alphabetically for canonical key
      const leadSpecies = nb.lead2_idx.map((idx) => nb.species_a[idx]);
      leadSpecies.sort();
      const pairKey = leadSpecies.join("+");
      leadCounts.set(pairKey, (leadCounts.get(pairKey) ?? 0) + 1);

      // Bring-4 individual mons
      const bringIndices = [...nb.lead2_idx, ...nb.back2_idx];
      for (const idx of bringIndices) {
        const sp = nb.species_a[idx];
        monCounts.set(sp, (monCounts.get(sp) ?? 0) + 1);
      }
    }

    // Sort descending by count, build output arrays
    const lead_pair_freq = [...leadCounts.entries()]
      .map(([pair, count]) => ({ pair, count, fraction: count / n }))
      .sort((a, b) => b.count - a.count);

    const mon_bring_freq = [...monCounts.entries()]
      .map(([species, count]) => ({ species, count, fraction: count / n }))
      .sort((a, b) => b.count - a.count);

    return {
      n_neighbors: n,
      mean_similarity: simSum / n,
      lead_pair_freq,
      mon_bring_freq,
    };
  }
}

// ── Helpers ──────────────────────────────────────────────────────────

/**
 * L2-normalize a single vector in-place.
 * If the vector is zero, it is left as-is.
 */
function normalizeVector(v: Float32Array): void {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  if (norm === 0) return;
  norm = 1 / Math.sqrt(norm);
  for (let i = 0; i < v.length; i++) v[i] *= norm;
}

/**
 * L2-normalize every row of a flat (rows x cols) Float32Array in-place.
 */
function normalizeRows(
  data: Float32Array,
  rows: number,
  cols: number,
): void {
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    let norm = 0;
    for (let j = 0; j < cols; j++) {
      const v = data[offset + j];
      norm += v * v;
    }
    if (norm === 0) continue;
    const inv = 1 / Math.sqrt(norm);
    for (let j = 0; j < cols; j++) {
      data[offset + j] *= inv;
    }
  }
}

/**
 * Fetch a binary file with download-progress reporting.
 *
 * Uses the `ReadableStream` API when available so `onProgress` can fire
 * incrementally.  Falls back to a plain `arrayBuffer()` call (no progress)
 * in environments without streaming support.
 *
 * @returns The raw ArrayBuffer, or `null` on failure.
 */
async function fetchWithProgress(
  url: string,
  expectedBytes: number,
  onProgress?: (pct: number) => void,
): Promise<ArrayBuffer | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;

    // If we have a body stream and a progress callback, stream it
    if (onProgress && res.body) {
      const contentLength =
        Number(res.headers.get("Content-Length")) || expectedBytes;
      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        onProgress(Math.min(received / contentLength, 1));
      }

      // Assemble into a single ArrayBuffer
      const combined = new Uint8Array(received);
      let pos = 0;
      for (const chunk of chunks) {
        combined.set(chunk, pos);
        pos += chunk.byteLength;
      }
      return combined.buffer;
    }

    // No progress callback or no streaming — simple path
    return await res.arrayBuffer();
  } catch (err) {
    console.error(`[fetchWithProgress] ${url}:`, err);
    return null;
  }
}
