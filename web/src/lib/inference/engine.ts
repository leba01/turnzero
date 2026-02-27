/**
 * ONNX Runtime inference engine for the TurnZero ensemble.
 *
 * Loads 5 transformer models, runs them in parallel via WASM backend,
 * and aggregates softmax probabilities for the 90-way action prediction.
 */

import * as ort from "onnxruntime-web";
import type {
  TeamSheet,
  PredictionResult,
  ActionPlan,
  FeatureSensitivity,
} from "@/types/pokemon";
import type { VocabMap } from "@/lib/data/vocab";
import { encodeTeam } from "@/lib/data/vocab";
import type { ReverseLexicon } from "@/lib/data/lexicon";
import { annotateTeam } from "@/lib/data/lexicon";
import { ACTION_TABLE } from "@/lib/data/action-space";
import { softmax, shannonEntropy, klDivergence, averageProbs } from "./math";
import { computeMarginals } from "./marginals";

/** Number of ensemble members. */
const N_MODELS = 5;

/** Abstention threshold on max(p_bar). */
const ABSTAIN_TAU = 0.04;

/** Model file paths. */
const MODEL_PATHS = Array.from(
  { length: N_MODELS },
  (_, i) => `/models/ensemble_${String(i + 1).padStart(3, "0")}.onnx`,
);

/**
 * Field groups for sensitivity analysis.
 * Maps a human-readable name to the column indices (within the 8-column encoding)
 * that should be masked to UNK (0).
 */
const SENSITIVITY_GROUPS: Record<keyof FeatureSensitivity, number[]> = {
  species: [0],
  items: [1],
  ability: [2],
  tera: [3],
  moves: [4, 5, 6, 7],
};

export class InferenceEngine {
  private sessions: ort.InferenceSession[] = [];

  /** Whether all models are loaded and ready. */
  get isReady(): boolean {
    return this.sessions.length === N_MODELS;
  }

  /**
   * Load all 5 ONNX model sessions with WASM backend.
   * Calls onProgress after each model loads.
   */
  async load(
    onProgress: (loaded: number, total: number) => void,
  ): Promise<void> {
    this.sessions = [];
    for (let i = 0; i < N_MODELS; i++) {
      const modelPath = MODEL_PATHS[i];
      const dataFileName = modelPath.split("/").pop() + ".data";
      // Fetch external data file as bytes — onnxruntime-web can't resolve
      // URL strings for external data in the browser WASM backend.
      const dataResponse = await fetch(modelPath + ".data");
      if (!dataResponse.ok) {
        throw new Error(`Failed to fetch ${modelPath}.data: ${dataResponse.status}`);
      }
      const dataBuffer = new Uint8Array(await dataResponse.arrayBuffer());
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        externalData: [
          {
            path: dataFileName,
            data: dataBuffer,
          },
        ],
      });
      this.sessions.push(session);
      onProgress(i + 1, N_MODELS);
    }
  }

  /**
   * Run all 5 models on encoded team tensors.
   * Returns per-model logits (1,90) and embeddings (1,128).
   */
  async predict(
    teamA: Int32Array,
    teamB: Int32Array,
  ): Promise<{ logits: Float32Array[]; embeddings: Float32Array[] }> {
    if (!this.isReady) {
      throw new Error("Models not loaded. Call load() first.");
    }

    const teamATensor = new ort.Tensor("int32", teamA, [1, 6, 8]);
    const teamBTensor = new ort.Tensor("int32", teamB, [1, 6, 8]);

    // Run sessions sequentially — the WASM backend shares a single thread pool
    // and cannot handle concurrent session.run() calls.
    const logits: Float32Array[] = [];
    const embeddings: Float32Array[] = [];
    for (const session of this.sessions) {
      const result = await session.run({ team_a: teamATensor, team_b: teamBTensor });
      logits.push(result.logits.data as Float32Array);
      embeddings.push(result.embedding.data as Float32Array);
    }

    return { logits, embeddings };
  }

  /**
   * Full prediction pipeline: encode, infer, aggregate, compute marginals.
   *
   * @param teamA - player's team (the one we predict leads for)
   * @param teamB - opponent's team
   * @param vocab - loaded vocabulary
   * @param lexicon - loaded reverse lexicon
   * @param temperature - temperature scaling factor (from temperature.json)
   * @returns PredictionResult with sensitivity and evidence initially null
   */
  async fullPredict(
    teamA: TeamSheet,
    teamB: TeamSheet,
    vocab: VocabMap,
    lexicon: ReverseLexicon,
    temperature: number,
  ): Promise<PredictionResult> {
    // 1. Encode both teams.
    const encodedA = encodeTeam(vocab, teamA);
    const encodedB = encodeTeam(vocab, teamB);

    // 2. Run all 5 models.
    const { logits } = await this.predict(encodedA, encodedB);

    // 3. Softmax each model's logits / T -> member probs.
    const memberProbs = logits.map((l) => {
      const scaled = new Float32Array(l.length);
      for (let i = 0; i < l.length; i++) {
        scaled[i] = l[i] / temperature;
      }
      return softmax(scaled);
    });

    // 4. Average probs -> p_bar (90,).
    const pBar = averageProbs(memberProbs);

    // 5. Ensemble entropy.
    const entropy = shannonEntropy(pBar);

    // 6. Mutual information = ensemble entropy - mean(member entropies).
    let memberEntropySum = 0;
    for (const mp of memberProbs) {
      memberEntropySum += shannonEntropy(mp);
    }
    const meanMemberEntropy = memberEntropySum / memberProbs.length;
    const mutualInformation = entropy - meanMemberEntropy;

    // 7. Confidence = max(p_bar).
    let confidence = 0;
    let topAction = 0;
    for (let i = 0; i < pBar.length; i++) {
      if (pBar[i] > confidence) {
        confidence = pBar[i];
        topAction = i;
      }
    }

    // 7b. Ensemble agreement: how many members' argmax matches the ensemble's top-1.
    let ensembleAgreement = 0;
    for (const mp of memberProbs) {
      let memberMax = 0;
      let memberArgmax = 0;
      for (let i = 0; i < mp.length; i++) {
        if (mp[i] > memberMax) {
          memberMax = mp[i];
          memberArgmax = i;
        }
      }
      if (memberArgmax === topAction) ensembleAgreement++;
    }

    // 8. Abstention decision.
    const abstain = confidence < ABSTAIN_TAU;

    // 9. Top-5 plans from ACTION_TABLE.
    const indexed = Array.from(pBar).map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);
    const topPlans: ActionPlan[] = indexed.slice(0, 5).map((entry, rank) => {
      const action = ACTION_TABLE[entry.i];
      // Bench = the 2 mons not in lead or back.
      const used = new Set([
        action.lead[0],
        action.lead[1],
        action.back[0],
        action.back[1],
      ]);
      const bench: [number, number] = [0, 0];
      let bi = 0;
      for (let m = 0; m < 6; m++) {
        if (!used.has(m)) {
          bench[bi++] = m;
        }
      }
      return {
        rank: rank + 1,
        action90_id: entry.i,
        probability: entry.p,
        lead: action.lead,
        back: action.back,
        bench,
      };
    });

    // 10. Compute marginals.
    const marginals = computeMarginals(pBar);

    // 11. Annotate opponent team with lexicon.
    const opponentCues = annotateTeam(lexicon, teamB);

    // 12. Return PredictionResult (sensitivity and evidence filled later).
    return {
      top_plans: topPlans,
      confidence,
      entropy,
      mutual_information: mutualInformation,
      ensemble_agreement: ensembleAgreement,
      abstain,
      marginals,
      opponent_cues: opponentCues,
      sensitivity: null,
      evidence: null,
    };
  }

  /**
   * Compute feature sensitivity via KL divergence from ablated inputs.
   *
   * For each field group (species, items, ability, tera, moves), masks the
   * corresponding columns in teamB to UNK (0), runs the ensemble, and
   * computes KL(baseline || masked).
   *
   * @param teamA - encoded player team (48 Int32)
   * @param teamB - encoded opponent team (48 Int32)
   * @param temperature - temperature scaling factor
   * @returns FeatureSensitivity with KL divergence per field group
   */
  async computeSensitivity(
    teamA: Int32Array,
    teamB: Int32Array,
    temperature: number,
  ): Promise<FeatureSensitivity> {
    // Baseline ensemble probs.
    const baselineResult = await this.predict(teamA, teamB);
    const baselineProbs = averageProbs(
      baselineResult.logits.map((l) => {
        const scaled = new Float32Array(l.length);
        for (let i = 0; i < l.length; i++) scaled[i] = l[i] / temperature;
        return softmax(scaled);
      }),
    );

    const sensitivity: Record<string, number> = {};

    for (const [groupName, cols] of Object.entries(SENSITIVITY_GROUPS)) {
      // Clone teamB and mask the specified columns to 0 (UNK).
      const maskedB = new Int32Array(teamB);
      for (let mon = 0; mon < 6; mon++) {
        for (const col of cols) {
          maskedB[mon * 8 + col] = 0;
        }
      }

      // Run ensemble on masked input.
      const maskedResult = await this.predict(teamA, maskedB);
      const maskedProbs = averageProbs(
        maskedResult.logits.map((l) => {
          const scaled = new Float32Array(l.length);
          for (let i = 0; i < l.length; i++) scaled[i] = l[i] / temperature;
          return softmax(scaled);
        }),
      );

      // KL(baseline || masked).
      sensitivity[groupName] = klDivergence(baselineProbs, maskedProbs);
    }

    return sensitivity as unknown as FeatureSensitivity;
  }

  /**
   * Get the embedding from the first model for retrieval queries.
   * Returns a Float32Array of length 128.
   */
  async getEmbedding(
    teamA: Int32Array,
    teamB: Int32Array,
  ): Promise<Float32Array> {
    if (!this.isReady) {
      throw new Error("Models not loaded. Call load() first.");
    }

    const teamATensor = new ort.Tensor("int32", teamA, [1, 6, 8]);
    const teamBTensor = new ort.Tensor("int32", teamB, [1, 6, 8]);

    const result = await this.sessions[0].run({
      team_a: teamATensor,
      team_b: teamBTensor,
    });

    return result.embedding.data as Float32Array;
  }
}
