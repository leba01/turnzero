// ONNX Runtime inference engine — loads 5 transformer models via WASM,
// aggregates softmax probabilities for the 90-way action prediction.

import * as ort from "onnxruntime-web";

// Suppress harmless "Unknown CPU vendor" warning in WSL2/virtualized environments.
ort.env.logSeverityLevel = 3;

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
  items: [1],
  ability: [2],
  tera: [3],
  moves: [4, 5, 6, 7],
};

export class InferenceEngine {
  private sessions: ort.InferenceSession[] = [];
  private static _wasmConfigured = false;

  /** Whether all models are loaded and ready. */
  get isReady(): boolean {
    return this.sessions.length === N_MODELS;
  }

  async load(
    onProgress: (loaded: number, total: number) => void,
  ): Promise<void> {
    // Single thread per session keeps 5 sessions within WASM memory limits.
    if (!InferenceEngine._wasmConfigured) {
      ort.env.wasm.numThreads = 1;
      InferenceEngine._wasmConfigured = true;
    }
    this.sessions = [];
    for (let i = 0; i < N_MODELS; i++) {
      const modelPath = MODEL_PATHS[i];
      const dataFileName = modelPath.split("/").pop() + ".data";
      // Fetch .data file as bytes — onnxruntime-web can't resolve URL strings for external data.
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

  async predict(
    teamA: Int32Array,
    teamB: Int32Array,
  ): Promise<{ logits: Float32Array[]; embeddings: Float32Array[] }> {
    if (!this.isReady) {
      throw new Error("Models not loaded. Call load() first.");
    }

    const teamATensor = new ort.Tensor("int32", teamA, [1, 6, 8]);
    const teamBTensor = new ort.Tensor("int32", teamB, [1, 6, 8]);

    // Sequential — WASM backend can't handle concurrent session.run() calls.
    const logits: Float32Array[] = [];
    const embeddings: Float32Array[] = [];
    for (const session of this.sessions) {
      const result = await session.run({ team_a: teamATensor, team_b: teamBTensor });
      // Copy before disposing — tensors are views into WASM heap.
      logits.push(new Float32Array(result.logits.data as Float32Array));
      embeddings.push(new Float32Array(result.embedding.data as Float32Array));
      result.logits.dispose();
      result.embedding.dispose();
    }

    teamATensor.dispose();
    teamBTensor.dispose();

    return { logits, embeddings };
  }

  async fullPredict(
    teamA: TeamSheet,
    teamB: TeamSheet,
    vocab: VocabMap,
    lexicon: ReverseLexicon,
    temperature: number,
  ): Promise<PredictionResult> {
    const encodedA = encodeTeam(vocab, teamA);
    const encodedB = encodeTeam(vocab, teamB);
    const { logits } = await this.predict(encodedA, encodedB);

    const memberProbs = logits.map((l) => {
      const scaled = new Float32Array(l.length);
      for (let i = 0; i < l.length; i++) {
        scaled[i] = l[i] / temperature;
      }
      return softmax(scaled);
    });

    const pBar = averageProbs(memberProbs);
    const entropy = shannonEntropy(pBar);

    // Mutual information = ensemble entropy - mean(member entropies).
    let memberEntropySum = 0;
    for (const mp of memberProbs) {
      memberEntropySum += shannonEntropy(mp);
    }
    const meanMemberEntropy = memberEntropySum / memberProbs.length;
    const mutualInformation = entropy - meanMemberEntropy;

    let confidence = 0;
    let topAction = 0;
    for (let i = 0; i < pBar.length; i++) {
      if (pBar[i] > confidence) {
        confidence = pBar[i];
        topAction = i;
      }
    }

    // How many members' argmax matches the ensemble's top-1.
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

    const abstain = confidence < ABSTAIN_TAU;

    const indexed = Array.from(pBar).map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);
    const topPlans: ActionPlan[] = indexed.slice(0, 5).map((entry, rank) => {
      const action = ACTION_TABLE[entry.i];
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

    const marginals = computeMarginals(pBar);
    const opponentCues = annotateTeam(lexicon, teamB);

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

  /** KL divergence per field group when opponent features are ablated to UNK. */
  async computeSensitivity(
    teamA: Int32Array,
    teamB: Int32Array,
    temperature: number,
  ): Promise<FeatureSensitivity> {
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
      const maskedB = new Int32Array(teamB);
      for (let mon = 0; mon < 6; mon++) {
        for (const col of cols) maskedB[mon * 8 + col] = 0;
      }

      const maskedResult = await this.predict(teamA, maskedB);
      const maskedProbs = averageProbs(
        maskedResult.logits.map((l) => {
          const scaled = new Float32Array(l.length);
          for (let i = 0; i < l.length; i++) scaled[i] = l[i] / temperature;
          return softmax(scaled);
        }),
      );

      sensitivity[groupName] = klDivergence(baselineProbs, maskedProbs);
    }

    return sensitivity as unknown as FeatureSensitivity;
  }

  async dispose(): Promise<void> {
    for (const session of this.sessions) {
      await session.release();
    }
    this.sessions = [];
  }

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

    const embedding = new Float32Array(result.embedding.data as Float32Array);
    result.embedding.dispose();
    teamATensor.dispose();
    teamBTensor.dispose();

    return embedding;
  }
}
