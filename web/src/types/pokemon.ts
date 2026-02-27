/** Core domain types for TurnZero web app. */

export interface Pokemon {
  species: string;
  item: string;
  ability: string;
  tera_type: string;
  moves: [string, string, string, string];
}

export interface TeamSheet {
  pokemon: [Pokemon, Pokemon, Pokemon, Pokemon, Pokemon, Pokemon];
}

/** 90-way action: which 2 to lead, which 2 to keep in back */
export interface ActionPlan {
  rank: number;
  action90_id: number;
  probability: number;
  lead: [number, number]; // indices into team_a
  back: [number, number]; // indices into team_a
  bench: [number, number]; // remaining 2
}

export interface Marginals {
  lead_probs: number[]; // length 6
  bring_probs: number[]; // length 6
  lead_pair_probs: number[]; // length 15
}

export interface OpponentCue {
  species: string;
  roles: string[];
}

export interface FeatureSensitivity {
  species: number;
  items: number;
  ability: number;
  tera: number;
  moves: number;
}

export interface RetrievalNeighbor {
  similarity: number;
  species_a: string[];
  species_b: string[];
  action90_id: number;
  lead2_idx: number[];
  back2_idx: number[];
}

export interface RetrievalEvidence {
  n_neighbors: number;
  mean_similarity: number;
  lead_pair_freq: { pair: string; count: number; fraction: number }[];
  mon_bring_freq: { species: string; count: number; fraction: number }[];
}

export interface PredictionResult {
  top_plans: ActionPlan[];
  confidence: number;
  entropy: number;
  mutual_information: number;
  abstain: boolean;
  marginals: Marginals;
  opponent_cues: OpponentCue[];
  sensitivity: FeatureSensitivity | null; // null while loading
  evidence: RetrievalEvidence | null; // null while loading or unavailable
}

/** Model loading state */
export type ModelLoadState =
  | { status: "idle" }
  | { status: "loading"; loaded: number; total: number }
  | { status: "ready" }
  | { status: "error"; message: string };
