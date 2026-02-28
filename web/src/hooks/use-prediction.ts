'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  TeamSheet,
  PredictionResult,
  ModelLoadState,
  FeatureSensitivity,
  RetrievalEvidence,
} from '@/types/pokemon';
import { InferenceEngine } from '@/lib/inference/engine';
import { loadVocab, type VocabMap, encodeTeam } from '@/lib/data/vocab';
import { loadLexicon, type ReverseLexicon } from '@/lib/data/lexicon';
import { loadPokemonData, type PokemonData } from '@/lib/data/pokemon-data';
import { RetrievalIndex } from '@/lib/retrieval/index';

// Module-level singleton to survive React StrictMode double-mount.
// Without this, the second mount tries to init the ONNX WASM backend
// while the first is still running → "Session already started" error.
type InitResult = {
  engine: InferenceEngine;
  vocab: VocabMap;
  lexicon: ReverseLexicon;
  pokemonData: PokemonData;
  temperature: number;
};

let initPromise: Promise<InitResult> | null = null;
let currentEngine: InferenceEngine | null = null;

function getInitPromise(onProgress: (loaded: number, total: number) => void) {
  if (!initPromise) {
    initPromise = (async () => {
      const [vocab, lexicon, pd, tempData] = await Promise.all([
        loadVocab(),
        loadLexicon(),
        loadPokemonData(),
        fetch('/data/temperature.json').then((r) => r.json()),
      ]);

      const engine = new InferenceEngine();
      await engine.load((loaded, total) => {
        onProgress(loaded, total);
      });

      currentEngine = engine;
      return { engine, vocab, lexicon, pokemonData: pd, temperature: tempData.T };
    })();
  }
  return initPromise;
}

/** Release ONNX sessions and reset singleton. Called on HMR dispose. */
async function cleanup() {
  if (currentEngine) {
    await currentEngine.dispose();
    currentEngine = null;
  }
  initPromise = null;
}

// HMR cleanup — release WASM memory when module is replaced during dev.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const hmr = (import.meta as any).hot ?? (typeof module !== 'undefined' && (module as any).hot);
if (hmr) {
  hmr.dispose(() => { cleanup(); });
}

export function usePrediction(onReady?: () => void) {
  const [modelState, setModelState] = useState<ModelLoadState>({ status: 'idle' });
  const [pokemonData, setPokemonData] = useState<PokemonData | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);

  const engineRef = useRef<InferenceEngine | null>(null);
  const vocabRef = useRef<VocabMap | null>(null);
  const lexiconRef = useRef<ReverseLexicon | null>(null);
  const temperatureRef = useRef<number>(1.0);
  const retrievalRef = useRef<RetrievalIndex | null>(null);
  const retrievalLoadingRef = useRef(false);
  const predictionGenRef = useRef(0);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  // Load models + static data on mount
  useEffect(() => {
    let cancelled = false;

    setModelState({ status: 'loading', loaded: 0, total: 5 });
    getInitPromise((loaded, total) => {
      if (!cancelled) setModelState({ status: 'loading', loaded, total });
    })
      .then((resources) => {
        if (cancelled) return;
        engineRef.current = resources.engine;
        vocabRef.current = resources.vocab;
        lexiconRef.current = resources.lexicon;
        setPokemonData(resources.pokemonData);
        temperatureRef.current = resources.temperature;
        setModelState({ status: 'ready' });
        onReadyRef.current?.();
      })
      .catch((err) => {
        if (!cancelled) {
          initPromise = null; // Allow retry on next mount
          setModelState({
            status: 'error',
            message: err instanceof Error ? err.message : 'Failed to load models',
          });
        }
      });

    return () => { cancelled = true; };
  }, []);

  // Lazy load retrieval index (called after first prediction)
  const loadRetrieval = useCallback(async () => {
    if (retrievalRef.current || retrievalLoadingRef.current) return;
    retrievalLoadingRef.current = true;
    const index = await RetrievalIndex.load();
    retrievalRef.current = index;
    retrievalLoadingRef.current = false;
  }, []);

  const predict = useCallback(
    async (teamA: TeamSheet, teamB: TeamSheet) => {
      const engine = engineRef.current;
      const vocab = vocabRef.current;
      const lexicon = lexiconRef.current;
      if (!engine || !vocab || !lexicon) return;

      const gen = ++predictionGenRef.current;
      setPredicting(true);
      setResult(null);

      try {
        const T = temperatureRef.current;
        const prediction = await engine.fullPredict(teamA, teamB, vocab, lexicon, T);
        if (predictionGenRef.current !== gen) return; // superseded by newer prediction
        setResult(prediction);

        // Fire-and-forget: sensitivity
        const teamAEnc = encodeTeam(vocab, teamA);
        const teamBEnc = encodeTeam(vocab, teamB);
        engine.computeSensitivity(teamAEnc, teamBEnc, T).then((sensitivity: FeatureSensitivity) => {
          if (predictionGenRef.current !== gen) return;
          setResult((prev) => (prev ? { ...prev, sensitivity } : null));
        });

        // Fire-and-forget: retrieval
        loadRetrieval().then(async () => {
          if (predictionGenRef.current !== gen) return;
          const index = retrievalRef.current;
          if (!index) return;
          const embedding = await engine.getEmbedding(teamAEnc, teamBEnc);
          if (predictionGenRef.current !== gen) return;
          const neighbors = index.query(embedding, 10);
          const evidence: RetrievalEvidence = index.evidenceSummary(neighbors);
          if (predictionGenRef.current !== gen) return;
          setResult((prev) => (prev ? { ...prev, evidence } : null));
        });
      } catch (err) {
        console.error('Prediction failed:', err);
      } finally {
        if (predictionGenRef.current === gen) setPredicting(false);
      }
    },
    [loadRetrieval],
  );

  return { modelState, pokemonData, result, predicting, predict };
}
