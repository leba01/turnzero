'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { TeamInputPanel } from '@/components/team-input/team-input-panel';
import { ResultsPanel } from '@/components/results/results-panel';
import { AboutDialog } from '@/components/about-dialog';
import { usePrediction } from '@/hooks/use-prediction';
import { EXAMPLES, EXAMPLE_TEAM_A, EXAMPLE_TEAM_B } from '@/lib/data/example-matchup';
import type { Pokemon, TeamSheet } from '@/types/pokemon';

function emptyMon(): Pokemon {
  return { species: 'UNK', item: 'UNK', ability: 'UNK', tera_type: 'UNK', moves: ['UNK', 'UNK', 'UNK', 'UNK'] };
}

function emptyTeam(): TeamSheet {
  return { pokemon: [emptyMon(), emptyMon(), emptyMon(), emptyMon(), emptyMon(), emptyMon()] };
}

function hasValidSpecies(team: TeamSheet): boolean {
  return team.pokemon.some((m) => m.species !== 'UNK' && m.species !== '');
}

function isAnyExample(teamA: TeamSheet, teamB: TeamSheet): boolean {
  return EXAMPLES.some(
    (ex) =>
      teamA.pokemon[0].species === ex.teamA.pokemon[0].species &&
      teamB.pokemon[0].species === ex.teamB.pokemon[0].species,
  );
}

export default function Home() {
  const [teamA, setTeamA] = useState<TeamSheet>(EXAMPLE_TEAM_A);
  const [teamB, setTeamB] = useState<TeamSheet>(EXAMPLE_TEAM_B);
  const autoPredictFired = useRef(false);
  const predictRef = useRef<((a: TeamSheet, b: TeamSheet) => void) | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const onReady = useCallback(() => {
    if (!autoPredictFired.current) {
      autoPredictFired.current = true;
      predictRef.current?.(EXAMPLE_TEAM_A, EXAMPLE_TEAM_B);
    }
  }, []);

  const { modelState, pokemonData, result, predicting, predict } = usePrediction(onReady);
  predictRef.current = predict;

  const canPredict =
    modelState.status === 'ready' &&
    !predicting &&
    hasValidSpecies(teamA) &&
    hasValidSpecies(teamB);

  const showingExample = isAnyExample(teamA, teamB);

  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const handleClear = () => {
    setTeamA(emptyTeam());
    setTeamB(emptyTeam());
  };

  const handleLoadExample = (exampleId: string) => {
    const ex = EXAMPLES.find((e) => e.id === exampleId);
    if (!ex) return;
    setTeamA(ex.teamA);
    setTeamB(ex.teamB);
    predict(ex.teamA, ex.teamB);
  };

  return (
    <main className="mx-auto max-w-5xl px-4 py-4 sm:py-8">
      {/* Header */}
      <header className="mb-8 text-center">
        <h1 className="mb-2 font-[family-name:var(--font-heading)] text-xl text-night">
          TURNZERO
        </h1>
        <p className="font-[family-name:var(--font-label)] text-sm text-rock">
          Turn-Zero OTS Coach for Pokémon VGC
        </p>
        <div className="mt-3">
          <AboutDialog />
        </div>
      </header>

      <Separator className="mb-8" />

      {/* Model loading status */}
      {modelState.status === 'loading' && (
        <div className="mb-6 flex flex-col items-center gap-2">
          <span className="font-[family-name:var(--font-label)] text-xs text-rock">
            Loading model {modelState.loaded}/{modelState.total}...
          </span>
          <Progress
            value={(modelState.loaded / modelState.total) * 100}
            className="h-4 w-full max-w-64"
          />
        </div>
      )}

      {modelState.status === 'error' && (
        <div className="mb-6 border-2 border-red-400 bg-red-50 p-4 text-center">
          <p className="font-[family-name:var(--font-label)] text-xs text-red-600">
            Failed to load models: {modelState.message}
          </p>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
            className="mt-3 border-red-400 text-red-600 hover:bg-red-100"
          >
            RETRY
          </Button>
        </div>
      )}

      {/* Team inputs */}
      <div className="flex flex-col gap-6">
        <TeamInputPanel
          label="Your Team"
          team={teamA}
          onChange={setTeamA}
          pokemonData={pokemonData}
        />

        <TeamInputPanel
          label="Opponent"
          team={teamB}
          onChange={setTeamB}
          pokemonData={pokemonData}
        />

        {/* Action buttons */}
        <div className="flex flex-col items-center gap-3">
          <div className="flex items-center gap-2">
            <Button
              size="lg"
              onClick={() => predict(teamA, teamB)}
              disabled={!canPredict}
              className="bg-jam px-8 py-3 text-white hover:bg-jam/90 disabled:bg-rock"
            >
              {predicting ? 'Predicting...' : modelState.status !== 'ready' ? 'Loading...' : 'PREDICT'}
            </Button>
            {showingExample && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClear}
                className="border-night text-night hover:bg-mist"
              >
                CLEAR
              </Button>
            )}
          </div>

          {/* Example matchup buttons */}
          <div className="flex flex-wrap items-center justify-center gap-2">
            <span className="font-[family-name:var(--font-label)] text-[10px] text-rock">
              EXAMPLES:
            </span>
            {EXAMPLES.map((ex) => (
              <Button
                key={ex.id}
                variant="outline"
                size="sm"
                onClick={() => handleLoadExample(ex.id)}
                disabled={!canPredict}
                className="border-night px-2 py-1 text-night hover:bg-mist disabled:border-rock disabled:text-rock"
              >
                <span className="flex flex-col items-start leading-tight">
                  <span className="font-[family-name:var(--font-label)] text-[10px]">{ex.label}</span>
                  <span className="font-[family-name:var(--font-body)] text-[8px] text-rock">{ex.description}</span>
                </span>
              </Button>
            ))}
          </div>
        </div>

        {/* Results */}
        <div ref={resultsRef}>
          {result && <ResultsPanel result={result} teamA={teamA} teamB={teamB} />}
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-12 border-t-2 border-night pt-4 text-center">
        <p className="font-[family-name:var(--font-body)] text-[10px] text-rock">
          CS229 Final Project · Stanford University ·{' '}
          <a
            href="https://github.com/leba01/turnzero"
            className="text-night underline hover:text-jam"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </p>
        <p className="mt-1 font-[family-name:var(--font-body)] text-[9px] text-rock">
          Sprites from Pokémon Showdown (community fan art). Not affiliated with Nintendo or The Pokémon Company.
        </p>
      </footer>
    </main>
  );
}
