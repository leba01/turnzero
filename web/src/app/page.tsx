'use client';

import { useState, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { TeamInputPanel } from '@/components/team-input/team-input-panel';
import { ResultsPanel } from '@/components/results/results-panel';
import { AboutDialog } from '@/components/about-dialog';
import { usePrediction } from '@/hooks/use-prediction';
import { EXAMPLE_TEAM_A, EXAMPLE_TEAM_B } from '@/lib/data/example-matchup';
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

function isExample(teamA: TeamSheet, teamB: TeamSheet): boolean {
  return (
    teamA.pokemon[0].species === EXAMPLE_TEAM_A.pokemon[0].species &&
    teamB.pokemon[0].species === EXAMPLE_TEAM_B.pokemon[0].species
  );
}

export default function Home() {
  const [teamA, setTeamA] = useState<TeamSheet>(EXAMPLE_TEAM_A);
  const [teamB, setTeamB] = useState<TeamSheet>(EXAMPLE_TEAM_B);
  const autoPredictFired = useRef(false);
  const predictRef = useRef<((a: TeamSheet, b: TeamSheet) => void) | null>(null);

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

  const showingExample = isExample(teamA, teamB);

  const handleClear = () => {
    setTeamA(emptyTeam());
    setTeamB(emptyTeam());
  };

  const handleLoadExample = () => {
    setTeamA(EXAMPLE_TEAM_A);
    setTeamB(EXAMPLE_TEAM_B);
    predict(EXAMPLE_TEAM_A, EXAMPLE_TEAM_B);
  };

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
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
            className="h-4 w-64"
          />
        </div>
      )}

      {modelState.status === 'error' && (
        <div className="mb-6 border-2 border-red-400 bg-red-50 p-4 text-center">
          <p className="font-[family-name:var(--font-label)] text-xs text-red-600">
            Failed to load models: {modelState.message}
          </p>
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
        <div className="flex items-center justify-center gap-3">
          {showingExample ? (
            <Button
              variant="outline"
              size="sm"
              onClick={handleClear}
              className="border-night text-night hover:bg-mist"
            >
              CLEAR
            </Button>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={handleLoadExample}
              className="border-night text-night hover:bg-mist"
            >
              TRY EXAMPLE
            </Button>
          )}

          <Button
            size="lg"
            onClick={() => predict(teamA, teamB)}
            disabled={!canPredict}
            className="bg-jam px-8 py-3 text-white hover:bg-jam/90 disabled:bg-rock"
          >
            {predicting ? 'Predicting...' : modelState.status !== 'ready' ? 'Loading...' : 'PREDICT'}
          </Button>
        </div>

        {/* Results */}
        {result && <ResultsPanel result={result} teamA={teamA} teamB={teamB} />}
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
