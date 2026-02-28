'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';
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

const LEFT_MIN = 340;
const LEFT_MAX = 700;
const LEFT_DEFAULT = 440;

export default function Home() {
  const [teamA, setTeamA] = useState<TeamSheet>(EXAMPLE_TEAM_A);
  const [teamB, setTeamB] = useState<TeamSheet>(EXAMPLE_TEAM_B);
  const autoPredictFired = useRef(false);
  const predictRef = useRef<((a: TeamSheet, b: TeamSheet) => void) | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [leftWidth, setLeftWidth] = useState(LEFT_MAX);
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false); // ref for stale-closure-safe mousemove; state drives CSS class
  const rightPanelRef = useRef<HTMLDivElement>(null);
  const leftPanelRef = useRef<HTMLElement>(null);
  const [showScrollHint, setShowScrollHint] = useState(true);

  const handleDragStart = useCallback(() => {
    isDraggingRef.current = true;
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  const handleDragEnd = useCallback(() => {
    isDraggingRef.current = false;
    setIsDragging(false);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      setLeftWidth(Math.min(LEFT_MAX, Math.max(LEFT_MIN, e.clientX)));
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', handleDragEnd);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', handleDragEnd);
    };
  }, [handleDragEnd]);

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
    if (result) setShowScrollHint(true);
  }, [result]);

  useEffect(() => {
    if (!result) return;
    if (rightPanelRef.current) {
      // Desktop: scroll the right panel to top so the blurb is always visible
      rightPanelRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (resultsRef.current) {
      // Mobile: jump the page down to the results section
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  useEffect(() => {
    const root = rightPanelRef.current;
    if (!root || !result) return;
    const onScroll = () => {
      if (root.scrollTop + root.clientHeight >= root.scrollHeight - 60) {
        setShowScrollHint(false);
      }
    };
    root.addEventListener('scroll', onScroll);
    return () => root.removeEventListener('scroll', onScroll);
  }, [result]);

  useEffect(() => {
    const root = rightPanelRef.current;
    if (!root) return;
    const handleArrow = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
      const focused = document.activeElement;
      if (!focused || focused.tagName !== 'SUMMARY') return;
      const summaries = Array.from(root.querySelectorAll<HTMLElement>('details > summary'));
      const idx = summaries.indexOf(focused as HTMLElement);
      if (idx === -1) return;
      e.preventDefault();
      const next = e.key === 'ArrowDown' ? summaries[idx + 1] : summaries[idx - 1];
      next?.focus();
    };
    root.addEventListener('keydown', handleArrow);
    return () => root.removeEventListener('keydown', handleArrow);
  }, [result]);

  useEffect(() => {
    const handleCtrlEnter = (e: KeyboardEvent) => {
      if (!e.ctrlKey || e.key !== 'Enter') return;
      const tag = (document.activeElement?.tagName ?? '').toUpperCase();
      const blocked =
        tag === 'TEXTAREA' || tag === 'INPUT' ||
        document.activeElement?.getAttribute('contenteditable') === 'true' ||
        document.activeElement?.getAttribute('role') === 'combobox' ||
        document.activeElement?.closest('[role="dialog"]') !== null;
      if (blocked) return;
      if (canPredict) predict(teamA, teamB);
    };
    window.addEventListener('keydown', handleCtrlEnter);
    return () => window.removeEventListener('keydown', handleCtrlEnter);
  }, [canPredict, predict, teamA, teamB]);

  useEffect(() => {
    const handlePanelSwap = (e: KeyboardEvent) => {
      if (!e.ctrlKey || (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft')) return;
      const tag = (document.activeElement?.tagName ?? '').toUpperCase();
      if (tag === 'TEXTAREA' || tag === 'INPUT' || document.activeElement?.getAttribute('contenteditable') === 'true') return;
      e.preventDefault();
      if (e.key === 'ArrowRight') {
        const right = rightPanelRef.current;
        if (!right) return;
        const first = right.querySelector<HTMLElement>('details > summary') ?? right;
        (first as HTMLElement).focus();
      } else {
        const left = leftPanelRef.current;
        if (!left) return;
        const first = left.querySelector<HTMLElement>('button, input, [role="combobox"]');
        first?.focus();
      }
    };
    window.addEventListener('keydown', handlePanelSwap);
    return () => window.removeEventListener('keydown', handlePanelSwap);
  }, []);

  useEffect(() => {
    const left = leftPanelRef.current;
    if (!left) return;
    const handleArrow = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
      const active = document.activeElement as HTMLElement | null;
      if (!active || !left.contains(active)) return;
      const tag = active.tagName.toUpperCase();
      if (tag === 'TEXTAREA' || tag === 'INPUT') return;
      const focusable = Array.from(
        left.querySelectorAll<HTMLElement>('button:not([disabled]), [role="combobox"]:not([disabled])')
      ).filter(el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length));
      const idx = focusable.indexOf(active);
      if (idx === -1) return;
      e.preventDefault();
      const next = e.key === 'ArrowDown' ? focusable[idx + 1] : focusable[idx - 1];
      next?.focus();
    };
    left.addEventListener('keydown', handleArrow);
    return () => left.removeEventListener('keydown', handleArrow);
  }, []);

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

  const handleSwap = () => {
    setTeamA(teamB);
    setTeamB(teamA);
    predict(teamB, teamA);
  };

  const headerBlock = (
    <header className="mb-6 text-center lg:text-left">
      <h1 className="mb-2 font-[family-name:var(--font-heading)] text-xl text-night">
        TURNZERO
      </h1>
      <p className="font-[family-name:var(--font-label)] text-sm sm:text-base text-rock">
        OTS Coach for Pokémon VGC Gen 9 Reg G
      </p>
      <div className="mt-3 max-w-md border-2 border-jam bg-jam/10 px-3 py-2.5 text-left">
        <ol className="list-inside list-decimal space-y-1 font-[family-name:var(--font-label)] text-xs sm:text-sm leading-relaxed text-night">
          <li>Enter your team and your opponent&apos;s OTS</li>
          <li>Hit <span className="font-[family-name:var(--font-heading)]">PREDICT</span></li>
          <li>See what to lead and bring</li>
        </ol>
        <p className="mt-1.5 font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
          Supports Pokepaste &amp; Showdown imports · Trained on 246K Bo3 matches
        </p>
      </div>
      <div className="mt-3">
        <AboutDialog />
      </div>
    </header>
  );

  const loadingBlock = modelState.status === 'loading' ? (
    <div className="mb-4 flex flex-col items-center gap-2">
      <span className="font-[family-name:var(--font-label)] text-xs sm:text-sm text-rock">
        Loading model {modelState.loaded}/{modelState.total}...
      </span>
      <Progress
        value={(modelState.loaded / modelState.total) * 100}
        className="h-4 w-full max-w-64"
      />
    </div>
  ) : modelState.status === 'error' ? (
    <div className="mb-4 border-2 border-red-400 bg-red-50 p-4 text-center">
      <p className="font-[family-name:var(--font-label)] text-xs sm:text-sm text-red-600">
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
  ) : null;

  const teamsBlock = (
    <div className="flex flex-col gap-4">
      <TeamInputPanel
        label="Your Team"
        team={teamA}
        onChange={setTeamA}
        onEditStart={() => setLeftWidth(LEFT_MAX)}
        pokemonData={pokemonData}
      />
      <div className="flex justify-center">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={handleSwap}
              disabled={!canPredict}
              className="flex flex-col items-center gap-0.5 text-night hover:text-jam disabled:cursor-not-allowed disabled:text-rock/40"
            >
              <span className="flex gap-1 font-[family-name:var(--font-heading)] text-base leading-none">
                <span>↑</span>
                <span>↓</span>
              </span>
              <span className="font-[family-name:var(--font-label)] text-[9px] sm:text-[10px] uppercase tracking-wider">
                swap
              </span>
            </button>
          </TooltipTrigger>
          <TooltipContent>Predict from opponent&apos;s perspective</TooltipContent>
        </Tooltip>
      </div>
      <TeamInputPanel
        label="Opponent"
        team={teamB}
        onChange={setTeamB}
        onEditStart={() => setLeftWidth(LEFT_MAX)}
        pokemonData={pokemonData}
      />
    </div>
  );

  const actionsBlock = (
    <div className="flex flex-col items-center gap-3 lg:items-start">
      <div className="flex items-center gap-2">
        <Button
          size="lg"
          onClick={() => predict(teamA, teamB)}
          disabled={!canPredict}
          className="bg-jam px-8 py-3 font-[family-name:var(--font-heading)] text-base text-white hover:bg-jam/90 disabled:bg-rock"
        >
          <span className="flex flex-col items-center leading-tight">
            <span>
              {predicting ? 'Predicting...' : modelState.status !== 'ready' ? 'Loading...' : 'PREDICT'}
            </span>
            {modelState.status === 'ready' && !predicting && (
              <span className="font-[family-name:var(--font-label)] text-[9px] opacity-70 tracking-widest">
                Ctrl+↵
              </span>
            )}
          </span>
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
      <div className="flex flex-col items-center gap-1 lg:items-start">
        <span className="font-[family-name:var(--font-label)] text-[10px] sm:text-sm text-rock">
          EXAMPLES:
        </span>
        <div className="flex flex-wrap justify-center gap-2 lg:justify-start">
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
              <span className="font-[family-name:var(--font-label)] text-[10px] sm:text-sm">{ex.label}</span>
              <span className="font-[family-name:var(--font-body)] text-[8px] sm:text-[10px] text-rock">{ex.description}</span>
            </span>
          </Button>
        ))}
        </div>
      </div>
    </div>
  );

  const resultsBlock = (
    <div ref={resultsRef}>
      {result && <ResultsPanel result={result} teamA={teamA} />}
    </div>
  );

  const footerBlock = (
    <footer className="mt-12 border-t-2 border-night pt-4 text-center lg:mt-6">
      <p className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
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
      <p className="mt-1 font-[family-name:var(--font-body)] text-[9px] sm:text-[11px] text-rock">
        Sprites from Pokémon Showdown (community fan art). Not affiliated with Nintendo or The Pokémon Company.
      </p>
    </footer>
  );

  return (
    <>
      {/* ── Mobile layout (< lg): vertical scroll, unchanged ── */}
      <main className="mx-auto max-w-5xl px-4 py-4 sm:py-8 lg:hidden">
        {headerBlock}
        <Separator className="mb-8" />
        {loadingBlock}
        <div className="flex flex-col gap-6">
          {teamsBlock}
          {actionsBlock}
          {resultsBlock}
        </div>
        {footerBlock}
      </main>

      {/* ── Desktop layout (lg+): resizable left/right panels ── */}
      <div className="hidden lg:flex lg:h-screen lg:overflow-hidden">
        {/* Left panel — input */}
        <aside
          ref={leftPanelRef}
          className={`flex shrink-0 flex-col gap-4 overflow-y-auto bg-milk px-5 py-6 ${isDragging ? '' : 'transition-[width] duration-300 ease-in-out'}`}
          style={{ width: leftWidth }}
        >
          {headerBlock}
          <Separator />
          {loadingBlock}
          {teamsBlock}
          {actionsBlock}
          {footerBlock}
        </aside>

        {/* Drag handle */}
        <div
          className="group relative z-10 flex w-8 shrink-0 cursor-col-resize flex-col items-center"
          onMouseDown={handleDragStart}
        >
          {/* Full-height night track */}
          <div className="absolute inset-y-0 left-1/2 w-[3px] -translate-x-1/2 bg-night" />
          {/* Centered knob */}
          <div className="pointer-events-none absolute top-1/2 -translate-y-1/2 flex h-14 w-8 flex-col items-center justify-center gap-[3px] border-2 border-night bg-jam shadow-[2px_2px_0px_#3D5A80] transition-shadow group-hover:shadow-[3px_3px_0px_#3D5A80]">
            <span className="font-[family-name:var(--font-label)] text-[8px] leading-none text-white">◀</span>
            <span className="font-[family-name:var(--font-label)] text-[8px] leading-none text-white">▶</span>
          </div>
        </div>

        {/* Right panel — results */}
        <div className="relative flex-1 overflow-y-auto bg-milk px-6 py-6 xl:px-10" ref={rightPanelRef}>
          {result ? (
            <div ref={resultsRef}><ResultsPanel result={result} teamA={teamA} /></div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="font-[family-name:var(--font-label)] text-sm sm:text-base text-rock">
                {modelState.status === 'ready'
                  ? 'Set up your teams and hit PREDICT'
                  : 'Loading models...'}
              </p>
            </div>
          )}
          {/* Scroll hint */}
          {result && showScrollHint && (
            <div className="pointer-events-none sticky inset-x-0 bottom-0 flex flex-col items-center pb-3">
              <div className="pointer-events-auto animate-bounce rounded-full bg-jam/90 px-6 py-1.5 font-[family-name:var(--font-label)] text-lg font-bold tracking-wider text-white shadow-lg">
                ▼ SCROLL FOR MORE ▼
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
