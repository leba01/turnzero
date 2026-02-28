'use client';

import Image from 'next/image';
import { spriteUrl, displayName } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import type { Marginals } from '@/types/pokemon';

interface MarginalsDisplayProps {
  marginals: Marginals;
  species: string[];
}

const LEAD_PAIRS: [number, number][] = [];
for (let i = 0; i < 6; i++) {
  for (let j = i + 1; j < 6; j++) {
    LEAD_PAIRS.push([i, j]);
  }
}

/** Horizontal strip of 6 sprites sorted by probability, top-N highlighted. */
function ProbStrip({
  probs,
  species,
  highlightCount,
  accentColor,
  label,
}: {
  probs: number[];
  species: string[];
  highlightCount: number;
  accentColor: string;
  label: string;
}) {
  const order = [...Array(6).keys()].sort((a, b) => probs[b] - probs[a]);

  // Near-miss detection: non-selected mons within σ/2 of the last selected prob
  // get a faded highlight to show they're statistically in the same cluster.
  const mean = probs.reduce((s, p) => s + p, 0) / probs.length;
  const std = Math.sqrt(probs.reduce((s, p) => s + (p - mean) ** 2, 0) / probs.length);
  const lastHighlightedProb = highlightCount > 0 ? probs[order[highlightCount - 1]] : 0;
  // Only show near-miss fading when the distribution is spread enough to be meaningful.
  // A compressed team (std < 10pp) means everything is ambiguous — highlighting clusters is noise.
  const diverse = std > 0.10;
  const nearMissThreshold = std * 0.5;

  return (
    <div>
      <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] sm:text-sm uppercase tracking-wider text-rock">
        {label}
      </h4>
      <div className="flex flex-wrap gap-2 sm:gap-3">
        {order.map((idx, rank) => {
          const highlighted = rank < highlightCount;
          const nearMiss = diverse && !highlighted && (lastHighlightedProb - probs[idx]) <= nearMissThreshold;
          // When distribution is compressed, full brightness would falsely signal confidence.
          const opacity = highlighted ? (diverse ? 1.0 : 0.55) : nearMiss ? 0.55 : 0.35;
          return (
            <div
              key={idx}
              className="flex w-20 sm:w-24 flex-col items-center gap-0.5 transition-opacity"
              style={{ opacity }}
            >
              <div
                className="flex items-center justify-center rounded-md border-2 p-1"
                style={{
                  borderColor: (highlighted || nearMiss) ? accentColor : 'transparent',
                  backgroundColor: (highlighted || nearMiss) ? `${accentColor}15` : 'transparent',
                }}
              >
                <Image
                  src={spriteUrl(species[idx])}
                  alt={displayName(species[idx])}
                  width={40}
                  height={40}
                  className="object-contain"
                  unoptimized
                />
              </div>
              <span className="w-full truncate text-center font-[family-name:var(--font-label)] text-[9px] sm:text-[10px] leading-tight text-night">
                <SpeciesName species={species[idx]} />
              </span>
              <span
                className="rounded-full px-1.5 py-0.5 font-[family-name:var(--font-body)] text-[9px] sm:text-[11px] font-medium"
                style={{
                  backgroundColor: (highlighted || nearMiss) ? `${accentColor}30` : '#f0f0f0',
                  color: (highlighted || nearMiss) ? accentColor : '#8D8D8D',
                }}
              >
                {(probs[idx] * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Lollipop chart: thin line + circle endpoint for each pair. */
function LollipopChart({
  pairs,
  species,
  probs,
}: {
  pairs: { indices: [number, number]; prob: number }[];
  species: string[];
  probs: number[];
}) {
  const maxProb = Math.max(...pairs.map((p) => p.prob));

  return (
    <div>
      <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] sm:text-sm uppercase tracking-wider text-rock">
        Top Lead Pairs
      </h4>
      <div className="flex flex-col gap-2">
        {pairs.map(({ indices: [a, b], prob }, i) => {
          const pct = maxProb > 0 ? (prob / maxProb) * 100 : 0;
          return (
            <div key={i} className="flex items-center gap-2">
              {/* Sprites + stacked names as a group */}
              <div className="flex w-40 sm:w-48 shrink-0 items-center gap-1.5">
                <div className="flex shrink-0 -space-x-2">
                  <Image
                    src={spriteUrl(species[a])}
                    alt={displayName(species[a])}
                    width={40}
                    height={40}
                    className="object-contain"
                    unoptimized
                  />
                  <Image
                    src={spriteUrl(species[b])}
                    alt={displayName(species[b])}
                    width={40}
                    height={40}
                    className="object-contain"
                    unoptimized
                  />
                </div>
                <div className="min-w-0 flex-1 break-words font-[family-name:var(--font-label)] text-[10px] sm:text-sm leading-snug text-night">
                  <SpeciesName species={species[a]} /> + <SpeciesName species={species[b]} />
                </div>
              </div>
              {/* Lollipop: thin line + dot */}
              <div className="relative h-4 flex-1">
                {/* Track */}
                <div className="absolute inset-y-0 left-0 right-0 flex items-center">
                  <div className="h-[1px] w-full bg-rock/20" />
                </div>
                {/* Line + dot */}
                <div className="absolute inset-y-0 left-0 flex items-center" style={{ width: `${pct}%` }}>
                  <div className="h-[2px] flex-1 bg-moss" />
                  <div className="size-3 shrink-0 rounded-full border-2 border-moss bg-grass" />
                </div>
              </div>
              {/* Value */}
              <span className="w-12 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-night">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function MarginalsDisplay({ marginals, species }: MarginalsDisplayProps) {
  // Top 5 lead pairs
  const pairOrder = [...Array(15).keys()]
    .sort((a, b) => marginals.lead_pair_probs[b] - marginals.lead_pair_probs[a])
    .slice(0, 5);

  const topPairs = pairOrder.map((j) => ({
    indices: LEAD_PAIRS[j],
    prob: marginals.lead_pair_probs[j],
  }));

  return (
    <div className="flex flex-col gap-4">
      <p className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
        Highlighted = model&apos;s pick · Faded highlight = near-miss, statistically tied · Dim = unlikely · All faded = model uncertain
      </p>
      <ProbStrip
        probs={marginals.lead_probs}
        species={species}
        highlightCount={2}
        accentColor="#F4845F"
        label="Lead Probabilities"
      />

      <ProbStrip
        probs={marginals.bring_probs}
        species={species}
        highlightCount={4}
        accentColor="#98C1D9"
        label="Bring Probabilities"
      />

      <LollipopChart
        pairs={topPairs}
        species={species}
        probs={marginals.lead_pair_probs}
      />
    </div>
  );
}
