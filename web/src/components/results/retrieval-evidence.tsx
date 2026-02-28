'use client';

import Image from 'next/image';
import { Skeleton } from '@/components/ui/skeleton';
import { spriteUrl, displayName } from '@/lib/sprites';
import { SpeciesName } from '@/components/ui/species-name';
import type { RetrievalEvidence } from '@/types/pokemon';

interface RetrievalEvidenceProps {
  evidence: RetrievalEvidence | null;
  loading?: boolean;
}

export function RetrievalEvidenceDisplay({ evidence, loading }: RetrievalEvidenceProps) {
  if (loading || evidence === null) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
        <span className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
          Loading similar matchups...
        </span>
      </div>
    );
  }

  if (evidence.n_neighbors === 0) {
    return (
      <p className="font-[family-name:var(--font-body)] text-xs sm:text-sm text-rock">
        No similar matchups found in training data.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-rock">
        From {evidence.n_neighbors} similar tournament game{evidence.n_neighbors !== 1 ? 's' : ''} — avg. match score {(evidence.mean_similarity * 100).toFixed(0)}%
      </p>

      {/* Top lead pairs */}
      {evidence.lead_pair_freq.length > 0 && (
        <div>
          <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] sm:text-sm uppercase tracking-wider text-rock">
            How Players with Teams Similar to Yours Led
          </h4>
          <div className="flex flex-col gap-2">
            {evidence.lead_pair_freq.slice(0, 5).map(({ pair, fraction }) => {
              const [spA, spB] = pair.split('+');
              const pct = fraction * 100;
              return (
                <div key={pair} className="flex items-center gap-2">
                  <div className="flex w-44 shrink-0 items-center gap-1.5 sm:w-52">
                    <div className="flex shrink-0 -space-x-2">
                      <Image
                        src={spriteUrl(spA)}
                        alt={displayName(spA)}
                        width={36}
                        height={36}
                        className="object-contain"
                        unoptimized
                      />
                      <Image
                        src={spriteUrl(spB)}
                        alt={displayName(spB)}
                        width={36}
                        height={36}
                        className="object-contain"
                        unoptimized
                      />
                    </div>
                    <div className="min-w-0 flex-1 break-words font-[family-name:var(--font-label)] text-[10px] sm:text-sm leading-snug text-night">
                      <SpeciesName species={spA} /> + <SpeciesName species={spB} />
                    </div>
                  </div>
                  <div className="h-3 flex-1 border border-night bg-muted">
                    <div className="h-full bg-grass" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="w-10 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-night">
                    {pct.toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Most brought mons */}
      {evidence.mon_bring_freq.length > 0 && (
        <div>
          <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] sm:text-sm uppercase tracking-wider text-rock">
            Most Brought in These Matchups
          </h4>
          <div className="flex flex-col gap-2">
            {evidence.mon_bring_freq.slice(0, 6).map(({ species, fraction }) => {
              const pct = fraction * 100;
              return (
                <div key={species} className="flex items-center gap-2">
                  <div className="flex w-44 shrink-0 items-center gap-1.5 sm:w-52">
                    <Image
                      src={spriteUrl(species)}
                      alt={displayName(species)}
                      width={32}
                      height={32}
                      className="shrink-0 object-contain"
                      unoptimized
                    />
                    <span className="min-w-0 flex-1 break-words font-[family-name:var(--font-label)] text-[10px] sm:text-sm leading-snug text-night">
                      <SpeciesName species={species} />
                    </span>
                  </div>
                  <div className="h-3 flex-1 border border-night bg-muted">
                    <div className="h-full bg-grass" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="w-10 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] sm:text-sm text-night">
                    {pct.toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
