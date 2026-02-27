'use client';

import { Skeleton } from '@/components/ui/skeleton';
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
        <span className="font-[family-name:var(--font-body)] text-[10px] text-rock">
          Loading similar matchups...
        </span>
      </div>
    );
  }

  if (evidence.n_neighbors === 0) {
    return (
      <p className="font-[family-name:var(--font-body)] text-xs text-rock">
        No similar matchups found in training data.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="font-[family-name:var(--font-body)] text-[10px] text-rock">
        k={evidence.n_neighbors} neighbors, mean similarity={evidence.mean_similarity.toFixed(3)}
      </p>

      {/* Top lead pairs */}
      {evidence.lead_pair_freq.length > 0 && (
        <div>
          <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-wider text-rock">
            Expert Lead Pairs
          </h4>
          <div className="flex flex-col gap-1">
            {evidence.lead_pair_freq.slice(0, 5).map(({ pair, fraction }) => {
              const pct = fraction * 100;
              return (
                <div key={pair} className="flex items-center gap-2">
                  <span className="w-24 shrink-0 truncate font-[family-name:var(--font-label)] text-[10px] text-night sm:w-36">
                    {pair}
                  </span>
                  <div className="h-3 flex-1 border border-night bg-muted">
                    <div
                      className="h-full bg-grass"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-10 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] text-night">
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
          <h4 className="mb-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-wider text-rock">
            Most Brought
          </h4>
          <div className="flex flex-col gap-1">
            {evidence.mon_bring_freq.slice(0, 6).map(({ species, fraction }) => {
              const pct = fraction * 100;
              return (
                <div key={species} className="flex items-center gap-2">
                  <span className="w-20 shrink-0 truncate font-[family-name:var(--font-label)] text-[10px] text-night sm:w-28">
                    {species}
                  </span>
                  <div className="h-3 flex-1 border border-night bg-muted">
                    <div
                      className="h-full bg-grass"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-10 shrink-0 text-right font-[family-name:var(--font-body)] text-[10px] text-night">
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
