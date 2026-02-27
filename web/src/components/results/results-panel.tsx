'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { TopPlans } from './top-plans';
import { MarginalsDisplay } from './marginals-display';
import { OpponentCues } from './opponent-cues';
import { SensitivityDisplay } from './sensitivity-display';
import { RetrievalEvidenceDisplay } from './retrieval-evidence';
import type { PredictionResult, TeamSheet } from '@/types/pokemon';

interface ResultsPanelProps {
  result: PredictionResult;
  teamA: TeamSheet;
  teamB: TeamSheet;
}

function DisclosureSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group border-2 border-night bg-white shadow-[2px_2px_0px_#3D5A80]">
      <summary className="flex cursor-pointer items-center justify-between p-3 font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night select-none hover:bg-mist/20">
        {title}
        <span aria-hidden="true" className="text-rock transition-transform group-open:rotate-90">
          ▶
        </span>
      </summary>
      <div className="border-t-2 border-night p-4">
        {children}
      </div>
    </details>
  );
}

export function ResultsPanel({ result, teamA }: ResultsPanelProps) {
  const species = teamA.pokemon.map((m) => m.species);

  return (
    <div className="flex flex-col gap-6">
      <Separator />

      {/* ── Tier 1: The Recommendation ── */}
      <Card>
        <CardContent className="p-4">
          <TopPlans
            plans={result.top_plans}
            species={species}
            abstain={result.abstain}
            confidence={result.confidence}
            ensembleAgreement={result.ensemble_agreement}
          />
        </CardContent>
      </Card>

      {/* ── Tier 2: Supporting Evidence (side by side on larger screens) ── */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardContent className="p-4">
            <h3 className="mb-3 font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night">
              Why These Leads
            </h3>
            <MarginalsDisplay marginals={result.marginals} species={species} />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <h3 className="mb-3 font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night">
              Similar Matchups
            </h3>
            <RetrievalEvidenceDisplay evidence={result.evidence} />
          </CardContent>
        </Card>
      </div>

      {/* ── Tier 3: Deep Dive (collapsed by default) ── */}
      <div className="flex flex-col gap-2">
        <DisclosureSection title="Opponent Cues">
          <OpponentCues cues={result.opponent_cues} />
        </DisclosureSection>

        <DisclosureSection title="Feature Sensitivity">
          <SensitivityDisplay sensitivity={result.sensitivity} />
        </DisclosureSection>
      </div>
    </div>
  );
}
