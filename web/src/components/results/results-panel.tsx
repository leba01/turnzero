'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { ConfidenceGauge } from './confidence-gauge';
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="font-[family-name:var(--font-label)] text-xs uppercase tracking-wider text-night">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export function ResultsPanel({ result, teamA }: ResultsPanelProps) {
  const species = teamA.pokemon.map((m) => m.species);

  return (
    <div className="flex flex-col gap-4">
      <Separator />

      <Section title="Confidence">
        <ConfidenceGauge
          confidence={result.confidence}
          entropy={result.entropy}
          mutual_information={result.mutual_information}
          abstain={result.abstain}
        />
      </Section>

      <Section title="Recommended Plans">
        <TopPlans plans={result.top_plans} species={species} abstain={result.abstain} />
      </Section>

      <Section title="Why These Leads">
        <MarginalsDisplay marginals={result.marginals} species={species} />
      </Section>

      <Section title="Key Opponent Cues">
        <OpponentCues cues={result.opponent_cues} />
      </Section>

      <Section title="Feature Sensitivity">
        <SensitivityDisplay sensitivity={result.sensitivity} />
      </Section>

      <Section title="Similar Matchups">
        <RetrievalEvidenceDisplay evidence={result.evidence} />
      </Section>
    </div>
  );
}
