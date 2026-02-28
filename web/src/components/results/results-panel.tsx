'use client';

import { Separator } from '@/components/ui/separator';
import { TopPlans } from './top-plans';
import { MarginalsDisplay } from './marginals-display';
import { OpponentCues } from './opponent-cues';
import { SensitivityDisplay } from './sensitivity-display';
import { RetrievalEvidenceDisplay } from './retrieval-evidence';
import { HelpTip } from '@/components/ui/help-tip';
import type { PredictionResult, TeamSheet } from '@/types/pokemon';

interface ResultsPanelProps {
  result: PredictionResult;
  teamA: TeamSheet;
}

function DisclosureSection({ title, help, defaultOpen = false, id, children }: { title: string; help?: string; defaultOpen?: boolean; id?: string; children: React.ReactNode }) {
  return (
    <details id={id} open={defaultOpen || undefined} className="group border-2 border-night bg-white shadow-[2px_2px_0px_#3D5A80]">
      <summary className="flex cursor-pointer items-center justify-between p-3 font-[family-name:var(--font-label)] text-xs sm:text-sm uppercase tracking-wider text-night select-none hover:bg-mist/20">
        <span className="flex items-center">
          {title}
          {help && <HelpTip text={help} />}
        </span>
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
    <div className="flex flex-col gap-3">
      <p className="font-[family-name:var(--font-body)] text-xs sm:text-sm text-rock leading-relaxed">
        Your recommendation is at the top. Everything below breaks down how the model got there — what it sees on their side, your individual bring odds, and past tournament games it pulled from.
      </p>
      <Separator />

      <DisclosureSection title="Recommended Plans" help="Your best lead pair (front) and back pair (dimmed) ranked by probability. #1 is what the model thinks you should go with." defaultOpen>
        <TopPlans
          plans={result.top_plans}
          species={species}
          abstain={result.abstain}
          confidence={result.confidence}
          ensembleAgreement={result.ensemble_agreement}
        />
      </DisclosureSection>

      <DisclosureSection title="Opponent Cues" help="What the model sees on the other side — these threats are shaping the lead recommendation above." defaultOpen>
        <OpponentCues cues={result.opponent_cues} />
      </DisclosureSection>

      <DisclosureSection title="Why These Leads" help="Per-mon odds of leading or being brought. Sprites are sorted by probability — highlighted ones are the model's picks." defaultOpen>
        <MarginalsDisplay marginals={result.marginals} species={species} />
      </DisclosureSection>

      <DisclosureSection title="Similar Matchups" help="Tournament games where both teams were similar to this matchup. Shows what players with a team like yours actually led and brought against opponents like this one." defaultOpen>
        <RetrievalEvidenceDisplay evidence={result.evidence} />
      </DisclosureSection>

      <DisclosureSection title="Feature Sensitivity" id="sensitivity-section" help="What happens if you swap items, abilities, or tera types. Shows which changes would shift the recommendation.">
        <SensitivityDisplay sensitivity={result.sensitivity} />
      </DisclosureSection>
    </div>
  );
}
