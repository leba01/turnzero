'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PokemonSlot } from './pokemon-slot';
import { PasteInput } from './paste-input';
import { TeamSummary } from './team-summary';
import type { Pokemon, TeamSheet } from '@/types/pokemon';

interface TeamInputPanelProps {
  label: string;
  team: TeamSheet;
  onChange: (team: TeamSheet) => void;
  onEditStart?: () => void;
  pokemonData: {
    species: string[];
    items: string[];
    abilities: string[];
    tera_types: string[];
    moves: string[];
  } | null;
}

function isTeamComplete(team: TeamSheet): boolean {
  return team.pokemon.every((m) => m.species !== 'UNK' && m.species !== '');
}

export function TeamInputPanel({ label, team, onChange, onEditStart, pokemonData }: TeamInputPanelProps) {
  const [tab, setTab] = React.useState('builder');
  const [editing, setEditing] = React.useState(false);
  const initialized = React.useRef(false);

  React.useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      if (window.matchMedia('(max-width: 640px)').matches) {
        setTab('paste');
      }
    }
  }, []);

  const handlePokemonChange = (index: number, pokemon: Pokemon) => {
    const newPokemon = [...team.pokemon] as TeamSheet['pokemon'];
    newPokemon[index] = pokemon;
    onChange({ pokemon: newPokemon });
  };

  const teamComplete = isTeamComplete(team);
  const showCompact = tab === 'builder' && teamComplete && !editing;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="font-[family-name:var(--font-label)] text-sm sm:text-base uppercase tracking-wider text-night">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={tab} onValueChange={(v) => { setTab(v); setEditing(false); }}>
          <TabsList className="mb-4 border-2 border-night bg-muted">
            <TabsTrigger value="builder">Builder</TabsTrigger>
            <TabsTrigger value="paste">Paste</TabsTrigger>
          </TabsList>

          <TabsContent value="builder">
            {showCompact ? (
              <TeamSummary team={team} onEdit={() => { setEditing(true); onEditStart?.(); }} />
            ) : (
              <div className="flex flex-col gap-3">
                {team.pokemon.map((mon, i) => (
                  <PokemonSlot
                    key={i}
                    index={i}
                    pokemon={mon}
                    onChange={(p) => handlePokemonChange(i, p)}
                    pokemonData={pokemonData}
                  />
                ))}
                {editing && teamComplete && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setEditing(false)}
                    className="self-end border-night text-night hover:bg-mist"
                  >
                    DONE
                  </Button>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="paste">
            <PasteInput onParsed={onChange} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
