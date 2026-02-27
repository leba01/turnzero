'use client';

import * as React from 'react';
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

export function TeamInputPanel({ label, team, onChange, pokemonData }: TeamInputPanelProps) {
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
        <CardTitle className="font-[family-name:var(--font-label)] text-sm uppercase tracking-wider text-night">
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
              <TeamSummary team={team} onEdit={() => setEditing(true)} />
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {team.pokemon.map((mon, i) => (
                  <PokemonSlot
                    key={i}
                    index={i}
                    pokemon={mon}
                    onChange={(p) => handlePokemonChange(i, p)}
                    pokemonData={pokemonData}
                  />
                ))}
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
