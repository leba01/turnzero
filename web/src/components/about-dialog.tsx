'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';

export function AboutDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button className="font-[family-name:var(--font-label)] text-[11px] text-rock underline decoration-dotted underline-offset-4 transition-colors hover:text-jam">
          About this project
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-lg border-3 border-night bg-white shadow-[4px_4px_0px_#3D5A80]">
        <DialogHeader>
          <DialogTitle className="font-[family-name:var(--font-heading)] text-sm text-night">
            TURNZERO
          </DialogTitle>
          <DialogDescription className="font-[family-name:var(--font-body)] text-xs text-rock">
            Turn-zero team preview advisor for Pokémon VGC Regulation G.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 font-[family-name:var(--font-body)] text-xs leading-relaxed text-night">
          <p>
            Given two Open Team Sheets, TurnZero predicts which Pokémon to
            lead and bring — the first decision in every VGC game, made before
            any moves are selected.
          </p>

          <Separator />

          {/* How it works */}
          <div>
            <h3 className="mb-2 font-[family-name:var(--font-label)] text-[11px] uppercase tracking-wider text-night">
              How it works
            </h3>
            <ul className="flex flex-col gap-1.5 text-[11px]">
              <li>
                <span className="text-jam">5 transformer ensemble</span> trained
                on 246K expert matches, running entirely in your browser via ONNX
              </li>
              <li>
                <span className="text-jam">90-way joint prediction</span> over
                all lead-2 + back-2 combinations
              </li>
              <li>
                <span className="text-jam">Honest uncertainty</span> — the model
                abstains when it doesn&apos;t know, because experts themselves
                disagree 59% of the time on similar matchups
              </li>
              <li>
                <span className="text-jam">Retrieval evidence</span> from 246K
                training examples grounds predictions in what pros actually did
              </li>
            </ul>
          </div>

          <Separator />

          {/* Links */}
          <div className="flex flex-wrap gap-4 text-[11px]">
            <a
              href="https://github.com/leba01/turnzero"
              target="_blank"
              rel="noopener noreferrer"
              className="text-night underline decoration-dotted underline-offset-4 hover:text-jam"
            >
              GitHub
            </a>
            <a
              href="https://github.com/leba01/turnzero/blob/main/paper/turnzero.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="text-night underline decoration-dotted underline-offset-4 hover:text-jam"
            >
              Paper (PDF)
            </a>
            <span className="text-rock">
              CS229 · Stanford · 2025
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
