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
        <button className="font-[family-name:var(--font-label)] text-[11px] sm:text-[13px] text-rock underline decoration-dotted underline-offset-4 transition-colors hover:text-jam">
          About this project
        </button>
      </DialogTrigger>
      <DialogContent className="w-[calc(100vw-2rem)] sm:max-w-lg border-3 border-night bg-white shadow-[4px_4px_0px_#3D5A80]">
        <DialogHeader>
          <DialogTitle className="font-[family-name:var(--font-heading)] text-sm sm:text-base text-night">
            TURNZERO
          </DialogTitle>
          <DialogDescription className="font-[family-name:var(--font-body)] text-xs sm:text-sm text-rock">
            Turn-zero team preview advisor for Pokémon VGC Regulation G.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 font-[family-name:var(--font-body)] text-xs sm:text-sm leading-relaxed text-night">
          <p>
            Given two Open Team Sheets, TurnZero predicts which Pokémon to
            lead and bring — the first decision in every VGC game, made before
            any moves are selected.
          </p>

          <Separator />

          {/* How it works */}
          <div>
            <h3 className="mb-2 font-[family-name:var(--font-label)] text-[11px] sm:text-[13px] uppercase tracking-wider text-night">
              How it works
            </h3>
            <ul className="flex flex-col gap-1.5 text-[11px] sm:text-[13px]">
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

          {/* About the author */}
          <div>
            <h3 className="mb-2 font-[family-name:var(--font-label)] text-[11px] sm:text-[13px] uppercase tracking-wider text-night">
              About
            </h3>
            <p className="text-[11px] sm:text-[13px] leading-relaxed text-rock">
              Hi! I&apos;m Lucas, a coterm at Stanford (BS/MS CS &apos;25). This started as my
              CS229 final project and turned into something I actually use. If you want to dig
              into how it works, the full paper and code are below — feedback always welcome.
            </p>
            <div className="mt-3 flex flex-wrap gap-4 text-[11px] sm:text-[13px]">
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
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
