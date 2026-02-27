import type { Metadata, Viewport } from "next";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#3D5A80",
};

export const metadata: Metadata = {
  title: "TurnZero — VGC Turn-Zero Coach",
  description:
    "AI-powered team preview advisor for Pokémon VGC. Predicts expert lead + bring decisions from Open Team Sheets using an ensemble of 5 transformers.",
  keywords: ["pokemon", "vgc", "team preview", "ots", "competitive"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
