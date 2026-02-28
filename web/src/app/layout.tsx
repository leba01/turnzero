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
  title: "TurnZero — VGC Team Preview Patterns",
  description:
    "Team preview patterns from 246K tournament games. See what experts typically led and brought in similar matchups — 5-model ensemble running in your browser.",
  keywords: ["pokemon", "vgc", "team preview", "ots", "competitive"],
  openGraph: {
    title: "TurnZero — VGC Team Preview Patterns",
    description:
      "Team preview patterns from 246K tournament games · 5-model ensemble · runs in your browser.",
    type: "website",
    images: ["/og.png"],
  },
  twitter: {
    card: "summary_large_image",
    title: "TurnZero — VGC Team Preview Patterns",
    description:
      "Team preview patterns from 246K tournament games · 5-model ensemble · runs in your browser.",
    images: ["/og.png"],
  },
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
