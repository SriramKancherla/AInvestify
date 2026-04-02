import { cn } from "@/lib/utils";

export type SiteFooterVariant = "app" | "landing";

type SiteFooterProps = {
  variant?: SiteFooterVariant;
  className?: string;
};

function CreatorsBlock({ emeraldAccent }: { emeraldAccent: boolean }) {
  const linkClass = emeraldAccent
    ? "text-emerald-400/90 hover:text-emerald-300 hover:underline underline-offset-2"
    : "text-primary hover:text-primary/90 hover:underline underline-offset-2";

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 sm:gap-8">
      <div className="space-y-2 text-left">
        <p className={cn("text-sm font-medium", emeraldAccent ? "text-foreground/95" : "text-foreground")}>Sriram Kancherla</p>
        <p className={cn("text-xs", emeraldAccent ? "text-muted-foreground/90" : "text-muted-foreground")}>
          <a href="mailto:kancherlasriram2006@gmail.com" className={linkClass}>
            kancherlasriram2006@gmail.com
          </a>
        </p>
        <p className="text-xs">
          <a href="https://www.linkedin.com/in/sriram-kancherla-80a7b028a/" target="_blank" rel="noopener noreferrer" className={linkClass}>
            LinkedIn
          </a>
        </p>
      </div>
      <div className="space-y-2 text-left">
        <p className={cn("text-sm font-medium", emeraldAccent ? "text-foreground/95" : "text-foreground")}>Viswanath Parashuram Yadavalli</p>
        <p className={cn("text-xs", emeraldAccent ? "text-muted-foreground/90" : "text-muted-foreground")}>
          <a href="mailto:vishwanaathh4@gmail.com" className={linkClass}>
            vishwanaathh4@gmail.com
          </a>
        </p>
        <p className="text-xs">
          <a href="https://linkedin.com/in/vishwa-yadavalli-65503628b" target="_blank" rel="noopener noreferrer" className={linkClass}>
            LinkedIn
          </a>
        </p>
      </div>
    </div>
  );
}

function DisclaimerText({ emeraldAccent }: { emeraldAccent: boolean }) {
  return (
    <p
      className={cn(
        "text-xs leading-relaxed max-w-3xl",
        emeraldAccent ? "text-muted-foreground/80" : "text-muted-foreground/90",
      )}
    >
      <span className={cn("font-medium", "text-foreground/90")}>Disclaimer:</span> AInvestify is an experimental, educational project. Outputs are
      model-based and may be wrong, incomplete, or out of date. This site is not financial, investment, tax, or legal advice. Do not use it to make
      investment decisions. Consult a qualified professional before investing.
    </p>
  );
}

/**
 * App: score guide + creators + disclaimer — reads as a soft page tail, not a heavy “footer slab”.
 * Landing: creators + disclaimer only — glass panel aligned with the auth card aesthetic.
 */
export default function SiteFooter({ variant = "app", className }: SiteFooterProps) {
  const emeraldAccent = variant === "landing";

  if (variant === "landing") {
    return (
      <footer className={cn("relative z-20 mt-10 w-full max-w-5xl mx-auto px-0 pb-8 pt-6 border-t border-white/10", className)}>
        <div className="rounded-xl border border-white/10 bg-card/35 backdrop-blur-md px-5 py-6 sm:px-6 space-y-6">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground/70 mb-4">Creators</p>
            <CreatorsBlock emeraldAccent={emeraldAccent} />
          </div>
          <div className="h-px bg-white/10" aria-hidden />
          <DisclaimerText emeraldAccent={emeraldAccent} />
        </div>
      </footer>
    );
  }

  return (
    <footer className={cn("container pb-4 pt-6 mt-4 border-t border-border space-y-6", className)}>
      <div className="rounded-xl border border-border p-4 bg-card shadow-sm">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">What the scores mean</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 text-sm">
          <div className="rounded-lg border border-border p-3 bg-secondary/20">
            <p className="font-semibold">Model view</p>
            <p className="text-muted-foreground mt-1">
              On Overview: fundamentals strength (strong / moderate / weak) plus news sentiment (bullish / neutral / bearish), in plain language.
            </p>
          </div>
          <div className="rounded-lg border border-border p-3 bg-secondary/20">
            <p className="font-semibold">Final Score</p>
            <p className="text-muted-foreground mt-1">Single blended number from fundamentals and sentiment. Higher generally means a stronger combined setup.</p>
          </div>
          <div className="rounded-lg border border-border p-3 bg-secondary/20">
            <p className="font-semibold">Fundamentals Score</p>
            <p className="text-muted-foreground mt-1">From valuation and financial metrics (P/E, P/B, market cap, EBITDA, etc.) using trained models.</p>
          </div>
          <div className="rounded-lg border border-border p-3 bg-secondary/20">
            <p className="font-semibold">Sentiment Score</p>
            <p className="text-muted-foreground mt-1">From recent stock news (classification + polarity blend).</p>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-border p-4 bg-card shadow-sm">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Creators</h2>
        <CreatorsBlock emeraldAccent={emeraldAccent} />
      </div>

      <p className="text-xs text-muted-foreground leading-relaxed max-w-3xl border border-border/80 rounded-lg p-4 bg-secondary/20">
        <span className="text-foreground font-semibold">Disclaimer:</span> AInvestify is an experimental, educational project. Outputs are model-based and
        may be wrong, incomplete, or out of date. This site is not financial, investment, tax, or legal advice. Do not use it to make investment decisions. Consult
        a qualified professional before investing.
      </p>
    </footer>
  );
}
