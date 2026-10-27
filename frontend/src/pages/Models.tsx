import { useNavigate } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import SiteFooter from "@/components/SiteFooter";

function ScoreScale({ bands }: { bands: [string, string, string] }) {
  return (
    <div className="space-y-2">
      <div className="h-3 rounded-full overflow-hidden flex">
        <div className="h-full" style={{ width: "40%", background: "hsl(var(--destructive) / 0.3)" }} />
        <div className="h-full" style={{ width: "20%", background: "hsl(var(--warning) / 0.3)" }} />
        <div className="h-full" style={{ width: "40%", background: "hsl(var(--success) / 0.35)" }} />
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-destructive font-medium">{bands[0]} <span className="text-muted-foreground">(&lt; 0.40)</span></span>
        <span className="text-warning font-medium">{bands[1]} <span className="text-muted-foreground">(0.40–0.59)</span></span>
        <span className="text-success font-medium">{bands[2]} <span className="text-muted-foreground">(≥ 0.60)</span></span>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header onHome={() => navigate("/app")} crumb="Models" />
      <main className="mx-auto max-w-[900px] px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tightish text-foreground">How our models work</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-xl">
            We compute a Fundamentals score and a News-sentiment score, then blend them into a single AI score.
          </p>
        </div>

        <section className="surface p-6 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">1 · Fundamentals model</h2>
          <p className="text-sm text-muted-foreground">
            Valuation and financial metrics (P/E, P/B, market cap, EBITDA, dividend yield…) produce a fundamentals score in [0..1].
          </p>
          <ScoreScale bands={["Weak", "Moderate", "Strong"]} />
        </section>

        <section className="surface p-6 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">2 · News sentiment model</h2>
          <p className="text-sm text-muted-foreground">
            Recent stock-news headlines are classified and blended into a sentiment score in [0..1].
          </p>
          <ScoreScale bands={["Bearish", "Neutral", "Bullish"]} />
        </section>

        <section className="surface p-6 space-y-5">
          <h2 className="text-sm font-semibold text-foreground">3 · Blended AI score</h2>
          <div className="flex items-center justify-center gap-3 sm:gap-5 py-4 flex-wrap">
            <div className="flex flex-col items-center gap-2">
              <span className="h-14 w-14 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-semibold">Fund.</span>
              <span className="text-xs text-muted-foreground">70%</span>
            </div>
            <span className="text-2xl text-muted-foreground">＋</span>
            <div className="flex flex-col items-center gap-2">
              <span className="h-14 w-14 rounded-full flex items-center justify-center text-xs font-semibold" style={{ background: "hsl(262 68% 55% / 0.12)", color: "hsl(262 68% 45%)" }}>Sent.</span>
              <span className="text-xs text-muted-foreground">30%</span>
            </div>
            <span className="text-2xl text-muted-foreground">→</span>
            <div className="flex flex-col items-center gap-2">
              <span className="h-16 w-16 rounded-full bg-foreground text-background flex items-center justify-center text-sm font-semibold">AI</span>
              <span className="text-xs text-muted-foreground">Final</span>
            </div>
          </div>
          <div className="rounded-lg bg-secondary/60 p-3 font-mono text-xs text-muted-foreground text-center">
            final = (1 − 0.30) × fundamentals + 0.30 × sentiment
          </div>
          <p className="text-sm text-muted-foreground">
            The final score is calibrated with model confidence and mapped to a plain-language “Model view” on each stock —
            e.g. <span className="text-foreground font-medium">“Strong fundamentals with bullish sentiment.”</span>
          </p>
        </section>
      </main>
      <SiteFooter variant="app" />
    </div>
  );
}
