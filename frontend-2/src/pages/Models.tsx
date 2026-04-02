import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import SiteFooter from "@/components/SiteFooter";

function TierRow({
  tier,
  range,
  description,
}: {
  tier: string;
  range: string;
  description: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-secondary/20 p-3 space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">{tier}</span>
        <span className="text-xs text-muted-foreground">({range})</span>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
    </div>
  );
}

export default function ModelsPage() {
  const navigate = useNavigate();

  const content = useMemo(
    () => (
      <main className="container py-6 space-y-6">
        <div className="rounded-xl border border-border bg-card p-4">
          <h1 className="text-2xl font-bold text-foreground">How our models work</h1>
          <p className="text-sm text-muted-foreground mt-1">
            We generate a <span className="font-medium text-foreground">Fundamentals</span> score and a{" "}
            <span className="font-medium text-foreground">Sentiment</span> score, then combine them into a final blended score.
            The “Model view” phrase on the Overview tab is derived from these tiers.
          </p>
        </div>

        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">1) Fundamentals model</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Uses valuation and financial metrics (P/E, P/B, market cap, EBITDA, etc.) to output a fundamentals score in{" "}
            <span className="font-medium text-foreground">[0..1]</span>.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <TierRow
              tier="Strong"
              range=">= 0.60"
              description="Valuation + financial signals look favorable in the model."
            />
            <TierRow
              tier="Moderate"
              range="0.40 – 0.59"
              description="Some positives, some negatives — mixed setup."
            />
            <TierRow
              tier="Weak"
              range="< 0.40"
              description="Valuation + financial signals look less favorable."
            />
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">2) News sentiment model</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Classifies recent stock news sentiment and blends polarity into a sentiment score in{" "}
            <span className="font-medium text-foreground">[0..1]</span>.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <TierRow
              tier="Bullish"
              range=">= 0.60"
              description="News flow is predominantly positive."
            />
            <TierRow
              tier="Neutral"
              range="0.40 – 0.59"
              description="News sentiment is mixed / close to neutral."
            />
            <TierRow
              tier="Bearish"
              range="< 0.40"
              description="News flow is predominantly negative."
            />
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">3) Final blended score</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            If both fundamentals and sentiment are available, we compute a raw blend:
          </p>
          <div className="rounded-lg border border-border bg-secondary/20 p-3 font-mono text-sm text-muted-foreground overflow-x-auto">
            (1 − sentiment_weight) * fundamentals_score + sentiment_weight * sentiment_score
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            By default, <span className="font-medium text-foreground">sentiment_weight = 0.30</span>. We also calibrate using model confidence, and then map the
            final score into labels such as <span className="font-medium text-foreground">Bullish</span>,{" "}
            <span className="font-medium text-foreground">Mildly Bullish</span>,{" "}
            <span className="font-medium text-foreground">Mixed / Uncertain</span>,{" "}
            <span className="font-medium text-foreground">Mildly Bearish</span>, or{" "}
            <span className="font-medium text-foreground">Bearish</span>.
          </p>
        </section>

        <section className="space-y-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">“Model view” on Overview</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The Overview badge turns the tiers into plain language:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-lg border border-border/60 bg-secondary/20 p-3 text-sm text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground">Strong fundamentals</span> + <span className="font-semibold text-foreground">Bullish sentiment</span>
              <div className="mt-1 text-xs">Example phrase: “Strong fundamentals with bullish sentiment.”</div>
            </div>
            <div className="rounded-lg border border-border/60 bg-secondary/20 p-3 text-sm text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground">Weak fundamentals</span> + <span className="font-semibold text-foreground">Bearish sentiment</span>
              <div className="mt-1 text-xs">Example phrase: “Weak fundamentals with bearish sentiment.”</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            If one of the two scores is missing, we fall back to a short message like “sentiment unavailable”.
          </p>
        </section>
      </main>
    ),
    []
  );

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header onHome={() => navigate("/app")} />
      {content}
      <SiteFooter variant="app" />
    </div>
  );
}

