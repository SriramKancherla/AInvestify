interface SentimentBarProps {
  /** 0–1 sentiment score. */
  score: number | null | undefined;
  label?: string;
}

/** Horizontal bearish/neutral/bullish split with a marker at the current score. */
export default function SentimentBar({ score, label }: SentimentBarProps) {
  const s = score == null || !Number.isFinite(score) ? null : Math.max(0, Math.min(1, score));
  const pct = s == null ? 50 : s * 100;

  return (
    <div className="space-y-2">
      <div className="relative h-2 rounded-full overflow-hidden bg-secondary">
        <div className="absolute inset-0 flex">
          <div className="h-full" style={{ width: "40%", background: "hsl(var(--destructive) / 0.25)" }} />
          <div className="h-full" style={{ width: "20%", background: "hsl(var(--warning) / 0.25)" }} />
          <div className="h-full" style={{ width: "40%", background: "hsl(var(--success) / 0.25)" }} />
        </div>
        {s != null && (
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3.5 w-3.5 rounded-full border-2 border-card shadow-sm transition-all duration-500"
            style={{
              left: `${pct}%`,
              background: pct >= 60 ? "hsl(var(--success))" : pct >= 40 ? "hsl(var(--warning))" : "hsl(var(--destructive))",
            }}
          />
        )}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>Bearish</span>
        <span>Neutral</span>
        <span>Bullish</span>
      </div>
      {label && <p className="text-xs text-muted-foreground">{label}</p>}
    </div>
  );
}
