import type { StockData } from "@/hooks/useStockStore";

interface FundamentalsListProps {
  data: StockData;
}

// Score rows live in the AnalysisCard; keep this list to hard financial metrics.
const SCORE_KEYS = new Set(["Final Score", "Fundamentals Score", "Sentiment Score"]);

export default function FundamentalsList({ data }: FundamentalsListProps) {
  const entries = Object.entries(data.fundamentals || {}).filter(([k]) => !SCORE_KEYS.has(k));
  if (!entries.length) return null;

  return (
    <div className="surface p-5">
      <h3 className="text-sm font-semibold text-foreground mb-3">Fundamentals</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8">
        {entries.map(([key, val]) => (
          <div key={key} className="flex items-center justify-between py-2.5 border-b border-border/70 last:border-b-0">
            <span className="text-sm text-muted-foreground">{key}</span>
            <span className="font-mono text-sm font-medium text-foreground">{val}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
