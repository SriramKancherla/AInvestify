import type { StockData } from "@/hooks/useStockStore";
import AIScoreGauge from "@/components/common/AIScoreGauge";
import SentimentBar from "@/components/common/SentimentBar";

interface AnalysisCardProps {
  data: StockData;
  onSeeFull?: () => void;
}

function num(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** AI Score gauge + one-line read + sentiment split. */
export default function AnalysisCard({ data, onSeeFull }: AnalysisCardProps) {
  const finalRaw = num(data.fundamentals?.["Final Score"]);
  const fundRaw = num(data.fundamentals?.["Fundamentals Score"]);
  const base = finalRaw ?? fundRaw;
  const score100 = base == null ? null : Math.round(base * 100);
  const sentiment01 = data.sentiment_score != null ? data.sentiment_score / 100 : null;

  return (
    <div className="surface p-5 space-y-5">
      <div className="flex items-center gap-4">
        <AIScoreGauge score={score100} label="AI Score" />
        <div className="min-w-0">
          <p className="label-caps">Model view</p>
          <p className="text-sm font-medium text-foreground mt-1 leading-snug">{data.recommendation || "—"}</p>
          {onSeeFull && (
            <button type="button" onClick={onSeeFull} className="mt-2 text-xs font-medium text-primary hover:underline">
              See full analysis →
            </button>
          )}
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <p className="label-caps">News sentiment</p>
          <span className="text-xs font-medium text-foreground">{data.sentiment_label || "—"}</span>
        </div>
        <SentimentBar score={sentiment01} />
      </div>
    </div>
  );
}
