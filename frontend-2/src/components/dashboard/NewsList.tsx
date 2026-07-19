import { ExternalLink } from "lucide-react";
import type { StockData } from "@/hooks/useStockStore";

interface NewsListProps {
  data: StockData;
}

function dotColor(sentiment: string): string {
  const s = sentiment.toLowerCase();
  if (s === "positive" || s === "bullish") return "hsl(var(--success))";
  if (s === "negative" || s === "bearish") return "hsl(var(--destructive))";
  return "hsl(var(--warning))";
}

export default function NewsList({ data }: NewsListProps) {
  const news = data.news || [];
  if (!news.length) return null;

  return (
    <div className="surface p-5">
      <h3 className="text-sm font-semibold text-foreground mb-3">Recent news</h3>
      <div className="space-y-1">
        {news.map((item, i) => (
          <a
            key={i}
            href={item.link || undefined}
            target={item.link ? "_blank" : undefined}
            rel="noopener noreferrer"
            className={`flex items-start gap-3 py-2.5 px-2 -mx-2 rounded-lg transition-colors ${item.link ? "hover:bg-secondary" : ""}`}
          >
            <span className="mt-1.5 h-2 w-2 rounded-full shrink-0" style={{ background: dotColor(item.sentiment) }} />
            <div className="min-w-0 flex-1">
              <p className="text-sm text-foreground leading-snug">{item.title}</p>
              <p className="text-xs text-muted-foreground mt-0.5 capitalize">{item.source}</p>
            </div>
            {item.link && <ExternalLink className="w-3.5 h-3.5 text-muted-foreground shrink-0 mt-1" />}
          </a>
        ))}
      </div>
    </div>
  );
}
