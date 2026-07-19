import { motion } from "framer-motion";
import { X } from "lucide-react";
import Monogram from "./Monogram";
import Sparkline from "./Sparkline";
import { useMiniQuote } from "@/hooks/useMiniQuote";
import { formatPrice, formatPct } from "@/lib/format";

interface StockCardProps {
  symbol: string;
  name?: string;
  onClick?: (symbol: string) => void;
  onRemove?: (symbol: string) => void;
  compact?: boolean;
}

/** Rich ticker card: monogram, name, live price + change, mini sparkline. */
export default function StockCard({ symbol, name, onClick, onRemove, compact = false }: StockCardProps) {
  const q = useMiniQuote(symbol);
  const isUp = (q.changePct ?? 0) >= 0;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 30 }}
      className="group relative"
    >
      <button
        type="button"
        onClick={() => onClick?.(symbol)}
        className="surface-interactive w-full text-left p-4 flex items-center gap-3 cursor-pointer"
      >
        <Monogram symbol={symbol} size={compact ? 34 : 40} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-semibold text-foreground">{symbol}</span>
          </div>
          {name && <div className="truncate text-xs text-muted-foreground mt-0.5">{name}</div>}
        </div>

        {!compact && (
          <div className="w-24 h-8 shrink-0">
            {q.loading ? (
              <div className="skeleton-shimmer h-full w-full" />
            ) : (
              <Sparkline data={q.spark} up={isUp} fill width={96} height={32} />
            )}
          </div>
        )}

        <div className="text-right shrink-0 min-w-[76px]">
          {q.loading ? (
            <>
              <div className="skeleton-shimmer h-4 w-16 ml-auto mb-1" />
              <div className="skeleton-shimmer h-3 w-10 ml-auto" />
            </>
          ) : (
            <>
              <div className="font-mono text-sm font-semibold text-foreground">{formatPrice(q.price)}</div>
              <div className={`font-mono text-xs font-medium ${isUp ? "kpi-positive" : "kpi-negative"}`}>
                {formatPct(q.changePct)}
              </div>
            </>
          )}
        </div>
      </button>

      {onRemove && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(symbol);
          }}
          className="absolute -top-2 -right-2 h-6 w-6 rounded-full bg-card border border-border text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center hover:text-destructive hover:border-destructive/30 shadow-sm"
          aria-label={`Remove ${symbol}`}
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </motion.div>
  );
}
