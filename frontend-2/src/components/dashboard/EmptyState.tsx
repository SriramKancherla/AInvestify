import { motion } from "framer-motion";
import { TrendingUp, Star } from "lucide-react";
import type { ReactNode } from "react";
import type { TickerOption } from "@/hooks/useStockStore";

interface EmptyStateProps {
  favorites: string[];
  onSelect: (ticker: string) => void;
  allTickers: TickerOption[];
  onOpenWatchlists: () => void;
  onOpenModels: () => void;
  footerAboveFavorites?: ReactNode;
}

export default function EmptyState({
  favorites,
  onSelect,
  allTickers,
  onOpenWatchlists,
  onOpenModels,
  footerAboveFavorites,
}: EmptyStateProps) {
  const sortedTickers = [...allTickers].sort((a, b) => a.symbol.localeCompare(b.symbol));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center pt-6 pb-12 sm:pt-8 sm:pb-16 px-4 text-center -mt-2"
    >
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
        <TrendingUp className="w-8 h-8 text-primary" />
      </div>
      <h2 className="text-xl font-bold text-foreground mb-2">Welcome to AInvestify</h2>
      <p className="text-sm text-muted-foreground mb-6 max-w-md">
        AI-powered stock insights at your fingertips. Pick a ticker below to get started with real-time analysis, sentiment data, and smart comparisons.
      </p>

      <div className="w-full max-w-xl mb-6 text-left space-y-2">
        <p className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">All stocks</p>
        <select
          onChange={(e) => {
            const sym = e.target.value;
            if (!sym) return;
            onSelect(sym);
            e.target.value = "";
          }}
          defaultValue=""
          className="w-full h-10 px-3 rounded-lg border border-border bg-secondary/50 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50"
        >
          <option value="" disabled>
            Select a ticker
          </option>
          {sortedTickers.map((t) => (
            <option key={t.symbol} value={t.symbol}>
              {t.symbol} - {t.name || t.symbol}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2 mb-6">
        <button
          type="button"
          onClick={onOpenWatchlists}
          className="px-3 py-1.5 rounded-lg border border-border bg-secondary/50 text-sm hover:border-primary/40 transition-colors"
        >
          Open Watchlists
        </button>
        <button
          type="button"
          onClick={onOpenModels}
          className="px-3 py-1.5 rounded-lg border border-border bg-secondary/50 text-sm hover:border-primary/40 transition-colors"
        >
          Open Models & Scoring
        </button>
      </div>

      {favorites.length > 0 && (
        <div className="mb-6">
          <span className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">Favorites</span>
          <div className="flex flex-wrap justify-center gap-2 mt-2">
            {favorites.map(t => (
              <button key={t} onClick={() => onSelect(t)} className="ticker-badge hover:bg-primary/20 transition-colors cursor-pointer">
                <Star className="w-3 h-3 mr-1 fill-warning text-warning" />{t}
              </button>
            ))}
          </div>
        </div>
      )}

      {footerAboveFavorites}
    </motion.div>
  );
}
