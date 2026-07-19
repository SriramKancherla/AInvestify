import { motion } from "framer-motion";
import type { ReactNode } from "react";
import type { TickerOption } from "@/hooks/useStockStore";
import { POPULAR_TICKERS } from "@/hooks/useStockStore";
import SearchBar from "@/components/common/SearchBar";
import StockCard from "@/components/common/StockCard";

interface EmptyStateProps {
  favorites: string[];
  onSelect: (ticker: string) => void;
  allTickers: TickerOption[];
  footerAboveFavorites?: ReactNode;
}

export default function EmptyState({ favorites, onSelect, allTickers, footerAboveFavorites }: EmptyStateProps) {
  const nameOf = (sym: string) => allTickers.find((t) => t.symbol === sym)?.name || "";
  const popular = POPULAR_TICKERS.filter((s) => !favorites.includes(s));

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-10">
      {/* Hero band */}
      <div className="rounded-2xl border border-border bg-card px-6 py-12 sm:py-16 flex flex-col items-center text-center">
        <h1 className="text-3xl sm:text-[32px] font-semibold tracking-tightish text-foreground">
          AI-powered stock insights, in seconds
        </h1>
        <p className="mt-2 text-[15px] text-muted-foreground max-w-md">
          Fundamentals, news sentiment, and a blended AI score for any ticker.
        </p>
        <div className="mt-6 w-full flex justify-center">
          <SearchBar size="lg" />
        </div>
      </div>

      {favorites.length > 0 && (
        <section className="space-y-3">
          <h2 className="label-caps">Favorites</h2>
          <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1">
            {favorites.map((sym) => (
              <div key={sym} className="min-w-[280px]">
                <StockCard symbol={sym} name={nameOf(sym)} onClick={onSelect} />
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="space-y-3">
        <h2 className="label-caps">Popular right now</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {popular.map((sym) => (
            <StockCard key={sym} symbol={sym} name={nameOf(sym)} onClick={onSelect} />
          ))}
        </div>
      </section>

      {footerAboveFavorites}
    </motion.div>
  );
}
