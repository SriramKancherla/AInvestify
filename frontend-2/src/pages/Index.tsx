import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeftRight, Bookmark, CalendarDays, Download, X } from "lucide-react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import ChartHero from "@/components/dashboard/ChartHero";
import StatGrid from "@/components/dashboard/StatGrid";
import AnalysisCard from "@/components/dashboard/AnalysisCard";
import FundamentalsList from "@/components/dashboard/FundamentalsList";
import NewsList from "@/components/dashboard/NewsList";
import CompareView from "@/components/dashboard/CompareView";
import ChatWidget from "@/components/dashboard/ChatWidget";
import EmptyState from "@/components/dashboard/EmptyState";
import SiteFooter from "@/components/SiteFooter";
import Monogram from "@/components/common/Monogram";
import { useStockStore } from "@/hooks/useStockStore";

function HeroSkeleton() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <div className="xl:col-span-2 space-y-4">
        <div className="elevated p-6"><div className="skeleton-shimmer h-[280px] w-full rounded-lg" /></div>
      </div>
      <div className="space-y-4">
        <div className="skeleton-shimmer h-40 w-full rounded-xl" />
        <div className="skeleton-shimmer h-52 w-full rounded-xl" />
      </div>
    </div>
  );
}

export default function Index() {
  const store = useStockStore();
  const navigate = useNavigate();
  const location = useLocation();
  const { symbol: symbolParam } = useParams<{ symbol?: string }>();
  const lastUrlTickerRef = useRef<string>("");
  const [showCompare, setShowCompare] = useState(false);
  const [compareInput, setCompareInput] = useState("");
  const [showWatchlistAdd, setShowWatchlistAdd] = useState(false);
  const [selectedWatchlist, setSelectedWatchlist] = useState("default");
  const [newWatchlistName, setNewWatchlistName] = useState("");

  const compareChoices = useMemo(
    () =>
      (store.allTickers.length ? store.allTickers : [])
        .filter((t) => t.symbol !== store.selectedTicker)
        .filter(
          (t) =>
            t.symbol.toLowerCase().includes(compareInput.toLowerCase()) ||
            t.name.toLowerCase().includes(compareInput.toLowerCase())
        )
        .slice(0, 40),
    [store.allTickers, store.selectedTicker, compareInput]
  );

  const handleSearch = (ticker: string) => {
    store.setSelectedTicker(ticker);
    store.fetchStock(ticker);
  };

  const handleHome = () => {
    lastUrlTickerRef.current = "";
    store.clearActiveStock();
    setShowCompare(false);
    navigate("/app", { replace: true });
  };

  const handleCompare = (ticker: string) => {
    store.setCompareTicker(ticker);
    store.fetchCompare(ticker);
    setCompareInput("");
  };

  const isFav = store.favorites.includes(store.selectedTicker);

  useEffect(() => {
    const routeSym = (symbolParam || "").trim().toUpperCase();
    if (routeSym) {
      if (lastUrlTickerRef.current === routeSym) return;
      lastUrlTickerRef.current = routeSym;
      store.setSelectedTicker(routeSym);
      store.fetchStock(routeSym);
      return;
    }
    const params = new URLSearchParams(location.search);
    const ticker = (params.get("ticker") || "").trim().toUpperCase();
    if (!ticker) return;
    if (lastUrlTickerRef.current === ticker) return;
    lastUrlTickerRef.current = ticker;
    store.setSelectedTicker(ticker);
    store.fetchStock(ticker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbolParam, location.search, store.fetchStock, store.setSelectedTicker]);

  const showEmpty = !store.stockData && !store.loading;

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header onHome={handleHome} crumb={store.stockData?.ticker} />

      <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6 space-y-6">
        {showEmpty ? (
          <EmptyState
            favorites={store.favorites}
            onSelect={handleSearch}
            allTickers={store.allTickers}
            footerAboveFavorites={<SiteFooter variant="app" className="mt-6" />}
          />
        ) : store.loading && !store.stockData ? (
          <HeroSkeleton />
        ) : store.stockData ? (
          <>
            {/* Actions toolbar */}
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-2">
              <button
                onClick={() => store.toggleFavorite(store.selectedTicker)}
                className={`inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border text-sm font-medium transition-colors ${
                  isFav ? "border-primary/30 bg-primary/8 text-primary" : "border-border bg-card text-muted-foreground hover:text-foreground"
                }`}
              >
                <Bookmark className={`w-4 h-4 ${isFav ? "fill-primary" : ""}`} />
                {isFav ? "Saved" : "Save"}
              </button>
              <button
                onClick={() => setShowWatchlistAdd(true)}
                className="inline-flex items-center h-9 px-3 rounded-lg border border-border bg-card text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                Add to watchlist
              </button>
              <button
                onClick={() => {
                  setShowCompare((p) => !p);
                  if (showCompare) store.setCompareTicker("");
                }}
                className={`inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border text-sm font-medium transition-colors ${
                  showCompare ? "border-primary/30 bg-primary/8 text-primary" : "border-border bg-card text-muted-foreground hover:text-foreground"
                }`}
              >
                <ArrowLeftRight className="w-4 h-4" />
                Compare
              </button>
              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={() => store.fetchEvents(store.selectedTicker)}
                  className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border border-border bg-card text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  <CalendarDays className="w-4 h-4" /> Events
                </button>
                <button
                  onClick={() =>
                    store.exportReport({
                      selected: store.stockData,
                      compare: store.compareData,
                      portfolio: store.portfolioSummary,
                      events: store.events,
                    })
                  }
                  className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border border-border bg-card text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Download className="w-4 h-4" /> Export PDF
                </button>
              </div>
            </motion.div>

            {showCompare ? (
              <CompareView
                primary={store.stockData}
                secondary={store.compareData}
                compareChoices={compareChoices}
                compareInput={compareInput}
                setCompareInput={setCompareInput}
                onSelectCompare={handleCompare}
                onClearCompare={() => store.setCompareTicker("")}
              />
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 space-y-4">
                  <ChartHero data={store.stockData} />
                  <FundamentalsList data={store.stockData} />
                  <NewsList data={store.stockData} />
                  {store.events.length > 0 && (
                    <div className="surface p-5">
                      <h3 className="text-sm font-semibold text-foreground mb-3">Upcoming events</h3>
                      <div className="flex gap-3 overflow-x-auto pb-1">
                        {store.events.map((e) => (
                          <div key={e.id} className="min-w-[180px] rounded-lg border border-border bg-secondary/40 p-3">
                            <p className="text-sm font-medium text-foreground">{e.event_type}</p>
                            <p className="text-xs text-muted-foreground mt-1">{e.event_date}</p>
                            <p className="text-[11px] text-muted-foreground mt-2">Source: {e.source || "N/A"}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <div className="space-y-4">
                  <StatGrid data={store.stockData} />
                  <AnalysisCard data={store.stockData} onSeeFull={() => navigate("/models")} />
                </div>
              </div>
            )}
          </>
        ) : null}
      </main>

      {store.stockData && !showCompare && <SiteFooter variant="app" />}
      <ChatWidget selectedSymbol={store.selectedTicker} />

      {/* Add-to-watchlist modal */}
      {showWatchlistAdd && store.stockData && (
        <div className="fixed inset-0 z-[95] bg-foreground/20 flex items-center justify-center p-4" onClick={() => setShowWatchlistAdd(false)}>
          <div className="w-full max-w-md rounded-2xl border border-border bg-card p-5 space-y-4 shadow-pop" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2.5">
                <Monogram symbol={store.stockData.ticker} size={36} />
                <div>
                  <p className="text-sm font-semibold text-foreground">Add {store.stockData.ticker}</p>
                  <p className="text-xs text-muted-foreground">to a watchlist</p>
                </div>
              </div>
              <button onClick={() => setShowWatchlistAdd(false)} className="h-8 w-8 rounded-lg border border-border flex items-center justify-center hover:bg-secondary" aria-label="Close">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex gap-2">
              <select
                value={selectedWatchlist}
                onChange={(e) => setSelectedWatchlist(e.target.value)}
                className="flex-1 h-11 px-3 rounded-lg border border-border bg-secondary text-sm"
              >
                {Object.keys(store.watchlists).map((wl) => (
                  <option key={wl} value={wl}>{wl}</option>
                ))}
              </select>
              <button
                onClick={() => {
                  const curr = store.watchlists[selectedWatchlist] || [];
                  store.saveWatchlist(selectedWatchlist, Array.from(new Set([...curr, store.stockData!.ticker])));
                  setShowWatchlistAdd(false);
                }}
                className="h-11 px-4 rounded-lg bg-primary text-primary-foreground text-sm font-medium"
              >
                Add
              </button>
            </div>
            <div className="flex gap-2">
              <input
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
                placeholder="Or create a new watchlist"
                className="flex-1 h-11 px-3 rounded-lg border border-transparent bg-secondary text-sm focus:outline-none focus:border-primary"
              />
              <button
                onClick={() => {
                  const nm = newWatchlistName.trim();
                  if (!nm) return;
                  store.saveWatchlist(nm, [store.stockData!.ticker]);
                  setSelectedWatchlist(nm);
                  setNewWatchlistName("");
                  setShowWatchlistAdd(false);
                }}
                className="h-11 px-4 rounded-lg border border-border bg-card text-sm font-medium hover:bg-secondary"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
