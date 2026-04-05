import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeftRight, Download, Star, X } from "lucide-react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import KPICards from "@/components/dashboard/KPICards";
import TabContent from "@/components/dashboard/TabContent";
import ChatWidget from "@/components/dashboard/ChatWidget";
import EmptyState from "@/components/dashboard/EmptyState";
import SiteFooter from "@/components/SiteFooter";
import { useStockStore } from "@/hooks/useStockStore";

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
        .filter((t) => t.symbol.toLowerCase().includes(compareInput.toLowerCase()) || t.name.toLowerCase().includes(compareInput.toLowerCase())),
    [store.allTickers, store.selectedTicker, compareInput]
  );
  const recentCompareChoices = useMemo(
    () => store.recentStocks.filter((t) => t !== store.selectedTicker).slice(0, 6),
    [store.recentStocks, store.selectedTicker]
  );

  const handleSearch = (ticker: string) => {
    store.setSelectedTicker(ticker);
    store.fetchStock(ticker);
  };

  const handleHome = () => {
    lastUrlTickerRef.current = "";
    store.clearActiveStock();
    navigate("/app", { replace: true });
  };

  const handleCompare = (ticker: string) => {
    store.setCompareTicker(ticker);
    store.fetchCompare(ticker);
    setShowCompare(true);
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
    // Intentionally omit `store` — only stable actions are listed to avoid refetch loops.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbolParam, location.search, store.fetchStock, store.setSelectedTicker]);

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header
        onHome={handleHome}
      />

      <main className="container py-6 space-y-6">
        {!store.stockData && !store.loading ? (
          <EmptyState
            favorites={store.favorites}
            onSelect={handleSearch}
            onOpenWatchlists={() => navigate("/watchlists")}
            onOpenModels={() => navigate("/models")}
            allTickers={store.allTickers}
            footerAboveFavorites={<SiteFooter variant="app" className="mt-4" />}
          />
        ) : (
          <>
            {store.stockData && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center gap-3">
                <h1 className="text-2xl font-bold text-foreground">{store.stockData.company_name}</h1>
                <span className="ticker-badge text-sm">{store.stockData.ticker}</span>
                <div className="flex items-center gap-2 ml-auto">
                  <button
                    onClick={() => store.toggleFavorite(store.selectedTicker)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-all cursor-pointer
                      ${isFav ? "border-warning/30 bg-warning/10 text-warning" : "border-border bg-secondary/50 text-muted-foreground hover:text-foreground"}`}
                  >
                    <Star className={`w-3.5 h-3.5 ${isFav ? "fill-warning" : ""}`} />
                    {isFav ? "Saved" : "Save"}
                  </button>
                  <button
                    onClick={() => setShowWatchlistAdd((p) => !p)}
                    className="px-3 py-1.5 rounded-lg border border-border bg-secondary/50 text-sm font-medium text-muted-foreground hover:text-foreground transition-all cursor-pointer"
                  >
                    Add to Watchlist
                  </button>
                  {!showCompare ? (
                    <button
                      onClick={() => setShowCompare(true)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-secondary/50 text-sm font-medium text-muted-foreground hover:text-foreground transition-all cursor-pointer"
                    >
                      <ArrowLeftRight className="w-3.5 h-3.5" />
                      Compare
                    </button>
                  ) : (
                    <button
                      onClick={() => { setShowCompare(false); store.setCompareTicker(""); }}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-destructive/30 bg-destructive/10 text-sm font-medium text-destructive transition-all cursor-pointer"
                    >
                      <X className="w-3.5 h-3.5" />
                      Close Compare
                    </button>
                  )}
                </div>
              </motion.div>
            )}

            {showWatchlistAdd && store.stockData && (
              <div className="fixed inset-0 z-[90] bg-black/40 flex items-center justify-center p-4">
                <div className="w-full max-w-lg rounded-xl border border-border bg-card p-4 space-y-3 shadow-xl">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm text-muted-foreground">
                      Add `{store.stockData.ticker}` to an existing watchlist or create a new one.
                    </p>
                    <button
                      type="button"
                      onClick={() => setShowWatchlistAdd(false)}
                      className="inline-flex h-7 w-7 items-center justify-center rounded border border-border hover:bg-secondary/40"
                      aria-label="Close add-to-watchlist popup"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <select value={selectedWatchlist} onChange={(e) => setSelectedWatchlist(e.target.value)} className="h-9 px-2 rounded border border-border bg-secondary/50 text-sm">
                      {Object.keys(store.watchlists).map((wl) => (
                        <option key={wl} value={wl}>{wl}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => {
                        const curr = store.watchlists[selectedWatchlist] || [];
                        store.saveWatchlist(selectedWatchlist, Array.from(new Set([...curr, store.stockData!.ticker])));
                        setShowWatchlistAdd(false);
                      }}
                      className="px-3 py-1.5 rounded border border-border text-sm"
                    >
                      Add to Selected
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <input
                      value={newWatchlistName}
                      onChange={(e) => setNewWatchlistName(e.target.value)}
                      placeholder="New watchlist name"
                      className="h-9 px-2 rounded border border-border bg-secondary/50 text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const name = newWatchlistName.trim();
                        if (!name) return;
                        store.saveWatchlist(name, [store.stockData!.ticker]);
                        setSelectedWatchlist(name);
                        setNewWatchlistName("");
                        setShowWatchlistAdd(false);
                      }}
                      className="px-3 py-1.5 rounded border border-border text-sm"
                    >
                      Create + Add
                    </button>
                  </div>
                </div>
              </div>
            )}

            {showCompare && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="glass-card p-4">
                <p className="text-sm text-muted-foreground mb-3">Select a stock to compare with {store.selectedTicker}:</p>
                <input
                  value={compareInput}
                  onChange={(e) => setCompareInput(e.target.value)}
                  placeholder="Search compare ticker..."
                  className="w-full h-9 px-3 mb-3 rounded-lg border border-border bg-secondary/50 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                />
                {recentCompareChoices.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs text-muted-foreground mb-1">Recent</p>
                    <div className="flex flex-wrap gap-2">
                      {recentCompareChoices.map((t) => (
                        <button key={`recent-${t}`} onClick={() => handleCompare(t)} className="px-2.5 py-1 rounded-md border border-border bg-secondary/50 text-xs font-mono text-foreground hover:border-primary/30 transition-all cursor-pointer">
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <div className="max-h-56 overflow-y-auto border border-border rounded-lg">
                  {compareChoices.map((t) => (
                    <button key={t.symbol} onClick={() => handleCompare(t.symbol)} className="w-full text-left px-3 py-2 text-sm font-mono font-medium text-foreground hover:bg-secondary/70 transition-all cursor-pointer border-b border-border last:border-b-0">
                      {t.label}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {showCompare && store.compareData ? (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div className="space-y-4 rounded-xl border border-border p-3">
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Primary · {store.stockData?.ticker}</h3>
                  <KPICards data={store.stockData} loading={store.loading} />
                  <TabContent data={store.stockData} loading={store.loading} />
                </div>
                <div className="space-y-4 rounded-xl border border-border p-3">
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Compare · {store.compareData?.ticker}</h3>
                  <KPICards data={store.compareData} loading={false} />
                  <TabContent data={store.compareData} loading={false} />
                </div>
              </div>
            ) : (
              <div className="space-y-4 rounded-xl border border-border p-3">
                <KPICards data={store.stockData} loading={store.loading} />
                <TabContent data={store.stockData} loading={store.loading} />
              </div>
            )}

            <div className="rounded-xl border border-border p-4 space-y-3">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Event Calendar</h3>
              <button type="button" onClick={() => store.fetchEvents(store.selectedTicker || "AAPL")} className="px-3 py-1.5 rounded border border-border text-sm">Load Detailed Events</button>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {(store.events || []).map((e) => (
                  <div key={e.id} className="rounded-lg border border-border p-3 bg-secondary/20">
                    <p className="text-sm font-semibold text-foreground">{e.event_type}</p>
                    <p className="text-xs text-muted-foreground mt-1">{e.event_date}</p>
                    <p className="text-[11px] text-muted-foreground mt-2">Source: {e.source || "N/A"}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-border p-4 space-y-2">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Export report</h3>
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() =>
                    store.exportReport({
                      selected: store.stockData,
                      compare: store.compareData,
                      portfolio: store.portfolioSummary,
                      events: store.events,
                    })
                  }
                  className="flex items-center gap-2 px-3 py-1.5 rounded border border-border text-sm hover:bg-secondary/40 transition-colors"
                >
                  <Download className="w-4 h-4" /> Export report to device (PDF)
                </button>
              </div>
            </div>
          </>
        )}
      </main>

      {(store.stockData || store.loading) ? <SiteFooter variant="app" /> : null}
      <ChatWidget selectedSymbol={store.selectedTicker} />
    </div>
  );
}
