import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import { useStockStore } from "@/hooks/useStockStore";
import { apiJson } from "@/lib/api";
import { resolveTickerForInsights } from "@/lib/tickers";
import SiteFooter from "@/components/SiteFooter";

type HeatmapRow = { ticker: string; price: number; pe: number; sentiment: number };

export default function WatchlistsPage() {
  const store = useStockStore();
  const navigate = useNavigate();
  const names = useMemo(() => {
    const all = Object.keys(store.watchlists);
    const filtered = all.filter((n) => n.toLowerCase() !== "default");
    return ["Default", ...filtered];
  }, [store.watchlists]);
  const [active, setActive] = useState(names[0] || "default");
  const [newName, setNewName] = useState("");
  const [tickerToEdit, setTickerToEdit] = useState<string>("");
  const [tickerFilter, setTickerFilter] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isHeatmapLoading, setIsHeatmapLoading] = useState(false);
  const [heatmapRows, setHeatmapRows] = useState<HeatmapRow[]>([]);
  const [watchlistNavError, setWatchlistNavError] = useState<string | null>(null);

  const activeKey = active.toLowerCase() === "default" ? "default" : active;
  const tickers = store.watchlists[activeKey] || [];

  const tickerSet = new Set<string>(tickers);
  const filteredTickerOptions = useMemo(() => {
    const q = tickerFilter.trim().toLowerCase();
    const source = store.allTickers || [];
    if (!q) return source;
    return source.filter((t) => t.symbol.toLowerCase().includes(q) || (t.name || "").toLowerCase().includes(q));
  }, [store.allTickers, tickerFilter]);

  useEffect(() => {
    if (!names.length) return;
    if (!names.includes(active)) {
      setActive(names[0]);
    }
  }, [names, active]);

  const toggleTickerInWatchlist = async (tickerSymbol: string) => {
    setWatchlistNavError(null);
    const sym = tickerSymbol.trim().toUpperCase();
    if (!sym) {
      setWatchlistNavError("Select a ticker.");
      return;
    }

    const curr = store.watchlists[activeKey] || [];
    const inWatchlist = curr.includes(sym);
    const next = inWatchlist ? curr.filter((t) => t !== sym) : Array.from(new Set([...curr, sym]));
    setIsSaving(true);
    try {
      await store.saveWatchlist(activeKey, next);
    } finally {
      setIsSaving(false);
    }
    setTickerToEdit("");
  };

  const generateHeatmap = async () => {
    setIsHeatmapLoading(true);
    const rows: HeatmapRow[] = [];
    for (const ticker of tickers) {
      const [fund, ins] = await Promise.all([
        apiJson<{ metrics?: { current_price?: number; pe_ratio?: number } }>(`/fundamentals/${encodeURIComponent(ticker)}`).catch(() => ({})),
        apiJson<{ sentiment?: { score?: number } }>(`/api/insights`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input: ticker, news_source: "auto", top_items: 3, sentiment_weight: 0.3, max_news: 10, train_missing: false }),
        }).catch(() => ({})),
      ]);
      rows.push({
        ticker,
        price: Number(fund.metrics?.current_price || 0),
        pe: Number(fund.metrics?.pe_ratio || 0),
        sentiment: Number(ins.sentiment?.score || 0),
      });
    }
    setHeatmapRows(rows);
    setIsHeatmapLoading(false);
  };

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header
        onHome={() => navigate("/app")}
      />
      <main className="container py-6 space-y-5">
        <div className="rounded-xl border border-border bg-card p-4">
          <h1 className="text-2xl font-bold text-foreground">Watchlists</h1>
          <p className="text-sm text-muted-foreground mt-1">Create multiple watchlists and generate heatmap views for each list.</p>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 space-y-3">
          <div className="flex flex-wrap gap-2">
            {names.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setActive(n)}
                className={`px-3 py-1.5 rounded border text-sm transition-colors ${
                  active === n ? "border-primary text-primary bg-primary/10" : "border-border hover:bg-secondary/30"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
          <div className="flex gap-2 flex-wrap">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                const name = newName.trim();
                if (!name || isSaving) return;
                setIsSaving(true);
                void store.saveWatchlist(name, []).finally(() => {
                  setIsSaving(false);
                });
                setActive(name);
                setNewName("");
              }}
              placeholder="Create watchlist"
              className="h-9 px-2 rounded border border-border bg-secondary/40 text-sm focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
            <button
              type="button"
              onClick={() => {
                const name = newName.trim();
                if (!name || isSaving) return;
                setIsSaving(true);
                void store.saveWatchlist(name, []).finally(() => {
                  setIsSaving(false);
                });
                setActive(name);
                setNewName("");
              }}
              className="px-3 py-1.5 rounded border border-primary/40 bg-primary/10 text-primary text-sm hover:bg-primary/15 transition-colors disabled:opacity-60"
              disabled={isSaving}
            >
              {isSaving ? "Saving..." : "Create"}
            </button>
          </div>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <p className="text-sm text-muted-foreground">
              Selected watchlist: <span className="font-semibold text-foreground">{active}</span>
              <span className="ml-2 text-xs text-muted-foreground">({tickers.length} ticker{tickers.length === 1 ? "" : "s"})</span>
            </p>
            {activeKey !== "default" && (
              <button
                type="button"
                onClick={() => {
                  if (isSaving) return;
                  setIsSaving(true);
                  void store.saveWatchlist(activeKey, []).finally(() => setIsSaving(false));
                }}
                className="px-2.5 py-1 rounded border border-destructive/30 bg-destructive/10 text-destructive text-xs hover:bg-destructive/15 transition-colors disabled:opacity-60"
                disabled={isSaving || tickers.length === 0}
              >
                Clear all tickers
              </button>
            )}
          </div>
          {watchlistNavError && <p className="text-sm text-destructive">{watchlistNavError}</p>}

          <div className="rounded-lg border border-border bg-card p-3 text-sm space-y-3">
            <p className="text-xs text-muted-foreground">
              Add tickers to <span className="font-semibold text-foreground">{active}</span>
            </p>
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">All stocks</p>
              <input
                value={tickerFilter}
                onChange={(e) => setTickerFilter(e.target.value)}
                placeholder="Filter tickers or company"
                className="h-8 px-2 rounded border border-border bg-secondary/40 text-xs focus:outline-none focus:ring-1 focus:ring-primary/40 min-w-[220px]"
              />
              <div className="flex gap-2 items-center flex-1 justify-end">
                <select
                  value={tickerToEdit}
                  onChange={(e) => setTickerToEdit(e.target.value)}
                  className="h-9 px-2 rounded border border-border bg-secondary/40 text-sm focus:outline-none focus:ring-1 focus:ring-primary/40 min-w-[220px]"
                >
                  <option value="">Select ticker...</option>
                  {filteredTickerOptions.map((t) => {
                    const inWatchlist = tickerSet.has(t.symbol);
                    return (
                      <option key={t.symbol} value={t.symbol}>
                        {t.name ? `${t.symbol} - ${t.name}${inWatchlist ? " (in watchlist)" : ""}` : `${t.symbol}${inWatchlist ? " (in watchlist)" : ""}`}
                      </option>
                    );
                  })}
                </select>
                <button
                  type="button"
                  onClick={() => void toggleTickerInWatchlist(tickerToEdit)}
                  className="px-3 py-1.5 rounded border border-primary/40 bg-primary/10 text-primary text-sm hover:bg-primary/15 transition-colors disabled:opacity-60"
                  disabled={!tickerToEdit || isSaving}
                >
                  {isSaving ? "Saving..." : tickerToEdit && tickerSet.has(tickerToEdit.trim().toUpperCase()) ? "Remove" : "Add"}
                </button>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-secondary/20 p-3 text-sm">
            <p className="text-xs text-muted-foreground mb-1">Tickers in this watchlist</p>
            {tickers.length === 0 ? (
              <p>none yet</p>
            ) : (
              <div className="space-y-1">
                {tickers.map((ticker) => (
                  <div key={ticker} className="flex items-center justify-between gap-2">
                    <button
                      type="button"
                      onClick={async () => {
                        setWatchlistNavError(null);
                        const sym = await resolveTickerForInsights(ticker);
                        if (!sym) {
                          setWatchlistNavError("Could not resolve that ticker.");
                          return;
                        }
                        navigate(`/stock/${encodeURIComponent(sym)}`);
                      }}
                      className="flex-1 min-w-0 text-left px-2 py-1 rounded border border-border bg-background hover:bg-secondary/40 transition-colors font-mono text-sm"
                      title="Open ticker insights"
                    >
                      {ticker}
                    </button>
                    <button
                      type="button"
                      onClick={() => void toggleTickerInWatchlist(ticker)}
                      className="px-2 py-1 rounded border border-destructive/30 bg-destructive/10 text-destructive text-sm hover:bg-destructive/15 transition-colors shrink-0 disabled:opacity-60"
                      title="Remove from watchlist"
                      disabled={isSaving}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={generateHeatmap}
            className="px-3 py-1.5 rounded border border-border text-sm hover:bg-secondary/40 transition-colors disabled:opacity-60"
            disabled={isHeatmapLoading || tickers.length === 0}
          >
            {isHeatmapLoading ? "Generating..." : "Generate Heatmap for this Watchlist"}
          </button>
        </div>

        {heatmapRows.length > 0 && (
          <div className="rounded-xl border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-2 text-foreground">Watchlist Heatmap</h3>
            <div className="overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-secondary/30">
                  <tr className="border-b border-border text-left">
                    <th className="py-2 px-3">Ticker</th>
                    <th className="py-2 px-3">Price</th>
                    <th className="py-2 px-3">P/E</th>
                    <th className="py-2 px-3">Sentiment</th>
                  </tr>
                </thead>
                <tbody>
                  {heatmapRows.map((r) => (
                    <tr key={r.ticker} className="border-b border-border hover:bg-secondary/20">
                      <td className="py-2 px-3 font-mono">{r.ticker}</td>
                      <td className="py-2 px-3">${r.price.toFixed(2)}</td>
                      <td className="py-2 px-3">{r.pe.toFixed(2)}</td>
                      <td className={`py-2 px-3 ${r.sentiment >= 0.6 ? "text-green-500" : r.sentiment <= 0.4 ? "text-red-500" : "text-yellow-500"}`}>
                        {r.sentiment.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
      <SiteFooter variant="app" />
    </div>
  );
}
