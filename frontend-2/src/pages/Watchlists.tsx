import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import Header from "@/components/dashboard/Header";
import SiteFooter from "@/components/SiteFooter";
import StockCard from "@/components/common/StockCard";
import { useStockStore } from "@/hooks/useStockStore";
import { apiJson } from "@/lib/api";

type HeatmapRow = { ticker: string; price: number; pe: number; sentiment: number; changePct: number };

/** Muted pastel gradient (light-mode-appropriate) for a -8%..+8% change. */
function heatColor(changePct: number): string {
  const clamped = Math.max(-8, Math.min(8, changePct)) / 8; // -1..1
  if (clamped >= 0) {
    const a = 0.10 + clamped * 0.28;
    return `hsl(152 60% 42% / ${a.toFixed(3)})`;
  }
  const a = 0.10 + Math.abs(clamped) * 0.28;
  return `hsl(4 70% 50% / ${a.toFixed(3)})`;
}

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
  const [tickerToAdd, setTickerToAdd] = useState("");
  const [tickerFilter, setTickerFilter] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [isHeatmapLoading, setIsHeatmapLoading] = useState(false);
  const [heatmapRows, setHeatmapRows] = useState<HeatmapRow[]>([]);

  const activeKey = active.toLowerCase() === "default" ? "default" : active;
  const tickers = store.watchlists[activeKey] || [];
  const tickerSet = new Set(tickers);

  const filteredOptions = useMemo(() => {
    const q = tickerFilter.trim().toLowerCase();
    const source = (store.allTickers || []).filter((t) => !tickerSet.has(t.symbol));
    if (!q) return source.slice(0, 60);
    return source.filter((t) => t.symbol.toLowerCase().includes(q) || (t.name || "").toLowerCase().includes(q)).slice(0, 60);
  }, [store.allTickers, tickerFilter, tickers]);

  useEffect(() => {
    if (!names.length) return;
    if (!names.includes(active)) setActive(names[0]);
  }, [names, active]);

  const nameOf = (sym: string) => store.allTickers.find((t) => t.symbol === sym)?.name || "";

  const addTicker = async (sym: string) => {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    const curr = store.watchlists[activeKey] || [];
    setIsSaving(true);
    try {
      await store.saveWatchlist(activeKey, Array.from(new Set([...curr, clean])));
    } finally {
      setIsSaving(false);
    }
    setTickerToAdd("");
  };

  const removeTicker = async (sym: string) => {
    const curr = store.watchlists[activeKey] || [];
    setIsSaving(true);
    try {
      await store.saveWatchlist(activeKey, curr.filter((t) => t !== sym));
    } finally {
      setIsSaving(false);
    }
  };

  const createWatchlist = () => {
    const nm = newName.trim();
    if (!nm || isSaving) return;
    setIsSaving(true);
    void store.saveWatchlist(nm, []).finally(() => setIsSaving(false));
    setActive(nm);
    setNewName("");
  };

  const generateHeatmap = async () => {
    setIsHeatmapLoading(true);
    const rows: HeatmapRow[] = [];
    for (const ticker of tickers) {
      const [fund, ins, chart] = await Promise.all([
        apiJson<{ metrics?: { current_price?: number; pe_ratio?: number } }>(`/fundamentals/${encodeURIComponent(ticker)}`).catch(
          () => ({} as { metrics?: { current_price?: number; pe_ratio?: number } })
        ),
        apiJson<{ sentiment?: { score?: number } }>(`/api/insights`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input: ticker, news_source: "auto", top_items: 3, sentiment_weight: 0.3, max_news: 10, train_missing: false }),
        }).catch(() => ({} as { sentiment?: { score?: number } })),
        apiJson<{ points?: Array<{ close?: number }> }>(`/chart/${encodeURIComponent(ticker)}?period=1mo&interval=1d`).catch(
          () => ({} as { points?: Array<{ close?: number }> })
        ),
      ]);
      const closes = (chart.points || []).map((p) => Number(p.close)).filter((n) => Number.isFinite(n));
      const changePct = closes.length > 1 ? ((closes[closes.length - 1] - closes[0]) / closes[0]) * 100 : 0;
      rows.push({
        ticker,
        price: Number(fund.metrics?.current_price || 0),
        pe: Number(fund.metrics?.pe_ratio || 0),
        sentiment: Number(ins.sentiment?.score || 0),
        changePct,
      });
    }
    setHeatmapRows(rows);
    setIsHeatmapLoading(false);
  };

  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      <Header onHome={() => navigate("/app")} crumb="Watchlists" />
      <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tightish text-foreground">Watchlists</h1>
          <p className="text-sm text-muted-foreground mt-1">Track groups of tickers and visualize them as a performance heatmap.</p>
        </div>

        {/* Underline tabs */}
        <div className="flex items-center gap-1 border-b border-border overflow-x-auto">
          {names.map((n) => (
            <button
              key={n}
              onClick={() => setActive(n)}
              className={`relative px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-colors ${
                active === n ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {n}
              {active === n && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full" />}
            </button>
          ))}
          <button
            onClick={() => setShowAdd((s) => !s)}
            className="ml-2 inline-flex items-center gap-1 px-3 py-1.5 text-sm text-primary hover:bg-primary/8 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" /> New watchlist
          </button>
        </div>

        {showAdd && (
          <div className="flex gap-2 max-w-md">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && createWatchlist()}
              placeholder="Watchlist name"
              className="flex-1 h-11 px-3.5 rounded-lg border border-transparent bg-secondary text-sm focus:outline-none focus:border-primary"
            />
            <button onClick={createWatchlist} disabled={isSaving} className="h-11 px-4 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-60">
              Create
            </button>
          </div>
        )}

        {/* Add ticker */}
        <div className="surface p-4 space-y-3">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <p className="text-sm font-medium text-foreground">Add tickers to <span className="font-mono">{active}</span></p>
            <input
              value={tickerFilter}
              onChange={(e) => setTickerFilter(e.target.value)}
              placeholder="Filter tickers…"
              className="h-9 px-3 rounded-lg border border-transparent bg-secondary text-sm focus:outline-none focus:border-primary min-w-[220px]"
            />
          </div>
          <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
            {filteredOptions.map((t) => (
              <button
                key={t.symbol}
                onClick={() => addTicker(t.symbol)}
                disabled={isSaving}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-border bg-card text-sm hover:border-primary/40 hover:bg-primary/5 transition-colors disabled:opacity-60"
              >
                <Plus className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="font-mono text-xs font-medium">{t.symbol}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Stock cards */}
        {tickers.length === 0 ? (
          <div className="surface p-10 text-center text-sm text-muted-foreground">No tickers in this watchlist yet.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {tickers.map((sym) => (
              <StockCard key={sym} symbol={sym} name={nameOf(sym)} onClick={(s) => navigate(`/stock/${encodeURIComponent(s)}`)} onRemove={removeTicker} />
            ))}
          </div>
        )}

        {tickers.length > 0 && (
          <button
            onClick={generateHeatmap}
            disabled={isHeatmapLoading}
            className="h-11 px-4 rounded-lg border border-border bg-card text-sm font-medium hover:bg-secondary transition-colors disabled:opacity-60"
          >
            {isHeatmapLoading ? "Generating…" : "Generate performance heatmap"}
          </button>
        )}

        {/* Heatmap grid */}
        {heatmapRows.length > 0 && (
          <div className="surface p-5">
            <h3 className="text-sm font-semibold text-foreground mb-4">Performance heatmap (1M)</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
              {heatmapRows.map((r) => (
                <button
                  key={r.ticker}
                  onClick={() => navigate(`/stock/${encodeURIComponent(r.ticker)}`)}
                  className="rounded-xl border border-border p-3 text-left transition-transform hover:-translate-y-0.5"
                  style={{ background: heatColor(r.changePct) }}
                >
                  <div className="font-mono text-sm font-semibold text-foreground">{r.ticker}</div>
                  <div className={`font-mono text-xs font-medium mt-1 ${r.changePct >= 0 ? "kpi-positive" : "kpi-negative"}`}>
                    {r.changePct >= 0 ? "+" : ""}{r.changePct.toFixed(2)}%
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-1.5">${r.price.toFixed(2)}</div>
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
      <SiteFooter variant="app" />
    </div>
  );
}
