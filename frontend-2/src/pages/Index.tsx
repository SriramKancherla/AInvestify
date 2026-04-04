import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeftRight, BellRing, Download, Mail, Star, X } from "lucide-react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import Header from "@/components/dashboard/Header";
import KPICards from "@/components/dashboard/KPICards";
import TabContent from "@/components/dashboard/TabContent";
import ChatWidget from "@/components/dashboard/ChatWidget";
import EmptyState from "@/components/dashboard/EmptyState";
import SiteFooter from "@/components/SiteFooter";
import { useStockStore } from "@/hooks/useStockStore";
import { useAuth } from "@/contexts/AuthContext";

export default function Index() {
  const store = useStockStore();
  const { session } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { symbol: symbolParam } = useParams<{ symbol?: string }>();
  const lastUrlTickerRef = useRef<string>("");
  const [showCompare, setShowCompare] = useState(false);
  const [compareInput, setCompareInput] = useState("");
  const [alertThreshold, setAlertThreshold] = useState("100");
  const [alertType, setAlertType] = useState<"price_above" | "price_below">("price_above");
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

  /** Same symbol as the insights header; read-only for alerts. */
  const alertPageTicker = useMemo(
    () => (store.stockData?.ticker || store.selectedTicker || "").trim().toUpperCase(),
    [store.stockData?.ticker, store.selectedTicker]
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

  const { stockData, evaluateAlertsNow, refreshAlerts } = store;

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

  // Re-check all pending one-shot rules when a signed-in user views a stock (no extra button).
  useEffect(() => {
    if (!session?.user?.id || !stockData?.ticker) return;
    let cancelled = false;
    void (async () => {
      try {
        await evaluateAlertsNow();
        if (!cancelled) await refreshAlerts();
      } catch {
        /* ignore — alerts are optional */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session?.user?.id, stockData?.ticker, evaluateAlertsNow, refreshAlerts]);

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
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-2"><BellRing className="w-4 h-4" /> Price alerts</h3>
              <p className="text-xs text-muted-foreground">
                Use <span className="font-medium text-foreground">Set one-time alert</span> to add one rule for the stock on this page. We use the latest daily close in our feed: if the rule is already satisfied, we email you once at your sign-in address and then remove the rule. If not, it stays pending until a later visit to this app fires it once, then it is deleted.
              </p>
              <div className="rounded-lg border border-border bg-secondary/25 p-3">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">Stock for this alert</p>
                <div
                  className="flex min-h-[2.75rem] items-center justify-center rounded-md border border-border/80 bg-background px-4 py-2"
                  role="status"
                  title="Matches the stock on this insights page; cannot be edited."
                >
                  <span className="pointer-events-none select-none text-xl font-bold font-mono tracking-tight text-foreground tabular-nums">
                    {alertPageTicker || "—"}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground">Notifications go to your account email (the one you signed in with).</p>
              <div className="flex flex-wrap gap-2 items-center">
                <select value={alertType} onChange={(e) => setAlertType(e.target.value as "price_above" | "price_below")} className="h-9 px-2 rounded border border-border bg-secondary/50 text-sm">
                  <option value="price_above">At or above</option>
                  <option value="price_below">At or below</option>
                </select>
                <input value={alertThreshold} onChange={(e) => setAlertThreshold(e.target.value)} placeholder="Threshold" className="w-24 h-9 px-2 rounded border border-border bg-secondary/50 text-sm" />
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      if (!alertPageTicker) {
                        window.alert("Open a stock first to set an alert.");
                        return;
                      }
                      const thresholdNum = Number(alertThreshold);
                      if (!Number.isFinite(thresholdNum) || thresholdNum <= 0) {
                        window.alert("Enter a valid positive threshold price.");
                        return;
                      }
                      const res = await store.createAlertRule({
                        ticker: alertPageTicker,
                        rule_type: alertType,
                        threshold: thresholdNum,
                      });
                      const n = res?.evaluate?.triggered_count ?? 0;
                      const first = res?.evaluate?.triggered?.[0] as
                        | {
                            delivered?: { email?: boolean };
                            email_delivery_error?: string | null;
                          }
                        | undefined;
                      const emailOk = first?.delivered?.email;
                      const emailErr = first?.email_delivery_error;
                      if (n > 0) {
                        const parts: string[] = [
                          "The price already satisfies your rule. We emailed you (if the server can send mail) and removed this rule — it will not fire again.",
                        ];
                        if (emailOk) {
                          parts.push("Email: sent (check inbox/spam).");
                        } else if (emailErr) {
                          parts.push(`Email: not sent — ${emailErr}`);
                        } else {
                          parts.push(
                            "Email: not sent — check server EMAIL_USER/EMAIL_PASS and that your JWT includes an email.",
                          );
                        }
                        window.alert(parts.join("\n"));
                      } else {
                        window.alert(
                          "Rule saved until it fires once. When the latest daily close in our feed meets your condition, we email you and then delete the rule automatically.",
                        );
                      }
                    } catch (e) {
                      window.alert(e instanceof Error ? e.message : "Could not set alert.");
                    }
                  }}
                  className="px-3 py-1.5 rounded border border-border text-sm font-medium bg-primary text-primary-foreground hover:opacity-90"
                >
                  Set one-time alert
                </button>
              </div>

              <div className="rounded-lg border border-border/80 bg-secondary/15 p-3 space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Pending one-time rules ({store.alerts.length})
                  </p>
                  {store.alerts.length > 0 && (
                    <button
                      type="button"
                      className="text-[11px] uppercase tracking-wide text-muted-foreground hover:text-destructive underline-offset-2 hover:underline"
                      onClick={async () => {
                        if (!window.confirm("Remove all pending price-alert rules for your account?")) return;
                        try {
                          await store.clearAllAlerts();
                        } catch (e) {
                          window.alert(e instanceof Error ? e.message : "Could not clear alerts.");
                        }
                      }}
                    >
                      Clear all pending
                    </button>
                  )}
                </div>
                {store.alerts.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No pending rules. Each new alert is one use only after it fires.</p>
                ) : (
                  <ul className="space-y-2 text-sm">
                    {store.alerts.map((a) => (
                      <li
                        key={a.id}
                        className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border/60 bg-background/80 px-2 py-1.5"
                      >
                        <span className="font-mono text-xs">
                          <span className="font-semibold">{a.ticker}</span>
                          {" · "}
                          {a.rule_type === "price_above" ? "≥" : "≤"} ${a.threshold.toFixed(2)}
                          {" · "}
                          Email
                        </span>
                        <button
                          type="button"
                          className="text-xs text-muted-foreground hover:text-destructive"
                          onClick={async () => {
                            try {
                              await store.deleteAlertRule(a.id);
                            } catch (e) {
                              window.alert(e instanceof Error ? e.message : "Could not remove rule.");
                            }
                          }}
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

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
                  onClick={async () => {
                    const raw = window.prompt("Recipient email(s), comma-separated");
                    if (!raw) return;
                    const recipients = raw.split(",").map((s) => s.trim()).filter(Boolean);
                    if (!recipients.length) return;
                    try {
                      await store.emailReport({
                        selected: store.stockData,
                        compare: store.compareData,
                        portfolio: store.portfolioSummary,
                        events: store.events,
                      }, recipients);
                      window.alert("Report emailed successfully.");
                    } catch (e) {
                      window.alert(e instanceof Error ? e.message : "Failed to send report email.");
                    }
                  }}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded border border-border text-sm hover:bg-secondary/40 transition-colors"
                >
                  <Mail className="w-4 h-4" /> Send report via email
                </button>
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
