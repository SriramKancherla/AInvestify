import { useState, useEffect, useCallback, useRef } from "react";
import { apiJson, apiBlob } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export interface StockData {
  ticker: string;
  company_name?: string;
  current_price?: number;
  change_percent?: number;
  market_cap?: string;
  pe_ratio?: number;
  volume?: string;
  high_52w?: number;
  low_52w?: number;
  sentiment_score?: number;
  sentiment_label?: string;
  /** Combined fundamentals tier + sentiment phrase (see `buildModelViewSummary`). */
  recommendation?: string;
  fundamentals?: Record<string, string | number>;
  news?: Array<{ title: string; sentiment: string; source: string; link?: string }>;
}

export interface TickerOption {
  symbol: string;
  name: string;
  label: string;
}

interface InsightsResponse {
  final?: { score?: number; label?: string };
  fundamentals?: { score?: number };
  sentiment?: {
    score?: number | null;
    source?: string;
    top_positive_examples?: string[];
    top_negative_examples?: string[];
  };
  resolved?: { fundamentals?: { company?: string; symbol?: string } };
}

interface FundamentalsResponse {
  metrics?: {
    market_cap?: number | null;
    current_price?: number | null;
    pe_ratio?: number | null;
    pb_ratio?: number | null;
    price_to_sales?: number | null;
    book_value?: number | null;
    ebitda?: number | null;
    dividend_yield?: number | null;
    fifty_two_week_high?: number | null;
    fifty_two_week_low?: number | null;
  };
}

interface ChartResponse {
  points?: Array<{ date?: string; close?: number; volume?: number }>;
}

interface NewsResponse {
  articles?: Array<{
    headline?: string;
    link?: string;
    sentiment_label?: string;
  }>;
}

const TICKER_REQUEST_TTL_MS = 10_000;
const _tickerInFlight = new Map<string, Promise<StockData>>();
const _tickerCache = new Map<string, { at: number; data: StockData }>();

export interface PortfolioHolding {
  id: string;
  ticker: string;
  quantity: number;
  avg_buy_price: number;
  last_price: number;
  market_value: number;
  cost_basis: number;
  pnl: number;
  pnl_pct: number;
}

const STORAGE_KEYS = {
  RECENT: "ainvestify_recent",
  FAVORITES: "ainvestify_favorites",
  AUTH_TOKEN: "ainvestify_auth_token",
};

function formatCompactMoney(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "N/A";
  const n = Number(v);
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(2)}K`;
  return `${sign}$${abs.toFixed(2)}`;
}

function formatCompactVolume(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "N/A";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(abs / 1e3).toFixed(2)}K`;
  return `${Math.round(abs)}`;
}

function sentimentLabelFromScore(score01?: number | null): string {
  if (score01 == null || !Number.isFinite(score01)) return "Unknown";
  if (score01 >= 0.6) return "Bullish";
  if (score01 >= 0.4) return "Neutral";
  return "Bearish";
}

/** Combines fundamentals model score (0–1) and news sentiment for the insights headline (replaces legacy “analyst” wording). */
export function buildModelViewSummary(
  fundamentalsScore01: number | null | undefined,
  sentiment01: number | null | undefined,
  sentimentDisplayLabel: string,
): string {
  const fundTier =
    fundamentalsScore01 != null && Number.isFinite(fundamentalsScore01)
      ? (() => {
          const x = Math.max(0, Math.min(1, fundamentalsScore01));
          if (x >= 0.6) return "strong" as const;
          if (x < 0.4) return "weak" as const;
          return "moderate" as const;
        })()
      : null;

  let tone: "bullish" | "neutral" | "bearish" | null =
    sentiment01 != null && Number.isFinite(sentiment01)
      ? sentiment01 >= 0.6
        ? "bullish"
        : sentiment01 < 0.4
          ? "bearish"
          : "neutral"
      : null;

  if (!tone) {
    const sl = sentimentDisplayLabel.toLowerCase();
    if (sl === "bullish" || sl === "bearish" || sl === "neutral") {
      tone = sl;
    }
  }

  const fAdj = fundTier === "strong" ? "Strong" : fundTier === "moderate" ? "Moderate" : fundTier === "weak" ? "Weak" : null;

  if (fAdj && tone) {
    return `${fAdj} fundamentals with ${tone} sentiment`;
  }
  if (fAdj && !tone) {
    return `${fAdj} fundamentals; sentiment unavailable`;
  }
  if (!fAdj && tone) {
    return `Fundamentals unavailable; ${tone} sentiment`;
  }
  return "Model read unavailable — fundamentals or sentiment data missing";
}

async function loadTickerData(ticker: string): Promise<StockData> {
  const clean = ticker.trim().toUpperCase();
  const insights = await apiJson<InsightsResponse>("/api/insights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: clean,
      news_source: "auto",
      top_items: 3,
      sentiment_weight: 0.3,
      max_news: 10,
      train_missing: false,
    }),
  });

  const [fundamentals, chart, news] = await Promise.all([
    apiJson<FundamentalsResponse>(`/fundamentals/${encodeURIComponent(clean)}`).catch(() => ({})),
    apiJson<ChartResponse>(`/chart/${encodeURIComponent(clean)}?period=1mo&interval=1d`).catch(() => ({})),
    apiJson<NewsResponse>(`/news/${encodeURIComponent(clean)}?max_articles=6`).catch(() => ({})),
  ]);

  const metrics = fundamentals.metrics || {};
  const points = chart.points || [];
  const firstClose = points.length > 0 ? Number(points[0]?.close || 0) : null;
  const lastClose = points.length > 0 ? Number(points[points.length - 1]?.close || 0) : null;
  const changePercent =
    firstClose && Number.isFinite(firstClose) && firstClose !== 0 && lastClose && Number.isFinite(lastClose)
      ? ((lastClose - firstClose) / firstClose) * 100
      : 0;
  const lastVolume = points.length > 0 ? Number(points[points.length - 1]?.volume || 0) : null;

  const sentiment01 = insights.sentiment?.score ?? null;
  const sentimentPct = sentiment01 == null ? undefined : Math.round(Math.max(0, Math.min(1, sentiment01)) * 100);
  const sentimentLbl = sentimentLabelFromScore(sentiment01);

  const safeMetric = (v: number | null | undefined, fallback: number) =>
    v == null || !Number.isFinite(Number(v)) ? fallback : Number(v);

  const currentPrice = safeMetric(metrics.current_price, 0);
  const high52 = safeMetric(metrics.fifty_two_week_high, currentPrice || 0);
  const low52 = safeMetric(metrics.fifty_two_week_low, currentPrice || 0);
  const marketCapNum = safeMetric(metrics.market_cap, 0);
  const peRatio = safeMetric(metrics.pe_ratio, 0);
  const pbRatio = safeMetric(metrics.pb_ratio, 0);
  const priceToSales = safeMetric(metrics.price_to_sales, 0);
  const bookValue = safeMetric(metrics.book_value, 0);
  const ebitda = safeMetric(metrics.ebitda, 0);
  const dividendYield = safeMetric(metrics.dividend_yield, 0);

  return {
    ticker: clean,
    company_name: insights.resolved?.fundamentals?.company || getMockName(clean),
    current_price: currentPrice,
    change_percent: Number.isFinite(changePercent) ? Number(changePercent.toFixed(2)) : 0,
    market_cap: formatCompactMoney(marketCapNum),
    pe_ratio: peRatio,
    volume: formatCompactVolume(lastVolume),
    high_52w: high52,
    low_52w: low52,
    sentiment_score: sentimentPct,
    sentiment_label: sentimentLbl,
    recommendation: buildModelViewSummary(insights.fundamentals?.score ?? null, sentiment01, sentimentLbl),
    fundamentals: {
      "Final Score": insights.final?.score != null ? insights.final.score.toFixed(3) : "N/A",
      "Fundamentals Score": insights.fundamentals?.score != null ? insights.fundamentals.score.toFixed(3) : "N/A",
      "Sentiment Score": sentiment01 != null ? sentiment01.toFixed(3) : "N/A",
      "Current Price": `$${currentPrice.toFixed(2)}`,
      "52W High": `$${high52.toFixed(2)}`,
      "52W Low": `$${low52.toFixed(2)}`,
      "Market Cap": formatCompactMoney(marketCapNum),
      "P/E Ratio": peRatio.toFixed(2),
      "P/B Ratio": pbRatio.toFixed(2),
      "Price/Sales": priceToSales.toFixed(2),
      "Book Value": bookValue.toFixed(2),
      "EBITDA": formatCompactMoney(ebitda),
      "Dividend Yield": `${(Math.abs(dividendYield) <= 1 ? dividendYield * 100 : dividendYield).toFixed(2)}%`,
    },
    news:
      (news.articles || []).map((a) => ({
        title: a.headline || "News item",
        sentiment: (a.sentiment_label || "neutral").toLowerCase(),
        source: insights.sentiment?.source || "news",
        link: a.link,
      })) || [],
  };
}

function getTickerDataSingleFlight(ticker: string): Promise<StockData> {
  const clean = ticker.trim().toUpperCase();
  if (!clean) {
    return Promise.reject(new Error("Ticker is required"));
  }
  const now = Date.now();
  const cached = _tickerCache.get(clean);
  if (cached && now - cached.at < TICKER_REQUEST_TTL_MS) {
    return Promise.resolve(cached.data);
  }
  const inflight = _tickerInFlight.get(clean);
  if (inflight) return inflight;
  const req = loadTickerData(clean)
    .then((data) => {
      _tickerCache.set(clean, { at: Date.now(), data });
      return data;
    })
    .finally(() => {
      _tickerInFlight.delete(clean);
    });
  _tickerInFlight.set(clean, req);
  return req;
}

export function useStockStore() {
  const { session } = useAuth();
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [compareTicker, setCompareTicker] = useState<string>("");
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [compareData, setCompareData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recentStocks, setRecentStocks] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEYS.RECENT) || "[]");
    } catch { return []; }
  });
  const [favorites, setFavorites] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEYS.FAVORITES) || "[]");
    } catch { return []; }
  });
  const [allTickers, setAllTickers] = useState<TickerOption[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);
  const [portfolioSummary, setPortfolioSummary] = useState<{ total_market_value: number; total_cost_basis: number; total_pnl: number; total_pnl_pct: number } | null>(null);
  const [watchlists, setWatchlists] = useState<Record<string, string[]>>({ default: [] });
  const [guestToken, setGuestToken] = useState<string>(() => localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN) || "");
  const [events, setEvents] = useState<Array<{ id: string; event_type: string; event_date: string; risk_level: string; details?: string; source?: string }>>([]);
  const [backtest, setBacktest] = useState<Record<string, unknown> | null>(null);
  const inFlightTickersRef = useRef<Set<string>>(new Set());
  const lastFetchAtRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.RECENT, JSON.stringify(recentStocks));
  }, [recentStocks]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.FAVORITES, JSON.stringify(favorites));
  }, [favorites]);

  useEffect(() => {
    let mounted = true;
    apiJson<{ tickers?: Array<{ symbol: string; name: string; label: string }> }>("/tickers")
      .then((data) => {
        if (!mounted) return;
        const list = (data.tickers || [])
          .map((t) => ({
            symbol: String(t.symbol || "").toUpperCase(),
            name: String(t.name || "").trim(),
            label: `${String(t.symbol || "").toUpperCase()} - ${String(t.name || "").trim()}`,
          }))
          .filter((t) => t.symbol);
        setAllTickers(list);
      })
      .catch(() => {
        // Keep dashboard usable even if tickers endpoint is temporarily unavailable.
        setAllTickers(
          POPULAR_TICKERS.map((s) => ({ symbol: s, name: getMockName(s), label: `${s} - ${getMockName(s)}` }))
        );
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Prevent showing stale event-calendar cards when user switches stocks.
  useEffect(() => {
    setEvents([]);
  }, [selectedTicker]);

  const refreshPortfolio = useCallback(async () => {
    const res = await apiJson<{ holdings: PortfolioHolding[]; summary: { total_market_value: number; total_cost_basis: number; total_pnl: number; total_pnl_pct: number } }>("/api/portfolio").catch(() => ({ holdings: [], summary: { total_market_value: 0, total_cost_basis: 0, total_pnl: 0, total_pnl_pct: 0 } }));
    setPortfolio(res.holdings || []);
    setPortfolioSummary(res.summary || null);
  }, []);

  const refreshWatchlists = useCallback(async () => {
    const res = await apiJson<{ watchlists: Array<{ name: string; tickers: string[] }> }>("/api/watchlists").catch(() => ({ watchlists: [] }));
    const map: Record<string, string[]> = {};
    for (const w of res.watchlists || []) {
      map[w.name] = [...(w.tickers || [])];
    }
    if (!Object.keys(map).length) map.default = [];
    setWatchlists(map);
  }, []);

  const addHolding = useCallback(async (ticker: string, quantity: number, avgBuyPrice: number) => {
    await apiJson("/api/portfolio", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticker, quantity, avg_buy_price: avgBuyPrice }) });
    await refreshPortfolio();
  }, [refreshPortfolio]);

  const removeHolding = useCallback(async (holdingId: string) => {
    await apiJson(`/api/portfolio/${encodeURIComponent(holdingId)}`, { method: "DELETE" });
    await refreshPortfolio();
  }, [refreshPortfolio]);

  const saveWatchlist = useCallback(async (name: string, tickers: string[]) => {
    await apiJson("/api/watchlists", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, tickers }) }).catch(() => null);
    await refreshWatchlists();
  }, [refreshWatchlists]);

  useEffect(() => {
    if (!session?.user?.id) return;
    void refreshPortfolio();
    void refreshWatchlists();
  }, [session?.user?.id, refreshPortfolio, refreshWatchlists]);

  const fetchEvents = useCallback(async (ticker: string) => {
    const res = await apiJson<{ events: Array<{ id: string; event_type: string; event_date: string; risk_level: string; details?: string; source?: string }> }>(`/api/events/${encodeURIComponent(ticker)}`).catch(() => ({ events: [] }));
    setEvents(res.events || []);
  }, []);

  const runBacktest = useCallback(async (ticker: string, period = "1y") => {
    const res = await apiJson<Record<string, unknown>>(`/api/backtest/${encodeURIComponent(ticker)}?period=${encodeURIComponent(period)}`);
    setBacktest(res);
  }, []);

  const exportReport = useCallback(async (payload: Record<string, unknown>) => {
    const blob = await apiBlob("/api/report/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    const sym = String((payload.selected as StockData | undefined)?.ticker || "Report").replace(/[^a-zA-Z0-9._-]/g, "") || "Report";
    a.download = `AInvestify-${sym}-report.pdf`;
    a.click();
    URL.revokeObjectURL(href);
  }, []);

  const ensureGuestAuth = useCallback(async () => {
    if (guestToken) return guestToken;
    const res = await apiJson<{ token: string }>("/api/auth/guest", { method: "POST" });
    setGuestToken(res.token);
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, res.token);
    return res.token;
  }, [guestToken]);

  const syncCloudState = useCallback(async () => null, []);

  const addRecent = useCallback((ticker: string) => {
    setRecentStocks(prev => {
      const filtered = prev.filter(t => t !== ticker);
      return [ticker, ...filtered].slice(0, 8);
    });
  }, []);

  const toggleFavorite = useCallback((ticker: string) => {
    setFavorites(prev =>
      prev.includes(ticker) ? prev.filter(t => t !== ticker) : [...prev, ticker]
    );
  }, []);

  const fetchStock = useCallback(async (ticker: string) => {
    if (!ticker) return;
    const clean = ticker.trim().toUpperCase();
    if (!clean) return;
    // Guard against accidental rapid repeat calls from UI/effects.
    if (inFlightTickersRef.current.has(clean)) return;
    const now = Date.now();
    const lastAt = lastFetchAtRef.current.get(clean) || 0;
    if (now - lastAt < 1500) return;
    inFlightTickersRef.current.add(clean);
    lastFetchAtRef.current.set(clean, now);
    setLoading(true);
    setError(null);
    addRecent(clean);
    try {
      const data = await getTickerDataSingleFlight(clean);
      setStockData(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch stock data.");
    } finally {
      inFlightTickersRef.current.delete(clean);
      setLoading(false);
    }
  }, [addRecent]);

  const fetchCompare = useCallback(async (ticker: string) => {
    if (!ticker) { setCompareData(null); return; }
    try {
      const data = await getTickerDataSingleFlight(ticker);
      setCompareData(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch compare stock.");
      setCompareData(null);
    }
  }, []);

  const clearActiveStock = useCallback(() => {
    setSelectedTicker("");
    setCompareTicker("");
    setStockData(null);
    setCompareData(null);
    setBacktest(null);
    setError(null);
    setLoading(false);
    inFlightTickersRef.current.clear();
  }, []);

  return {
    selectedTicker, setSelectedTicker,
    compareTicker, setCompareTicker,
    stockData, compareData,
    loading, error,
    recentStocks, favorites, allTickers,
    portfolio, portfolioSummary, watchlists, events, backtest,
    toggleFavorite,
    refreshPortfolio, refreshWatchlists,
    addHolding, removeHolding,
    saveWatchlist,
    fetchEvents, runBacktest, exportReport, ensureGuestAuth, syncCloudState,
    fetchStock, fetchCompare,
    clearActiveStock,
  };
}

function getMockName(ticker: string): string {
  const names: Record<string, string> = {
    AAPL: "Apple Inc.", GOOGL: "Alphabet Inc.", MSFT: "Microsoft Corp.",
    AMZN: "Amazon.com Inc.", TSLA: "Tesla Inc.", META: "Meta Platforms",
    NVDA: "NVIDIA Corp.", NFLX: "Netflix Inc.", JPM: "JPMorgan Chase",
    V: "Visa Inc.", DIS: "Walt Disney Co.", AMD: "Advanced Micro Devices",
  };
  return names[ticker.toUpperCase()] || `${ticker.toUpperCase()} Corp.`;
}

export const POPULAR_TICKERS = [
  "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "V", "DIS", "AMD"
];
