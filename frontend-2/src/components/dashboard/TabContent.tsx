import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, Brain, LineChart, Layout, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { StockData } from "@/hooks/useStockStore";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface TabContentProps {
  data: StockData | null;
  loading: boolean;
}

const tabs = [
  { id: "overview", label: "Overview", icon: Layout },
  { id: "fundamentals", label: "Fundamentals", icon: BarChart3 },
  { id: "sentiment", label: "Sentiment", icon: Brain },
  { id: "chart", label: "Chart", icon: LineChart },
] as const;

type TabId = typeof tabs[number]["id"];
const PERIOD_MAP: Record<"1W" | "1M" | "3M" | "1Y" | "MAX", string> = {
  "1W": "5d",
  "1M": "1mo",
  "3M": "3mo",
  "1Y": "1y",
  "MAX": "max",
};

export default function TabContent({ data, loading }: TabContentProps) {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  if (!data && !loading) return null;

  return (
    <div className="glass-card overflow-hidden">
      {/* Tab Bar */}
      <div className="flex border-b border-border overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`relative flex items-center gap-2 px-5 py-3.5 text-sm font-medium transition-colors whitespace-nowrap cursor-pointer
              ${activeTab === tab.id ? "text-foreground" : "text-muted-foreground hover:text-foreground/80"}`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
            {activeTab === tab.id && (
              <motion.div
                layoutId="tab-indicator"
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-5 min-h-[300px]">
        {loading ? (
          <div className="space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="skeleton-shimmer h-10 w-full" />
            ))}
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === "overview" && data && <OverviewTab data={data} />}
              {activeTab === "fundamentals" && data && <FundamentalsTab data={data} />}
              {activeTab === "sentiment" && data && <SentimentTab data={data} />}
              {activeTab === "chart" && data && <ChartTab data={data} />}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

function OverviewTab({ data }: { data: StockData }) {
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold text-foreground">{data.company_name}</h2>
        <span className="ticker-badge">{data.ticker}</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {[
          ["52W High", `$${data.high_52w}`],
          ["52W Low", `$${data.low_52w}`],
          ["P/E Ratio", data.pe_ratio],
          ["Volume", data.volume],
          ["Market Cap", data.market_cap],
          ["Model view", data.recommendation],
        ].map(([label, value]) => (
          <div key={label as string} className="p-3 rounded-lg bg-secondary/50">
            <div className="text-xs text-muted-foreground mb-1">{label}</div>
            <div className="text-sm font-semibold text-foreground">{value}</div>
          </div>
        ))}
      </div>
      {data.recommendation && (
        <div
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold ${
            (data.sentiment_label || "").toLowerCase() === "bullish"
              ? "bg-success/10 text-success"
              : (data.sentiment_label || "").toLowerCase() === "bearish"
                ? "bg-destructive/10 text-destructive"
                : "bg-warning/10 text-warning"
          }`}
        >
          {(data.sentiment_label || "").toLowerCase() === "bullish" ? (
            <TrendingUp className="w-4 h-4" />
          ) : (data.sentiment_label || "").toLowerCase() === "bearish" ? (
            <TrendingDown className="w-4 h-4" />
          ) : (
            <Minus className="w-4 h-4" />
          )}
          <span>
            <span className="block text-[10px] font-medium uppercase tracking-wide text-muted-foreground/90">Our models</span>
            {data.recommendation}
          </span>
        </div>
      )}
    </div>
  );
}

function FundamentalsTab({ data }: { data: StockData }) {
  if (!data.fundamentals) return <EmptyState message="No fundamentals data available." />;
  return (
    <div className="space-y-1">
      {Object.entries(data.fundamentals).map(([key, val], i) => (
        <motion.div
          key={key}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05 }}
          className="flex items-center justify-between py-3 px-3 rounded-lg hover:bg-secondary/50 transition-colors"
        >
          <span className="text-sm text-muted-foreground">{key}</span>
          <span className="text-sm font-semibold font-mono text-foreground">{val}</span>
        </motion.div>
      ))}
    </div>
  );
}

function SentimentTab({ data }: { data: StockData }) {
  const score = data.sentiment_score ?? 50;
  const [impactFilter, setImpactFilter] = useState<"all" | "high" | "medium" | "low">("all");
  const filteredNews = useMemo(() => {
    if (impactFilter === "all") return data.news || [];
    // Heuristic fallback mapping using sentiment label when impact metadata is unavailable in current shape.
    return (data.news || []).filter((n) =>
      impactFilter === "high" ? n.sentiment !== "neutral" : impactFilter === "low" ? n.sentiment === "neutral" : true
    );
  }, [data.news, impactFilter]);
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="relative w-20 h-20">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none" stroke="hsl(var(--border))" strokeWidth="3" />
            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none" stroke={score >= 60 ? "hsl(var(--success))" : score >= 40 ? "hsl(var(--warning))" : "hsl(var(--destructive))"}
              strokeWidth="3" strokeDasharray={`${score}, 100`} strokeLinecap="round" />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center text-lg font-bold text-foreground">{score}</div>
        </div>
        <div>
          <div className={`text-lg font-bold ${score >= 60 ? "kpi-positive" : score >= 40 ? "text-warning" : "kpi-negative"}`}>
            {data.sentiment_label}
          </div>
          <div className="text-sm text-muted-foreground">Market sentiment score</div>
        </div>
      </div>

      {data.news && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">Recent News</h3>
            <select
              value={impactFilter}
              onChange={(e) => setImpactFilter(e.target.value as "all" | "high" | "medium" | "low")}
              className="text-xs rounded-md border border-border bg-background px-2 py-1"
            >
              <option value="all">All impact</option>
              <option value="high">High impact</option>
              <option value="medium">Medium impact</option>
              <option value="low">Low impact</option>
            </select>
          </div>
          {filteredNews.map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
            >
              <div className={`w-2 h-2 mt-1.5 rounded-full shrink-0 ${
                item.sentiment === "positive" ? "bg-success" : item.sentiment === "negative" ? "bg-destructive" : "bg-warning"
              }`} />
              <div className="min-w-0">
                <div className="text-sm text-foreground">{item.title}</div>
                <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
                  <span>{item.source}</span>
                  {item.link && (
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                    >
                      redirect
                    </a>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}


function ChartTab({ data }: { data: StockData }) {
  const [period, setPeriod] = useState<"1W" | "1M" | "3M" | "1Y" | "MAX">("1M");
  const [chartPoints, setChartPoints] = useState<Array<{ date: string; close: number }>>([]);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setChartLoading(true);
    setChartError(null);
    fetch(`/chart/${encodeURIComponent(data.ticker)}?period=${PERIOD_MAP[period]}&interval=1d`)
      .then(async (resp) => {
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          const msg = payload?.detail?.error?.message || payload?.detail || "Chart request failed";
          throw new Error(msg);
        }
        const next = (payload?.points || [])
          .map((p: { date?: string; close?: number }) => ({
            date: String(p.date || ""),
            close: Number(p.close),
          }))
          .filter((p: { date: string; close: number }) => p.date && Number.isFinite(p.close));
        if (mounted) {
          setChartPoints(next);
        }
      })
      .catch((e) => {
        if (mounted) {
          setChartPoints([]);
          setChartError(e instanceof Error ? e.message : "Chart unavailable.");
        }
      })
      .finally(() => {
        if (mounted) setChartLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [data.ticker, period]);

  const { min, max, isUp } = useMemo(() => {
    const values = chartPoints.length > 1 ? chartPoints.map((p) => p.close) : [data.current_price ?? 0, data.current_price ?? 0];
    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    return {
      min: minV,
      max: maxV,
      isUp: values[values.length - 1] >= values[0],
    };
  }, [chartPoints, data.current_price]);
  const chartData = useMemo(
    () =>
      chartPoints.map((p) => ({
        date: p.date,
        dateObj: new Date(`${p.date}T00:00:00`),
        year: p.date.slice(0, 4),
        close: Number(p.close.toFixed(2)),
      })),
    [chartPoints]
  );

  const xAxisConfig = useMemo(() => {
    if (period === "1W") {
      return {
        label: "Time (7 days)",
        minTickGap: 8,
        tickFormatter: (v: string) => {
          const d = new Date(`${v}T00:00:00`);
          return `${d.getMonth() + 1}/${d.getDate()}`;
        },
      };
    }
    if (period === "1M") {
      return {
        label: "Days (28-31)",
        minTickGap: 10,
        tickFormatter: (v: string) => {
          const d = new Date(`${v}T00:00:00`);
          return `${d.getDate()}`;
        },
      };
    }
    if (period === "1Y") {
      return {
        label: "Months (12)",
        minTickGap: 20,
        tickFormatter: (v: string) => {
          const d = new Date(`${v}T00:00:00`);
          return d.toLocaleString("en-US", { month: "short" });
        },
      };
    }
    if (period === "3M") {
      return {
        label: "Months",
        minTickGap: 18,
        tickFormatter: (v: string) => {
          const d = new Date(`${v}T00:00:00`);
          return `${d.toLocaleString("en-US", { month: "short" })} ${d.getDate()}`;
        },
      };
    }
    return {
      label: "Year",
      minTickGap: 24,
      tickFormatter: (v: string) => String(v).slice(0, 4),
    };
  }, [period]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Price History ({period})</h3>
        <div className="flex gap-2">
          {["1W", "1M", "3M", "1Y", "MAX"].map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p as "1W" | "1M" | "3M" | "1Y" | "MAX")}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors cursor-pointer ${
                p === period ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>
      <div className="w-full aspect-[2.5/1] rounded-lg bg-secondary/30 p-4">
        {chartLoading ? (
          <div className="h-full w-full flex items-center justify-center text-sm text-muted-foreground">Loading chart...</div>
        ) : chartError ? (
          <div className="h-full w-full flex items-center justify-center text-sm text-destructive">{chartError}</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 4, bottom: 8 }}>
              <defs>
                <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={isUp ? "hsl(var(--success))" : "hsl(var(--destructive))"} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={isUp ? "hsl(var(--success))" : "hsl(var(--destructive))"} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.35} vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={xAxisConfig.tickFormatter}
                minTickGap={xAxisConfig.minTickGap}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                label={{ value: xAxisConfig.label, position: "insideBottomRight", offset: -4, fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => `$${Number(v).toFixed(0)}`}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={58}
                label={{ value: "Price ($)", angle: -90, position: "insideLeft", fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              />
              <Tooltip
                formatter={(value: number) => [`$${Number(value).toFixed(2)}`, "Price"]}
                labelFormatter={(label: string) => `Date: ${label}`}
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "10px",
                  color: "hsl(var(--foreground))",
                }}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={isUp ? "hsl(var(--success))" : "hsl(var(--destructive))"}
                strokeWidth={2}
                fill="url(#priceFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
        {!chartLoading && !chartError && (
          <div className="mt-1 text-xs text-muted-foreground flex justify-between">
            <span>Low: ${min.toFixed(2)}</span>
            <span>High: ${max.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <BarChart3 className="w-10 h-10 text-muted-foreground/40 mb-3" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
