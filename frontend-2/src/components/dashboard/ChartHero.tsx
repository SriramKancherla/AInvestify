import { useEffect, useMemo, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { StockData } from "@/hooks/useStockStore";
import Monogram from "@/components/common/Monogram";
import CountUp from "@/components/common/CountUp";

interface ChartHeroProps {
  data: StockData;
}

type Period = "1W" | "1M" | "3M" | "1Y" | "MAX";
const PERIOD_MAP: Record<Period, string> = { "1W": "5d", "1M": "1mo", "3M": "3mo", "1Y": "1y", MAX: "max" };
const PERIODS: Period[] = ["1W", "1M", "3M", "1Y", "MAX"];

export default function ChartHero({ data }: ChartHeroProps) {
  const [period, setPeriod] = useState<Period>("1M");
  const [points, setPoints] = useState<Array<{ date: string; close: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    fetch(`/chart/${encodeURIComponent(data.ticker)}?period=${PERIOD_MAP[period]}&interval=1d`)
      .then(async (resp) => {
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(payload?.detail?.error?.message || payload?.detail || "Chart request failed");
        const next = (payload?.points || [])
          .map((p: { date?: string; close?: number }) => ({ date: String(p.date || ""), close: Number(p.close) }))
          .filter((p: { date: string; close: number }) => p.date && Number.isFinite(p.close));
        if (mounted) setPoints(next);
      })
      .catch((e) => {
        if (mounted) {
          setPoints([]);
          setError(e instanceof Error ? e.message : "Chart unavailable.");
        }
      })
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, [data.ticker, period]);

  const { first, last, isUp, periodChangePct } = useMemo(() => {
    const vals = points.map((p) => p.close);
    const first = vals.length ? vals[0] : data.current_price ?? 0;
    const last = vals.length ? vals[vals.length - 1] : data.current_price ?? 0;
    const isUp = last >= first;
    const periodChangePct = first ? ((last - first) / first) * 100 : 0;
    return { first, last, isUp, periodChangePct };
  }, [points, data.current_price]);

  const price = data.current_price ?? last ?? 0;
  const chartData = useMemo(() => points.map((p) => ({ date: p.date, close: Number(p.close.toFixed(2)) })), [points]);
  const color = isUp ? "hsl(var(--success))" : "hsl(var(--destructive))";

  return (
    <div className="elevated p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <Monogram symbol={data.ticker} size={44} />
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold text-foreground tracking-tightish">{data.company_name}</h1>
              <span className="ticker-badge">{data.ticker}</span>
            </div>
            <div className="mt-1 flex items-baseline gap-3">
              <CountUp value={price} prefix="$" decimals={2} className="font-mono text-[32px] leading-none font-semibold text-foreground" />
              <span className={`inline-flex items-center gap-1 font-mono text-sm font-medium ${isUp ? "kpi-positive" : "kpi-negative"}`}>
                {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {periodChangePct > 0 ? "+" : ""}{periodChangePct.toFixed(2)}%
                <span className="text-muted-foreground font-sans">· {period}</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex gap-1 rounded-lg border border-border p-1">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                p === period ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 w-full aspect-[2.6/1] min-h-[220px]">
        {loading ? (
          <div className="h-full w-full skeleton-shimmer rounded-lg" />
        ) : error ? (
          <div className="h-full w-full flex items-center justify-center text-sm text-destructive">{error}</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="heroFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.14} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.6} vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                minTickGap={40}
                tickFormatter={(v: string) => {
                  const d = new Date(`${v}T00:00:00`);
                  return period === "1Y" || period === "MAX"
                    ? d.toLocaleString("en-US", { month: "short", year: "2-digit" })
                    : `${d.getMonth() + 1}/${d.getDate()}`;
                }}
              />
              <YAxis
                domain={["auto", "auto"]}
                orientation="right"
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={52}
                tickFormatter={(v: number) => `$${Number(v).toFixed(0)}`}
              />
              <Tooltip
                formatter={(value: number) => [`$${Number(value).toFixed(2)}`, "Price"]}
                labelFormatter={(label: string) => `Date: ${label}`}
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "10px",
                  color: "hsl(var(--foreground))",
                  fontSize: "12px",
                  boxShadow: "var(--shadow-pop)",
                }}
                cursor={{ stroke: "hsl(var(--muted-foreground))", strokeDasharray: "3 3", strokeOpacity: 0.5 }}
              />
              <Area type="monotone" dataKey="close" stroke={color} strokeWidth={2} fill="url(#heroFill)" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
