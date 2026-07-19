import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";
import { Check, X } from "lucide-react";
import type { StockData, TickerOption } from "@/hooks/useStockStore";
import Monogram from "@/components/common/Monogram";

interface CompareViewProps {
  primary: StockData;
  secondary: StockData | null;
  compareChoices: TickerOption[];
  compareInput: string;
  setCompareInput: (v: string) => void;
  onSelectCompare: (ticker: string) => void;
  onClearCompare: () => void;
}

type Period = "1M" | "3M" | "1Y";
const PERIOD_MAP: Record<Period, string> = { "1M": "1mo", "3M": "3mo", "1Y": "1y" };
const PERIODS: Period[] = ["1M", "3M", "1Y"];

const COLOR_A = "hsl(var(--primary))";
const COLOR_B = "hsl(262 68% 55%)"; // slate-purple secondary series

async function fetchCloses(symbol: string, period: string): Promise<Array<{ date: string; close: number }>> {
  try {
    const resp = await fetch(`/chart/${encodeURIComponent(symbol)}?period=${period}&interval=1d`);
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) return [];
    return (payload?.points || [])
      .map((p: { date?: string; close?: number }) => ({ date: String(p.date || ""), close: Number(p.close) }))
      .filter((p: { date: string; close: number }) => p.date && Number.isFinite(p.close));
  } catch {
    return [];
  }
}

function rebase(series: Array<{ date: string; close: number }>): Map<string, number> {
  const m = new Map<string, number>();
  if (!series.length) return m;
  const base = series[0].close || 1;
  for (const p of series) m.set(p.date, ((p.close - base) / base) * 100);
  return m;
}

function num(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export default function CompareView({
  primary,
  secondary,
  compareChoices,
  compareInput,
  setCompareInput,
  onSelectCompare,
  onClearCompare,
}: CompareViewProps) {
  const [period, setPeriod] = useState<Period>("3M");
  const [merged, setMerged] = useState<Array<{ date: string; a: number | null; b: number | null }>>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!secondary) {
      setMerged([]);
      return;
    }
    let mounted = true;
    setLoading(true);
    Promise.all([fetchCloses(primary.ticker, PERIOD_MAP[period]), fetchCloses(secondary.ticker, PERIOD_MAP[period])])
      .then(([sa, sb]) => {
        if (!mounted) return;
        const ra = rebase(sa);
        const rb = rebase(sb);
        const dates = Array.from(new Set([...ra.keys(), ...rb.keys()])).sort();
        setMerged(dates.map((d) => ({ date: d, a: ra.get(d) ?? null, b: rb.get(d) ?? null })));
      })
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, [primary.ticker, secondary, period]);

  const rows = useMemo(() => {
    if (!secondary) return [];
    return [
      { label: "Price", a: primary.current_price, b: secondary.current_price, better: "high" as const, fmt: (v: number) => `$${v.toFixed(2)}` },
      { label: "Change %", a: primary.change_percent, b: secondary.change_percent, better: "high" as const, fmt: (v: number) => `${v.toFixed(2)}%` },
      { label: "P/E Ratio", a: primary.pe_ratio, b: secondary.pe_ratio, better: "low" as const, fmt: (v: number) => v.toFixed(1) },
      { label: "Sentiment", a: primary.sentiment_score, b: secondary.sentiment_score, better: "high" as const, fmt: (v: number) => `${v.toFixed(0)}` },
      {
        label: "AI Score",
        a: num(primary.fundamentals?.["Final Score"]),
        b: num(secondary.fundamentals?.["Final Score"]),
        better: "high" as const,
        fmt: (v: number) => v.toFixed(3),
      },
    ];
  }, [primary, secondary]);

  return (
    <div className="space-y-4">
      {/* Ticker pickers */}
      <div className="surface p-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 h-11 px-3 rounded-lg border border-border bg-secondary/50">
          <Monogram symbol={primary.ticker} size={28} />
          <span className="font-mono text-sm font-semibold text-foreground">{primary.ticker}</span>
        </div>
        <span className="text-sm text-muted-foreground font-medium">vs</span>
        {secondary ? (
          <div className="flex items-center gap-2 h-11 px-3 rounded-lg border border-border" style={{ background: "hsl(262 68% 55% / 0.08)" }}>
            <Monogram symbol={secondary.ticker} size={28} />
            <span className="font-mono text-sm font-semibold text-foreground">{secondary.ticker}</span>
            <button onClick={onClearCompare} className="ml-1 text-muted-foreground hover:text-destructive" aria-label="Clear compare">
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex-1 min-w-[240px]">
            <input
              value={compareInput}
              onChange={(e) => setCompareInput(e.target.value)}
              placeholder="Search a ticker to compare…"
              className="w-full h-11 px-3.5 rounded-lg border border-transparent bg-secondary text-sm focus:outline-none focus:border-primary"
            />
          </div>
        )}
        <div className="ml-auto flex gap-1 rounded-lg border border-border p-1">
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

      {!secondary ? (
        <div className="surface p-2 max-h-72 overflow-y-auto">
          {compareChoices.map((t) => (
            <button
              key={t.symbol}
              onClick={() => onSelectCompare(t.symbol)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-secondary transition-colors"
            >
              <Monogram symbol={t.symbol} size={28} />
              <span className="font-mono text-sm font-semibold text-foreground">{t.symbol}</span>
              <span className="text-xs text-muted-foreground truncate">{t.name}</span>
            </button>
          ))}
        </div>
      ) : (
        <>
          {/* Overlay chart (rebased %) */}
          <div className="elevated p-5 sm:p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-foreground">Relative performance ({period}, rebased to 0%)</h3>
            </div>
            <div className="w-full aspect-[2.6/1] min-h-[240px]">
              {loading ? (
                <div className="h-full w-full skeleton-shimmer rounded-lg" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={merged} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.6} vertical={false} />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={40}
                      tickFormatter={(v: string) => {
                        const d = new Date(`${v}T00:00:00`);
                        return `${d.getMonth() + 1}/${d.getDate()}`;
                      }}
                    />
                    <YAxis
                      orientation="right"
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={52}
                      tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => [`${Number(value).toFixed(2)}%`, name]}
                      contentStyle={{
                        background: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "10px",
                        fontSize: "12px",
                        boxShadow: "var(--shadow-pop)",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: "12px" }} />
                    <Line type="monotone" dataKey="a" name={primary.ticker} stroke={COLOR_A} strokeWidth={2} dot={false} connectNulls />
                    <Line type="monotone" dataKey="b" name={secondary.ticker} stroke={COLOR_B} strokeWidth={2} dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Comparison table */}
          <div className="surface overflow-hidden">
            <div className="grid grid-cols-3 px-5 py-3 bg-secondary/40 label-caps">
              <span>Metric</span>
              <span className="text-center">{primary.ticker}</span>
              <span className="text-center">{secondary.ticker}</span>
            </div>
            {rows.map((r) => {
              const a = Number(r.a);
              const b = Number(r.b);
              const valid = Number.isFinite(a) && Number.isFinite(b);
              const aWins = valid ? (r.better === "low" ? a < b : a > b) : null;
              return (
                <div key={r.label} className="grid grid-cols-3 px-5 py-3 border-t border-border items-center">
                  <span className="text-sm text-muted-foreground">{r.label}</span>
                  <span className="text-center font-mono text-sm font-medium text-foreground inline-flex items-center justify-center gap-1.5">
                    {aWins === true && <Check className="w-3.5 h-3.5 text-success" />}
                    {Number.isFinite(a) ? r.fmt(a) : "—"}
                  </span>
                  <span className="text-center font-mono text-sm font-medium text-foreground inline-flex items-center justify-center gap-1.5">
                    {aWins === false && <Check className="w-3.5 h-3.5 text-success" />}
                    {Number.isFinite(b) ? r.fmt(b) : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
