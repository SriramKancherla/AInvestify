import { useEffect, useMemo, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import { Activity, RefreshCw, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ForecastChartPanelProps {
  ticker: string;
}

type HorizonDays = 5 | 21 | 63 | 126 | 189 | 252;

const HORIZONS: { days: HorizonDays; label: string }[] = [
  { days: 5, label: "1W" },
  { days: 21, label: "1M" },
  { days: 63, label: "3M" },
  { days: 126, label: "6M" },
  { days: 189, label: "9M" },
  { days: 252, label: "1Y" },
];

type PriceEstimate = {
  last_price?: number;
  predicted_price?: number;
  predicted_pct_change?: number;
  range_low?: number;
  range_high?: number;
  expected_move_pct?: number;
  annualised_vol_pct?: number;
  ensemble_members?: number;
  ensemble_spread_pct?: number;
  prices?: number[];
  band_low?: number[];
  band_high?: number[];
  /** "volatility_cone" when the honest cone produced this. */
  method?: string;
  /** The band's probability. 0.68 — measured at 67.8–72.5% coverage. */
  confidence_level?: number;
  /** Half-width at each future point, as a percent of today's price. */
  uncertainty_pct?: number[];
  trading_days_out?: number[];
  note?: string;
};

type ForecastPoint = {
  date: string;
  price: number;
  low: number;
  high: number;
  sigma?: number;
  model_sigma?: number;
  /** This point's own uncertainty — the number the panel leads with. */
  uncertainty_pct?: number;
  /** Trading days from today, so "day 3 of 5" is readable without counting. */
  days_out?: number;
};

type ForecastSeries = {
  history?: { date: string; close: number }[];
  forecast?: ForecastPoint[];
  anchor_date?: string;
  anchor_price?: number;
};

type ForecastPayload = {
  ticker?: string;
  horizon_days?: number;
  window_end?: string;
  target_end?: string;
  inference_seconds?: number;
  image_base64?: string;
  price_estimate?: PriceEstimate | null;
  render_mode?: string;
  forecast_mode?: string;
  ensemble_members?: number;
  series?: ForecastSeries | null;
  conditioning_note?: string;
  detail?: { error?: { message?: string } } | string;
};

const LIVE_CONDITIONING_NOTE =
  "Fundamentals are the SEC values that were public on the window end. Form 8-K tone is the item-code score of filings in the window. A day with no filing stays neutral.";

/** One row per date. `close` is real history; the rest is forecast.
 *  The anchor date carries both, which is what visually joins the two. */
type ChartRow = {
  date: string;
  close?: number;
  forecast?: number;
  band?: [number, number];
  sigma?: number;
  modelSigma?: number;
  uncertaintyPct?: number;
  daysOut?: number;
};

function formatUsd(n: number | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

function formatCompactUsd(n: number | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `$${n >= 1000 ? n.toFixed(0) : n.toFixed(2)}`;
}

function formatDay(v: string | undefined, long: boolean): string {
  if (!v) return "—";
  const d = new Date(`${v}T00:00:00`);
  if (Number.isNaN(d.getTime())) return v;
  return long
    ? d.toLocaleString("en-US", { month: "short", year: "2-digit" })
    : d.toLocaleString("en-US", { month: "short", day: "numeric" });
}

type TooltipEntry = { dataKey?: string | number; value?: number | number[]; payload?: ChartRow };

function ForecastTooltip({
  active,
  payload,
  label,
  isCone,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string;
  isCone?: boolean;
}) {
  if (!active || !payload?.length) return null;

  const row = payload[0]?.payload;
  const close = payload.find((p) => p.dataKey === "close")?.value;
  const forecast = payload.find((p) => p.dataKey === "forecast")?.value;

  const rows: { label: string; value: string; accent?: boolean }[] = [];
  if (typeof close === "number") rows.push({ label: "Actual", value: formatUsd(close) });

  if (isCone) {
    // A cone has no "predicted price" to report — the centre is today's price by
    // construction. What varies from day to day, and is the whole point of
    // hovering, is how uncertain THIS day is.
    if (row?.daysOut != null)
      rows.push({ label: "Trading days out", value: String(row.daysOut) });
    if (typeof forecast === "number")
      rows.push({ label: "Today's price", value: formatUsd(forecast) });
    if (row?.uncertaintyPct != null)
      rows.push({
        label: "Uncertainty",
        value: `± ${row.uncertaintyPct.toFixed(2)}%`,
        accent: true,
      });
    const b = row?.band;
    if (Array.isArray(b) && b.length === 2 && b[0] !== b[1])
      rows.push({ label: "Range", value: `${formatUsd(b[0])} – ${formatUsd(b[1])}` });
  } else {
    if (typeof forecast === "number")
      rows.push({ label: "Predicted", value: formatUsd(forecast), accent: true });
    const band = row?.band;
    if (Array.isArray(band) && band.length === 2 && band[0] !== band[1])
      rows.push({ label: "Range", value: `${formatUsd(band[0])} – ${formatUsd(band[1])}` });
    if (row?.sigma) rows.push({ label: "Uncertainty", value: `± ${formatUsd(row.sigma)}` });
    if (row?.modelSigma != null && row.modelSigma > 0)
      rows.push({ label: "Model spread", value: `± ${formatUsd(row.modelSigma)}` });
  }
  if (!rows.length) return null;

  return (
    <div
      className="rounded-[10px] border border-border px-3 py-2 text-xs"
      style={{ background: "hsl(var(--card))", boxShadow: "var(--shadow-pop)" }}
    >
      <div className="label-caps mb-1.5">{formatDay(label, false)}</div>
      {rows.map((r) => (
        <div key={r.label} className="flex items-baseline justify-between gap-6 leading-relaxed">
          <span className="text-muted-foreground">{r.label}</span>
          <span className={`font-mono ${r.accent ? "font-medium" : ""} text-foreground`}>
            {r.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Identity is never carried by colour alone: solid vs dashed distinguishes the
 *  two series, and this names them. */
function ChartLegend({ accent, isCone }: { accent: string; isCone: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <svg width="16" height="8" aria-hidden="true">
          <line x1="0" y1="4" x2="16" y2="4" stroke="hsl(var(--foreground))" strokeWidth="1.5" strokeOpacity="0.75" />
        </svg>
        Actual
      </span>
      <span className="inline-flex items-center gap-1.5">
        <svg width="16" height="8" aria-hidden="true">
          <line x1="0" y1="4" x2="16" y2="4" stroke={accent} strokeWidth="2" strokeDasharray="5 3" />
        </svg>
        {isCone ? "Today's price, held flat" : "Predicted"}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <svg width="16" height="8" aria-hidden="true">
          <rect x="0" y="1" width="16" height="6" rx="1" fill={accent} fillOpacity="0.2" />
        </svg>
        {isCone ? "Uncertainty — widens each day" : "Uncertainty range"}
      </span>
    </div>
  );
}

function ForecastChart({
  series,
  accent,
  isCone,
  longDates,
}: {
  series: ForecastSeries;
  accent: string;
  isCone: boolean;
  longDates: boolean;
}) {
  const rows = useMemo<ChartRow[]>(() => {
    const history = series.history ?? [];
    const forecast = series.forecast ?? [];
    const anchorPrice = series.anchor_price;

    const out: ChartRow[] = history.map((h, i) => {
      // The last real point also seeds the forecast line and collapses every
      // band to zero width, so the two lines join instead of floating apart.
      const isAnchor = i === history.length - 1;
      if (!(isAnchor && anchorPrice != null)) return { date: h.date, close: h.close };
      return {
        date: h.date,
        close: h.close,
        forecast: anchorPrice,
        band: [anchorPrice, anchorPrice],
        uncertaintyPct: 0,
        daysOut: 0,
      };
    });

    for (const f of forecast) {
      out.push({
        date: f.date,
        forecast: f.price,
        band: [f.low, f.high],
        sigma: f.sigma,
        modelSigma: f.model_sigma,
        uncertaintyPct: f.uncertainty_pct,
        daysOut: f.days_out,
      });
    }
    return out;
  }, [series]);

  const last = series.forecast?.[series.forecast.length - 1];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 10, right: 8, left: 0, bottom: 0 }}>
        <defs>
          {/* Soft neutral wash under real history — present, but never competing
              with the forecast, which is the thing worth looking at. */}
          <linearGradient id="fcHistory" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--foreground))" stopOpacity={0.1} />
            <stop offset="100%" stopColor="hsl(var(--foreground))" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="fcBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity={0.05} />
            <stop offset="50%" stopColor={accent} stopOpacity={0.2} />
            <stop offset="100%" stopColor={accent} stopOpacity={0.05} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.55} vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          minTickGap={44}
          tickFormatter={(v: string) => formatDay(v, longDates)}
        />
        <YAxis
          domain={["auto", "auto"]}
          orientation="right"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={58}
          tickFormatter={(v: number) => formatCompactUsd(Number(v))}
        />
        <Tooltip
          content={<ForecastTooltip isCone={isCone} />}
          cursor={{ stroke: "hsl(var(--muted-foreground))", strokeDasharray: "3 3", strokeOpacity: 0.45 }}
        />

        {/* One band. Its width at any date IS that date's uncertainty, so the
            shape of the shading and the numbers in the table are the same fact. */}
        <Area
          dataKey="band"
          type="monotone"
          stroke="none"
          fill="url(#fcBand)"
          connectNulls={false}
          activeDot={false}
          animationDuration={550}
        />

        <Area
          dataKey="close"
          type="monotone"
          stroke="hsl(var(--foreground))"
          strokeOpacity={0.75}
          strokeWidth={1.5}
          fill="url(#fcHistory)"
          connectNulls={false}
          activeDot={{ r: 3, fill: "hsl(var(--foreground))", stroke: "hsl(var(--card))", strokeWidth: 2 }}
          animationDuration={550}
        />
        <Line
          dataKey="forecast"
          type="monotone"
          stroke={accent}
          strokeWidth={2.25}
          strokeDasharray="6 4"
          strokeLinecap="round"
          dot={false}
          connectNulls={false}
          activeDot={{ r: 3.5, fill: accent, stroke: "hsl(var(--card))", strokeWidth: 2 }}
          animationDuration={550}
        />

        {series.anchor_date && (
          <ReferenceLine
            x={series.anchor_date}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="2 4"
            strokeOpacity={0.5}
            label={{
              value: "now",
              position: "top",
              fill: "hsl(var(--muted-foreground))",
              fontSize: 10,
            }}
          />
        )}
        {series.anchor_date && series.anchor_price != null && (
          <ReferenceDot
            x={series.anchor_date}
            y={series.anchor_price}
            r={3.5}
            fill="hsl(var(--foreground))"
            stroke="hsl(var(--card))"
            strokeWidth={2}
            isFront
          />
        )}
        {last && (
          <ReferenceDot
            x={last.date}
            y={last.price}
            r={4}
            fill={accent}
            stroke="hsl(var(--card))"
            strokeWidth={2}
            isFront
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Every plotted value, readable without hovering. A tooltip should enhance a
 *  chart, never be the only way to get a number out of it. */
function ForecastTable({
  points,
  isCone,
  accentClass,
}: {
  points: ForecastPoint[];
  isCone: boolean;
  accentClass: string;
}) {
  // The widest bar in the column sets the scale, so the bars read as "how much
  // more uncertain is the last day than the first" rather than as decoration.
  const maxUnc = Math.max(...points.map((p) => p.uncertainty_pct ?? 0), 0.0001);

  return (
    <div className="mt-3 max-h-72 overflow-y-auto rounded-lg border border-border">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-card">
          <tr className="border-b border-border">
            <th className="label-caps text-left font-medium px-3 py-2">Date</th>
            {isCone ? (
              <>
                <th className="label-caps text-right font-medium px-3 py-2">Day</th>
                <th className="label-caps text-right font-medium px-3 py-2">Price</th>
                <th className="label-caps text-right font-medium px-3 py-2">Uncertainty</th>
                <th className="label-caps text-right font-medium px-3 py-2">Range</th>
              </>
            ) : (
              <>
                <th className="label-caps text-right font-medium px-3 py-2">Predicted</th>
                <th className="label-caps text-right font-medium px-3 py-2">Low</th>
                <th className="label-caps text-right font-medium px-3 py-2">High</th>
                <th className="label-caps text-right font-medium px-3 py-2">Uncertainty</th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {points.map((p, i) => (
            <tr key={p.date} className={i % 2 ? "bg-secondary/20" : undefined}>
              <td className="px-3 py-1.5 text-muted-foreground whitespace-nowrap">
                {formatDay(p.date, false)}
              </td>
              {isCone ? (
                <>
                  <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                    {p.days_out ?? i + 1}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-foreground">
                    {formatUsd(p.price)}
                  </td>
                  <td className="px-3 py-1.5 text-right whitespace-nowrap">
                    <span className="inline-flex items-center justify-end gap-1.5">
                      <span
                        aria-hidden="true"
                        className="h-1 rounded-full bg-primary/45"
                        style={{ width: `${Math.max(3, ((p.uncertainty_pct ?? 0) / maxUnc) * 36)}px` }}
                      />
                      <span className="font-mono font-medium text-foreground tabular-nums">
                        ± {(p.uncertainty_pct ?? 0).toFixed(2)}%
                      </span>
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted-foreground whitespace-nowrap">
                    {formatUsd(p.low)} – {formatUsd(p.high)}
                  </td>
                </>
              ) : (
                <>
                  <td className={`px-3 py-1.5 text-right font-mono font-medium ${accentClass}`}>
                    {formatUsd(p.price)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                    {formatUsd(p.low)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                    {formatUsd(p.high)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                    ± {formatUsd(p.sigma)}
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ForecastChartPanel({ ticker }: ForecastChartPanelProps) {
  const [horizon, setHorizon] = useState<HorizonDays>(63);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ windowEnd?: string; targetEnd?: string; seconds?: number } | null>(null);
  const [priceEstimate, setPriceEstimate] = useState<PriceEstimate | null>(null);
  const [series, setSeries] = useState<ForecastSeries | null>(null);
  const [renderMode, setRenderMode] = useState<string | null>(null);
  const [conditioningNote, setConditioningNote] = useState(LIVE_CONDITIONING_NOTE);
  const [loading, setLoading] = useState(false);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  // Bumping this re-runs the effect with refresh=1, bypassing the server cache.
  const [reloadKey, setReloadKey] = useState(0);
  const [showTable, setShowTable] = useState(false);
  const prevTicker = useRef(ticker);

  const horizonLabel = useMemo(
    () => HORIZONS.find((h) => h.days === horizon)?.label ?? `${horizon}d`,
    [horizon]
  );

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setUnavailable(null);

    // Only blank the panel when the *stock* changes — showing AAPL's chart
    // while MSFT loads would be a lie. Switching horizon or regenerating keeps
    // the previous render (dimmed, see below) instead of flashing a skeleton
    // and jumping the layout every time a tab is clicked.
    if (prevTicker.current !== ticker) {
      prevTicker.current = ticker;
      setImageSrc(null);
      setMeta(null);
      setPriceEstimate(null);
      setSeries(null);
      setRenderMode(null);
    }

    const url =
      `/api/forecast/chart/${encodeURIComponent(ticker)}?horizon_days=${horizon}` +
      (reloadKey > 0 ? "&refresh=1" : "");

    fetch(apiUrl(url), { signal: controller.signal })
      .then(async (resp) => {
        const payload = (await resp.json().catch(() => ({}))) as ForecastPayload;
        if (!resp.ok) {
          const msg =
            (typeof payload.detail === "object" && payload.detail?.error?.message) ||
            (typeof payload.detail === "string" ? payload.detail : null) ||
            "Forecast is not available on this server.";
          throw new Error(msg);
        }
        if (!payload.image_base64) throw new Error("Forecast response did not include an image.");
        setImageSrc(`data:image/png;base64,${payload.image_base64}`);
        setMeta({
          windowEnd: payload.window_end,
          targetEnd: payload.target_end,
          seconds: payload.inference_seconds,
        });
        setPriceEstimate(payload.price_estimate ?? null);
        setSeries(payload.series ?? null);
        setConditioningNote(payload.conditioning_note || LIVE_CONDITIONING_NOTE);
        // Absent entirely => the running server predates this field, i.e. the
        // backend process wasn't restarted after the update.
        setRenderMode(payload.render_mode ?? "stale_server");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        const msg = e instanceof Error ? e.message : "Forecast unavailable.";
        setUnavailable(
          /socket hang up|Failed to fetch|NetworkError/i.test(msg)
            ? "Forecast server busy or restarted — wait a moment and refresh."
            : msg
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [ticker, horizon, reloadKey]);

  const isCone = priceEstimate?.method === "volatility_cone" || renderMode === "cone";
  const pct = priceEstimate?.predicted_pct_change ?? 0;
  const isUp = pct >= 0;
  const hasSeries = Boolean(series?.forecast?.length);
  // The cone's centre is flat by construction, so colouring it green or red
  // would imply a direction it explicitly does not claim.
  const accent = isCone
    ? "hsl(var(--primary))"
    : isUp
      ? "hsl(var(--success))"
      : "hsl(var(--destructive))";
  const points = series?.forecast ?? [];
  // Short horizons are a handful of rows, and those rows are the answer — make
  // them visible without a click. Long horizons stay collapsed so a 1Y view
  // doesn't open with fifty rows of table.
  const autoOpen = points.length > 0 && points.length <= 10;
  const tableOpen = showTable || autoOpen;

  return (
    <div className="elevated p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {isCone ? (
              <Activity className="w-4 h-4 text-primary" />
            ) : (
              <Sparkles className="w-4 h-4 text-primary" />
            )}
            <h2 className="text-sm font-semibold text-foreground tracking-tightish">
              {isCone ? "Outcome range" : "AI prediction"}
            </h2>
            <span className="ticker-badge">{horizonLabel}</span>
          </div>

          {priceEstimate && isCone ? (
            <div className="mt-2">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-mono text-[28px] leading-none font-semibold text-foreground">
                  {formatUsd(priceEstimate.last_price)}
                </span>
                <span className="text-sm text-muted-foreground">today</span>
              </div>
              <p className="mt-2 text-xs text-foreground/90 max-w-xl">
                Uncertainty grows from{" "}
                <span className="font-mono font-medium">
                  ± {(points[0]?.uncertainty_pct ?? 0).toFixed(2)}%
                </span>{" "}
                on {formatDay(points[0]?.date, false)} to{" "}
                <span className="font-mono font-medium">
                  ± {(points[points.length - 1]?.uncertainty_pct ?? 0).toFixed(2)}%
                </span>{" "}
                by {formatDay(meta?.targetEnd, true)} — across {points.length} trading{" "}
                {points.length === 1 ? "day" : "days"}.
              </p>
            </div>
          ) : priceEstimate ? (
            <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="font-mono text-[28px] leading-none font-semibold text-foreground">
                {formatUsd(priceEstimate.predicted_price)}
              </span>
              <span
                className={`inline-flex items-center gap-1 font-mono text-sm font-medium ${
                  isUp ? "kpi-positive" : "kpi-negative"
                }`}
              >
                {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {pct > 0 ? "+" : ""}
                {pct.toFixed(2)}%
                <span className="text-muted-foreground font-sans">
                  · from {formatUsd(priceEstimate.last_price)}
                </span>
              </span>
            </div>
          ) : (
            <p className="mt-2 text-xs text-muted-foreground max-w-xl">
              How far this stock could plausibly travel, drawn against real history.
            </p>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setReloadKey((k) => k + 1)}
            disabled={loading}
            title="Recalculate, ignoring the cached result"
            aria-label="Recalculate forecast"
            className="rounded-lg border border-border p-[7px] text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
          <div className="flex gap-1 rounded-lg border border-border p-1">
            {HORIZONS.map((h) => (
              <button
                key={h.days}
                type="button"
                onClick={() => setHorizon(h.days)}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                  horizon === h.days
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {h.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {renderMode === "stale_server" && (
        <p className="mt-4 rounded-lg border border-border bg-secondary/40 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
          This response came from a backend process started before the last update, so it's still
          serving the old chart. Restart the API server to pick up the new renderer.
        </p>
      )}

      <div
        className={`mt-5 w-full aspect-[2.6/1] min-h-[240px] transition-opacity duration-200 ${
          loading && (hasSeries || imageSrc) ? "opacity-40" : "opacity-100"
        }`}
      >
        {loading && !hasSeries && !imageSrc ? (
          <div className="h-full w-full skeleton-shimmer rounded-lg" />
        ) : unavailable ? (
          <div className="h-full w-full flex items-center justify-center px-6 text-center">
            <div>
              <p className="text-sm font-medium text-foreground">Forecast not ready</p>
              <p className="mt-2 text-xs text-muted-foreground leading-relaxed max-w-md">
                {unavailable}
              </p>
            </div>
          </div>
        ) : hasSeries && series ? (
          <ForecastChart
            series={series}
            accent={accent}
            isCone={isCone}
            longDates={horizon >= 126}
          />
        ) : imageSrc ? (
          // No series — fall back to the server-rendered image.
          <div className="h-full w-full flex items-center justify-center">
            <img
              src={imageSrc}
              alt={`${ticker} ${horizonLabel} outcome range`}
              className="max-h-full w-auto rounded-lg"
            />
          </div>
        ) : (
          <div className="h-full w-full flex items-center justify-center px-6 text-center">
            <p className="text-xs text-muted-foreground leading-relaxed max-w-md">
              No price history available for this ticker.
            </p>
          </div>
        )}
      </div>

      {hasSeries && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <ChartLegend accent={accent} isCone={isCone} />
          {!autoOpen && (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              className="text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2 decoration-border"
            >
              {showTable ? "Hide" : "Show"} all {points.length} days
            </button>
          )}
        </div>
      )}

      {tableOpen && points.length ? (
        <ForecastTable
          points={points}
          isCone={isCone}
          accentClass={isUp ? "kpi-positive" : "kpi-negative"}
        />
      ) : null}

      {priceEstimate && (
        <div className="mt-4 border-t border-border pt-3 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex flex-wrap items-baseline gap-x-2">
            {isCone ? (
              <>
                <span className="label-caps">Range at {horizonLabel}</span>
                <span className="font-mono text-foreground">
                  {formatUsd(priceEstimate.range_low)} – {formatUsd(priceEstimate.range_high)}
                </span>
                {priceEstimate.annualised_vol_pct != null && (
                  <span>· {priceEstimate.annualised_vol_pct.toFixed(0)}% volatility a year</span>
                )}
              </>
            ) : (
              <>
                <span className="label-caps">Range</span>
                <span className="font-mono text-foreground">
                  {formatUsd(priceEstimate.range_low)} – {formatUsd(priceEstimate.range_high)}
                </span>
                {priceEstimate.ensemble_members != null && (
                  <span>
                    · {priceEstimate.ensemble_members} samples, ±
                    {(priceEstimate.ensemble_spread_pct ?? 0).toFixed(1)}% agreement
                  </span>
                )}
              </>
            )}
          </span>
          <span className="font-mono">
            through {formatDay(meta?.targetEnd, true)}
            {meta?.seconds != null ? ` · ${meta.seconds}s` : ""}
          </span>
        </div>
      )}

      <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground/80">
        {isCone
          ? "Not a price prediction. The centre line is today's price held flat, because no method tested here forecast direction better than that. Each day's ± is how far the price would typically travel by then — inside that range about 2 times in 3, measured at 68–72% across 18 years of outcomes."
          : "Illustrative only. The model produces the shape of the path; the price scale comes from this stock's own historical volatility, so this is not a calibrated price target."}
      </p>
      <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground/80">
        {conditioningNote}
      </p>
    </div>
  );
}
