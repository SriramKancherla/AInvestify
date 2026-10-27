import { useEffect, useMemo, useRef, useState } from "react";
import { ExternalLink, FileText } from "lucide-react";
import { apiUrl } from "@/lib/api";

interface SecFilingsPanelProps {
  ticker: string;
}

type Filing = {
  form: string;
  form_label: string;
  is_amendment?: boolean;
  filed: string;
  period?: string | null;
  description?: string | null;
  items?: string[];
  size_bytes?: number | null;
  url?: string | null;
  index_url: string;
  accession: string;
};

type FilingsPayload = {
  ticker?: string;
  cik?: string;
  company?: string;
  exchange?: string | null;
  sic_description?: string | null;
  total_available?: number;
  counts_by_form?: Record<string, number>;
  forms_included?: string[];
  filings?: Filing[];
  company_url?: string;
  detail?: { error?: { message?: string; code?: string } } | string;
};

/** Ordered so the filter row reads the way people think about filings, rather
 *  than alphabetically or by however many happen to exist. */
const FORM_ORDER = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"];

function formatDay(v: string | undefined | null): string {
  if (!v) return "—";
  const d = new Date(`${v}T00:00:00`);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatSize(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return "";
  const mb = bytes / 1_048_576;
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** A 10-K and an 8-K are different kinds of thing; colour lets the eye group a
 *  long list without reading every row. Never the only cue — the form code is
 *  always spelled out beside it. */
function formTone(form: string): string {
  if (form.startsWith("10-K")) return "bg-primary/10 text-primary border-primary/20";
  if (form.startsWith("10-Q")) return "bg-secondary text-foreground border-border";
  if (form.startsWith("8-K")) return "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20";
  return "bg-secondary/60 text-muted-foreground border-border";
}

export default function SecFilingsPanel({ ticker }: SecFilingsPanelProps) {
  const [data, setData] = useState<FilingsPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [expanded, setExpanded] = useState(false);
  const prevTicker = useRef(ticker);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    if (prevTicker.current !== ticker) {
      prevTicker.current = ticker;
      setData(null);
      setFilter("all");
      setExpanded(false);
    }

    fetch(apiUrl(`/api/filings/${encodeURIComponent(ticker)}?limit=60`), { signal: controller.signal })
      .then(async (resp) => {
        const payload = (await resp.json().catch(() => ({}))) as FilingsPayload;
        if (!resp.ok) {
          throw new Error(
            (typeof payload.detail === "object" && payload.detail?.error?.message) ||
              (typeof payload.detail === "string" ? payload.detail : null) ||
              "SEC filings are not available."
          );
        }
        setData(payload);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : "SEC filings unavailable.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [ticker]);

  const counts = data?.counts_by_form ?? {};
  const availableForms = useMemo(
    () => FORM_ORDER.filter((f) => (counts[f] ?? 0) > 0),
    [counts]
  );

  const filtered = useMemo(() => {
    const all = data?.filings ?? [];
    return filter === "all" ? all : all.filter((f) => f.form.startsWith(filter));
  }, [data, filter]);

  const shown = expanded ? filtered : filtered.slice(0, 8);

  return (
    <div className="elevated p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold text-foreground tracking-tightish">SEC filings</h2>
            {data?.cik && <span className="ticker-badge">CIK {data.cik.replace(/^0+/, "")}</span>}
          </div>
          <p className="mt-1.5 text-xs text-muted-foreground">
            {data?.company ? (
              <>
                {data.company}
                {data.sic_description ? ` · ${data.sic_description}` : ""}
              </>
            ) : (
              "Filed with the SEC, straight from EDGAR."
            )}
          </p>
        </div>
        {data?.company_url && (
          <a
            href={data.company_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            All filings on EDGAR
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>

      {availableForms.length > 1 && (
        <div className="mt-4 flex flex-wrap gap-1">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-md border transition-colors ${
              filter === "all"
                ? "bg-primary text-primary-foreground border-primary"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            All {data?.total_available ?? 0}
          </button>
          {availableForms.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`px-2.5 py-1 text-[11px] font-medium rounded-md border transition-colors ${
                filter === f
                  ? "bg-primary text-primary-foreground border-primary"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {f} {counts[f]}
            </button>
          ))}
        </div>
      )}

      <div className="mt-4">
        {loading && !data ? (
          <div className="space-y-2">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-12 w-full skeleton-shimmer rounded-lg" />
            ))}
          </div>
        ) : error ? (
          <div className="rounded-lg border border-border bg-secondary/40 px-3 py-3">
            <p className="text-xs font-medium text-foreground">Filings unavailable</p>
            <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{error}</p>
          </div>
        ) : !filtered.length ? (
          <p className="text-xs text-muted-foreground py-2">
            No {filter === "all" ? "" : `${filter} `}filings on record for {ticker}.
          </p>
        ) : (
          <ul className="divide-y divide-border rounded-lg border border-border overflow-hidden">
            {shown.map((f) => (
              <li key={f.accession} className="group">
                <a
                  href={f.url || f.index_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 px-3 py-2.5 hover:bg-secondary/40 transition-colors"
                >
                  <span
                    className={`shrink-0 mt-0.5 inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium ${formTone(
                      f.form
                    )}`}
                  >
                    {f.form}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-baseline gap-x-2">
                      <span className="text-xs font-medium text-foreground">{f.form_label}</span>
                      {f.is_amendment && (
                        <span className="text-[10px] text-muted-foreground">amended</span>
                      )}
                      {f.period && (
                        <span className="text-[10px] text-muted-foreground">
                          for period ending {formatDay(f.period)}
                        </span>
                      )}
                    </span>
                    {f.items?.length ? (
                      <span className="mt-0.5 block text-[11px] leading-snug text-muted-foreground">
                        {f.items.join(" · ")}
                      </span>
                    ) : null}
                  </span>
                  <span className="shrink-0 text-right">
                    <span className="block font-mono text-[11px] text-foreground">
                      {formatDay(f.filed)}
                    </span>
                    {formatSize(f.size_bytes) && (
                      <span className="block text-[10px] text-muted-foreground">
                        {formatSize(f.size_bytes)}
                      </span>
                    )}
                  </span>
                  <ExternalLink className="w-3 h-3 shrink-0 mt-1 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>

      {filtered.length > 8 && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-3 text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors underline underline-offset-2 decoration-border"
        >
          {expanded ? "Show fewer" : `Show all ${filtered.length}`}
        </button>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground/80">
        Annual (10-K), quarterly (10-Q), material events (8-K), proxy statements and registrations.
        Insider transaction forms (3/4/5) are excluded — they are most of a large company's filing
        volume and would bury everything else.
      </p>
    </div>
  );
}
