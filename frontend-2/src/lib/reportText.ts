import type { StockData } from "@/hooks/useStockStore";

type PortfolioSummary = {
  total_market_value?: number;
  total_cost_basis?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
} | null;

type CalendarEvent = {
  event_type?: string;
  event_date?: string;
  risk_level?: string;
  details?: string;
  source?: string;
};

function section(title: string, body: string[]): string[] {
  return ["", "=== " + title + " ===", ...body, ""];
}

function stockLines(label: string, data: StockData | null | undefined): string[] {
  if (!data?.ticker) {
    return [`${label}: (not included)`];
  }
  const lines: string[] = [];
  lines.push(`${label}`);
  lines.push(`  Company: ${data.company_name || "—"}`);
  lines.push(`  Ticker: ${data.ticker}`);
  if (data.recommendation) lines.push(`  Model view: ${data.recommendation}`);
  lines.push("  --- Price & activity ---");
  lines.push(`  Current price: ${data.current_price ?? "—"}`);
  lines.push(`  Change % (chart window): ${data.change_percent ?? "—"}`);
  lines.push(`  Market cap: ${data.market_cap ?? "—"}`);
  lines.push(`  Volume: ${data.volume ?? "—"}`);
  lines.push(`  52W high: ${data.high_52w ?? "—"}`);
  lines.push(`  52W low: ${data.low_52w ?? "—"}`);
  lines.push(`  Sentiment: ${data.sentiment_score ?? "—"} (${data.sentiment_label ?? "—"})`);
  const f = data.fundamentals;
  if (f && Object.keys(f).length) {
    lines.push("  --- Model & fundamentals ---");
    for (const [k, v] of Object.entries(f)) {
      lines.push(`    - ${k}: ${v}`);
    }
  }
  const news = data.news;
  if (news?.length) {
    lines.push("  --- Headlines (sample) ---");
    for (const n of news.slice(0, 12)) {
      lines.push(`    - [${n.sentiment}] ${n.title}${n.source ? ` (${n.source})` : ""}`);
      if (n.link) lines.push(`      ${n.link}`);
    }
  }
  return lines;
}

/** Full plain-text report for email (keep reasonably short for mailto limits). */
export function buildReportPlainText(opts: {
  selected: StockData | null | undefined;
  compare: StockData | null | undefined;
  portfolio: PortfolioSummary;
  events: CalendarEvent[] | undefined;
}): string {
  const out: string[] = [];
  out.push("AINVESTIFY — EQUITY INSIGHT REPORT");
  out.push(`Generated: ${new Date().toISOString()}`);
  out.push("");
  out.push(
    "This mirrors your dashboard: model scores, metrics, and news samples. For education/research only — not investment advice."
  );
  out.push(...section("PRIMARY", stockLines("Primary holding", opts.selected)));
  if (opts.compare?.ticker) {
    out.push(...section("COMPARE", stockLines("Comparison holding", opts.compare)));
  }

  const p = opts.portfolio;
  if (p && (p.total_market_value != null || p.total_cost_basis != null)) {
    out.push(...section("PORTFOLIO (session)", [
      `Total market value: ${p.total_market_value ?? "—"}`,
      `Total cost basis: ${p.total_cost_basis ?? "—"}`,
      `Total P/L: ${p.total_pnl ?? "—"}`,
      `Total P/L %: ${p.total_pnl_pct ?? "—"}`,
      "",
      "Note: Not tied to a user account yet; server session only.",
    ]));
  }

  const ev = opts.events;
  if (ev?.length) {
    const elines: string[] = ["Calendar entries (illustrative templates, not exchange filings):"];
    for (const e of ev.slice(0, 8)) {
      elines.push(`- ${e.event_type || "Event"} (${e.event_date || "—"}, ${e.risk_level || "—"})`);
      if (e.details) elines.push(`  ${e.details}`);
      if (e.source) elines.push(`  Source: ${e.source}`);
    }
    out.push(...section("CALENDAR", elines));
  }

  out.push(
    ...section("DISCLAIMER", [
      "Outputs may be wrong or incomplete. Verify facts; consult a licensed professional before investing.",
    ])
  );

  return out.join("\n");
}

export const MAILTO_BODY_MAX = 2800;
