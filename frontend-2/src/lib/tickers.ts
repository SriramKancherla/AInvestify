import { apiJson } from "@/lib/api";

/** Resolve watchlist/header ticker against GET /tickers; fallback to uppercase input if missing from list. */
export async function resolveTickerForInsights(raw: string): Promise<string> {
  const want = raw.trim().toUpperCase();
  if (!want) return "";
  try {
    const data = await apiJson<{ tickers?: Array<{ symbol?: string }> }>("/tickers");
    const list = data.tickers || [];
    const hit = list.find((t) => String(t.symbol || "").toUpperCase() === want);
    if (hit?.symbol) return String(hit.symbol).toUpperCase();
  } catch {
    // offline or API error — still allow navigation with normalized symbol
  }
  return want;
}
