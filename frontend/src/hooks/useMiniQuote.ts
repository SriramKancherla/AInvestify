import { useEffect, useState } from "react";
import { apiJson } from "@/lib/api";

export interface MiniQuote {
  symbol: string;
  price: number | null;
  changePct: number | null;
  spark: number[];
  loading: boolean;
}

interface ChartResponse {
  points?: Array<{ close?: number }>;
}

const TTL_MS = 60_000;
const _cache = new Map<string, { at: number; data: Omit<MiniQuote, "loading"> }>();
const _inflight = new Map<string, Promise<Omit<MiniQuote, "loading">>>();

async function load(symbol: string): Promise<Omit<MiniQuote, "loading">> {
  const clean = symbol.trim().toUpperCase();
  const cached = _cache.get(clean);
  if (cached && Date.now() - cached.at < TTL_MS) return cached.data;
  const existing = _inflight.get(clean);
  if (existing) return existing;

  const req = apiJson<ChartResponse>(`/chart/${encodeURIComponent(clean)}?period=1mo&interval=1d`)
    .then((res) => {
      const closes = (res.points || [])
        .map((p) => Number(p.close))
        .filter((n) => Number.isFinite(n));
      const price = closes.length ? closes[closes.length - 1] : null;
      const first = closes.length ? closes[0] : null;
      const changePct =
        first && first !== 0 && price != null ? ((price - first) / first) * 100 : null;
      const data = { symbol: clean, price, changePct, spark: closes };
      _cache.set(clean, { at: Date.now(), data });
      return data;
    })
    .catch(() => ({ symbol: clean, price: null, changePct: null, spark: [] as number[] }))
    .finally(() => {
      _inflight.delete(clean);
    });
  _inflight.set(clean, req);
  return req;
}

/** Lightweight per-symbol quote (last price + change% + sparkline) for cards. */
export function useMiniQuote(symbol: string | undefined | null): MiniQuote {
  const clean = (symbol || "").trim().toUpperCase();
  const [state, setState] = useState<MiniQuote>({
    symbol: clean,
    price: null,
    changePct: null,
    spark: [],
    loading: Boolean(clean),
  });

  useEffect(() => {
    if (!clean) return;
    let mounted = true;
    setState((s) => ({ ...s, symbol: clean, loading: true }));
    load(clean).then((data) => {
      if (mounted) setState({ ...data, loading: false });
    });
    return () => {
      mounted = false;
    };
  }, [clean]);

  return state;
}
