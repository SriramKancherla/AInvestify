import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiJson } from "@/lib/api";
import { POPULAR_TICKERS } from "@/hooks/useStockStore";
import CommandPalette from "@/components/common/CommandPalette";

export interface SearchTicker {
  symbol: string;
  name: string;
}

interface SearchContextValue {
  open: boolean;
  openSearch: () => void;
  closeSearch: () => void;
  tickers: SearchTicker[];
}

const SearchContext = createContext<SearchContextValue | null>(null);

const RECENT_KEY = "ainvestify_recent";

export function SearchProvider({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [tickers, setTickers] = useState<SearchTicker[]>([]);

  useEffect(() => {
    let mounted = true;
    apiJson<{ tickers?: Array<{ symbol?: string; name?: string }> }>("/tickers")
      .then((data) => {
        if (!mounted) return;
        const list = (data.tickers || [])
          .map((t) => ({ symbol: String(t.symbol || "").toUpperCase(), name: String(t.name || "").trim() }))
          .filter((t) => t.symbol);
        setTickers(list);
      })
      .catch(() => {
        setTickers(POPULAR_TICKERS.map((s) => ({ symbol: s, name: "" })));
      });
    return () => {
      mounted = false;
    };
  }, []);

  const openSearch = useCallback(() => setOpen(true), []);
  const closeSearch = useCallback(() => setOpen(false), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onSelect = useCallback(
    (symbol: string) => {
      setOpen(false);
      try {
        const prev: string[] = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
        const next = [symbol, ...prev.filter((t) => t !== symbol)].slice(0, 8);
        localStorage.setItem(RECENT_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      navigate(`/stock/${encodeURIComponent(symbol)}`);
    },
    [navigate]
  );

  const recent = useMemo<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    } catch {
      return [];
    }
  }, [open]);

  const value = useMemo(() => ({ open, openSearch, closeSearch, tickers }), [open, openSearch, closeSearch, tickers]);

  return (
    <SearchContext.Provider value={value}>
      {children}
      <CommandPalette
        open={open}
        onClose={closeSearch}
        onSelect={onSelect}
        tickers={tickers}
        recent={recent}
        popular={POPULAR_TICKERS}
      />
    </SearchContext.Provider>
  );
}

export function useSearch() {
  const ctx = useContext(SearchContext);
  if (!ctx) throw new Error("useSearch must be used within SearchProvider");
  return ctx;
}
