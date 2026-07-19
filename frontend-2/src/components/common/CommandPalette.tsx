import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Search } from "lucide-react";
import Monogram from "./Monogram";
import { useMiniQuote } from "@/hooks/useMiniQuote";
import { formatPrice, formatPct } from "@/lib/format";
import type { SearchTicker } from "@/contexts/SearchContext";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onSelect: (symbol: string) => void;
  tickers: SearchTicker[];
  recent: string[];
  popular: string[];
}

interface Row {
  symbol: string;
  name: string;
  section: "Recent" | "Popular" | "Results";
}

function PaletteRow({ row, active, onSelect }: { row: Row; active: boolean; onSelect: (s: string) => void }) {
  const q = useMiniQuote(row.symbol);
  const isUp = (q.changePct ?? 0) >= 0;
  return (
    <button
      type="button"
      data-active={active}
      onClick={() => onSelect(row.symbol)}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
        active ? "bg-primary/8" : "hover:bg-secondary"
      }`}
    >
      <Monogram symbol={row.symbol} size={30} />
      <div className="min-w-0 flex-1">
        <span className="font-mono text-sm font-semibold text-foreground">{row.symbol}</span>
        {row.name && <span className="ml-2 text-xs text-muted-foreground truncate">{row.name}</span>}
      </div>
      <div className="text-right shrink-0">
        {!q.loading && q.price != null && (
          <>
            <div className="font-mono text-xs text-foreground">{formatPrice(q.price)}</div>
            <div className={`font-mono text-[11px] ${isUp ? "kpi-positive" : "kpi-negative"}`}>{formatPct(q.changePct)}</div>
          </>
        )}
      </div>
    </button>
  );
}

export default function CommandPalette({ open, onClose, onSelect, tickers, recent, popular }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(0);
      setTimeout(() => inputRef.current?.focus(), 30);
    }
  }, [open]);

  const nameBySymbol = useMemo(() => {
    const m = new Map<string, string>();
    for (const t of tickers) m.set(t.symbol, t.name);
    return m;
  }, [tickers]);

  const rows = useMemo<Row[]>(() => {
    const q = query.trim().toLowerCase();
    if (q) {
      return tickers
        .filter((t) => t.symbol.toLowerCase().includes(q) || t.name.toLowerCase().includes(q))
        .slice(0, 8)
        .map((t) => ({ symbol: t.symbol, name: t.name, section: "Results" as const }));
    }
    const recentRows: Row[] = recent
      .slice(0, 5)
      .map((s) => ({ symbol: s, name: nameBySymbol.get(s) || "", section: "Recent" as const }));
    const popularRows: Row[] = popular
      .filter((s) => !recent.includes(s))
      .slice(0, 6)
      .map((s) => ({ symbol: s, name: nameBySymbol.get(s) || "", section: "Popular" as const }));
    return [...recentRows, ...popularRows];
  }, [query, tickers, recent, popular, nameBySymbol]);

  useEffect(() => {
    setCursor(0);
  }, [query]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setCursor((c) => Math.min(c + 1, rows.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const r = rows[cursor];
        if (r) onSelect(r.symbol);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, rows, cursor, onClose, onSelect]);

  let lastSection = "";

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[100] flex items-start justify-center px-4 pt-[12vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <div className="absolute inset-0 bg-foreground/20" onClick={onClose} />
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 400, damping: 32 }}
            className="relative w-full max-w-[560px] rounded-2xl border border-border bg-card shadow-pop overflow-hidden"
          >
            <div className="flex items-center gap-3 px-4 border-b border-border">
              <Search className="w-4 h-4 text-muted-foreground shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search any stock, e.g. AAPL, TSLA…"
                className="flex-1 h-14 bg-transparent text-[15px] text-foreground placeholder:text-muted-foreground focus:outline-none"
              />
              <kbd className="hidden sm:inline-flex items-center rounded border border-border bg-secondary px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
                ESC
              </kbd>
            </div>
            <div className="max-h-[52vh] overflow-y-auto p-2">
              {rows.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-muted-foreground">No matches found.</p>
              ) : (
                rows.map((row, i) => {
                  const showHeader = row.section !== lastSection;
                  lastSection = row.section;
                  return (
                    <div key={`${row.section}-${row.symbol}`}>
                      {showHeader && <p className="label-caps px-3 pt-3 pb-1">{row.section}</p>}
                      <PaletteRow row={row} active={i === cursor} onSelect={onSelect} />
                    </div>
                  );
                })
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
