import { Search } from "lucide-react";
import { useSearch } from "@/contexts/SearchContext";

interface SearchBarProps {
  size?: "lg" | "sm";
  className?: string;
}

/** Command-style search trigger. Opens the global palette; never a native dropdown. */
export default function SearchBar({ size = "lg", className = "" }: SearchBarProps) {
  const { openSearch } = useSearch();
  const isLg = size === "lg";
  return (
    <button
      type="button"
      onClick={openSearch}
      className={`group flex items-center gap-3 rounded-full border border-border bg-card text-left transition-all hover:border-[hsl(230_6%_84%)] hover:shadow-hero ${
        isLg ? "h-14 px-6 w-full max-w-xl" : "h-10 px-4"
      } ${className}`}
    >
      <Search className={`text-muted-foreground shrink-0 ${isLg ? "w-5 h-5" : "w-4 h-4"}`} />
      <span className={`flex-1 text-muted-foreground ${isLg ? "text-[15px]" : "text-sm"}`}>
        {isLg ? "Search any stock, e.g. AAPL, TSLA…" : "Search stocks…"}
      </span>
      <kbd className="hidden sm:inline-flex items-center gap-0.5 rounded-md border border-border bg-secondary px-2 py-1 text-[11px] font-mono text-muted-foreground">
        ⌘K
      </kbd>
    </button>
  );
}
