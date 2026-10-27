import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronRight, LayoutGrid, BookOpen, Home, ExternalLink, Workflow, PlaySquare } from "lucide-react";
import SearchBar from "@/components/common/SearchBar";

interface HeaderProps {
  onHome: () => void;
  /** Trailing breadcrumb segment, e.g. a ticker. */
  crumb?: string;
}

export default function Header({ onHome, crumb }: HeaderProps) {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="mx-auto max-w-[1400px] flex items-center gap-4 h-16 px-4 sm:px-6">
        <button type="button" onClick={onHome} className="flex items-center gap-2 shrink-0 cursor-pointer">
          <img src="/favicon.svg" alt="AInvestify" className="w-8 h-8 rounded-lg" />
          <span className="text-lg font-semibold tracking-tightish hidden sm:block">
            <span className="gradient-text">AI</span>
            <span className="text-foreground">nvestify</span>
          </span>
        </button>

        <div className="flex items-center gap-1.5 text-sm text-muted-foreground min-w-0">
          <ChevronRight className="w-4 h-4 shrink-0 opacity-50" />
          <button type="button" onClick={onHome} className="hover:text-foreground transition-colors">
            Dashboard
          </button>
          {crumb && (
            <>
              <ChevronRight className="w-4 h-4 shrink-0 opacity-50" />
              <span className="font-mono font-medium text-foreground truncate">{crumb}</span>
            </>
          )}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <div className="hidden md:block">
            <SearchBar size="sm" />
          </div>

          <div className="relative" ref={menuRef}>
            <button
              type="button"
              onClick={() => setMenuOpen((o) => !o)}
              className="h-10 w-10 rounded-full bg-primary/10 text-primary font-semibold text-sm flex items-center justify-center hover:bg-primary/15 transition-colors"
              aria-label="Menu"
            >
              AI
            </button>
            {menuOpen && (
              <div className="absolute right-0 mt-2 w-60 rounded-xl border border-border bg-card shadow-pop p-1.5 z-50">
                <button
                  type="button"
                  onClick={() => { setMenuOpen(false); navigate("/"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <Home className="w-4 h-4 text-muted-foreground" /> Landing
                </button>
                <button
                  type="button"
                  onClick={() => { setMenuOpen(false); navigate("/watchlists"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <LayoutGrid className="w-4 h-4 text-muted-foreground" /> Watchlists
                </button>
                <button
                  type="button"
                  onClick={() => { setMenuOpen(false); navigate("/models"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <BookOpen className="w-4 h-4 text-muted-foreground" /> How models work
                </button>
                <div className="my-1.5 border-t border-border" />
                <p className="px-3 pb-1 pt-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                  Reviewer tools
                </p>
                <a
                  href="http://127.0.0.1:8766/"
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => setMenuOpen(false)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <Workflow className="w-4 h-4 text-muted-foreground" /> Backend walkthrough
                  <ExternalLink className="ml-auto w-3.5 h-3.5 text-muted-foreground" />
                </a>
                <a
                  href="http://127.0.0.1:8767/"
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => setMenuOpen(false)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <PlaySquare className="w-4 h-4 text-muted-foreground" /> Inference pipeline
                  <ExternalLink className="ml-auto w-3.5 h-3.5 text-muted-foreground" />
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
