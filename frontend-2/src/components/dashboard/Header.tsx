import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { Session } from "@supabase/supabase-js";
import { ChevronRight, LogOut, LayoutGrid, BookOpen } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import SearchBar from "@/components/common/SearchBar";

interface HeaderProps {
  onHome: () => void;
  /** Trailing breadcrumb segment, e.g. a ticker. */
  crumb?: string;
}

function displayName(session: Session | null): string {
  const meta = session?.user?.user_metadata as { first_name?: string; last_name?: string } | undefined;
  const first = (meta?.first_name ?? "").trim();
  const last = (meta?.last_name ?? "").trim();
  if (first && last) return `${first} ${last}`;
  if (first || last) return first || last;
  const email = session?.user?.email?.trim();
  if (email) return email.split("@")[0] || email;
  return "there";
}

function initialsOf(name: string): string {
  const parts = name.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return (name.slice(0, 2) || "?").toUpperCase();
}

export default function Header({ onHome, crumb }: HeaderProps) {
  const { session, signOut } = useAuth();
  const navigate = useNavigate();
  const name = displayName(session);
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
              aria-label="Account menu"
            >
              {initialsOf(name)}
            </button>
            {menuOpen && (
              <div className="absolute right-0 mt-2 w-60 rounded-xl border border-border bg-card shadow-pop p-1.5 z-50">
                <div className="px-3 py-2.5">
                  <p className="text-xs text-muted-foreground">Welcome back,</p>
                  <p className="text-sm font-semibold text-foreground truncate">{name}</p>
                </div>
                <div className="h-px bg-border my-1" />
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
                <div className="h-px bg-border my-1" />
                <button
                  type="button"
                  onClick={async () => {
                    try { await signOut(); } catch { /* still route out */ } finally { window.location.assign("/"); }
                  }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary transition-colors"
                >
                  <LogOut className="w-4 h-4 text-muted-foreground" /> Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
