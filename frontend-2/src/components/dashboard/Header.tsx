import { useMemo, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { useAuth } from "@/contexts/AuthContext";

interface HeaderProps {
  onHome: () => void;
}

function headerDisplayName(session: Session | null): string {
  const meta = session?.user?.user_metadata as { first_name?: string; last_name?: string } | undefined;
  const first = (meta?.first_name ?? "").trim();
  const last = (meta?.last_name ?? "").trim();
  if (first && last) return `${first} ${last}`;
  if (first || last) return first || last;
  const email = session?.user?.email?.trim();
  if (email) return email.split("@")[0] || email;
  return "there";
}

export default function Header({ onHome }: HeaderProps) {
  const { session, signOut } = useAuth();
  const welcomeName = headerDisplayName(session);

  return (
    <header className="sticky top-0 z-50 w-full max-w-[100vw] overflow-x-hidden border-b border-border bg-background/95 backdrop-blur-xl">
      <div className="mx-auto max-w-[1400px] flex items-center justify-between gap-3 h-14 sm:h-16 px-4 sm:px-6">
        {/* Logo */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={onHome}
            title="Back to dashboard"
            aria-label="Back to dashboard"
            className="flex items-center gap-2.5 bg-transparent border-none p-0 m-0 cursor-pointer"
          >
            <img src="/favicon.svg" alt="AInvestify" className="w-8 h-8 rounded-lg object-cover" />
            <span className="text-lg font-bold tracking-tight">
              <span className="gradient-text">AI</span>
              <span className="text-foreground">nvestify</span>
            </span>
          </button>
        </div>

        <div className="flex-1 text-center text-sm sm:text-base font-semibold text-foreground tracking-tight">
          Welcome, <span className="text-primary">{welcomeName}</span>!!!
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={async () => {
              try {
                await signOut();
              } catch {
                // ignore signOut issues, still route to landing
              } finally {
                window.location.assign("/");
              }
            }}
            className="px-2.5 py-1.5 rounded border border-border text-xs text-muted-foreground hover:bg-secondary/60"
          >
            Sign out
          </button>
        </div>
      </div>
    </header>
  );
}
