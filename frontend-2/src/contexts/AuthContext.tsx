import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";
import { apiJson } from "@/lib/api";

type AuthContextValue = {
  session: Session | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  /** Returns the new session when email confirmation is off; null when confirmation email was sent. */
  signUp: (email: string, password: string, metadata?: { first_name?: string; last_name?: string }) => Promise<Session | null>;
  signOut: () => Promise<void>;
  accessToken: string;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session || null);
      setLoading(false);
    });
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession || null);
      setLoading(false);
    });
    return () => {
      mounted = false;
      data.subscription.unsubscribe();
    };
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      session,
      loading,
      accessToken: session?.access_token || "",
      signIn: async (email: string, password: string) => {
        const trySignIn = async () => supabase.auth.signInWithPassword({ email, password });
        const { error } = await trySignIn();
        if (error) {
          const msg = (error.message || "").toLowerCase();
          const code = String((error as { code?: string }).code || "").toLowerCase();
          const looksUnconfirmed =
            code === "email_not_confirmed" || msg.includes("email not confirmed") || msg.includes("not confirmed");
          if (looksUnconfirmed) {
            try {
              await apiJson<{ ok: boolean }>("/api/auth/unstick-email-confirm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
              });
            } catch {
              throw error;
            }
            const second = await trySignIn();
            if (second.error) throw second.error;
            return;
          }
          throw error;
        }
      },
      signUp: async (email: string, password: string, metadata?: { first_name?: string; last_name?: string }) => {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: metadata || {},
          },
        });
        if (error) throw error;
        return data.session ?? null;
      },
      signOut: async () => {
        const { error } = await supabase.auth.signOut();
        if (error) throw error;
      },
    }),
    [session, loading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
