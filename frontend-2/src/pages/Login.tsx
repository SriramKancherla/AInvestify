import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { isSupabaseConfigured } from "@/lib/supabase";

export default function LoginPage() {
  const { signIn, signUp, session, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const nextPath = useMemo(() => {
    const from = (location.state as { from?: string } | null)?.from;
    return from || "/app";
  }, [location.state]);

  if (!loading && session) {
    navigate(nextPath, { replace: true });
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-card p-5 space-y-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">AInvestify</h1>
          <p className="text-sm text-muted-foreground mt-1">Sign in to continue.</p>
        </div>

        <div className="flex gap-2 text-sm">
          <button
            type="button"
            onClick={() => {
              setMode("signin");
              setInfo(null);
            }}
            className={`px-3 py-1.5 rounded border ${mode === "signin" ? "border-primary text-primary" : "border-border"}`}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("signup");
              setInfo(null);
            }}
            className={`px-3 py-1.5 rounded border ${mode === "signup" ? "border-primary text-primary" : "border-border"}`}
          >
            Sign up
          </button>
        </div>

        <div className="space-y-2">
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="email"
            placeholder="Email"
            className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm"
          />
          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            placeholder="Password"
            className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm"
          />
        </div>

        {error && <p className="text-sm text-red-500">{error}</p>}
        {info && <p className="text-xs text-emerald-400">{info}</p>}
        {!isSupabaseConfigured && (
          <p className="text-xs text-amber-500">
            Supabase is not configured. Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in `frontend-2/.env`.
          </p>
        )}

        <button
          type="button"
          disabled={busy || !isSupabaseConfigured}
          onClick={async () => {
            setError(null);
            setInfo(null);
            setBusy(true);
            try {
              if (!email || !password) throw new Error("Email and password are required.");
              if (mode === "signin") {
                await signIn(email, password);
                navigate(nextPath, { replace: true });
              } else {
                const newSession = await signUp(email, password);
                if (newSession) {
                  navigate(nextPath, { replace: true });
                } else {
                  setInfo("Account created. If your project requires email confirmation, check your inbox and then sign in.");
                }
              }
            } catch (e) {
              setError(e instanceof Error ? e.message : "Authentication failed.");
            } finally {
              setBusy(false);
            }
          }}
          className="w-full h-10 rounded bg-primary text-primary-foreground text-sm font-medium disabled:opacity-60"
        >
          {busy ? "Please wait..." : mode === "signin" ? "Sign in" : "Create account"}
        </button>
      </div>
    </div>
  );
}
