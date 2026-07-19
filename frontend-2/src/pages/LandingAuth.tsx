import { useEffect, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { BarChart3, Brain, Gauge, Mail, Lock, User } from "lucide-react";
import SiteFooter from "@/components/SiteFooter";
import Sparkline from "@/components/common/Sparkline";
import { useAuth } from "@/contexts/AuthContext";
import { isSupabaseConfigured } from "@/lib/supabase";

const VALUE_PROPS = [
  { icon: BarChart3, title: "Fundamentals", desc: "valuation & financial health" },
  { icon: Brain, title: "News sentiment", desc: "live headline classification" },
  { icon: Gauge, title: "Blended AI score", desc: "one number, both signals" },
];

const PREVIEW_SPARK = [12, 13, 12.4, 14, 13.6, 15.2, 14.8, 16, 15.4, 17.1, 16.6, 18.4, 19];

export default function LandingAuthPage() {
  const { signIn, signUp, session, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const nextPath = useMemo(() => {
    const from = (location.state as { from?: string } | null)?.from;
    return from || "/app";
  }, [location.state]);

  useEffect(() => {
    setInfo(null);
    setError(null);
  }, [mode]);

  if (!loading && session) return <Navigate to={nextPath} replace />;

  const submit = async () => {
    setError(null);
    setInfo(null);
    setBusy(true);
    try {
      if (!email || !password) throw new Error("Email and password are required.");
      if (mode === "signin") {
        await signIn(email, password);
        navigate(nextPath, { replace: true });
      } else {
        if (!firstName.trim() || !lastName.trim()) throw new Error("First and last name are required.");
        const newSession = await signUp(email, password, { first_name: firstName.trim(), last_name: lastName.trim() });
        if (newSession) navigate(nextPath, { replace: true });
        else setInfo("Account created. If confirmation is required, check your inbox, then sign in.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Nav — logo + wordmark only, nothing else until signed in. */}
      <nav className="w-full border-b border-border/70">
        <div className="mx-auto max-w-6xl flex items-center px-6 h-16">
          <div className="flex items-center gap-2">
            <img src="/favicon.svg" alt="AInvestify" className="w-8 h-8 rounded-lg" />
            <span className="text-lg font-semibold tracking-tightish">
              <span className="gradient-text">AI</span>
              <span className="text-foreground">nvestify</span>
            </span>
          </div>
        </div>
      </nav>

      <div className="flex-1 mx-auto max-w-6xl w-full px-6 py-14 sm:py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Hero — one confident idea: kicker, headline, one line, three rows, preview. */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <p className="label-caps text-primary">AI-driven analysis</p>
            <h1 className="mt-3 text-4xl sm:text-[44px] leading-[1.1] font-semibold tracking-tightish text-foreground">
              Stock insights, in seconds
            </h1>
            <p className="mt-3 text-base text-muted-foreground max-w-sm">
              Fundamentals and live news sentiment, blended into one explainable score.
            </p>
            <div className="mt-7 space-y-2.5">
              {VALUE_PROPS.map((v) => (
                <div key={v.title} className="flex items-center gap-2.5">
                  <span className="h-7 w-7 rounded-md bg-primary/8 text-primary flex items-center justify-center shrink-0">
                    <v.icon className="w-3.5 h-3.5" />
                  </span>
                  <p className="text-sm text-foreground">
                    <span className="font-medium">{v.title}</span>
                    <span className="text-muted-foreground"> — {v.desc}</span>
                  </p>
                </div>
              ))}
            </div>

            {/* Product preview — the second thing the eye lands on. */}
            <div className="mt-9 hidden sm:block max-w-sm elevated p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="ticker-badge">AAPL</span>
                  <span className="text-xs text-muted-foreground">Apple Inc.</span>
                </div>
                <span className="font-mono text-xs kpi-positive">+2.14%</span>
              </div>
              <Sparkline data={PREVIEW_SPARK} up fill draw width={320} height={56} className="mt-3 w-full" />
            </div>
          </motion.div>

          {/* Auth card */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="w-full max-w-md mx-auto lg:ml-auto elevated p-8"
          >
            {/* Segmented control — only the active pill is a shape; inactive is bare text. */}
            <div className="flex p-1 rounded-full bg-secondary/70 text-sm font-medium mb-6">
              {(["signin", "signup"] as const).map((m) => (
                <button key={m} onClick={() => setMode(m)} className="relative flex-1 h-9 rounded-full">
                  {mode === m && (
                    <motion.span
                      layoutId="authSeg"
                      transition={{ type: "spring", stiffness: 400, damping: 32 }}
                      className="absolute inset-0 rounded-full bg-card shadow-sm"
                    />
                  )}
                  <span className={`relative z-10 transition-colors ${mode === m ? "text-foreground" : "text-muted-foreground"}`}>
                    {m === "signin" ? "Sign in" : "Sign up"}
                  </span>
                </button>
              ))}
            </div>

            <div className="space-y-3">
              {mode === "signup" && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <input value={firstName} onChange={(e) => setFirstName(e.target.value)} placeholder="First name" className="w-full h-11 pl-9 pr-3 rounded-lg border border-transparent bg-secondary text-sm transition-shadow focus:outline-none focus:border-primary/40 focus:ring-2 focus:ring-primary/20" />
                  </div>
                  <input value={lastName} onChange={(e) => setLastName(e.target.value)} placeholder="Last name" className="w-full h-11 px-3 rounded-lg border border-transparent bg-secondary text-sm transition-shadow focus:outline-none focus:border-primary/40 focus:ring-2 focus:ring-primary/20" />
                </div>
              )}
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" placeholder="Email" className="w-full h-11 pl-9 pr-3 rounded-lg border border-transparent bg-secondary text-sm transition-shadow focus:outline-none focus:border-primary/40 focus:ring-2 focus:ring-primary/20" />
              </div>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" onKeyDown={(e) => e.key === "Enter" && submit()} className="w-full h-11 pl-9 pr-3 rounded-lg border border-transparent bg-secondary text-sm transition-shadow focus:outline-none focus:border-primary/40 focus:ring-2 focus:ring-primary/20" />
              </div>
            </div>

            {error && <p className="text-sm text-destructive mt-3">{error}</p>}
            {info && <p className="text-xs text-success mt-3">{info}</p>}
            {!isSupabaseConfigured && <p className="text-xs text-warning mt-3">Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable sign-in.</p>}

            <button
              onClick={submit}
              disabled={busy || !isSupabaseConfigured}
              className="mt-5 w-full h-11 rounded-lg bg-primary text-primary-foreground text-sm font-medium transition-transform active:scale-[0.98] disabled:opacity-60"
            >
              {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
            </button>
            <p className="mt-3 text-center text-xs text-muted-foreground">
              {mode === "signin" ? "New here? Switch to Sign up." : "Already have an account? Switch to Sign in."}
            </p>
          </motion.div>
        </div>

        <SiteFooter variant="landing" />
      </div>
    </div>
  );
}
