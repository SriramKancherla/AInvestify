import { useEffect, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import ArrowCanvas from "@/components/landing/ArrowCanvas";
import SiteFooter from "@/components/SiteFooter";
import { useAuth } from "@/contexts/AuthContext";
import { isSupabaseConfigured } from "@/lib/supabase";
import { apiJson } from "@/lib/api";

export default function LandingAuthPage() {
  const { signIn, session, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [mode, setMode] = useState<"signin" | "signup" | "forgot">("signin");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  const [otpVerified, setOtpVerified] = useState(false);
  const [otp, setOtp] = useState("");
  const [otpCooldownUntil, setOtpCooldownUntil] = useState(0);
  const [nowTs, setNowTs] = useState(() => Date.now());

  const nextPath = useMemo(() => {
    const from = (location.state as { from?: string } | null)?.from;
    return from || "/app";
  }, [location.state]);

  useEffect(() => {
    // Reset OTP flow when switching auth mode.
    setOtpSent(false);
    setOtpVerified(false);
    setOtp("");
    setOtpCooldownUntil(0);
    setNowTs(Date.now());
    setInfo(null);
    setError(null);
  }, [mode]);

  useEffect(() => {
    const id = window.setInterval(() => setNowTs(Date.now()), 500);
    return () => window.clearInterval(id);
  }, []);

  const otpCooldownLeft = Math.max(0, Math.ceil((otpCooldownUntil - nowTs) / 1000));

  if (!loading && session) return <Navigate to={nextPath} replace />;

  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      <div
        className="absolute inset-0 z-0 bg-gradient-animated"
        style={{ backgroundImage: "linear-gradient(160deg, hsl(222 47% 5%) 0%, hsl(230 50% 10%) 50%, hsl(222 47% 5%) 100%)" }}
      />
      <div className="grid-overlay grid-overlay-animated absolute inset-[-80px] z-[1] opacity-40" />
      <motion.svg
        className="absolute inset-0 z-[2] h-full w-full opacity-[0.08]"
        preserveAspectRatio="none"
        viewBox="0 0 1440 900"
        initial={{ opacity: 0.03 }}
        animate={{ opacity: [0.04, 0.1, 0.04] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      >
        <motion.g
          animate={{ x: [0, 20, 0], y: [0, -10, 0] }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        >
          <polyline fill="none" stroke="hsl(150 100% 64%)" strokeWidth="2" points="0,600 120,580 240,520 360,550 480,420 600,460 720,380 840,400 960,350 1080,370 1200,300 1320,320 1440,280" />
        </motion.g>
        <motion.g
          animate={{ x: [0, -24, 0], y: [0, 8, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
        >
          <polyline fill="none" stroke="hsl(0 100% 65%)" strokeWidth="2" points="0,400 120,430 240,480 360,450 480,500 600,470 720,520 840,510 960,560 1080,540 1200,590 1320,570 1440,620" />
        </motion.g>
      </motion.svg>

      <motion.nav
        data-no-arrow-spawn="true"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="fixed top-0 left-0 right-0 z-40 glass-surface"
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <button
            type="button"
            onClick={() => window.location.reload()}
            title="AInvestify home"
            aria-label="AInvestify home"
            className="flex items-center gap-2.5 bg-transparent border-none p-0 m-0 cursor-pointer"
          >
            <img src="/favicon.svg" alt="AInvestify" className="w-8 h-8 rounded-lg object-cover" />
            <span className="font-display text-lg font-semibold tracking-tight text-foreground">
              <span className="text-gradient-brand">AInvestify</span>
            </span>
          </button>
          <span className="text-xs text-muted-foreground">Stock Insights</span>
        </div>
      </motion.nav>

      <div className="relative z-20 flex min-h-screen flex-col px-6 py-20 pt-28">
        <div className="flex flex-1 w-full min-h-0 items-center justify-center">
          <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
          <div className="pointer-events-none">
            <motion.h1 initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="font-display text-5xl sm:text-7xl font-bold tracking-tight">
              <span className="text-gradient-brand">AInvestify</span>
            </motion.h1>
            <motion.p initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-4 font-mono text-sm tracking-widest text-muted-foreground">
              STOCK INSIGHTS
            </motion.p>
          </div>

          <div data-no-arrow-spawn="true" className="rounded-xl border border-border bg-card/80 backdrop-blur p-5 space-y-4">
            <div>
              <h2 className="text-xl font-semibold text-foreground">
                {mode === "signin" ? "Sign in" : mode === "signup" ? "Create account" : "Forgot password"}
              </h2>
              <p className="text-sm text-muted-foreground mt-1">Use your email to access the dashboard.</p>
            </div>
            <div className="flex gap-2 text-sm">
              <button type="button" onClick={() => setMode("signin")} className={`px-3 py-1.5 rounded border ${mode === "signin" ? "border-primary text-primary" : "border-border"}`}>Sign in</button>
              <button type="button" onClick={() => setMode("signup")} className={`px-3 py-1.5 rounded border ${mode === "signup" ? "border-primary text-primary" : "border-border"}`}>Sign up</button>
              <button type="button" onClick={() => setMode("forgot")} className={`px-3 py-1.5 rounded border ${mode === "forgot" ? "border-primary text-primary" : "border-border"}`}>Forgot</button>
            </div>
            <div className="space-y-2">
              {(mode === "signup") && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <input value={firstName} onChange={(e) => setFirstName(e.target.value)} type="text" placeholder="First name" className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm" />
                  <input value={lastName} onChange={(e) => setLastName(e.target.value)} type="text" placeholder="Last name" className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm" />
                </div>
              )}
              <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" placeholder="Email" className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm" />
              <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder={mode === "forgot" ? "New password" : "Password"} className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm" />
              {otpSent && (
                <input
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  type="text"
                  inputMode="numeric"
                  placeholder="Enter OTP from your email"
                  className="w-full h-10 px-3 rounded border border-border bg-secondary/40 text-sm"
                />
              )}
            </div>
            {error && <p className="text-sm text-red-500">{error}</p>}
            {info && <p className="text-xs text-emerald-400">{info}</p>}
            {!isSupabaseConfigured && <p className="text-xs text-amber-500">Configure `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in `frontend-2/.env`.</p>}
            {(mode === "signup" || mode === "forgot") && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={busy || !isSupabaseConfigured || otpCooldownLeft > 0}
                  onClick={async () => {
                    setError(null);
                    setInfo(null);
                    setBusy(true);
                    try {
                      if (!email) throw new Error("Enter email before requesting OTP.");
                      await apiJson<{ ok: boolean }>(mode === "forgot" ? "/api/auth/forgot-otp/send" : "/api/auth/signup-otp/send", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ email }),
                      });
                      setOtpSent(true);
                      setOtpVerified(false);
                      setOtpCooldownUntil(Date.now() + 30_000);
                      setInfo("OTP sent to your email.");
                    } catch (e) {
                      setError(e instanceof Error ? e.message : "Failed to send OTP.");
                    } finally {
                      setBusy(false);
                    }
                  }}
                  className="h-10 rounded border border-border text-sm hover:bg-secondary/40 disabled:opacity-60"
                >
                  {otpSent ? (otpCooldownLeft > 0 ? `Resend OTP (${otpCooldownLeft}s)` : "Resend OTP") : "Email Verification OTP"}
                </button>
                <button
                  type="button"
                  disabled={busy || !otpSent || !isSupabaseConfigured}
                  onClick={async () => {
                    setError(null);
                    setInfo(null);
                    setBusy(true);
                    try {
                      if (!otp.trim()) throw new Error("Enter OTP.");
                      if (mode === "forgot") {
                        await apiJson<{ ok: boolean; reset: boolean }>("/api/auth/forgot-otp/reset", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ email, otp: otp.trim(), new_password: password }),
                        });
                        setOtpVerified(true);
                        setInfo("Password reset complete. You can now sign in.");
                        setMode("signin");
                        setOtp("");
                        setOtpSent(false);
                      } else {
                        const status = await apiJson<{ ok: boolean; exists: boolean }>("/api/auth/signup-email-status", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ email }),
                        });
                        if (status.exists) {
                          setOtpVerified(false);
                          setError("Email already exists. Please sign in with this email or use another email.");
                          return;
                        }
                        await apiJson<{ ok: boolean; verified: boolean }>("/api/auth/signup-otp/verify", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ email, otp: otp.trim() }),
                        });
                        setOtpVerified(true);
                        setInfo("OTP verified. Complete signup now.");
                      }
                    } catch (e) {
                      setOtpVerified(false);
                      setError(e instanceof Error ? e.message : "OTP verification failed.");
                    } finally {
                      setBusy(false);
                    }
                  }}
                  className="h-10 rounded border border-border text-sm hover:bg-secondary/40 disabled:opacity-60"
                >
                  {mode === "forgot" ? "Verify OTP & Reset" : "Verify OTP"}
                </button>
              </div>
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
                  } else if (mode === "forgot") {
                    throw new Error("Use Email Verification OTP and Verify OTP & Reset for forgot password.");
                  } else {
                    if (!firstName.trim() || !lastName.trim()) throw new Error("First name and last name are required.");
                    if (!otpVerified) throw new Error("Please verify OTP first.");
                    await apiJson<{ ok: boolean; created: boolean }>("/api/auth/signup-complete", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        email,
                        password,
                        first_name: firstName.trim(),
                        last_name: lastName.trim(),
                      }),
                    });
                    await signIn(email, password);
                    setInfo("Account ready. Redirecting…");
                    navigate(nextPath, { replace: true });
                  }
                } catch (e) {
                  setError(e instanceof Error ? e.message : "Authentication failed.");
                } finally {
                  setBusy(false);
                }
              }}
              className="w-full h-10 rounded bg-primary text-primary-foreground text-sm font-medium disabled:opacity-60"
            >
              {busy ? "Please wait..." : mode === "signin" ? "Sign in" : mode === "forgot" ? "Reset via OTP section" : "Complete Signup"}
            </button>
          </div>
        </div>
        </div>
        <SiteFooter variant="landing" />
      </div>

      <ArrowCanvas />
    </div>
  );
}
