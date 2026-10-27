import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  TrendingUp,
  BarChart3,
  Newspaper,
  LineChart,
  Sparkles,
  ShieldCheck,
  Zap,
  Check,
  ChevronDown,
  Layers,
  type LucideIcon,
} from "lucide-react";
import SiteFooter from "@/components/SiteFooter";

const PILLARS: {
  icon: LucideIcon;
  tag: string;
  title: string;
  description: string;
  metrics: string[];
  accent: string;
}[] = [
  {
    icon: BarChart3,
    tag: "01 / FUNDAMENTALS",
    title: "Valuation & balance sheet clarity",
    description:
      "Key multiples, margin trajectories, and cash flow durability distilled automatically from quarterly filings—without digging through raw spreadsheets.",
    metrics: ["P/E & PEG Context", "Free Cash Flow Yield", "Balance Sheet Health"],
    accent: "text-teal-700 bg-teal-50 border-teal-200/80",
  },
  {
    icon: Newspaper,
    tag: "02 / NEWS TONE",
    title: "Real-time narrative & sentiment scoring",
    description:
      "Instant NLP tone analysis across breaking financial press, analyst revisions, and earnings transcripts to separate material catalysts from market noise.",
    metrics: ["Bull / Bear Drift", "Catalyst Tagging", "Source Credibility"],
    accent: "text-sky-700 bg-sky-50 border-sky-200/80",
  },
  {
    icon: LineChart,
    tag: "03 / CHART FORECASTS",
    title: "Probabilistic price trajectory bands",
    description:
      "Forward-looking confidence cones that map bull, base, and bear scenarios alongside historical price structure so you can frame risk in seconds.",
    metrics: ["30-Day Scenario Cone", "Volatility Regimes", "Support & Resistance"],
    accent: "text-emerald-700 bg-emerald-50 border-emerald-200/80",
  },
];

const STEPS = [
  {
    step: "01",
    title: "Enter any ticker symbol",
    detail:
      "Type a US equity ticker and AInvestify immediately pulls the latest price structure, filing ratios, and media coverage.",
  },
  {
    step: "02",
    title: "Read the unified brief",
    detail:
      "See fundamentals, news tone, and chart forecasts aligned side-by-side instead of jumping across ten browser tabs.",
  },
  {
    step: "03",
    title: "Act with structured conviction",
    detail:
      "Validate your thesis in seconds with transparent scoring and scenario ranges before entering the full dashboard.",
  },
];

const FAQS = [
  {
    q: "What does AInvestify analyze for each stock?",
    a: "Every lookup combines three core signals in one view: fundamental financial ratios from official filings, real-time news sentiment tone across financial media, and statistical chart forecast ranges.",
  },
  {
    q: "Do I need an account to explore the landing page?",
    a: "No. This public landing page requires zero authentication, and you can jump straight into the application workspace via the Enter Dashboard button whenever you are ready.",
  },
  {
    q: "How fast are insights generated?",
    a: "AInvestify synthesizes fundamentals, news tone, and chart forecasts in just a few seconds per ticker so you can evaluate ideas at market speed.",
  },
];

export default function LandingPage() {
  const [openFaq, setOpenFaq] = useState(0);

  return (
    <div className="landing-root min-h-screen flex flex-col bg-white text-slate-900 selection:bg-teal-500/15 selection:text-teal-900">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');
        .landing-root {
          --landing-ink: #0f172a;
          --landing-slate: #475569;
          --landing-accent: #0d9488;
          --landing-accent-light: #f0fdfa;
          font-family: "Instrument Sans", system-ui, -apple-system, sans-serif;
        }
        .landing-display {
          font-family: "Newsreader", Georgia, serif;
          font-optical-sizing: auto;
        }
        @keyframes fadeUp {
          from {
            opacity: 0;
            transform: translateY(18px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes floatSlow {
          0%, 100% {
            transform: translateY(0px) scale(1);
          }
          50% {
            transform: translateY(-10px) scale(1.03);
          }
        }
        @keyframes drawLine {
          from {
            stroke-dashoffset: 1600;
          }
          to {
            stroke-dashoffset: 0;
          }
        }
        .anim-fade-1 {
          animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        .anim-fade-2 {
          opacity: 0;
          animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.12s forwards;
        }
        .anim-fade-3 {
          opacity: 0;
          animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.24s forwards;
        }
        .anim-fade-4 {
          opacity: 0;
          animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.36s forwards;
        }
        .ambient-orb {
          animation: floatSlow 11s ease-in-out infinite;
        }
        .hero-curve {
          stroke-dasharray: 1600;
          stroke-dashoffset: 1600;
          animation: drawLine 2.2s cubic-bezier(0.22, 1, 0.36, 1) 0.25s forwards;
        }
      `}</style>

      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none bg-white">
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(85% 55% at 50% 0%, rgba(20, 184, 166, 0.09) 0%, rgba(14, 165, 233, 0.05) 45%, transparent 75%), radial-gradient(55% 45% at 85% 20%, rgba(45, 212, 191, 0.08) 0%, transparent 65%), radial-gradient(60% 50% at 15% 85%, rgba(241, 245, 249, 0.9) 0%, transparent 70%)",
          }}
        />
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "linear-gradient(rgba(15, 23, 42, 0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(15, 23, 42, 0.04) 1px, transparent 1px)",
            backgroundSize: "64px 64px",
            maskImage:
              "radial-gradient(ellipse 80% 60% at 50% 30%, black 25%, transparent 80%)",
          }}
        />
        <div
          className="ambient-orb absolute -right-24 top-20 h-[440px] w-[440px] rounded-full blur-3xl"
          style={{
            background:
              "radial-gradient(circle, rgba(20, 184, 166, 0.14), transparent 70%)",
          }}
        />
        <svg
          className="absolute inset-x-0 bottom-0 h-[38vh] w-full"
          viewBox="0 0 1440 360"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <linearGradient id="lightWaveArea" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0d9488" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#0d9488" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path
            d="M0,260 C240,245 360,280 560,210 C760,140 940,185 1140,120 C1280,75 1360,95 1440,70 L1440,360 L0,360 Z"
            fill="url(#lightWaveArea)"
          />
          <path
            className="hero-curve"
            d="M0,260 C240,245 360,280 560,210 C760,140 940,185 1140,120 C1280,75 1360,95 1440,70"
            fill="none"
            stroke="#0d9488"
            strokeOpacity="0.32"
            strokeWidth="2"
          />
          <path
            d="M0,300 C260,285 460,315 680,255 C900,195 1120,235 1440,175"
            fill="none"
            stroke="#64748b"
            strokeOpacity="0.16"
            strokeWidth="1.5"
            strokeDasharray="5 5"
          />
        </svg>
      </div>

      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 sm:px-10">
          <a href="#top" className="flex items-center gap-2.5 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-600 text-white shadow-sm shadow-teal-600/20 transition-transform group-hover:scale-105">
              <TrendingUp className="h-4 w-4" />
            </div>
            <span className="landing-display text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
              AInvestify
            </span>
          </a>

          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <a href="#pillars" className="transition-colors hover:text-slate-950">
              Capabilities
            </a>
            <a href="#workflow" className="transition-colors hover:text-slate-950">
              How It Works
            </a>
            <a href="#faq" className="transition-colors hover:text-slate-950">
              FAQ
            </a>
          </nav>

          <div className="flex items-center gap-4">
            <Link
              to="/app"
              className="hidden text-sm font-medium text-slate-600 transition-colors hover:text-slate-950 sm:inline"
            >
              Open app
            </Link>
            <Link
              to="/app"
              className="inline-flex items-center gap-1.5 rounded-full bg-slate-900 px-4 py-2 text-xs sm:text-sm font-semibold text-white transition-all hover:bg-teal-700 shadow-sm"
            >
              <span>Enter dashboard</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </header>

      <main id="top" className="relative z-10 flex-1">
        <section className="mx-auto flex min-h-[78vh] max-w-6xl flex-col justify-center px-6 pt-16 pb-20 sm:px-10">
          <div className="max-w-3xl">
            <div className="anim-fade-1 inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50/90 px-3.5 py-1.5 text-xs font-semibold text-teal-800 shadow-sm">
              <Sparkles className="h-3.5 w-3.5 text-teal-600" />
              <span>AI-Powered Equity Research</span>
            </div>

            <p className="anim-fade-2 landing-display mt-6 text-[clamp(3.2rem,9vw,6.25rem)] font-medium leading-[0.96] tracking-tight text-slate-950">
              AInvestify
            </p>

            <h1 className="anim-fade-3 mt-5 max-w-2xl text-xl sm:text-2xl md:text-[1.65rem] font-normal leading-snug text-slate-600">
              Stock insights in seconds —{" "}
              <span className="font-semibold text-slate-900">
                fundamentals, news tone, and chart forecasts
              </span>{" "}
              in one place.
            </h1>

            <div className="anim-fade-4 mt-9 flex flex-wrap items-center gap-4">
              <Link
                to="/app"
                className="group inline-flex items-center gap-2.5 rounded-full bg-teal-600 px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-teal-600/20 transition-all hover:bg-teal-700 hover:scale-[1.02] active:scale-[0.98]"
              >
                <span>Enter dashboard</span>
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </Link>

              <a
                href="#pillars"
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950"
              >
                <span>Explore what’s inside</span>
              </a>
            </div>

            <div className="anim-fade-4 mt-14 grid grid-cols-1 gap-4 pt-8 border-t border-slate-200/80 sm:grid-cols-3">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-50 text-teal-700 border border-teal-200/60">
                  <Zap className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-900">Instant Synthesis</p>
                  <p className="text-xs text-slate-500">No multi-tab spreadsheet hunting</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-700 border border-sky-200/60">
                  <Layers className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-900">Three Signal Pillars</p>
                  <p className="text-xs text-slate-500">Filings, NLP sentiment & price cones</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200/60">
                  <ShieldCheck className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-900">Clean & Uncluttered</p>
                  <p className="text-xs text-slate-500">Built for fast, high-conviction reads</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="pillars" className="border-t border-slate-200/80 bg-slate-50/70 py-20">
          <div className="mx-auto max-w-6xl px-6 sm:px-10">
            <div className="max-w-2xl">
              <span className="text-xs font-semibold uppercase tracking-widest text-teal-700">
                Core Capabilities
              </span>
              <h2 className="landing-display mt-2 text-3xl sm:text-4xl font-medium tracking-tight text-slate-950">
                Everything required to size up a stock, nothing that gets in the way.
              </h2>
              <p className="mt-3 text-base text-slate-600">
                AInvestify brings together quantitative fundamentals, qualitative news tone, and forward chart forecasts into a single cohesive readout.
              </p>
            </div>

            <div className="mt-12 grid grid-cols-1 gap-6 md:grid-cols-3">
              {PILLARS.map((item) => {
                const Icon = item.icon;
                return (
                  <div
                    key={item.tag}
                    className="flex flex-col justify-between rounded-2xl border border-slate-200/90 bg-white p-7 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md"
                  >
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-semibold tracking-wider text-slate-400">
                          {item.tag}
                        </span>
                        <div className={`flex h-10 w-10 items-center justify-center rounded-xl border ${item.accent}`}>
                          <Icon className="h-5 w-5" />
                        </div>
                      </div>
                      <h3 className="mt-5 text-lg font-semibold text-slate-950">{item.title}</h3>
                      <p className="mt-2.5 text-sm leading-relaxed text-slate-600">{item.description}</p>
                    </div>
                    <ul className="mt-6 space-y-2 border-t border-slate-100 pt-4">
                      {item.metrics.map((metric) => (
                        <li key={metric} className="flex items-center gap-2 text-xs font-medium text-slate-700">
                          <Check className="h-3.5 w-3.5 text-teal-600 shrink-0" />
                          <span>{metric}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section id="workflow" className="py-20 bg-white border-t border-slate-200/70">
          <div className="mx-auto max-w-6xl px-6 sm:px-10">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
              <div>
                <span className="text-xs font-semibold uppercase tracking-widest text-teal-700">
                  Workflow
                </span>
                <h2 className="landing-display mt-2 text-3xl sm:text-4xl font-medium tracking-tight text-slate-950">
                  From ticker symbol to thesis in three steps.
                </h2>
              </div>
              <Link to="/app" className="inline-flex items-center gap-2 text-sm font-semibold text-teal-700 hover:text-teal-800">
                <span>Go straight to dashboard</span>
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            <div className="mt-12 grid grid-cols-1 gap-6 md:grid-cols-3">
              {STEPS.map((step) => (
                <div key={step.step} className="relative rounded-2xl border border-slate-200/80 bg-slate-50/50 p-6">
                  <span className="landing-display text-3xl font-medium text-teal-600/80">{step.step}</span>
                  <h3 className="mt-3 text-base font-semibold text-slate-950">{step.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600">{step.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="faq" className="border-t border-slate-200/70 bg-slate-50/60 py-20">
          <div className="mx-auto max-w-3xl px-6 sm:px-10">
            <div className="text-center">
              <span className="text-xs font-semibold uppercase tracking-widest text-teal-700">
                Questions & Answers
              </span>
              <h2 className="landing-display mt-2 text-3xl font-medium text-slate-950">
                Frequently asked questions
              </h2>
            </div>

            <div className="mt-10 space-y-3">
              {FAQS.map((item, idx) => {
                const isOpen = openFaq === idx;
                return (
                  <div key={item.q} className="rounded-2xl border border-slate-200/90 bg-white overflow-hidden transition-colors">
                    <button
                      type="button"
                      onClick={() => setOpenFaq(isOpen ? -1 : idx)}
                      aria-expanded={isOpen}
                      className="flex w-full items-center justify-between px-6 py-4 text-left text-sm font-semibold text-slate-900 hover:bg-slate-50/80"
                    >
                      <span>{item.q}</span>
                      <ChevronDown
                        className={`h-4 w-4 text-slate-500 transition-transform ${isOpen ? "rotate-180 text-teal-600" : ""}`}
                      />
                    </button>
                    {isOpen && (
                      <div className="px-6 pb-5 text-sm leading-relaxed text-slate-600 border-t border-slate-100 pt-3">
                        {item.a}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="py-16 bg-white border-t border-slate-200/70">
          <div className="mx-auto max-w-6xl px-6 sm:px-10">
            <div className="rounded-3xl border border-teal-200/80 bg-gradient-to-br from-teal-50/90 via-white to-sky-50/70 p-8 sm:p-12 shadow-sm">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
                <div className="max-w-xl">
                  <h2 className="landing-display text-3xl sm:text-4xl font-medium text-slate-950">
                    Ready to research your next stock?
                  </h2>
                  <p className="mt-2.5 text-base text-slate-600">
                    Launch the AInvestify dashboard for instant fundamentals, news tone, and chart forecasts.
                  </p>
                </div>
                <div>
                  <Link
                    to="/app"
                    className="group inline-flex items-center gap-2.5 rounded-full bg-teal-600 px-7 py-3.5 text-sm font-semibold text-white shadow-md shadow-teal-600/15 transition-all hover:bg-teal-700 hover:scale-[1.02]"
                  >
                    <span>Enter dashboard</span>
                    <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative z-10 border-t border-slate-200/80 bg-white/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl flex-col sm:flex-row items-center justify-between gap-4 px-6 py-6 sm:px-10 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-teal-600 text-white">
              <TrendingUp className="h-3.5 w-3.5" />
            </div>
            <span className="font-semibold text-slate-900">AInvestify</span>
            <span>— Stock insights in seconds.</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#pillars" className="hover:text-teal-700 transition-colors">
              Capabilities
            </a>
            <a href="#workflow" className="hover:text-teal-700 transition-colors">
              Workflow
            </a>
            <Link to="/app" className="font-medium text-teal-700 hover:text-teal-800">
              Open app →
            </Link>
          </div>
        </div>
        <SiteFooter variant="landing" className="mx-auto max-w-6xl border-0 px-6 pb-8 pt-0 sm:px-10" />
      </footer>
    </div>
  );
}
