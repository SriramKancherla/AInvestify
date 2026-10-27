import { useEffect, useState } from "react";

interface AIScoreGaugeProps {
  /** 0–100. */
  score: number | null | undefined;
  size?: number;
  label?: string;
}

function bandColor(score: number): string {
  if (score >= 60) return "hsl(var(--success))";
  if (score >= 40) return "hsl(var(--warning))";
  return "hsl(var(--destructive))";
}

/** Circular 0–100 gauge, colored by band, with an animated sweep on mount. */
export default function AIScoreGauge({ score, size = 96, label }: AIScoreGaugeProps) {
  const target = score == null || !Number.isFinite(score) ? 0 : Math.max(0, Math.min(100, score));
  const [shown, setShown] = useState(0);

  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / 700);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(from + (target - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);

  const stroke = 8;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = (shown / 100) * c;
  const color = score == null ? "hsl(var(--muted-foreground))" : bandColor(target);

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--secondary))" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-xl font-semibold text-foreground">
          {score == null ? "—" : Math.round(shown)}
        </span>
        {label && <span className="text-[10px] text-muted-foreground mt-0.5">{label}</span>}
      </div>
    </div>
  );
}
