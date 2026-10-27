import { tickerHue } from "@/lib/format";

interface MonogramProps {
  symbol: string;
  size?: number;
  className?: string;
}

/** Circular monogram with a deterministic soft tint derived from the ticker. */
export default function Monogram({ symbol, size = 36, className = "" }: MonogramProps) {
  const s = (symbol || "?").trim().toUpperCase();
  const initials = s.slice(0, s.length >= 4 ? 2 : Math.min(2, s.length)) || "?";
  const hue = tickerHue(s);
  return (
    <span
      className={`inline-flex items-center justify-center rounded-full font-semibold shrink-0 ${className}`}
      style={{
        width: size,
        height: size,
        fontSize: size * 0.36,
        background: `hsl(${hue} 70% 95%)`,
        color: `hsl(${hue} 55% 38%)`,
        border: `1px solid hsl(${hue} 45% 88%)`,
      }}
      aria-hidden
    >
      {initials}
    </span>
  );
}
