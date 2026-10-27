import { useEffect, useMemo, useRef, useState } from "react";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  /** Force a color; otherwise derived from first vs last point. */
  up?: boolean;
  className?: string;
  strokeWidth?: number;
  fill?: boolean;
  /** Animate the line drawing itself once on mount. */
  draw?: boolean;
}

/** Minimal, axis-less price sparkline (emerald up / red down). */
export default function Sparkline({
  data,
  width = 96,
  height = 32,
  up,
  className = "",
  strokeWidth = 1.5,
  fill = false,
  draw = false,
}: SparklineProps) {
  const pathRef = useRef<SVGPathElement>(null);
  const [len, setLen] = useState(0);
  const [drawn, setDrawn] = useState(!draw);

  useEffect(() => {
    if (!draw || !pathRef.current) return;
    const total = pathRef.current.getTotalLength();
    setLen(total);
    setDrawn(false);
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setDrawn(true)));
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draw, data, width, height]);
  const { line, area, isUp } = useMemo(() => {
    const pts = (data || []).filter((n) => Number.isFinite(n));
    if (pts.length < 2) return { line: "", area: "", isUp: true };
    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const span = max - min || 1;
    const stepX = width / (pts.length - 1);
    const coords = pts.map((v, i) => {
      const x = i * stepX;
      const y = height - ((v - min) / span) * (height - strokeWidth * 2) - strokeWidth;
      return [x, y] as const;
    });
    const line = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
    const area = `${line} L${width},${height} L0,${height} Z`;
    const derivedUp = pts[pts.length - 1] >= pts[0];
    return { line, area, isUp: up ?? derivedUp };
  }, [data, width, height, up, strokeWidth]);

  const color = isUp ? "hsl(var(--success))" : "hsl(var(--destructive))";
  const gradId = useMemo(() => `spark-${Math.random().toString(36).slice(2, 8)}`, []);

  if (!line) {
    return <svg width={width} height={height} className={className} aria-hidden />;
  }

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={className} aria-hidden>
      {fill && (
        <>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.18} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={area} fill={`url(#${gradId})`} stroke="none" />
        </>
      )}
      <path
        ref={pathRef}
        d={line}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={
          draw
            ? {
                strokeDasharray: len,
                strokeDashoffset: drawn ? 0 : len,
                transition: "stroke-dashoffset 380ms ease-out",
              }
            : undefined
        }
      />
    </svg>
  );
}
