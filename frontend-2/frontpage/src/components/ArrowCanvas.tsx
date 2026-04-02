import { useRef, useEffect, useCallback } from "react";

interface Arrow {
  x: number;
  y: number;
  angle: number;
  speed: number;
  length: number;
  isGreen: boolean;
  opacity: number;
  birth: number;
  lifetime: number;
  scale: number;
}

const ARROW_COUNT = 14;
const MIN_LIFETIME = 2000;
const MAX_LIFETIME = 4000;

const GREEN = { r: 77, g: 255, b: 136 };
const RED = { r: 255, g: 77, b: 77 };

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  angle: number,
  length: number,
  color: typeof GREEN,
  opacity: number,
  scale: number
) {
  const len = length * scale;
  const headLen = 10 * scale;
  const endX = x + Math.cos(angle) * len;
  const endY = y + Math.sin(angle) * len;

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = `rgba(${color.r},${color.g},${color.b},${opacity})`;
  ctx.fillStyle = `rgba(${color.r},${color.g},${color.b},${opacity})`;
  ctx.lineWidth = 2 * scale;
  ctx.lineCap = "round";

  // Glow
  ctx.shadowColor = `rgba(${color.r},${color.g},${color.b},${opacity * 0.8})`;
  ctx.shadowBlur = 12 * scale;

  // Line
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(
    endX - headLen * Math.cos(angle - Math.PI / 6),
    endY - headLen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    endX - headLen * Math.cos(angle + Math.PI / 6),
    endY - headLen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

export default function ArrowCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const arrowsRef = useRef<Arrow[]>([]);
  const animFrameRef = useRef<number>(0);

  const spawn = useCallback((cx: number, cy: number) => {
    const now = performance.now();
    for (let i = 0; i < ARROW_COUNT; i++) {
      const isGreen = Math.random() > 0.45;
      // Green: random upward half, Red: random downward half
      const angle = isGreen
        ? -(Math.random() * Math.PI) // anywhere in upper semicircle
        : Math.random() * Math.PI;   // anywhere in lower semicircle
      arrowsRef.current.push({
        x: cx,
        y: cy,
        angle,
        speed: 50 + Math.random() * 140,
        length: 50 + Math.random() * 90,
        isGreen,
        opacity: 0.6 + Math.random() * 0.4,
        birth: now + Math.random() * 150,
        lifetime: MIN_LIFETIME + Math.random() * (MAX_LIFETIME - MIN_LIFETIME),
        scale: 0,
      });
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    let lastTime = performance.now();

    const loop = (time: number) => {
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

      arrowsRef.current = arrowsRef.current.filter((a) => {
        const age = time - a.birth;
        if (age > a.lifetime) return false;

        const progress = age / a.lifetime;
        // scale in fast, fade out
        a.scale = progress < 0.1 ? progress / 0.1 : 1;
        const fadeStart = 0.6;
        const currentOpacity =
          progress > fadeStart
            ? a.opacity * (1 - (progress - fadeStart) / (1 - fadeStart))
            : a.opacity;

        // move outward
        a.x += Math.cos(a.angle) * a.speed * dt;
        a.y += Math.sin(a.angle) * a.speed * dt;
        // decelerate
        a.speed *= 0.995;

        drawArrow(
          ctx,
          a.x,
          a.y,
          a.angle,
          a.length,
          a.isGreen ? GREEN : RED,
          Math.max(0, currentOpacity),
          a.scale
        );
        return true;
      });

      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      spawn(e.clientX, e.clientY);
    },
    [spawn]
  );

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="absolute inset-0 z-10 cursor-crosshair"
    />
  );
}
