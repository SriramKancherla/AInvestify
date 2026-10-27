import { useEffect, useState } from "react";
import Monogram from "./Monogram";
import { logoSrc } from "@/lib/logos";

interface CompanyLogoProps {
  symbol: string;
  size?: number;
  className?: string;
}

/** Company logo from /public/logos when available; falls back to tinted monogram. */
export default function CompanyLogo({ symbol, size = 36, className = "" }: CompanyLogoProps) {
  const [failed, setFailed] = useState(false);
  const sym = (symbol || "?").trim().toUpperCase();

  useEffect(() => {
    setFailed(false);
  }, [sym]);

  if (!sym || failed) {
    return <Monogram symbol={sym} size={size} className={className} />;
  }

  return (
    <img
      key={sym}
      src={logoSrc(sym)}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      decoding="async"
      className={`inline-block rounded-full object-contain bg-white border border-border shrink-0 ${className}`}
      style={{ width: size, height: size }}
      onError={() => setFailed(true)}
    />
  );
}
