import { motion } from "framer-motion";
import { ArrowLeftRight, TrendingUp, TrendingDown } from "lucide-react";
import type { StockData } from "@/hooks/useStockStore";

interface CompareViewProps {
  primary: StockData | null;
  secondary: StockData | null;
}

export default function CompareView({ primary, secondary }: CompareViewProps) {
  if (!primary || !secondary) return null;

  const metrics = [
    { label: "Price", a: `$${primary.current_price?.toFixed(2)}`, b: `$${secondary.current_price?.toFixed(2)}`, aNum: primary.current_price, bNum: secondary.current_price },
    { label: "Change %", a: `${primary.change_percent?.toFixed(2)}%`, b: `${secondary.change_percent?.toFixed(2)}%`, aNum: primary.change_percent, bNum: secondary.change_percent },
    { label: "Market Cap", a: primary.market_cap, b: secondary.market_cap },
    { label: "P/E Ratio", a: primary.pe_ratio?.toFixed(1), b: secondary.pe_ratio?.toFixed(1), aNum: primary.pe_ratio, bNum: secondary.pe_ratio, lower: true },
    { label: "Volume", a: primary.volume, b: secondary.volume },
    { label: "Sentiment", a: `${primary.sentiment_score}`, b: `${secondary.sentiment_score}`, aNum: primary.sentiment_score, bNum: secondary.sentiment_score },
    { label: "Model view", a: primary.recommendation, b: secondary.recommendation },
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-card overflow-hidden">
      <div className="flex items-center gap-3 p-5 border-b border-border">
        <ArrowLeftRight className="w-5 h-5 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">Comparison</h3>
      </div>

      {/* Header Row */}
      <div className="grid grid-cols-3 px-5 py-3 bg-secondary/30 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        <span>Metric</span>
        <span className="text-center">{primary.ticker}</span>
        <span className="text-center">{secondary.ticker}</span>
      </div>

      {metrics.map((m, i) => {
        const aWins = m.aNum !== undefined && m.bNum !== undefined
          ? m.lower ? m.aNum < m.bNum : m.aNum > m.bNum
          : undefined;
        return (
          <motion.div
            key={m.label}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.05 }}
            className="grid grid-cols-3 px-5 py-3 border-b border-border/50 hover:bg-secondary/30 transition-colors"
          >
            <span className="text-sm text-muted-foreground">{m.label}</span>
            <span className={`text-sm font-mono font-semibold text-center ${aWins === true ? "kpi-positive" : aWins === false ? "kpi-negative" : "text-foreground"}`}>
              {m.a}
            </span>
            <span className={`text-sm font-mono font-semibold text-center ${aWins === false ? "kpi-positive" : aWins === true ? "kpi-negative" : "text-foreground"}`}>
              {m.b}
            </span>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
