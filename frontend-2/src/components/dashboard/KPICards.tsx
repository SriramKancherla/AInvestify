import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Activity, Target } from "lucide-react";
import type { StockData } from "@/hooks/useStockStore";

interface KPICardsProps {
  data: StockData | null;
  loading: boolean;
}

const SkeletonCard = () => (
  <div className="glass-card p-5 space-y-3">
    <div className="skeleton-shimmer h-3 w-20" />
    <div className="skeleton-shimmer h-7 w-28" />
    <div className="skeleton-shimmer h-3 w-16" />
  </div>
);

export default function KPICards({ data, loading }: KPICardsProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    );
  }
  if (!data) return null;

  const isPositive = (data.change_percent ?? 0) >= 0;

  const cards = [
    {
      label: "Price",
      value: `$${data.current_price?.toFixed(2)}`,
      sub: `${isPositive ? "+" : ""}${data.change_percent?.toFixed(2)}%`,
      subColor: isPositive ? "kpi-positive" : "kpi-negative",
      icon: DollarSign,
      iconBg: "bg-primary/10 text-primary",
    },
    {
      label: "Market Cap",
      value: data.market_cap,
      sub: "Total Valuation",
      subColor: "text-muted-foreground",
      icon: BarChart3,
      iconBg: "bg-accent/10 text-accent",
    },
    {
      label: "P/E Ratio",
      value: data.pe_ratio?.toFixed(1),
      sub: data.pe_ratio && data.pe_ratio > 25 ? "Above avg" : "Below avg",
      subColor: data.pe_ratio && data.pe_ratio > 25 ? "text-warning" : "kpi-positive",
      icon: Target,
      iconBg: "bg-warning/10 text-warning",
    },
    {
      label: "Volume",
      value: data.volume,
      sub: "Daily avg",
      subColor: "text-muted-foreground",
      icon: Activity,
      iconBg: "bg-success/10 text-success",
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.08 }}
          className="glass-card-hover p-5 group"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{card.label}</span>
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${card.iconBg} transition-transform group-hover:scale-110`}>
              <card.icon className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-bold text-foreground tracking-tight">{card.value}</div>
          <div className={`text-xs font-medium mt-1 flex items-center gap-1 ${card.subColor}`}>
            {card.label === "Price" && (isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />)}
            {card.sub}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
