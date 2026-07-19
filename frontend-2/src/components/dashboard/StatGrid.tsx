import type { StockData } from "@/hooks/useStockStore";
import { formatPrice } from "@/lib/format";

interface StatGridProps {
  data: StockData;
}

/** Compact 2x2 KPI grid for the dashboard right rail. */
export default function StatGrid({ data }: StatGridProps) {
  const cells: Array<{ label: string; value: string }> = [
    { label: "Price", value: formatPrice(data.current_price) },
    { label: "Market Cap", value: data.market_cap || "—" },
    { label: "P/E Ratio", value: data.pe_ratio != null ? data.pe_ratio.toFixed(1) : "—" },
    { label: "Volume", value: data.volume || "—" },
  ];
  return (
    <div className="surface p-1 grid grid-cols-2">
      {cells.map((c, i) => (
        <div
          key={c.label}
          className={`p-4 ${i % 2 === 0 ? "border-r border-border" : ""} ${i < 2 ? "border-b border-border" : ""}`}
        >
          <p className="label-caps">{c.label}</p>
          <p className="mt-1 font-mono text-lg font-semibold text-foreground">{c.value}</p>
        </div>
      ))}
    </div>
  );
}
