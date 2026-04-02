import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import ArrowCanvas from "@/components/ArrowCanvas";

const Index = () => {
  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      {/* Background gradient */}
      <div
        className="absolute inset-0 z-0"
        style={{
          background:
            "linear-gradient(160deg, hsl(222 47% 5%) 0%, hsl(230 50% 10%) 50%, hsl(222 47% 5%) 100%)",
        }}
      />

      {/* Grid overlay */}
      <div className="grid-overlay absolute inset-0 z-[1] opacity-40" />

      {/* Faint chart lines */}
      <svg
        className="absolute inset-0 z-[2] h-full w-full opacity-[0.06]"
        preserveAspectRatio="none"
        viewBox="0 0 1440 900"
      >
        <polyline
          fill="none"
          stroke="hsl(150 100% 64%)"
          strokeWidth="2"
          points="0,600 120,580 240,520 360,550 480,420 600,460 720,380 840,400 960,350 1080,370 1200,300 1320,320 1440,280"
        />
        <polyline
          fill="none"
          stroke="hsl(0 100% 65%)"
          strokeWidth="2"
          points="0,400 120,430 240,480 360,450 480,500 600,470 720,520 840,510 960,560 1080,540 1200,590 1320,570 1440,620"
        />
      </svg>

      <Navbar />

      {/* Hero content */}
      <div className="pointer-events-none relative z-20 flex min-h-screen flex-col items-center justify-center px-6">
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="font-display text-6xl font-bold tracking-tight sm:text-8xl"
        >
          <span className="text-gradient-brand">AInvestify</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-4 font-mono text-sm tracking-widest text-muted-foreground"
        >
          STOCK INSIGHTS
        </motion.p>
      </div>

      {/* Canvas on top for clicks */}
      <ArrowCanvas />
    </div>
  );
};

export default Index;
