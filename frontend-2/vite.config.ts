import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/chart": "http://127.0.0.1:8000",
      "/fundamentals": "http://127.0.0.1:8000",
      "/news": "http://127.0.0.1:8000",
      "/tickers": "http://127.0.0.1:8000",
      "/chatbot": "http://127.0.0.1:8000",
      // SPA deep link /stock/TICKER — backend serves index.html
      "/stock": "http://127.0.0.1:8000",
    },
    hmr: {
      overlay: false,
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime"],
  },
}));
