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
      // Forecast can take 30–90s on first run; default proxy timeout is too short.
      "/api": {
        target: "http://127.0.0.1:8000",
        timeout: 180_000,
        proxyTimeout: 180_000,
      },
      "/chart": "http://127.0.0.1:8000",
      "/fundamentals": "http://127.0.0.1:8000",
      "/news": "http://127.0.0.1:8000",
      "/tickers": "http://127.0.0.1:8000",
      "/chatbot": "http://127.0.0.1:8000",
      // Do NOT proxy /stock — that is a React Router page. Proxying it to FastAPI
      // breaks SPA navigation/refresh on /stock/:symbol during Vite dev.
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
