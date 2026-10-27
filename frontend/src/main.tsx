import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { preloadLogoManifest } from "@/lib/logos";

preloadLogoManifest();

createRoot(document.getElementById("root")!).render(<App />);
