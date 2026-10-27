import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SearchProvider } from "@/contexts/SearchContext";
import DashboardLayout from "./layouts/DashboardLayout";
import Index from "./pages/Index.tsx";
import ModelsPage from "./pages/Models.tsx";
import WatchlistsPage from "./pages/Watchlists.tsx";
import NotFound from "./pages/NotFound.tsx";
import LandingPage from "./pages/Landing.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <SearchProvider>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route element={<DashboardLayout />}>
              <Route path="/app" element={<Index />} />
              <Route path="/stock/:symbol" element={<Index />} />
              <Route path="/watchlists" element={<WatchlistsPage />} />
              <Route path="/models" element={<ModelsPage />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </SearchProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
