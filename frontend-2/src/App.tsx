import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/contexts/AuthContext";
import { SearchProvider } from "@/contexts/SearchContext";
import RequireAuth from "@/components/auth/RequireAuth";
import DashboardLayout from "./layouts/DashboardLayout";
import Index from "./pages/Index.tsx";
import ModelsPage from "./pages/Models.tsx";
import WatchlistsPage from "./pages/Watchlists.tsx";
import NotFound from "./pages/NotFound.tsx";
import LandingAuthPage from "./pages/LandingAuth.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <SearchProvider>
            <Routes>
              <Route path="/" element={<LandingAuthPage />} />
              <Route
                element={
                  <RequireAuth>
                    <DashboardLayout />
                  </RequireAuth>
                }
              >
                <Route path="/app" element={<Index />} />
                <Route path="/stock/:symbol" element={<Index />} />
                <Route path="/watchlists" element={<WatchlistsPage />} />
                <Route path="/models" element={<ModelsPage />} />
              </Route>
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </SearchProvider>
        </BrowserRouter>
      </TooltipProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
