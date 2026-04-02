import { Outlet } from "react-router-dom";

/**
 * Wraps all authenticated app routes so the global footer is always last in the document (below page + chat widget).
 */
export default function DashboardLayout() {
  return (
    <div
      id="dashboard-theme-scope"
      className="dashboard-theme-scope dark min-h-screen bg-background overflow-x-hidden"
    >
      <Outlet />
    </div>
  );
}
