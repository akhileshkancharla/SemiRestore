import { Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { InspectionPage } from "./pages/InspectionPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { OperationsPage } from "./pages/OperationsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { ApiStatusProvider } from "./state/ApiStatusContext";

export function App() {
  return (
    <ApiStatusProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<OverviewPage />} />
          <Route path="inspect" element={<InspectionPage />} />
          <Route path="operations" element={<OperationsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </ApiStatusProvider>
  );
}
