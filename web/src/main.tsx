import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";

import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

const app = import.meta.env.MODE === "pages" ? (
  <HashRouter>
    <App />
  </HashRouter>
) : (
  <BrowserRouter>
    <App />
  </BrowserRouter>
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      {app}
    </ErrorBoundary>
  </StrictMode>,
);
