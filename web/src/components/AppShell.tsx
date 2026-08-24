import { NavLink, Outlet } from "react-router-dom";

import { useApiStatus, type ConnectionState } from "../state/ApiStatusContext";

const navigation = [
  { to: "/", label: "Overview", code: "OV" },
  { to: "/inspect", label: "Inspection", code: "IN" },
  { to: "/operations", label: "Operations", code: "OP" },
];

const statusCopy: Record<ConnectionState, string> = {
  checking: "Checking service",
  ready: "Model ready",
  unready: "Model unavailable",
  offline: "API offline",
};

export function AppShell() {
  const { connection, version } = useApiStatus();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>SemiRestore</strong>
            <small>Inspection console</small>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          {navigation.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => (isActive ? "nav-item nav-item--active" : "nav-item")}
            >
              <span className="nav-item__code">{item.code}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <span className={`status-dot status-dot--${connection}`} aria-hidden="true" />
          <div>
            <strong>{statusCopy[connection]}</strong>
            <small>{version ? `Service ${version.version}` : "Awaiting service identity"}</small>
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div>
            <span className="topbar__context">SEM image restoration</span>
            <span className="topbar__divider" aria-hidden="true" />
            <span className="topbar__mode">Scientific assurance mode</span>
          </div>
          <div className={`readiness-chip readiness-chip--${connection}`}>
            <span className="status-dot" aria-hidden="true" />
            {statusCopy[connection]}
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
