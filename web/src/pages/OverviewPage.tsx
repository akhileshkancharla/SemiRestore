import { Link } from "react-router-dom";

import { useApiStatus } from "../state/ApiStatusContext";

function LoadingOverview() {
  return (
    <section aria-label="Synchronizing service state" aria-busy="true">
      <div className="page-heading">
        <div>
          <div className="skeleton skeleton--eyebrow" />
          <div className="skeleton skeleton--title" />
        </div>
      </div>
      <div className="status-grid">
        {[0, 1, 2].map((item) => (
          <div className="panel metric-card" key={item}>
            <div className="skeleton skeleton--label" />
            <div className="skeleton skeleton--metric" />
          </div>
        ))}
      </div>
    </section>
  );
}

export function OverviewPage() {
  const { connection, ready, model, version, error, refresh } = useApiStatus();

  if (connection === "checking") return <LoadingOverview />;

  const isReady = connection === "ready";
  return (
    <section>
      <div className="page-heading">
        <div>
          <p className="eyebrow">System overview</p>
          <h1>{isReady ? "Ready for inspection" : "Service attention required"}</h1>
          <p className="page-heading__description">
            {isReady
              ? "The verified model boundary is online and accepting SEM restoration work."
              : error || ready?.unavailable_reason || "The model service is not ready."}
          </p>
        </div>
        <button className="button button--secondary" type="button" onClick={() => void refresh()}>
          Refresh status
        </button>
      </div>

      <div className="status-grid">
        <article className="panel metric-card">
          <span className="metric-card__label">API connection</span>
          <strong className={`metric-card__value metric-card__value--${connection}`}>
            {connection === "offline" ? "Offline" : "Live"}
          </strong>
          <span className="metric-card__detail">Operational liveness boundary</span>
        </article>
        <article className="panel metric-card">
          <span className="metric-card__label">Model readiness</span>
          <strong className={`metric-card__value metric-card__value--${connection}`}>
            {model?.ready ? "Ready" : "Unavailable"}
          </strong>
          <span className="metric-card__detail">{model?.state || "No status received"}</span>
        </article>
        <article className="panel metric-card">
          <span className="metric-card__label">Compute device</span>
          <strong className="metric-card__value">{model?.device || "—"}</strong>
          <span className="metric-card__detail">Process-local model runtime</span>
        </article>
      </div>

      <div className="overview-layout">
        <article className="panel assurance-panel">
          <div className="panel__header">
            <div>
              <p className="eyebrow">Inspection assurance</p>
              <h2>Traceable restoration boundary</h2>
            </div>
            <span className="index-label">01</span>
          </div>
          <div className="wafer-visual" aria-hidden="true">
            <div className="wafer-visual__disc">
              <span className="wafer-visual__scan" />
              <span className="wafer-visual__node wafer-visual__node--one" />
              <span className="wafer-visual__node wafer-visual__node--two" />
              <span className="wafer-visual__node wafer-visual__node--three" />
            </div>
            <div className="wafer-visual__readout">
              <span>MODEL</span>
              <strong>{model?.model_version || "Not loaded"}</strong>
              <span>CHECKPOINT</span>
              <strong>
                {model?.checkpoint_checksum
                  ? `${model.checkpoint_checksum.slice(0, 12)}…`
                  : "Not verified"}
              </strong>
            </div>
          </div>
          <p className="assurance-panel__note">
            Diagnostics are advisory scientific indicators. Restored output is not ground truth.
          </p>
        </article>

        <article className="panel launch-panel">
          <div className="panel__header">
            <div>
              <p className="eyebrow">Workspace</p>
              <h2>Inspection staging</h2>
            </div>
            <span className="index-label">02</span>
          </div>
          <div className="launch-panel__body">
            <div className="launch-panel__icon" aria-hidden="true">
              <span />
            </div>
            <p>
              The dashboard foundation is connected. Upload and comparison workflows arrive in
              the next product milestone.
            </p>
            <Link className="button" to="/inspect">
              View workspace status
            </Link>
          </div>
        </article>
      </div>

      <footer className="service-signature">
        <span>APPLICATION</span>
        <strong>{version?.application || "semirestore"}</strong>
        <span>VERSION</span>
        <strong>{version?.version || "unavailable"}</strong>
        <span>READINESS</span>
        <strong>{ready?.state || connection}</strong>
      </footer>
    </section>
  );
}
