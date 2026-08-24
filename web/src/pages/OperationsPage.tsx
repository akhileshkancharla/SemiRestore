import { API_BASE_URL } from "../api/client";
import { useApiStatus } from "../state/ApiStatusContext";

export function OperationsPage() {
  const { connection, ready, model, version, error, refresh } = useApiStatus();

  return (
    <section>
      <div className="page-heading">
        <div>
          <p className="eyebrow">Operations</p>
          <h1>Service boundary</h1>
          <p className="page-heading__description">
            Safe operational metadata from the platform health and version contracts.
          </p>
        </div>
        <button className="button button--secondary" type="button" onClick={() => void refresh()}>
          Poll now
        </button>
      </div>

      {error ? <div className="notice notice--error">{error}</div> : null}

      <dl className="panel detail-list">
        <div>
          <dt>Connection</dt>
          <dd>{connection}</dd>
        </div>
        <div>
          <dt>API base URL</dt>
          <dd>{API_BASE_URL}</dd>
        </div>
        <div>
          <dt>Application version</dt>
          <dd>{version?.version || "Unavailable"}</dd>
        </div>
        <div>
          <dt>Model state</dt>
          <dd>{ready?.state || "Unavailable"}</dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>{model?.device || "Not reported"}</dd>
        </div>
        <div>
          <dt>Model version</dt>
          <dd>{model?.model_version || "Not reported"}</dd>
        </div>
      </dl>
    </section>
  );
}
