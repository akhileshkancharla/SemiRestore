import type { ModelHealthResponse } from "../api/types";
import type { ConnectionState } from "../state/ApiStatusContext";
import type { ErrorPresentation } from "../workspace/errorPresentation";
import type { WorkspaceResult } from "../workspace/types";

interface DiagnosticsPanelProps {
  result: WorkspaceResult | null;
  connection: ConnectionState;
  modelHealth?: ModelHealthResponse | null;
  failure?: ErrorPresentation | null;
}

type SafeRecord = Record<string, unknown>;
type DisplayValue = string | number | boolean;

function record(value: unknown): SafeRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as SafeRecord)
    : {};
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function friendlyName(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function displayValue(value: DisplayValue): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(4);
    return value.toPrecision(5);
  }
  return value;
}

function scalarRows(source: unknown): Array<[string, DisplayValue]> {
  const rows: Array<[string, DisplayValue]> = [];
  for (const [key, value] of Object.entries(record(source))) {
    if (typeof value === "string" || typeof value === "boolean") rows.push([key, value]);
    else if (typeof value === "number" && Number.isFinite(value)) rows.push([key, value]);
    if (Array.isArray(value) && value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
      rows.push([key, value.map((item) => String(item)).join(", ")]);
    }
  }
  return rows;
}

function Unavailable({ children = "Not returned by this operation." }: { children?: string }) {
  return <p className="unavailable-value">{children}</p>;
}

function MetadataTable({ source, empty }: { source: unknown; empty?: string }) {
  const rows = scalarRows(source);
  if (rows.length === 0) return <Unavailable>{empty}</Unavailable>;
  return (
    <dl className="assurance-table">
      {rows.map(([key, value]) => (
        <div key={key}>
          <dt>{friendlyName(key)}</dt>
          <dd>{displayValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function MeasurementTable({ source }: { source: unknown }) {
  const measurements = record(record(source).measurements);
  const entries = Object.entries(measurements).flatMap(([name, candidate]) => {
    const measurement = record(candidate);
    const value = measurement.value;
    if (
      !(
        typeof value === "string" ||
        typeof value === "boolean" ||
        (typeof value === "number" && Number.isFinite(value))
      )
    ) {
      return [];
    }
    return [[name, value, measurement] as const];
  });
  if (entries.length === 0) return <Unavailable />;
  return (
    <div className="metric-table" role="table" aria-label="Returned diagnostic measurements">
      {entries.map(([name, value, measurement]) => {
        const interpretation = text(measurement.interpretation);
        const units = text(measurement.units);
        const label = text(measurement.qualitative_label);
        return (
          <div className="metric-row" role="row" key={name}>
            <div role="cell" className="metric-row__name">
              <span>{friendlyName(name)}</span>
              {interpretation ? (
                <span
                  className="metric-help"
                  title={interpretation}
                  aria-label={`${friendlyName(name)}: ${interpretation}`}
                  tabIndex={0}
                >
                  ?
                </span>
              ) : null}
            </div>
            <div role="cell" className="metric-row__value">
              <strong>{displayValue(value)}</strong>
              {units ? <small>{friendlyName(units)}</small> : null}
            </div>
            <div role="cell" className="metric-row__kind">
              <span>Measured</span>
              {label ? <small>{friendlyName(label)}</small> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DiagnosticGroup({ title, source }: { title: string; source: unknown }) {
  const group = record(source);
  return (
    <section className="assurance-subsection">
      <h4>{title}</h4>
      <MeasurementTable source={group} />
      {text(group.qualitative_label) ? (
        <p className="returned-label">
          Returned assessment: <strong>{friendlyName(text(group.qualitative_label)!)}</strong>
        </p>
      ) : null}
    </section>
  );
}

function TimingTable({ source, title }: { source: unknown; title: string }) {
  const rows = Object.entries(record(source)).flatMap(([name, value]) => {
    const latency = finiteNumber(value);
    return latency === null ? [] : [[name, latency] as const];
  });
  return (
    <section className="assurance-subsection">
      <h4>{title}</h4>
      {rows.length ? (
        <dl className="timing-list">
          {rows.map(([name, latency]) => (
            <div key={name}>
              <dt>{friendlyName(name)}</dt>
              <dd>{latency.toFixed(2)} ms</dd>
            </div>
          ))}
        </dl>
      ) : (
        <Unavailable>No timing values were returned.</Unavailable>
      )}
    </section>
  );
}

function QualityIndicators({ source }: { source: unknown }) {
  const rows = scalarRows(source);
  if (!rows.length) return <Unavailable />;
  return (
    <dl className="indicator-list">
      {rows.map(([name, value]) => (
        <div key={name}>
          <dt>
            {friendlyName(name)}
            <span
              className="metric-help"
              title="Value returned by the no-reference assurance contract."
              aria-label={`${friendlyName(name)}: returned no-reference assurance value`}
              tabIndex={0}
            >
              ?
            </span>
          </dt>
          <dd>
            <strong>{displayValue(value)}</strong>
            <small>{typeof value === "boolean" ? "Contract check" : "Heuristic indicator"}</small>
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function DiagnosticsPanel({
  result,
  connection,
  modelHealth = null,
  failure = null,
}: DiagnosticsPanelProps) {
  if (!result) {
    if (!failure) return null;
    return (
      <section className="panel assurance-workspace" aria-labelledby="assurance-heading">
        <p className="eyebrow">Scientific assurance</p>
        <h2 id="assurance-heading">Assurance unavailable</h2>
        <div className="workflow-notice workflow-notice--server" role="status">
          <strong>No assurance result</strong>
          <span>No diagnostic result was returned. The selected image remains available.</span>
        </div>
      </section>
    );
  }

  const isRestoration = result.kind === "restoration";
  const diagnostics = record(result.data.diagnostics);
  const inputDiagnostics = isRestoration ? record(diagnostics.input) : diagnostics;
  const restoredDiagnostics = isRestoration ? record(diagnostics.restored) : {};
  const suitability = isRestoration ? record(diagnostics.suitability) : result.data.suitability;
  const reasons = stringList(record(suitability).reasons);
  const recommendation = text(record(suitability).recommendation);
  const warnings = result.data.warnings;
  const limitations = stringList(diagnostics.limitations);
  const restorationMetadata = isRestoration ? record(diagnostics.restoration) : {};
  const serviceTimings = record(restorationMetadata.latency_ms);
  const model = isRestoration ? result.data.model : null;
  const device = isRestoration ? result.data.inference.device : modelHealth?.device;
  const modelVersion = model?.version ?? modelHealth?.model_version ?? null;
  const restoredWidth = isRestoration ? result.data.image.width : null;
  const restoredHeight = isRestoration ? result.data.image.height : null;
  const phaseTimings = isRestoration
    ? result.data.inference.phase_latency_ms
    : { analysis: result.data.analysis.latency_ms };

  return (
    <section className="panel assurance-workspace" aria-labelledby="assurance-heading">
      <div className="panel__header assurance-heading">
        <div>
          <p className="eyebrow">Scientific assurance</p>
          <h2 id="assurance-heading">Returned diagnostics and provenance</h2>
        </div>
        <span className="assurance-kind">No-reference / advisory</span>
      </div>

      <dl className="assurance-summary">
        <div>
          <dt>Service readiness</dt>
          <dd>{connection === "ready" ? "Ready" : friendlyName(connection)}</dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>{device || "Unavailable"}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{model?.name || "Unavailable"}</dd>
        </div>
        <div>
          <dt>Model version</dt>
          <dd>{modelVersion || "Unavailable"}</dd>
        </div>
        <div>
          <dt>Original resolution</dt>
          <dd>{result.data.input.width} × {result.data.input.height} px</dd>
        </div>
        <div>
          <dt>Restored resolution</dt>
          <dd>{restoredWidth && restoredHeight ? `${restoredWidth} × ${restoredHeight} px` : "Unavailable"}</dd>
        </div>
      </dl>

      <div className="assurance-grid">
        <article className="assurance-card assurance-card--suitability">
          <p className="eyebrow">Heuristic assessment</p>
          <h3>Suitability</h3>
          {recommendation ? (
            <>
              <span className={`recommendation recommendation--${recommendation}`}>
                Recommendation: {friendlyName(recommendation)}
              </span>
              <p className="advisory-note">Rule-based advisory; not a probability.</p>
              {reasons.length ? (
                <ul className="assurance-list">
                  {reasons.map((reason) => <li key={reason}>{reason}</li>)}
                </ul>
              ) : (
                <Unavailable>No suitability reasons were returned.</Unavailable>
              )}
            </>
          ) : (
            <Unavailable />
          )}
        </article>

        <article className="assurance-card">
          <p className="eyebrow">Measured input</p>
          <h3>Input diagnostics</h3>
          <DiagnosticGroup title="Intensity" source={inputDiagnostics.intensity} />
          <DiagnosticGroup title="Structure" source={inputDiagnostics.structure} />
          {Object.keys(record(inputDiagnostics.preprocessing)).length ? (
            <section className="assurance-subsection">
              <h4>Preprocessing</h4>
              <MetadataTable source={inputDiagnostics.preprocessing} />
            </section>
          ) : null}
        </article>

        <article className="assurance-card">
          <p className="eyebrow">Measured output</p>
          <h3>Restored-output diagnostics</h3>
          {isRestoration ? (
            <>
              <DiagnosticGroup title="Intensity" source={restoredDiagnostics.intensity} />
              <DiagnosticGroup title="Structure" source={restoredDiagnostics.structure} />
            </>
          ) : (
            <Unavailable>Restored-output diagnostics are unavailable for analyze-only results.</Unavailable>
          )}
        </article>

        <article className="assurance-card">
          <p className="eyebrow">No-reference indicators</p>
          <h3>Structural quality indicators</h3>
          <QualityIndicators source={diagnostics.quality_indicators} />
        </article>

        <article className="assurance-card">
          <p className="eyebrow">Execution</p>
          <h3>Latency</h3>
          <TimingTable title="Pipeline phases" source={phaseTimings} />
          {Object.keys(serviceTimings).length ? (
            <TimingTable title="Service phases and queue wait" source={serviceTimings} />
          ) : null}
        </article>

        <article className="assurance-card">
          <p className="eyebrow">Spatial execution</p>
          <h3>Spatial and tiled inference</h3>
          <section className="assurance-subsection">
            <h4>Spatial plan</h4>
            <MetadataTable source={diagnostics.spatial} />
          </section>
          <section className="assurance-subsection">
            <h4>Tile metadata</h4>
            <MetadataTable
              source={diagnostics.tiles}
              empty="No tile metadata was returned for this operation."
            />
          </section>
        </article>

        <article className="assurance-card">
          <p className="eyebrow">Output bounds</p>
          <h3>Clipping metadata</h3>
          <MetadataTable source={diagnostics.clipping} />
        </article>

        <article className="assurance-card assurance-card--warnings">
          <p className="eyebrow">Qualification</p>
          <h3>Warnings and limitations</h3>
          {warnings.length ? (
            <section className="assurance-subsection">
              <h4>Warnings</h4>
              <ul className="assurance-list assurance-list--warning">
                {warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </section>
          ) : (
            <Unavailable>No warnings were returned.</Unavailable>
          )}
          {limitations.length ? (
            <section className="assurance-subsection">
              <h4>Limitations</h4>
              <ul className="assurance-list">
                {limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
              </ul>
            </section>
          ) : (
            <Unavailable>No limitations were returned.</Unavailable>
          )}
        </article>
      </div>
    </section>
  );
}
