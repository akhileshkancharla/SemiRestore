export function InspectionPage() {
  return (
    <section>
      <div className="page-heading">
        <div>
          <p className="eyebrow">Inspection workspace</p>
          <h1>Workspace foundation</h1>
          <p className="page-heading__description">
            Routing and service connectivity are ready. Image upload, analysis controls, restored
            comparison, and result export intentionally begin in Milestone 19.
          </p>
        </div>
      </div>
      <div className="panel placeholder-stage">
        <div className="placeholder-stage__grid" aria-hidden="true" />
        <span className="placeholder-stage__tag">FOUNDATION / NO USER DATA</span>
        <h2>Restoration workspace reserved</h2>
        <p>No image is selected, transmitted, cached, or persisted by this dashboard foundation.</p>
      </div>
    </section>
  );
}
