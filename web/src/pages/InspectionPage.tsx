import { UploadWorkflow } from "../components/UploadWorkflow";
import { useApiStatus } from "../state/ApiStatusContext";

export function InspectionPage() {
  const { connection, ready } = useApiStatus();

  return (
    <section>
      <div className="page-heading">
        <div>
          <p className="eyebrow">Inspection workspace</p>
          <h1>SEM restoration workspace</h1>
          <p className="page-heading__description">
            Validate one local capture, choose the scientific operation, and submit it through the
            production model-service boundary. Images are not permanently stored by the dashboard.
          </p>
        </div>
      </div>
      <UploadWorkflow
        connection={connection}
        unavailableReason={ready?.unavailable_reason ?? null}
      />
    </section>
  );
}
