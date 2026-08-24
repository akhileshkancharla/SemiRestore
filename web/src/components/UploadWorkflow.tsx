import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";

import { RequestCancelledError, apiClient } from "../api/client";
import type { ConnectionState } from "../state/ApiStatusContext";
import { presentRequestError, type ErrorPresentation } from "../workspace/errorPresentation";
import {
  formatFileSize,
  inspectImageFile,
  LocalImageValidationError,
  SUPPORTED_IMAGE_TYPES,
} from "../workspace/imageFile";
import type { OperationMode, SelectedImage, WorkspaceResult } from "../workspace/types";

interface UploadWorkflowProps {
  connection: ConnectionState;
  unavailableReason: string | null;
}

type WorkflowPhase = "idle" | "validating" | "ready" | "processing" | "success" | "cancelled";

const operationLabels: Record<OperationMode, string> = {
  analyze: "Analyze only",
  restore: "Restore only",
  "restore-and-analyze": "Restore + analyze",
};

export function UploadWorkflow({ connection, unavailableReason }: UploadWorkflowProps) {
  const [selected, setSelected] = useState<SelectedImage | null>(null);
  const [operation, setOperation] = useState<OperationMode>("restore-and-analyze");
  const [phase, setPhase] = useState<WorkflowPhase>("idle");
  const [error, setError] = useState<ErrorPresentation | null>(null);
  const [result, setResult] = useState<WorkspaceResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const selectedRef = useRef<SelectedImage | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (selectedRef.current) URL.revokeObjectURL(selectedRef.current.previewUrl);
    },
    [],
  );

  async function selectFile(file: File | undefined): Promise<void> {
    if (!file) return;
    abortRef.current?.abort();
    setPhase("validating");
    setError(null);
    setResult(null);
    try {
      const dimensions = await inspectImageFile(file);
      const previewUrl = URL.createObjectURL(file);
      setSelected((previous) => {
        if (previous) URL.revokeObjectURL(previous.previewUrl);
        return {
          file,
          previewUrl,
          previewSupported: file.type !== "image/tiff",
          ...dimensions,
        };
      });
      setPhase("ready");
    } catch (cause) {
      setPhase("idle");
      if (cause instanceof LocalImageValidationError) {
        setError({ title: "Image rejected", message: cause.message, category: "validation" });
      } else {
        setError({
          title: "Image could not be inspected",
          message: "Choose another PNG, JPEG, or single-frame TIFF image.",
          category: "validation",
        });
      }
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function submit(): Promise<void> {
    if (!selected || phase === "processing" || connection !== "ready") return;
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase("processing");
    setError(null);
    setResult(null);
    try {
      if (operation === "analyze") {
        const data = await apiClient.analyze(selected.file, controller.signal);
        setResult({ kind: "analysis", operation, data });
      } else {
        const data =
          operation === "restore"
            ? await apiClient.restore(selected.file, controller.signal)
            : await apiClient.restoreAndAnalyze(selected.file, controller.signal);
        setResult({ kind: "restoration", operation, data });
      }
      setPhase("success");
    } catch (cause) {
      if (cause instanceof RequestCancelledError || controller.signal.aborted) {
        setPhase("cancelled");
      } else {
        setPhase("ready");
        setError(presentRequestError(cause));
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>): void {
    void selectFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    if (phase !== "processing") void selectFile(event.dataTransfer.files[0]);
  }

  const serviceReady = connection === "ready";
  const processing = phase === "processing";
  const statusMessage =
    phase === "validating"
      ? "Validating the selected image locally."
      : processing
        ? `${operationLabels[operation]} is in progress.`
        : phase === "success"
          ? `${operationLabels[operation]} completed successfully.`
          : phase === "cancelled"
            ? "The request was cancelled. The selected image is still available."
            : selected
              ? "Image validated and ready to submit."
              : "No image selected.";

  return (
    <div className="workflow-layout">
      <section className="panel upload-panel" aria-labelledby="upload-heading">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Input specimen</p>
            <h2 id="upload-heading">Select one SEM image</h2>
          </div>
          <span className="index-label">01</span>
        </div>

        <div
          className={`drop-zone${processing ? " drop-zone--disabled" : ""}`}
          role="group"
          aria-label="SEM image drop zone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept={SUPPORTED_IMAGE_TYPES.join(",")}
            aria-label="Choose SEM image"
            onChange={onInputChange}
            disabled={processing}
          />
          <div className="drop-zone__glyph" aria-hidden="true">
            +
          </div>
          <strong>Drop a validated SEM capture here</strong>
          <span>PNG, JPEG, or single-frame TIFF</span>
          <button
            className="button button--secondary"
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={processing}
          >
            Browse files
          </button>
        </div>

        {selected ? (
          <article className="selected-image" aria-label="Selected image details">
            <div className="selected-image__preview">
              {selected.previewSupported ? (
                <img src={selected.previewUrl} alt="Local preview of the selected SEM image" />
              ) : (
                <div className="selected-image__preview-fallback">
                  <span>TIFF</span>
                  <small>Metadata preview</small>
                </div>
              )}
            </div>
            <dl>
              <div>
                <dt>Filename</dt>
                <dd>{selected.file.name}</dd>
              </div>
              <div>
                <dt>Media type</dt>
                <dd>{selected.file.type}</dd>
              </div>
              <div>
                <dt>Dimensions</dt>
                <dd>{selected.width} × {selected.height} px</dd>
              </div>
              <div>
                <dt>File size</dt>
                <dd>{formatFileSize(selected.file.size)}</dd>
              </div>
            </dl>
          </article>
        ) : null}
      </section>

      <section className="panel operation-panel" aria-labelledby="operation-heading">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Operation</p>
            <h2 id="operation-heading">Choose processing mode</h2>
          </div>
          <span className="index-label">02</span>
        </div>

        <fieldset className="operation-options" disabled={processing}>
          <legend className="visually-hidden">Processing mode</legend>
          {(Object.keys(operationLabels) as OperationMode[]).map((value) => (
            <label key={value} className={operation === value ? "mode-card mode-card--active" : "mode-card"}>
              <input
                type="radio"
                name="operation"
                value={value}
                checked={operation === value}
                onChange={() => setOperation(value)}
              />
              <span>
                <strong>{operationLabels[value]}</strong>
                <small>
                  {value === "analyze"
                    ? "Input diagnostics without restoration"
                    : value === "restore"
                      ? "Lossless restored PNG"
                      : "Restoration with complete diagnostics"}
                </small>
              </span>
            </label>
          ))}
        </fieldset>

        {!serviceReady ? (
          <div className="workflow-notice workflow-notice--readiness" role="status">
            <strong>Submission unavailable</strong>
            <span>{unavailableReason || "The model service is not ready for restoration work."}</span>
          </div>
        ) : null}

        {error ? (
          <div className={`workflow-notice workflow-notice--${error.category}`} role="alert">
            <strong>{error.title}</strong>
            <span>{error.message}</span>
          </div>
        ) : null}

        <div className="workflow-actions">
          <button
            className="button"
            type="button"
            onClick={() => void submit()}
            disabled={!selected || !serviceReady || processing || phase === "validating"}
          >
            {processing ? "Processing…" : operationLabels[operation]}
          </button>
          {processing ? (
            <button className="button button--danger" type="button" onClick={() => abortRef.current?.abort()}>
              Cancel request
            </button>
          ) : null}
        </div>

        <p className="workflow-status" role="status" aria-live="polite">
          {statusMessage}
        </p>

        {result ? (
          <article className="result-receipt" aria-label="Operation result">
            <span className="result-receipt__mark" aria-hidden="true">✓</span>
            <div>
              <strong>API result received</strong>
              <p>
                {result.kind === "analysis"
                  ? `Suitability: ${result.data.suitability.recommendation}.`
                  : `Lossless PNG: ${result.data.image.width} × ${result.data.image.height} px.`}
              </p>
              <small>Scientific comparison and detailed assurance panels follow in the next milestones.</small>
            </div>
          </article>
        ) : null}
      </section>
    </div>
  );
}
