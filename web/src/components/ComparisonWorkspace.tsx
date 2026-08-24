import { useEffect, useRef, useState, type PointerEvent } from "react";

import type { RestoreResponse } from "../api/types";
import { restoredDownloadName, restoredPngBlob } from "../workspace/restoredImage";
import type { SelectedImage } from "../workspace/types";

interface ComparisonWorkspaceProps {
  original: SelectedImage;
  restoration: RestoreResponse;
}

type ViewMode = "fit" | "actual";

export function ComparisonWorkspace({ original, restoration }: ComparisonWorkspaceProps) {
  const [restoredUrl, setRestoredUrl] = useState<string | null>(null);
  const [payloadInvalid, setPayloadInvalid] = useState(false);
  const [divider, setDivider] = useState(50);
  const [viewMode, setViewMode] = useState<ViewMode>("fit");
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  useEffect(() => {
    setPayloadInvalid(false);
    try {
      const url = URL.createObjectURL(restoredPngBlob(restoration.image.content));
      setRestoredUrl(url);
      return () => URL.revokeObjectURL(url);
    } catch {
      setRestoredUrl(null);
      setPayloadInvalid(true);
      return undefined;
    }
  }, [restoration.image.content]);

  function setMode(mode: ViewMode): void {
    setViewMode(mode);
    setPan({ x: 0, y: 0 });
  }

  function resetView(): void {
    setViewMode("fit");
    setDivider(50);
    setPan({ x: 0, y: 0 });
  }

  function beginPan(event: PointerEvent<HTMLDivElement>): void {
    if (viewMode !== "actual") return;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function movePan(event: PointerEvent<HTMLDivElement>): void {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    setPan((current) => ({ x: current.x + deltaX, y: current.y + deltaY }));
  }

  function endPan(event: PointerEvent<HTMLDivElement>): void {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  }

  const restored = restoration.image;
  const isTwoX = restored.width === original.width * 2 && restored.height === original.height * 2;
  const contentStyle =
    viewMode === "actual"
      ? {
          width: `${restored.width}px`,
          height: `${restored.height}px`,
          transform: `translate(${pan.x}px, ${pan.y}px)`,
        }
      : undefined;

  return (
    <section className="panel comparison-workspace" aria-labelledby="comparison-heading">
      <div className="panel__header comparison-header">
        <div>
          <p className="eyebrow">Scientific comparison</p>
          <h2 id="comparison-heading">Original and restored capture</h2>
        </div>
        <div className="comparison-controls" aria-label="Comparison view controls">
          <button
            className={viewMode === "fit" ? "button button--compact button--active" : "button button--compact button--secondary"}
            type="button"
            onClick={() => setMode("fit")}
            aria-pressed={viewMode === "fit"}
          >
            Fit to view
          </button>
          <button
            className={viewMode === "actual" ? "button button--compact button--active" : "button button--compact button--secondary"}
            type="button"
            onClick={() => setMode("actual")}
            aria-pressed={viewMode === "actual"}
          >
            100% pixels
          </button>
          <button className="button button--compact button--secondary" type="button" onClick={resetView}>
            Reset view
          </button>
        </div>
      </div>

      {payloadInvalid ? (
        <div className="workflow-notice workflow-notice--server" role="alert">
          <strong>Restored preview unavailable</strong>
          <span>The returned payload could not be verified as a lossless PNG.</span>
        </div>
      ) : (
        <>
          <div
            className={`comparison-viewport comparison-viewport--${viewMode}`}
            onPointerDown={beginPan}
            onPointerMove={movePan}
            onPointerUp={endPan}
            onPointerCancel={endPan}
            aria-label="Synchronized original and restored image viewport"
          >
            <div className="comparison-content" style={contentStyle} data-testid="comparison-content">
              {original.previewSupported ? (
                <img
                  className="comparison-image comparison-image--original"
                  src={original.previewUrl}
                  alt="Original SEM capture"
                  draggable="false"
                />
              ) : (
                <div className="comparison-tiff-fallback">
                  Original TIFF preview is not supported by this browser.
                </div>
              )}
              {restoredUrl ? (
                <div
                  className="comparison-restored-layer"
                  style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}
                  data-testid="restored-layer"
                >
                  <img
                    className="comparison-image comparison-image--restored"
                    src={restoredUrl}
                    alt="Restored SEM capture"
                    draggable="false"
                  />
                </div>
              ) : null}
              <span className="comparison-label comparison-label--original">Original</span>
              <span className="comparison-label comparison-label--restored">Restored</span>
            </div>
          </div>

          <label className="comparison-slider">
            <span>Before / after boundary</span>
            <input
              type="range"
              min="0"
              max="100"
              value={divider}
              onChange={(event) => setDivider(Number(event.target.value))}
              aria-label="Before and after comparison position"
            />
            <output>{divider}% restored</output>
          </label>
        </>
      )}

      <div className="comparison-footer">
        <dl className="comparison-dimensions">
          <div>
            <dt>Original resolution</dt>
            <dd>{original.width} × {original.height} px</dd>
          </div>
          <div>
            <dt>Restored resolution</dt>
            <dd>{restored.width} × {restored.height} px</dd>
          </div>
          <div>
            <dt>Model scale</dt>
            <dd>{isTwoX ? "2× spatial scale" : "Derived from returned dimensions"}</dd>
          </div>
          <div>
            <dt>Display mode</dt>
            <dd>{viewMode === "fit" ? "Display-scaled to fit" : "100% restored pixels"}</dd>
          </div>
        </dl>
        {restoredUrl ? (
          <a
            className="button comparison-download"
            href={restoredUrl}
            download={restoredDownloadName(original.file.name)}
          >
            Download lossless PNG
          </a>
        ) : null}
      </div>
      <p className="comparison-disclaimer">
        Previews use native browser rendering only. No sharpening, filtering, or visual enhancement is applied.
      </p>
    </section>
  );
}
