import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiRequestError, apiClient } from "../api/client";
import type { AnalyzeResponse, RestoreResponse } from "../api/types";
import { UploadWorkflow } from "./UploadWorkflow";

function pngFile(name = "wafer.png"): File {
  const bytes = new Uint8Array(24);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, 8);
  view.setUint32(20, 6);
  return new File([bytes], name, { type: "image/png" });
}

const analysis: AnalyzeResponse = {
  input: { width: 8, height: 6, media_type: "image/png" },
  analysis: { latency_ms: 2 },
  diagnostics: {},
  suitability: { recommendation: "restore", reasons: [], advisory_not_probability: true },
  warnings: [],
};

const restoration: RestoreResponse = {
  image: {
    encoding: "base64",
    media_type: "image/png",
    content: "iVBORw0KGgo=",
    width: 16,
    height: 12,
  },
  input: { width: 8, height: 6, media_type: "image/png" },
  inference: { latency_ms: 12, device: "cpu", phase_latency_ms: {} },
  model: { name: "SemiRestore", version: "test", training_revision: null, checkpoint_checksum: null },
  diagnostics: {},
  warnings: [],
};

describe("upload workflow", () => {
  const createObjectURL = vi.fn(() => "blob:local-preview");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
  });

  async function choose(file = pngFile()): Promise<void> {
    fireEvent.change(screen.getByLabelText("Choose SEM image"), { target: { files: [file] } });
    expect(await screen.findByText(file.name)).toBeVisible();
  }

  it("validates a file, shows local metadata, and submits each operation", async () => {
    vi.spyOn(apiClient, "analyze").mockResolvedValue(analysis);
    vi.spyOn(apiClient, "restore").mockResolvedValue(restoration);
    vi.spyOn(apiClient, "restoreAndAnalyze").mockResolvedValue(restoration);
    render(<UploadWorkflow connection="ready" unavailableReason={null} />);

    await choose();
    expect(screen.getByText("image/png")).toBeVisible();
    expect(screen.getByText("8 × 6 px")).toBeVisible();
    expect(screen.getByAltText("Local preview of the selected SEM image")).toHaveAttribute(
      "src",
      "blob:local-preview",
    );

    fireEvent.click(screen.getByRole("radio", { name: /Analyze only/i }));
    fireEvent.click(screen.getByRole("button", { name: "Analyze only" }));
    expect(await screen.findByText("Suitability: restore.")).toBeVisible();
    expect(apiClient.analyze).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("radio", { name: /Restore only/i }));
    fireEvent.click(screen.getByRole("button", { name: "Restore only" }));
    expect(await screen.findByText("Lossless PNG: 16 × 12 px.")).toBeVisible();
    expect(apiClient.restore).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("radio", { name: /Restore \+ analyze/i }));
    fireEvent.click(screen.getByRole("button", { name: "Restore + analyze" }));
    await waitFor(() => expect(apiClient.restoreAndAnalyze).toHaveBeenCalledTimes(1));
  });

  it("prevents duplicate submissions and supports cancellation", async () => {
    vi.spyOn(apiClient, "restoreAndAnalyze").mockImplementation(
      (_file, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    render(<UploadWorkflow connection="ready" unavailableReason={null} />);
    await choose();

    const submit = screen.getByRole("button", { name: "Restore + analyze" });
    fireEvent.click(submit);
    expect(await screen.findByRole("button", { name: "Processing…" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel request" }));

    expect(await screen.findByText(/request was cancelled/i)).toBeVisible();
    expect(apiClient.restoreAndAnalyze).toHaveBeenCalledTimes(1);
    expect(screen.getByText("wafer.png")).toBeVisible();
  });

  it("distinguishes backpressure and retains the selected image", async () => {
    vi.spyOn(apiClient, "restoreAndAnalyze").mockRejectedValue(
      new ApiRequestError("unsafe service detail", 503, "inference_busy", "req-1"),
    );
    render(<UploadWorkflow connection="ready" unavailableReason={null} />);
    await choose();

    fireEvent.click(screen.getByRole("button", { name: "Restore + analyze" }));
    expect(await screen.findByText("Service is busy")).toBeVisible();
    expect(screen.getByText(/queue is full/i)).toBeVisible();
    expect(screen.queryByText("unsafe service detail")).not.toBeInTheDocument();
    expect(screen.getByText("wafer.png")).toBeVisible();
  });

  it("disables submission when readiness is unavailable", async () => {
    render(
      <UploadWorkflow connection="unready" unavailableReason="Checkpoint is not available." />,
    );
    await choose();

    expect(screen.getByText("Checkpoint is not available.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Restore + analyze" })).toBeDisabled();
  });

  it("supports drag-and-drop and revokes object URLs on replacement and unmount", async () => {
    const { unmount } = render(<UploadWorkflow connection="ready" unavailableReason={null} />);
    fireEvent.drop(screen.getByRole("group", { name: "SEM image drop zone" }), {
      dataTransfer: { files: [pngFile("first.png")] },
    });
    expect(await screen.findByText("first.png")).toBeVisible();
    await choose(pngFile("second.png"));

    expect(revokeObjectURL).toHaveBeenCalledWith("blob:local-preview");
    unmount();
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);
  });
});
