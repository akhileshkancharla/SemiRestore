import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RestoreResponse } from "../api/types";
import type { SelectedImage } from "../workspace/types";
import { ComparisonWorkspace } from "./ComparisonWorkspace";

const selected: SelectedImage = {
  file: new File(["original"], "wafer field.tiff", { type: "image/tiff" }),
  previewUrl: "blob:original",
  previewSupported: true,
  width: 8,
  height: 6,
};

const restoration: RestoreResponse = {
  image: {
    encoding: "base64",
    media_type: "image/png",
    content: "iVBORw0KGgoAAAANSUhEUg==",
    width: 16,
    height: 12,
  },
  input: { width: 8, height: 6, media_type: "image/tiff" },
  inference: { latency_ms: 10, device: "cpu", phase_latency_ms: {} },
  model: { name: "SemiRestore", version: "test", training_revision: null, checkpoint_checksum: null },
  diagnostics: {},
  warnings: [],
};

describe("comparison workspace", () => {
  const createObjectURL = vi.fn<(blob: Blob) => string>(() => "blob:restored");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeObjectURL });
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
  });

  it("presents direct comparison, exact dimensions, scale, and download", async () => {
    render(<ComparisonWorkspace original={selected} restoration={restoration} />);

    expect(screen.getByAltText("Original SEM capture")).toHaveAttribute("src", "blob:original");
    expect(await screen.findByAltText("Restored SEM capture")).toHaveAttribute(
      "src",
      "blob:restored",
    );
    expect(screen.getByText("8 × 6 px")).toBeVisible();
    expect(screen.getByText("16 × 12 px")).toBeVisible();
    expect(screen.getByText("2× spatial scale")).toBeVisible();
    expect(screen.getByRole("link", { name: "Download lossless PNG" })).toHaveAttribute(
      "download",
      "wafer_field-restored.png",
    );
    expect(createObjectURL.mock.calls[0]?.[0]).toBeInstanceOf(Blob);
  });

  it("supports keyboard slider positioning and fit/actual/reset controls", async () => {
    render(<ComparisonWorkspace original={selected} restoration={restoration} />);
    await screen.findByAltText("Restored SEM capture");
    const slider = screen.getByRole("slider", { name: "Before and after comparison position" });

    fireEvent.change(slider, { target: { value: "72" } });
    expect(screen.getByText("72% restored")).toBeVisible();
    expect(screen.getByTestId("restored-layer")).toHaveStyle({ clipPath: "inset(0 28% 0 0)" });

    fireEvent.click(screen.getByRole("button", { name: "100% pixels" }));
    expect(screen.getByText("100% restored pixels")).toBeVisible();
    expect(screen.getByTestId("comparison-content")).toHaveStyle({ width: "16px", height: "12px" });

    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(screen.getByText("50% restored")).toBeVisible();
    expect(screen.getByText("Display-scaled to fit")).toBeVisible();
  });

  it("uses one synchronized transform for pointer panning", async () => {
    render(<ComparisonWorkspace original={selected} restoration={restoration} />);
    await screen.findByAltText("Restored SEM capture");
    fireEvent.click(screen.getByRole("button", { name: "100% pixels" }));
    const viewport = screen.getByLabelText("Synchronized original and restored image viewport");

    fireEvent.pointerDown(viewport, { pointerId: 1, clientX: 10, clientY: 10 });
    fireEvent.pointerMove(viewport, { pointerId: 1, clientX: 26, clientY: 31 });
    fireEvent.pointerUp(viewport, { pointerId: 1 });

    await waitFor(() => {
      expect(screen.getByTestId("comparison-content")).toHaveStyle({
        transform: "translate(16px, 21px)",
      });
    });
  });

  it("revokes only the restored object URL on cleanup", async () => {
    const { unmount } = render(
      <ComparisonWorkspace original={selected} restoration={restoration} />,
    );
    await screen.findByAltText("Restored SEM capture");
    unmount();

    expect(revokeObjectURL).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:restored");
    expect(revokeObjectURL).not.toHaveBeenCalledWith("blob:original");
  });

  it("fails safely when returned content is not a PNG", async () => {
    render(
      <ComparisonWorkspace
        original={selected}
        restoration={{ ...restoration, image: { ...restoration.image, content: "bm90LXBuZw==" } }}
      />,
    );

    expect(await screen.findByText("Restored preview unavailable")).toBeVisible();
    expect(screen.queryByRole("link", { name: /Download/i })).not.toBeInTheDocument();
  });
});
