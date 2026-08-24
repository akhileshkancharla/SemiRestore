import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AnalyzeResponse, RestoreResponse } from "../api/types";
import type { WorkspaceResult } from "../workspace/types";
import { DiagnosticsPanel } from "./DiagnosticsPanel";

const measurement = (value: number, interpretation: string) => ({
  value,
  units: "normalized_intensity",
  interpretation,
  qualitative_label: "moderate",
});

const completeRestoration: RestoreResponse = {
  image: {
    encoding: "base64",
    media_type: "image/png",
    content: "iVBORw0KGgoAAAANSUhEUg==",
    width: 20,
    height: 16,
  },
  input: { width: 10, height: 8, media_type: "image/png" },
  inference: {
    latency_ms: 12,
    device: "cpu",
    phase_latency_ms: {
      preprocessing: 1.25,
      model_inference: 6.5,
      postprocessing: 2.25,
      total: 12,
    },
  },
  model: {
    name: "naf_sr",
    version: "controlled-v1",
    training_revision: "revision-7",
    checkpoint_checksum: "a".repeat(64),
  },
  diagnostics: {
    input: {
      intensity: { measurements: { mean: measurement(0.42, "Arithmetic mean brightness.") } },
      structure: {
        measurements: {
          edge_density: {
            ...measurement(0.18, "Fraction above the returned edge threshold."),
            units: "fraction_of_pixels",
          },
        },
      },
    },
    restored: {
      intensity: { measurements: { mean: measurement(0.44, "Arithmetic mean brightness.") } },
      structure: {
        measurements: {
          approximate_noise_sigma: measurement(0.02, "Returned no-reference noise estimate."),
        },
      },
    },
    suitability: {
      recommendation: "restore",
      reasons: ["No structural warning threshold was triggered."],
      advisory_not_probability: true,
    },
    quality_indicators: {
      dimension_contract_satisfied: true,
      sharpness_proxy_ratio: 1.125,
    },
    restoration: {
      latency_ms: {
        preprocessing: 0.9,
        inference_wait: 0.25,
        model_inference: 6.5,
        postprocessing: 1.8,
        total: 10.2,
      },
    },
    spatial: { alignment: 16, scale_factor: 2, internal_padding_required: true },
    tiles: { tile_count: 4, overlap: 32, global_conditioning_reused: true },
    clipping: { fraction_below_zero: 0.001, fraction_above_one: 0.002 },
    limitations: ["No-reference indicators cannot prove reconstruction correctness."],
  },
  warnings: ["Restored preview is advisory."],
};

const restorationResult: WorkspaceResult = {
  kind: "restoration",
  operation: "restore-and-analyze",
  data: completeRestoration,
};

const analysis: AnalyzeResponse = {
  input: { width: 10, height: 8, media_type: "image/png" },
  analysis: { latency_ms: 3.75 },
  diagnostics: {
    preprocessing: { original_mode: "L", normalized: true },
    intensity: { measurements: { mean: measurement(0.4, "Arithmetic mean brightness.") } },
    structure: { measurements: {} },
  },
  suitability: {
    recommendation: "warn",
    reasons: ["Texture and noise are ambiguous."],
    advisory_not_probability: true,
  },
  warnings: [],
};

describe("diagnostics and assurance panel", () => {
  it("renders complete returned diagnostics, metadata, and accessible metric explanations", () => {
    render(
      <DiagnosticsPanel
        result={restorationResult}
        connection="ready"
        modelHealth={{
          ready: true,
          state: "ready",
          unavailable_reason: null,
          device: "cpu",
          model_version: "controlled-v1",
          checkpoint_checksum: "a".repeat(64),
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Returned diagnostics and provenance" })).toBeVisible();
    expect(screen.getByText("naf_sr")).toBeVisible();
    expect(screen.getByText("10 × 8 px")).toBeVisible();
    expect(screen.getByText("20 × 16 px")).toBeVisible();
    expect(screen.getByText("Recommendation: Restore")).toBeVisible();
    expect(screen.getByText("Rule-based advisory; not a probability.")).toBeVisible();
    expect(screen.getAllByLabelText("Mean: Arithmetic mean brightness.")[0]).toHaveAttribute(
      "tabindex",
      "0",
    );
    expect(screen.getByText("Inference Wait")).toBeVisible();
    expect(screen.getByText("Tile Count")).toBeVisible();
    expect(screen.getByText("Fraction Below Zero")).toBeVisible();
    expect(screen.getByText("Sharpness Proxy Ratio")).toBeVisible();
  });

  it("renders partial analyze-only fields and marks unavailable output sections", () => {
    render(
      <DiagnosticsPanel
        result={{ kind: "analysis", operation: "analyze", data: analysis }}
        connection="ready"
        modelHealth={null}
      />,
    );

    expect(screen.getByText("Restored-output diagnostics are unavailable for analyze-only results.")).toBeVisible();
    expect(screen.getByText("3.75 ms")).toBeVisible();
    expect(screen.getByText("Original Mode")).toBeVisible();
    expect(screen.getByText("Recommendation: Warn")).toBeVisible();
  });

  it("separates returned warnings and limitations", () => {
    render(<DiagnosticsPanel result={restorationResult} connection="ready" />);
    const panel = screen.getByRole("heading", { name: "Warnings and limitations" }).closest("article");
    expect(panel).not.toBeNull();
    expect(within(panel!).getByText("Restored preview is advisory.")).toBeVisible();
    expect(
      within(panel!).getByText("No-reference indicators cannot prove reconstruction correctness."),
    ).toBeVisible();
  });

  it("labels readiness and unavailable provenance without invented values", () => {
    render(
      <DiagnosticsPanel
        result={{ kind: "analysis", operation: "analyze", data: analysis }}
        connection="unready"
      />,
    );

    expect(screen.getByText("Unready")).toBeVisible();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThanOrEqual(3);
    expect(screen.queryByText(/confidence score/i)).not.toBeInTheDocument();
  });

  it("renders a safe failure state without fabricating diagnostics", () => {
    render(
      <DiagnosticsPanel
        result={null}
        connection="ready"
        failure={{
          title: "Request failed safely",
          message: "unsafe internal path C:\\private\\model.pt",
          category: "server",
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Assurance unavailable" })).toBeVisible();
    expect(screen.getByText("No diagnostic result was returned. The selected image remains available.")).toBeVisible();
    expect(screen.queryByText(/private\\model\.pt/i)).not.toBeInTheDocument();
  });
});
