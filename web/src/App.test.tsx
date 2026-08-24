import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";

const readyPayloads: Record<string, unknown> = {
  "/health/live": { status: "alive" },
  "/health/ready": { ready: true, state: "ready", unavailable_reason: null },
  "/health/model": {
    ready: true,
    state: "ready",
    unavailable_reason: null,
    device: "cpu",
    model_version: "controlled-test-model",
    checkpoint_checksum: "abcdef0123456789",
  },
  "/version": { application: "semirestore", version: "0.1.0" },
};

function installApi(payloads: Record<string, unknown> = readyPayloads): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input), "http://dashboard.test").pathname.replace(
        /^\/service/,
        "",
      );
      const body = payloads[path];
      if (body === undefined) return new Response(null, { status: 404 });
      const isUnready = path === "/health/ready" && (body as { ready?: boolean }).ready === false;
      return new Response(JSON.stringify(body), {
        status: isUnready ? 503 : 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
}

function renderApp(path = "/"): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("dashboard foundation", () => {
  it("shows loading state before rendering verified service readiness", async () => {
    installApi();
    renderApp();

    expect(screen.getByLabelText("Synchronizing service state")).toHaveAttribute(
      "aria-busy",
      "true",
    );
    expect(await screen.findByRole("heading", { name: "Ready for inspection" })).toBeVisible();
    expect(screen.getByText("controlled-test-model")).toBeVisible();
    expect(screen.getByText("cpu")).toBeVisible();
  });

  it("reports an unready model without presenting the API as offline", async () => {
    installApi({
      ...readyPayloads,
      "/health/ready": {
        ready: false,
        state: "unavailable",
        unavailable_reason: "Checkpoint is not available.",
      },
      "/health/model": {
        ready: false,
        state: "unavailable",
        unavailable_reason: "Checkpoint is not available.",
        device: null,
        model_version: null,
        checkpoint_checksum: null,
      },
    });
    renderApp();

    expect(
      await screen.findByRole("heading", { name: "Service attention required" }),
    ).toBeVisible();
    expect(screen.getByText("Checkpoint is not available.")).toBeVisible();
    expect(screen.getByText("Live")).toBeVisible();
  });

  it("routes to the restoration upload workflow", async () => {
    installApi();
    renderApp("/inspect");

    expect(
      await screen.findByRole("heading", { name: "SEM restoration workspace" }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Browse files" })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /Restore \+ analyze/i })).toBeChecked();
  });

  it("surfaces a safe offline state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("private network details")));
    renderApp();

    expect(
      await screen.findByRole("heading", { name: "Service attention required" }),
    ).toBeVisible();
    expect(screen.getByText("The SemiRestore API could not be reached.")).toBeVisible();
    expect(screen.queryByText("private network details")).not.toBeInTheDocument();
  });

  it("provides a global, non-sensitive render failure boundary", () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    function Failure(): never {
      throw new Error("C:\\private\\checkpoint.pt");
    }

    render(
      <ErrorBoundary>
        <Failure />
      </ErrorBoundary>,
    );

    expect(
      screen.getByRole("heading", {
        name: "The inspection console could not be rendered.",
      }),
    ).toBeVisible();
    expect(screen.queryByText(/checkpoint\.pt/i)).not.toBeInTheDocument();
  });
});
