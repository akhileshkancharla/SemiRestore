import { describe, expect, it, vi } from "vitest";

import { API_BASE_URL, ApiRequestError, RequestCancelledError, apiClient } from "./client";

function jsonResponse(body: unknown, status = 200, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("typed API client", () => {
  it("uses a same-origin service prefix by default", () => {
    expect(API_BASE_URL).toBe("/service");
  });

  it("treats an explicit readiness 503 as typed operational state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { ready: false, state: "unavailable", unavailable_reason: "Model unavailable" },
          503,
        ),
      ),
    );

    await expect(apiClient.getReady()).resolves.toEqual({
      ready: false,
      state: "unavailable",
      unavailable_reason: "Model unavailable",
    });
  });

  it("maps safe API envelopes without exposing arbitrary response bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "model_unavailable",
              message: "The model service is unavailable.",
              details: null,
              request_id: "req-safe",
            },
          },
          503,
        ),
      ),
    );

    const failure = await apiClient.restore(new File(["image"], "sample.png")).catch(
      (cause: unknown) => cause,
    );

    expect(failure).toBeInstanceOf(ApiRequestError);
    expect(failure).toMatchObject({
      status: 503,
      code: "model_unavailable",
      requestId: "req-safe",
      message: "The model service is unavailable.",
    });
  });

  it("suppresses non-JSON error content", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("C:\\private\\checkpoint.pt stack trace", {
          status: 500,
          headers: { "x-request-id": "req-header" },
        }),
      ),
    );

    await expect(apiClient.getVersion()).rejects.toMatchObject({
      code: "unexpected_response",
      message: "The SemiRestore API returned an unexpected response.",
      requestId: "req-header",
    });
  });

  it("uses a safe offline error when the network fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("sensitive network detail")));

    await expect(apiClient.getLive()).rejects.toMatchObject({
      status: 0,
      code: "offline",
      message: "The SemiRestore API could not be reached.",
    });
  });

  it("preserves cancellation as a distinct client state", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        controller.abort();
        return Promise.reject(
          init?.signal?.aborted
            ? new DOMException("operation aborted", "AbortError")
            : new Error("unexpected"),
        );
      }),
    );

    await expect(
      apiClient.analyze(new File(["image"], "sample.png"), controller.signal),
    ).rejects.toBeInstanceOf(RequestCancelledError);
  });
});
