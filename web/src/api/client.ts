import type {
  AnalyzeResponse,
  ErrorResponse,
  LiveResponse,
  ModelHealthResponse,
  ReadyResponse,
  RestoreResponse,
  VersionResponse,
} from "./types";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "/service";
export const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");

export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(message: string, status: number, code: string, requestId: string | null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export class RequestCancelledError extends Error {
  constructor() {
    super("The request was cancelled.");
    this.name = "RequestCancelledError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  acceptedStatuses: readonly number[] = [200],
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch (cause) {
    if (init?.signal?.aborted || (cause instanceof DOMException && cause.name === "AbortError")) {
      throw new RequestCancelledError();
    }
    throw new ApiRequestError("The SemiRestore API could not be reached.", 0, "offline", null);
  }

  if (acceptedStatuses.includes(response.status)) {
    try {
      return (await response.json()) as T;
    } catch {
      throw new ApiRequestError(
        "The SemiRestore API returned an unexpected response.",
        response.status,
        "unexpected_response",
        response.headers.get("x-request-id"),
      );
    }
  }

  let envelope: ErrorResponse | null = null;
  try {
    envelope = (await response.json()) as ErrorResponse;
  } catch {
    // The client never exposes an arbitrary non-JSON response body.
  }
  const error = envelope?.error;
  throw new ApiRequestError(
    error?.message || "The SemiRestore API returned an unexpected response.",
    response.status,
    error?.code || "unexpected_response",
    error?.request_id || response.headers.get("x-request-id"),
  );
}

function imageForm(image: File): FormData {
  const form = new FormData();
  form.append("image", image);
  return form;
}

export const apiClient = {
  getLive: () => request<LiveResponse>("/health/live"),
  getReady: () => request<ReadyResponse>("/health/ready", undefined, [200, 503]),
  getModelHealth: () => request<ModelHealthResponse>("/health/model"),
  getVersion: () => request<VersionResponse>("/version"),
  analyze: (image: File, signal?: AbortSignal) =>
    request<AnalyzeResponse>("/api/v1/analyze", {
      method: "POST",
      body: imageForm(image),
      signal,
    }),
  restore: (image: File, signal?: AbortSignal) =>
    request<RestoreResponse>("/api/v1/restore", {
      method: "POST",
      body: imageForm(image),
      signal,
    }),
  restoreAndAnalyze: (image: File, signal?: AbortSignal) =>
    request<RestoreResponse>("/api/v1/restore-and-analyze", {
      method: "POST",
      body: imageForm(image),
      signal,
    }),
};
