export type ModelServiceState = "starting" | "ready" | "unavailable" | "stopped";

export interface LiveResponse {
  status: "alive";
}

export interface ReadyResponse {
  ready: boolean;
  state: ModelServiceState;
  unavailable_reason: string | null;
}

export interface ModelHealthResponse extends ReadyResponse {
  device: string | null;
  model_version: string | null;
  checkpoint_checksum: string | null;
}

export interface VersionResponse {
  application: "semirestore";
  version: string;
}

export interface ErrorBody {
  code: string;
  message: string;
  details: Record<string, unknown> | null;
  request_id: string | null;
}

export interface ErrorResponse {
  error: ErrorBody;
}

export interface RestoreResponse {
  image: {
    encoding: "base64";
    media_type: "image/png";
    content: string;
    width: number;
    height: number;
  };
  input: {
    width: number;
    height: number;
    media_type: "image/png" | "image/jpeg" | "image/tiff";
  };
  inference: {
    latency_ms: number | null;
    device: string | null;
    phase_latency_ms: Record<string, number>;
  };
  model: {
    name: string | null;
    version: string | null;
    training_revision: string | null;
    checkpoint_checksum: string | null;
  };
  diagnostics: Record<string, unknown>;
  warnings: string[];
}

export interface AnalyzeResponse {
  input: {
    width: number;
    height: number;
    media_type: "image/png" | "image/jpeg" | "image/tiff";
  };
  analysis: { latency_ms: number };
  diagnostics: Record<string, unknown>;
  suitability: {
    recommendation: "restore" | "warn" | "bypass";
    reasons: string[];
    advisory_not_probability: true;
  };
  warnings: string[];
}
