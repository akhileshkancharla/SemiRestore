import type { AnalyzeResponse, RestoreResponse } from "../api/types";

export type OperationMode = "analyze" | "restore" | "restore-and-analyze";

export interface SelectedImage {
  file: File;
  previewUrl: string;
  previewSupported: boolean;
  width: number;
  height: number;
}

export type WorkspaceResult =
  | { kind: "analysis"; operation: "analyze"; data: AnalyzeResponse }
  | {
      kind: "restoration";
      operation: "restore" | "restore-and-analyze";
      data: RestoreResponse;
    };
