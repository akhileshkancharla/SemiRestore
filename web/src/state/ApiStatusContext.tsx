import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { ApiRequestError, apiClient } from "../api/client";
import type { ModelHealthResponse, ReadyResponse, VersionResponse } from "../api/types";

export type ConnectionState = "checking" | "ready" | "unready" | "offline";

interface ApiStatusValue {
  connection: ConnectionState;
  ready: ReadyResponse | null;
  model: ModelHealthResponse | null;
  version: VersionResponse | null;
  error: string | null;
  refresh: () => Promise<void>;
}

const ApiStatusContext = createContext<ApiStatusValue | null>(null);

export function ApiStatusProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [ready, setReady] = useState<ReadyResponse | null>(null);
  const [model, setModel] = useState<ModelHealthResponse | null>(null);
  const [version, setVersion] = useState<VersionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [, readiness, modelHealth, applicationVersion] = await Promise.all([
        apiClient.getLive(),
        apiClient.getReady(),
        apiClient.getModelHealth(),
        apiClient.getVersion(),
      ]);
      if (!mounted.current) return;
      setReady(readiness);
      setModel(modelHealth);
      setVersion(applicationVersion);
      setConnection(readiness.ready ? "ready" : "unready");
    } catch (cause) {
      if (!mounted.current) return;
      setConnection("offline");
      setReady(null);
      setModel(null);
      setError(
        cause instanceof ApiRequestError
          ? cause.message
          : "The service state could not be determined.",
      );
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    const interval = window.setInterval(() => void refresh(), 15_000);
    return () => {
      mounted.current = false;
      window.clearInterval(interval);
    };
  }, [refresh]);

  const value = useMemo(
    () => ({ connection, ready, model, version, error, refresh }),
    [connection, ready, model, version, error, refresh],
  );
  return <ApiStatusContext.Provider value={value}>{children}</ApiStatusContext.Provider>;
}

export function useApiStatus(): ApiStatusValue {
  const value = useContext(ApiStatusContext);
  if (value === null) {
    throw new Error("useApiStatus must be used inside ApiStatusProvider");
  }
  return value;
}
