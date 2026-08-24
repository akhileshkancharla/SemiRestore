import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, process.cwd(), "");
  const developmentApi = environment.SEMIRESTORE_DEV_API_URL ?? "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    build: {
      sourcemap: false,
      target: "es2022",
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/service": {
          target: developmentApi,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/service/, ""),
        },
      },
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: "./src/test/setup.ts",
      css: true,
    },
  };
});
