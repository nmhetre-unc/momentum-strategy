/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API the dev server forwards /api to: the local Compose stack's api
// service (compose.yaml publishes it on 127.0.0.1:8000, or QBT_API_PORT).
const apiTarget = process.env.QBT_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    // The browser only ever talks to the Vite server, so the API needs no
    // CORS headers in development: /api/strategies -> {apiTarget}/strategies.
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
