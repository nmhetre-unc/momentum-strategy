/**
 * Playwright: end-to-end tests of the production build, served by nginx in
 * the Compose stack. Run them with `npm run test:e2e` (e2e/run.mjs), which
 * builds the stack fresh and runs this inside the Playwright image, where
 * nginx is http://web:8080.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  // One stack, shared state (the prediction log): run the tests in order.
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://web:8080",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
