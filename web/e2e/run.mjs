#!/usr/bin/env node
/**
 * npm run test:e2e: the Playwright tests against the Compose stack.
 *
 * 1. e2e/stack.mjs up: every image built fresh (--build), started, SPY
 *    ingested, a model trained.
 * 2. The tests, in the Playwright image (Chromium with its system
 *    libraries, the same version as @playwright/test), on the stack's
 *    network, so nginx is http://web:8080 and nothing depends on host ports.
 *    Runs as the current user, so test-results/ and playwright-report/
 *    belong to you.
 * 3. e2e/stack.mjs down, unless --keep. Extra arguments go to
 *    `playwright test`, e.g. `npm run test:e2e -- --grep Model`.
 */
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { BASE_URL, NETWORK, PLAYWRIGHT_IMAGE, down, up } from "./stack.mjs";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const keep = args.includes("--keep");
const extra = args.filter((a) => a !== "--keep");

let status;
try {
  up();
  const user = `${String(process.getuid?.() ?? 0)}:${String(process.getgid?.() ?? 0)}`;
  const result = spawnSync(
    "docker",
    [
      "run",
      "--rm",
      "--network",
      NETWORK,
      "--user",
      user,
      "--ipc=host",
      "-e",
      "HOME=/tmp",
      "-e",
      `E2E_BASE_URL=${BASE_URL}`,
      "-e",
      `CI=${process.env.CI ?? ""}`,
      "-v",
      `${WEB}:/work`,
      "-w",
      "/work",
      PLAYWRIGHT_IMAGE,
      "npx",
      "playwright",
      "test",
      ...extra,
    ],
    { stdio: "inherit" },
  );
  status = result.status ?? 1;
} finally {
  if (keep) console.error("--keep: the stack is still running (node e2e/stack.mjs down)");
  else down();
}
process.exit(status);
