#!/usr/bin/env node
/**
 * The Compose stack the end-to-end tests and the web benchmark run against,
 * as its own Compose project so it never touches the development stack or
 * its data:
 *
 *   node e2e/stack.mjs up     build every image fresh, start, ingest, train
 *   node e2e/stack.mjs down   remove it, volumes included
 *
 * `up` always passes --build: an image left over from an earlier build
 * silently lacks newer routes (a stale api image answered /series with a
 * generic 404). With nothing changed, Docker's layer cache makes it quick.
 *
 * Environment: QBT_E2E_PROJECT (default qbt-e2e). The stack publishes no
 * ports a developer would use: the tests reach nginx over the project's
 * network (http://web:8080), from a container (e2e/run.mjs).
 */
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
export const PROJECT = process.env.QBT_E2E_PROJECT ?? "qbt-e2e";
export const NETWORK = `${PROJECT}_default`;
/** Chromium and its system libraries; the same version as @playwright/test. */
export const PLAYWRIGHT_IMAGE =
  "mcr.microsoft.com/playwright:v1.64.1-noble@sha256:2f05f8a734509a7b3f26a912639488ee08f58a7b292d14d62271bcff9038768f";

export const BASE_URL = "http://web:8080";

// Prices for the backtests the tests submit (2015-2025) and for training,
// which needs five years of history before its first out-of-sample month.
const TICKER = "SPY";
const START = "2008-01-01";
const END = "2025-01-01";

// Off the development stack's ports (5432/8000/8080) and the smoke test's.
const ENV = {
  ...process.env,
  QBT_DB_PORT: process.env.QBT_DB_PORT ?? "56432",
  QBT_API_PORT: process.env.QBT_API_PORT ?? "58100",
  QBT_WEB_PORT: process.env.QBT_WEB_PORT ?? "58180",
};

function stamp() {
  return new Date().toISOString().slice(11, 19);
}

export function compose(...args) {
  const command = ["compose", "-p", PROJECT, "-f", resolve(ROOT, "compose.yaml"), ...args];
  console.error(`[${stamp()}] docker ${command.join(" ")}`);
  const result = spawnSync("docker", command, { env: ENV, stdio: "inherit", cwd: ROOT });
  if (result.status !== 0) {
    throw new Error(`docker ${command.join(" ")} exited ${String(result.status)}`);
  }
}

export function up() {
  compose("up", "-d", "--build", "--wait", "--wait-timeout", "300");
  compose("run", "--rm", "ingest", TICKER, "--start", START, "--end", END);
  compose("run", "--rm", "train", TICKER, "--start", START, "--end", END);
}

export function down() {
  compose("down", "-v", "--remove-orphans");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const action = process.argv[2];
  if (action === "up") up();
  else if (action === "down") down();
  else {
    console.error("usage: node e2e/stack.mjs up|down");
    process.exit(2);
  }
}
