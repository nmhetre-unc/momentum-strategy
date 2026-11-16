#!/usr/bin/env node
/**
 * npm run docs:screenshots: the README's screenshots (docs/dashboard-*.png),
 * from a stack built fresh. Like npm run test:e2e: e2e/stack.mjs up, then
 * scripts/screenshots.mjs in the Playwright image on the stack's network,
 * then down (unless --keep).
 */
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { BASE_URL, NETWORK, PLAYWRIGHT_IMAGE, down, up } from "./stack.mjs";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = resolve(WEB, "..", "docs");
const keep = process.argv.includes("--keep");

let status;
try {
  up();
  const user = `${String(process.getuid?.() ?? 0)}:${String(process.getgid?.() ?? 0)}`;
  const result = spawnSync(
    "docker",
    [
      ...["run", "--rm", "--network", NETWORK, "--user", user, "--ipc=host", "-e", "HOME=/tmp"],
      ...["-v", `${WEB}:/work`, "-v", `${DOCS}:/docs`, "-w", "/work", PLAYWRIGHT_IMAGE],
      ...["node", "scripts/screenshots.mjs", "--base", BASE_URL, "--out", "/docs"],
    ],
    { stdio: "inherit" },
  );
  status = result.status ?? 1;
} finally {
  if (!keep) down();
}
process.exit(status);
