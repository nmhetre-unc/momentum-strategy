// npm run api:check: fails if the generated API client is stale.
//
// 1. openapi.json must equal the schema the Python API produces now
//    (scripts/export_openapi.py, compared as parsed JSON). Needs the
//    repo's Python environment; set PYTHON to choose the interpreter.
//    --types-only skips this step, for machines without it.
// 2. src/api/schema.d.ts must equal what openapi-typescript generates from
//    openapi.json, byte for byte.
//
// Fix a failure with `npm run api:export && npm run api:generate`.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("..", import.meta.url));
const typesOnly = process.argv.includes("--types-only");
let stale = false;

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((k) => [k, canonical(value[k])]),
    );
  }
  return value;
}

const committed = readFileSync(join(web, "openapi.json"), "utf8");

if (typesOnly) {
  console.log("api:check: skipping the Python schema comparison (--types-only)");
} else {
  const python = process.env.PYTHON ?? "python";
  let current;
  try {
    current = execFileSync(
      python,
      [join(web, "..", "scripts", "export_openapi.py"), "--out", "-"],
      {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "inherit"],
      },
    );
  } catch (error) {
    console.error(
      `api:check: could not run ${python} ../scripts/export_openapi.py (${error.message}).`,
    );
    console.error("Activate the repo's Python environment, set PYTHON, or pass --types-only.");
    process.exit(1);
  }
  const same =
    JSON.stringify(canonical(JSON.parse(committed))) ===
    JSON.stringify(canonical(JSON.parse(current)));
  if (same) {
    console.log("api:check: openapi.json matches the API");
  } else {
    console.error("api:check: STALE openapi.json: the API's schema has changed.");
    stale = true;
  }
}

const tmp = mkdtempSync(join(tmpdir(), "qbt-api-check-"));
try {
  const generated = join(tmp, "schema.d.ts");
  execFileSync(
    process.execPath,
    [
      join(web, "node_modules", "openapi-typescript", "bin", "cli.js"),
      join(web, "openapi.json"),
      "--output",
      generated,
    ],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
  if (
    readFileSync(generated, "utf8") === readFileSync(join(web, "src", "api", "schema.d.ts"), "utf8")
  ) {
    console.log("api:check: src/api/schema.d.ts matches openapi.json");
  } else {
    console.error("api:check: STALE src/api/schema.d.ts: regenerate it from openapi.json.");
    stale = true;
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

if (stale) {
  console.error("Fix: npm run api:export && npm run api:generate");
  process.exit(1);
}
