// npm run api:fetch: download openapi.json from a running API.
//
//   npm run api:fetch                                   # http://127.0.0.1:8000
//   QBT_API_URL=http://127.0.0.1:58000 npm run api:fetch
//
// Then `npm run api:generate`. Without a running API, `npm run api:export`
// writes the same schema straight from the Python code.
import { writeFile } from "node:fs/promises";

const base = process.env.QBT_API_URL ?? "http://127.0.0.1:8000";
const url = `${base.replace(/\/$/, "")}/openapi.json`;

let response;
try {
  response = await fetch(url);
} catch (error) {
  console.error(`api:fetch: no API at ${url} (${error.cause?.code ?? error.message}).`);
  console.error("Start it (docker compose up -d api), or use `npm run api:export`.");
  process.exit(1);
}
if (!response.ok) {
  console.error(`api:fetch: ${url} answered ${response.status}`);
  process.exit(1);
}
const schema = await response.json();
await writeFile(
  new URL("../openapi.json", import.meta.url),
  JSON.stringify(schema, null, 2) + "\n",
);
console.log(`api:fetch: wrote openapi.json from ${url}`);
