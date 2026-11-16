# web: the qbt dashboard

A browser front end for the backtest service (`src/qbt_service`). This is
step E1, the tooling only: there are no features yet. The page is a heading
that says so. What is in place is everything features will be built on:

- a typed API client generated from the service's own schema;
- one place where API errors are handled;
- server state, charts, linting, formatting and tests;
- a development server that talks to the local API.

This README assumes you haven't used React before.

## Quick start

```bash
cd web
nvm use              # Node 26.13.1, from .nvmrc (install nvm first if needed)
npm ci               # install exactly what package-lock.json lists
docker compose up -d api   # from the repo root: the API the dev server talks to
npm run dev          # http://localhost:5173
```

| Command                | What it does                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `npm run dev`          | Development server with instant reload; `/api` is forwarded to the API                   |
| `npm run build`        | Type-check, then build the static site into `dist/`                                      |
| `npm run preview`      | Serve `dist/` locally, to check a build                                                  |
| `npm run lint`         | ESLint, then Prettier's formatting check; any warning fails                              |
| `npm run format`       | Rewrite every file in Prettier's format                                                  |
| `npm run typecheck`    | TypeScript only, no output files                                                         |
| `npm test`             | Run the tests once (`npm run test:watch` re-runs on save)                                |
| `npm run api:export`   | Write `openapi.json` from the Python code (no server needed)                             |
| `npm run api:fetch`    | Write `openapi.json` from a running API (`QBT_API_URL`, default `http://127.0.0.1:8000`) |
| `npm run api:generate` | Regenerate `src/api/schema.d.ts` from `openapi.json`                                     |
| `npm run api:check`    | Fail if `openapi.json` or `schema.d.ts` is stale (below)                                 |

## The ideas, briefly

- **React** builds the page out of _components_. A component is a function
  that returns markup written in JSX, which looks like HTML inside
  TypeScript (`src/App.tsx`). When a component's data changes, React
  re-runs it and updates only what changed on the page. `src/main.tsx` is
  where React takes over the `<div id="root">` in `index.html`.
- **TypeScript** is JavaScript with types. A wrong field name or a missing
  `null` check is a compile error rather than a bug in the browser. The
  project runs in strict mode with extra checks on (`tsconfig.app.json`).
  For example, `noUncheckedIndexedAccess` makes `list[0]` possibly
  `undefined`, which it is.
- **Server state** is data that lives in the API: job statuses, results,
  models. It goes stale, has to be refetched, and can fail to load. The
  page doesn't own that data; it keeps a cached copy. TanStack Query
  manages that cache (below), so components don't do it by hand.

## Folder structure

```
web/
├── index.html            the one HTML page; the app is mounted into <div id="root">
├── openapi.json          the API's schema, exported from the Python code (committed)
├── package.json          dependencies (exact versions) and the npm scripts above
├── package-lock.json     the full dependency tree, pinned (committed)
├── .nvmrc  .npmrc        Node version; npm settings (exact versions, strict engines)
├── vite.config.ts        dev server, /api proxy, build, and the test runner's settings
├── tsconfig*.json        TypeScript: app code (strict) and the Vite config file
├── eslint.config.js      lint rules
├── .prettierrc.json      formatting rules
├── scripts/
│   ├── fetch-schema.mjs  npm run api:fetch
│   └── check-api.mjs     npm run api:check
└── src/
    ├── main.tsx          entry point: creates the query client, renders <App>
    ├── App.tsx           the top-level component (a placeholder for now)
    ├── queryClient.ts    TanStack Query defaults: retries, caching, error type
    ├── api/
    │   ├── schema.d.ts   GENERATED from openapi.json; never edit by hand
    │   ├── client.ts     the typed client and unwrap(): data, or a thrown ApiProblem
    │   └── problem.ts    ApiProblem: the one error type the app sees
    ├── test/setup.ts     test setup (DOM matchers)
    └── *.test.ts(x)      tests sit next to the code they test
```

Features will go in `src/features/<name>/`, with a feature's components,
queries and tests together. Shared building blocks will go in
`src/components/`.

## The API client

`openapi.json` is the API's OpenAPI schema. The service generates it from
its own request and response models, so it is the exact contract.
**openapi-typescript** turns it into TypeScript types
(`src/api/schema.d.ts`), and **openapi-fetch** makes calls using those
types:

```ts
const job = await unwrap(api.GET("/backtests/{job_id}", { params: { path: { job_id } } }));
job.status; // "queued" | "running" | "succeeded" | "failed", from the schema
```

A path, parameter or body field the API doesn't have is a compile error.
So is forgetting that a field can be `null`.

**Why openapi-fetch rather than a thin wrapper of our own:**

- **It adds almost nothing at runtime.** It's a few kB around `fetch`.
  Everything it adds is types, and those come from the schema, so there is
  no client code to maintain.
- **It's built for these types.** It comes from the same project as
  openapi-typescript and reads its `paths` type directly. A wrapper of our
  own would mean re-deriving "the response type of this method on this
  path" with conditional types. That's exactly the code openapi-fetch
  already provides, with tests.
- **Generating a full client was rejected.** Generators that write a class
  per endpoint (openapi-generator, orval) produce far more code. That code
  changes in large diffs whenever the API changes, and still wraps `fetch`.
- **Its one quirk is handled in one place.** openapi-fetch returns
  `{ data, error, response }` instead of throwing. `unwrap()` in
  `client.ts` turns that into "data, or throw", which is what TanStack
  Query expects.

**Keeping it current.** When the API changes:

```bash
npm run api:export     # or: npm run api:fetch, against a running API
npm run api:generate
npm run api:check      # passes once both are current
```

`api:check` fails if either generated file is behind:

- **`openapi.json` vs the API:** compared, as parsed JSON, with what
  `scripts/export_openapi.py` produces from the Python code now. This step
  needs the repo's Python environment. Set `PYTHON` to choose the
  interpreter, or pass `-- --types-only` to skip it.
- **`schema.d.ts` vs `openapi.json`:** regenerated and compared byte for
  byte.

The comparison is of parsed JSON because the two ways of getting the schema
format numbers differently: Python's export writes `5.0` where a fetched
copy says `5`. The committed file comes from `api:export`. `api:check` is
not in CI yet; adding a web job is a later step.

## Errors: problem details, handled in one place

The API reports every error as RFC 9457 _problem details_, with content
type `application/problem+json`. Its fields:

- `type` names the kind of problem, e.g. `/problems/job-not-found`. This is
  what code should compare.
- `title` is a short summary, and `detail` explains this occurrence.
- `status` is the HTTP status code.
- Validation problems (422) also list `errors`, one per invalid field.

`src/api/problem.ts` turns **every** failure into one `ApiProblem` error:

| What happened                                                   | `status`        | `type`                                           |
| --------------------------------------------------------------- | --------------- | ------------------------------------------------ |
| The API answered with problem details                           | the HTTP status | the API's, e.g. `/problems/validation-error`     |
| A non-problem error response (a proxy's 502 page, an empty 500) | the HTTP status | `about:blank` (RFC 9457: the status says it all) |
| No response at all (API down, network gone)                     | `0`             | `network-error`                                  |

How the rest of the app gets that:

- **Calls:** `unwrap()` is the only place that builds an `ApiProblem`.
- **Typing:** `queryClient.ts` registers `ApiProblem` as TanStack Query's
  error type, so every `error` a component sees is already an `ApiProblem`.
- **Retries:** failed queries are retried twice, except 4xx problems. The
  same request would fail again, so a 422 shows at once.

## Library choices

Exact versions are in `package.json`, and the whole tree is pinned in
`package-lock.json`. `.npmrc` has `save-exact=true`, so `npm install x`
never writes a `^` range. Install with `npm ci`, which installs exactly the
lock file and fails if `package.json` disagrees with it.

**Every direct dependency is a release at least two weeks old** (on or
before 2026-11-01 when E1 was set up), so a compromised release that gets
pulled within days never lands here. The lock file pins transitive
dependencies, but they weren't individually checked for age.

| Library                                | What it's for            | Why this one                                                                                                                                                                                                                                  |
| -------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **React 19**                           | Components and rendering | The most widely used UI library, so its documentation, examples and tooling are the deepest. The project uses only function components and hooks, React's current style                                                                       |
| **TypeScript 5.9 (strict)**            | Types                    | TypeScript 7 (the native compiler) and 6.0 exist, but typescript-eslint supports `<6.1` and openapi-typescript asks for `^5`. 5.9.3 is the newest version all the tools support                                                               |
| **Vite 8**                             | Dev server and build     | Starts in well under a second, reloads modules on save, and builds a static `dist/`. It also proxies `/api` (below)                                                                                                                           |
| **TanStack Query 5**                   | Server state             | Caching, background refetching, deduplicating identical requests, retries, and loading/error states, all of which hand-written `useEffect` + `fetch` gets wrong. It's built for polling a job until it finishes, which this dashboard will do |
| **Recharts 3**                         | Charts                   | Charts as React components (`<LineChart>`, `<Line>`), configured in JSX rather than imperatively as with D3. Its SVG output is easy to test and style. That's enough for equity curves and metric comparisons; installed, not used yet        |
| **openapi-typescript + openapi-fetch** | The API client           | See [The API client](#the-api-client)                                                                                                                                                                                                         |
| **ESLint 10 + typescript-eslint**      | Lint                     | `strictTypeChecked` uses the type checker to catch bugs: floating promises, unnecessary conditions, unsafe `any`. `react-hooks` enforces the rules of hooks, and `react-refresh` keeps hot reload working                                     |
| **Prettier 3**                         | Formatting               | Formatting is decided by a tool, not in review. `eslint-config-prettier` turns off the ESLint rules that would disagree with it. `npm run lint` checks both                                                                                   |
| **Vitest 5**                           | Test runner              | Uses Vite's own config and transforms, so tests run the code the way the app does, with no separate Babel/Jest setup. The API is Jest's (`test`, `expect`)                                                                                    |
| **Testing Library**                    | Component tests          | Tests find elements the way a user does (by role, label and text), not by component internals, so they survive refactoring. `jsdom` provides the browser DOM in Node                                                                          |

## Talking to the API

**In development,** the browser only ever talks to Vite on `localhost:5173`.
Vite forwards every request under `/api` to the API, stripping the prefix:
`/api/backtests/…` goes to `http://127.0.0.1:8000/backtests/…`. Set
`QBT_API_URL` to point it elsewhere. To the browser it's all one origin,
so the API needs no CORS headers, and none were added. This was checked
against a running API: `/api/health` returned 200, an unknown job id
returned a `application/problem+json` 404, and an invalid body returned a
problem+json 422, all through the proxy.

**In production,** `npm run build` produces static files. The API needs no
change _if_ they are served from the same origin as the API, with a reverse
proxy (nginx, Caddy, a load balancer) sending `/api/*` to the API and
everything else to `dist/`. That's what the dev proxy imitates.

Two cases would need changes, not made here:

- **The site and the API on different origins** (say the site on a CDN and
  the API on its own host). The API would then need CORS:
  FastAPI's `CORSMiddleware`, allowing that one origin, in
  `qbt_service.api.create_app`.
- **The deployed API as it stands** (`infra/`) is plain HTTP on port 8000,
  open to one IP. A page served over HTTPS can't call it: browsers block
  mixed content. It would need HTTPS in front, or the same-origin proxy
  above.
