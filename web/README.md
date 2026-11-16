# web: the qbt dashboard

A browser front end for the backtest service (`src/qbt_service`). It has
four pages:

| Page    | URL               | What it does                                                                                                                                                                                                       |
| ------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Run     | `/`               | Pick a strategy, fill in a parameter form generated from its JSON Schema, submit, and follow the job until it succeeds or fails                                                                                    |
| Results | `/backtests/{id}` | Equity curve, drawdown, the Sharpe ratio with its bootstrap interval, the other metrics, the strategy version                                                                                                      |
| Compare | `/compare?ids=…`  | Two or more succeeded backtests: one metrics table and their equity curves on one chart                                                                                                                            |
| Model   | `/model`          | The latest model's registry entry and whether it beats its baseline (`beats`, and the baseline accuracy it is up against). A prediction is made only when you press Predict, and shown no larger than that verdict |

Underneath:

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
docker compose up -d --build api   # from the repo root: the API the dev server talks to
npm run dev          # http://localhost:5173
```

**Always pass `--build`.** Without it, Compose reuses whatever `api` image
it built last. That image silently lacks every route added since, and the
page gets a generic 404 instead. This happened with `/backtests/{id}/series`.
`--build` takes seconds when nothing changed, because Docker's layer cache
is reused. The end-to-end tests and CI use `docker compose up --build` for
the same reason.

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
    ├── App.tsx           the header, navigation, and the page for the current URL
    ├── router.ts         the client-side router (below)
    ├── queryClient.ts    TanStack Query defaults: retries, caching, error type
    ├── theme.ts          chart colors, light and dark
    ├── format.ts         number formatting
    ├── index.css         the styles
    ├── api/
    │   ├── schema.d.ts   GENERATED from openapi.json; never edit by hand
    │   ├── client.ts     the typed client and unwrap(): data, or a thrown ApiProblem
    │   ├── problem.ts    ApiProblem: the one error type the app sees
    │   ├── context.ts    which client components use (tests swap in a fake)
    │   ├── queries.ts    every API call as a hook: polling, submit with retries
    │   └── types.ts      short names for the generated types
    ├── components/       shared pieces: Link, charts, loading/error/empty states
    ├── features/
    │   ├── run/          RunView.tsx, and form.ts: the form generated from JSON Schema
    │   ├── results/      ResultsView.tsx, and the metrics table shared with Compare
    │   ├── compare/      CompareView.tsx, and recentRuns.ts (this browser's runs)
    │   └── model/        ModelView.tsx
    ├── test/             setup, a fake API at the fetch layer, typed fixtures
    └── *.test.ts(x)      tests sit next to the code they test
```

Each page is a _feature_ folder with its components, logic and tests
together. What more than one page uses lives in `src/components/` and
`src/api/`.

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

## How the React code works

This walks through the Run page, from the URL to a finished job. Every
other page is built the same way. The code is in
`src/features/run/RunView.tsx`; each hook there has a comment saying why
it's used.

### 1. The URL picks the page

`main.tsx` renders `<App>` inside a `QueryClientProvider`, which gives every
component below it access to one shared cache of server data.

`App` calls `useLocation()`, which returns the address. `matchRoute("/")`
returns `{ name: "run" }`, so `App` renders `<RunView />`. When the address
changes, `useLocation()` makes `App` render again and pick the new page.

> **Rendering** means React calls the component function, and the function
> returns JSX describing what the page should show. React compares that with
> the page as it is and changes only what differs. A component renders again
> whenever its state, its props (the arguments it's given) or a hook's value
> changes. So a component is a description of the page _for the current
> data_, not a list of steps that change the page.

### 2. Loading the strategies: loading, error, empty

```tsx
const strategies = useStrategies();
```

`useStrategies()` (in `src/api/queries.ts`) is a `useQuery` call. It asks
`GET /strategies` through the typed client and `unwrap()`. It returns an
object that is always in exactly one state:

- **`isPending`** on the first render, because nothing has arrived yet.
  `RunView` shows `<Loading>`.
- **`isError`** if the call failed. `strategies.error` is an `ApiProblem`,
  whatever went wrong, and `<ProblemView>` shows its title, detail and
  type, with a "Try again" button that calls `strategies.refetch()`.
- **Success, but an empty list.** `<Empty>` says there's nothing to run.
- **Success with strategies.** `RunView` renders `<RunForm>`.

The data arriving is itself a change, so React renders `RunView` again and
the `if` chain picks the next branch. Nothing in `RunView` waits for or
fetches anything by hand.

**Why `RunForm` is a separate component.** Hooks (`useState`, `useQuery`,
...) must be called in the same order on every render; React tells them
apart by their order. The form's state needs a strategy to start from, which
doesn't exist during loading. So the form's hooks live in a component that
only renders once the strategies exist.

### 3. The form's state, and what is derived from it

```tsx
const [selected, setSelected] = useState(strategies[0]?.name ?? "");
const [common, setCommon] = useState(DEFAULT_VALUES);
const [paramsByStrategy, setParamsByStrategy] = useState({});
const [showErrors, setShowErrors] = useState(false);
const [submission, setSubmission] = useState<Submission | null>(null);
```

`useState` keeps a value between renders and returns a setter. Calling the
setter stores the new value and renders the component again. These five
are everything the user has done: picked a strategy, typed into fields,
pressed submit.

Everything else is **computed from them on each render**, never stored:

```tsx
const strategy = strategies.find((s) => s.name === selected) ?? strategies[0];
const params = paramsByStrategy[strategy.name] ?? defaultParams(strategy);
const checked = validate(strategy, { ...common, params });
```

So the version shown, the fields, and the validation errors can't fall
out of step with what the user typed. There is no second copy to update.

### 4. A form generated from JSON Schema

`paramFields(strategy.params_schema)` (in `form.ts`) reads the schema the
API sent. Each `integer` or `number` property becomes a field with its
title, description, minimum, maximum and default. The form renders one
`<NumberField>` per field. A new parameter in the API appears in the form
with no change here.

Each input is **controlled**: its `value` comes from state, and its
`onChange` calls a setter.

```tsx
<input value={value} onChange={(e) => onChange(e.target.value)} />
```

Typing calls the setter, which re-renders with the new value. React state,
not the input element, holds the truth. `useId()` gives each input a unique
id, so its `<label>`, hint and error can point at it.

### 5. Validation that matches the API

`validate()` checks what `POST /backtests` checks:

- each parameter's type and bounds, from the schema;
- the rules the schema can't express, mirrored from
  `src/qbt_service/models.py`: short window below long window, the ticker
  pattern, real calendar dates with start before end, and the `cost_bps`,
  `n_boot` and `seed` bounds.

**The shared cases.** Copying rules by hand can drift, so both sides are
tested against one file: `contracts/backtest_requests.json` at the repo
root. It lists request bodies, valid and invalid, and whether the API
accepts each one. It covers every strategy's bounds, the cross-field
rules, the ticker pattern, dates and the request bounds.

- `tests/service/test_request_contract.py` sends every case to the real
  `POST /backtests` and checks the API agrees.
- `src/features/run/form.contract.test.ts` checks `validate()` agrees.

To change a rule, add or change a case there first. Both test suites then
show which side still disagrees. The one known difference is left out of
the file: seeds above 2^53 − 1, which the API accepts but the form refuses
(a JavaScript number can't hold them exactly).

It returns either the request body or the errors. Errors are shown only
after the first submit (`showErrors`), then update as the user types.

If the API rejects a request anyway (a rule changed, or no price data),
its 422 lists where each error is. `fieldErrorsFromProblem()` puts those
messages on the same fields.

### 6. Submitting, with an Idempotency-Key

```tsx
const onSubmit = (event) => {
  event.preventDefault(); // keep the browser from reloading the page
  setShowErrors(true);
  if (checked.errors) return;
  const next = { key: newIdempotencyKey(), body: checked.body };
  setSubmission(next);
  send(next);
};
```

`send` calls `submit.mutate(...)`. `submit` is a `useMutation` (in
`queries.ts`): a write that runs only when asked. It tracks its own state
(`isPending`, `isError`, `error`, `data`).

The key is what makes a retry safe. A request can fail after the API has
already created the job, for example when the response is lost. Sending
it again with the **same key** returns that job instead of creating a
second one. So:

- **Automatic retries** of a transient failure (no response, or a 5xx)
  reuse the key, because the mutation retries with the same variables.
- **"Try again"** after the automatic retries are used up resends the
  stored `submission`: the same body and the same key.
- **Pressing "Run backtest" again** is a new submission with a new key,
  even if nothing changed. Editing a field after a failure clears the
  error (`submit.reset()`), so the next attempt is a new submission.

A 4xx is not retried: the same request would fail the same way.

### 7. Polling the job until it is final

```tsx
const job = useBacktestJob(submit.data?.id, pollIntervalMs);
```

Until the POST succeeds there is no id, and the query is disabled: it
doesn't run. Once `submit.data` holds the accepted job, the same hook
starts fetching `GET /backtests/{id}`. `refetchInterval` asks `nextPoll()`
after every response whether to fetch again:

- every second while the job is `queued` or `running`;
- **never again** once it is `succeeded` or `failed`, or after an error.

TanStack Query owns the timer. When the component unmounts (the user goes
to another page), the timer goes with it, so there is no `setInterval` to
remember to clear.

`<JobProgress>` renders whatever the latest response says. When the job
succeeds it shows a `<Link>` to `/backtests/{id}`. Clicking it calls
`navigate`, `App` re-renders, `matchRoute` returns the results route, and
`<ResultsView>` takes over. Its own `useBacktestJob` call finds the job
already in the cache, so it appears at once.

### 8. The tests

`RunView.test.tsx` renders the real component with the real client,
`unwrap()` and TanStack Query. Only `fetch` is fake (`src/test/api.tsx`),
answering from a list of handlers and recording every request. The tests
check what a user sees and what was sent:

- the generated fields and their defaults;
- the client-side errors, with nothing posted;
- the request body;
- that the same key was used across retries, and a new key for a new
  submission;
- that the number of polls stops growing once the job is final or the
  page is closed.

They find elements the way a user would, by label, role and text.

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
| **Recharts 3**                         | Charts                   | Charts as React components (`<LineChart>`, `<Line>`), configured in JSX rather than imperatively as with D3. Its SVG output is easy to test and style. That's enough for equity curves and metric comparisons                                 |
| **openapi-typescript + openapi-fetch** | The API client           | See [The API client](#the-api-client)                                                                                                                                                                                                         |
| **ESLint 10 + typescript-eslint**      | Lint                     | `strictTypeChecked` uses the type checker to catch bugs: floating promises, unnecessary conditions, unsafe `any`. `react-hooks` enforces the rules of hooks, and `react-refresh` keeps hot reload working                                     |
| **Prettier 3**                         | Formatting               | Formatting is decided by a tool, not in review. `eslint-config-prettier` turns off the ESLint rules that would disagree with it. `npm run lint` checks both                                                                                   |
| **Vitest 5**                           | Test runner              | Uses Vite's own config and transforms, so tests run the code the way the app does, with no separate Babel/Jest setup. The API is Jest's (`test`, `expect`)                                                                                    |
| **Testing Library**                    | Component tests          | Tests find elements the way a user does (by role, label and text), not by component internals, so they survive refactoring. `jsdom` provides the browser DOM in Node                                                                          |

## Routing: why no router library

The four pages need only this: turn the address into a page, change the
address without reloading, and follow the back and forward buttons. That
takes about 60 lines on the browser's History API (`src/router.ts`):

- `navigate(to)` calls `history.pushState`, which changes the address bar
  without loading anything, then announces the change.
- `useLocation()` gives a component the current address and re-renders it
  when the address changes, by `navigate` or by back/forward (`popstate`).
- `matchRoute()` turns the address into one of five routes, typed, so
  `route.id` exists only on the results route.
- `<Link to>` renders a real `<a href>` (open in new tab and copy link
  work) but handles a plain click with `navigate`.

A library (React Router, TanStack Router) earns its size with nested
layouts, per-route data loading, lazy routes and search-param validation.
None is needed here. If they become needed, the routes are all in
`matchRoute` and `App.tsx`, so the switch is small.

Hash URLs (`/#/model`) would need no server support at all, but they make
every URL odd. Real paths need the server to answer every page's URL with
`index.html` (an "SPA fallback"). Vite's dev server does this already; the
production server will need the same rule.

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
