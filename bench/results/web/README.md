# Dashboard measurements

The production build of `web/`, served by nginx in the Compose stack, measured by
[`bench/web_bench.py`](../../web_bench.py) and
[`web/scripts/measure-web.mjs`](../../../web/scripts/measure-web.mjs). Every number below
comes from the JSON files in this folder.

| File | What it is |
|---|---|
| `bundle_before_split.json` | The bundle before code splitting (`--only bundle`): the app source as committed at `8f2067f`, built fresh, with the new nginx files from this step |
| `web.json` | Everything, after code splitting: bundle, Lighthouse, time to first chart |

## Results

**Bundle, as nginx serves it** (gzip level 6, bytes on the wire):

| | Before (`bundle_before_split.json`) | After (`web.json`) |
|---|---|---|
| Initial load: what `index.html` loads | 203.4 kB (one 201.9 kB script + 1.5 kB CSS) | **90.4 kB** (88.9 kB script + 1.5 kB CSS) |
| Loaded when a chart page opens | none | 115.7 kB: Recharts and the shared metrics code, 112.5 kB, plus `ResultsView` 1.3 kB and `CompareView` 1.9 kB |

- **The Run and Model pages** now load 113 kB less: 90.4 kB instead of 203.4 kB.
- **A results page opened directly** loads 90.4 + 112.5 + 1.3 = 204.1 kB. That is 0.7 kB more than before, the cost of the extra chunks. The point of the split is that pages without charts never pay for Recharts.

**Lighthouse performance** (3 runs per page, each in a fresh Chrome; Lighthouse 13.6.0 defaults: mobile emulation, simulated slow 4G, 4x CPU slowdown):

| Page | Median | Range | Notes |
|---|---|---|---|
| Run (`/`) | **100** | 100–100 | FCP 0.75 s, LCP 1.65 s, TBT 0–5 ms, CLS 0 |
| Results (`/backtests/{id}`) | **70** | 70–72 | FCP 0.75 s, LCP 2.40–2.47 s, TBT 1.39–1.74 s, CLS 0 |

The results page loses its points to **Total Blocking Time**. That's
main-thread work rendering two SVG charts of about 2,500 daily points each
(equity and drawdown, 2015–2025), under the 4x CPU slowdown. Downsampling
the plotted series, or drawing to a canvas, are the levers if this matters.
Neither was tried here.

**Time to first chart**, on the results page: 10 cold runs (fresh browser
context each, after one unrecorded warm-up), no throttling. Each is the
time from the start of navigation until the equity curve's SVG path exists
with data in it, recorded in the page by a MutationObserver.

- Median **559 ms**, range 520–630 ms.
- That covers loading `index.html`, the entry script and the lazy chart chunk, `GET /backtests/{id}`, `GET /backtests/{id}/series` (2,516 days) and rendering.

## Machine and method

- **Host:** Intel Core 7 150U (10 cores, 12 threads), 15.7 GiB RAM, Windows power plan Balanced, **on AC power**.
  - The power guard (`guarded` in `bench/service_bench.py`) checked the Windows event log: no sleep, wake or power-source change during either measurement, and no attempt was discarded.
  - Windows was kept awake throughout.
- **Docker:** Docker Desktop 29.9.3 (WSL2 backend; VM: 12 CPUs, 7.6 GiB), Compose 5.7.0.
  - The browser ran in the Playwright image (`mcr.microsoft.com/playwright:v1.64.1-noble`, Chromium 156.0.8078.80) on the stack's own network, so it shares that VM with the stack.
  - The development stack (`momentum-strategy-*`: db, api, worker) was running and idle. The 1-minute load average when `web.json` was recorded was 1.57.
- **Data:** the stack is built fresh (`docker compose up --build`), SPY 2008–2025 is ingested and a model trained (`web/e2e/stack.mjs`). Then one `sma_crossover` backtest with the Run page's defaults is submitted; its results page is the one measured.
- **Rerun:**
  ```bash
  .venv/bin/python bench/web_bench.py                                          # about 2 minutes
  .venv/bin/python bench/web_bench.py --only bundle --name bundle_before_split  # bundle only
  ```
