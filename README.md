# momentum-strategy

[![CI](https://github.com/nmhetre-unc/momentum-strategy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nmhetre-unc/momentum-strategy/actions/workflows/ci.yml)

**[Live demo: Streamlit dashboard](https://momentum-strategy-hbht78g3wpzyfyhjggwhjt.streamlit.app/)**

A backtesting engine for single-asset and cross-sectional equity strategies,
built to test one question: does any of them beat buy-and-hold? None of them
do by a detectable margin. The findings below are the point of the project.

All results use SPY or the S&P 100 over 2008-2025, 5bps one-way costs,
90% confidence intervals from a stationary block bootstrap, and deflated
Sharpe ratios following Bailey & López de Prado (2014).

## At a glance

- 11 timing strategies tested on SPY (4,278 trading days, 2008-2025): none beats
  buy-and-hold at 90% confidence. The 7 that learn from history are scored only
  out of sample, against buy-and-hold over the same window.
- `adaptive_ensemble` looked best when it was scored on the days it had learned
  from: a deflated Sharpe against the benchmark of 0.906 at one trial, 0.549 at
  five, 0.380 at eleven. Out of sample it is 0.295, 0.042 and 0.015.
- 12-1 momentum on the S&P 100: long-leg Sharpe 1.071 [0.727, 1.441], short leg
  -0.882; 88.7% of the loss traces to a -0.214 net beta mismatch, not the signal.
- Monthly ML (logistic regression, LightGBM, HMM features) in nested purged
  walk-forward: no model beats its bar. The best, a regime model, is right
  80.8% of the time against a lookup table's 82.7%. It is served through the
  API anyway, with those numbers in every response. See
  [Machine learning](#machine-learning).
- The stationary bootstrap, 62% of the full results run, now runs in C++ and
  gives bit-identical output. The bootstrap call is 19.4x faster and the whole
  pipeline 2.4x faster, against a ceiling of about 2.7x.
  See [Performance](#performance).
- The engine also runs as a service: an HTTP API, a Postgres job queue with
  leases and fencing, and Docker Compose. With 4 workers it runs 926
  backtests/min, a job starts 8 ms after it is submitted at 50% load, and a
  SIGKILLed worker loses no jobs. See [Running as a service](#running-as-a-service).
- 653 tests at 94% line coverage (91% for the engine, 98% for the service).
  CI runs on every push: it builds the C++ extension, runs the suite against
  Postgres with coverage gates, and runs an end-to-end test of the Compose
  stack.

## Running as a service

The engine also runs as an HTTP service. You POST a backtest and get a job
id at once. A worker runs the job, and you GET the result. The queue is
Postgres itself, with no broker. Any number of workers can share it, and a
worker can be killed at any point without losing a job or recording one
twice.

### Architecture

```
            curl / client
                 │  POST /backtests  ·  GET /backtests/{id}[/series]  ·  GET /strategies
                 │  GET /models  ·  GET /models/{version}  ·  POST /predict
                 ▼
 ┌─ api ────────────────────────────┐
 │ FastAPI · 2 uvicorn processes    │   validate, pin a price snapshot,
 │ cpu_shares 4096                  │   INSERT the job + NOTIFY (one transaction)
 └────────────────┬─────────────────┘
                  ▼
 ┌─ db: Postgres 17 ───────────────────────────────────────────────┐
 │ jobs (the queue) · results · metrics · strategies               │
 │ price_series / price_bars (immutable, content-hashed snapshots) │
 │ model_registry (immutable versions) · predictions (the log)     │
 └───────▲─────────────────────────────────────────────▲───────────┘
         │ claim, heartbeat, complete, reap             │ LISTEN qbt_jobs
 ┌─ worker × N ─────────────────────────────────────────────────────┐
 │ supervisor (DB, leases, reaper) ──pipe──► compute child          │
 │                                           (qbt.runner, 1 thread) │
 └──────────────────────────────────────────────────────────────────┘

 one-shots:  migrate (alembic upgrade head, before api and workers start)
             ingest  (download from Yahoo Finance into price_series)
             train   (fit a model on a pinned snapshot, register a version)
```

Each worker is two processes. The **supervisor** owns the database
connection: it claims jobs, keeps their leases alive, and writes the
outcomes. A long-lived **compute child** runs the backtests. Heartbeats
keep flowing however long a backtest holds the GIL, and a job that must
stop is ended by killing the child, which can't write anything. Code:
[`src/qbt_service/`](src/qbt_service/); the stack is in
[`compose.yaml`](compose.yaml).

### The queue

- **Claiming:** `SELECT … FOR UPDATE SKIP LOCKED` takes the oldest due job,
  so concurrent workers never block on each other or take the same row.
- **Leases and heartbeats:** a claimed job gets a 30 s lease, which the
  supervisor extends every 10 s. Nothing holds a row lock while a backtest
  runs.
- **Fencing tokens:** each claim writes a fresh random `lease_token`, and
  every later write (heartbeat, complete, fail, release) is conditioned on
  it. A worker that stalls past its lease, wakes up after the job was
  requeued and claimed by someone else, can neither extend the new lease
  nor write its result.
- **The reaper:** every 15 s each worker requeues jobs whose lease expired,
  with backoff of 5 s × 2^(attempt−1), up to 3 attempts. All times come
  from the database's `now()`, never a worker's clock.
- **Wake-up:** every change that makes a job claimable (enqueue, retry,
  release, reaper requeue) sends `NOTIFY` in the same transaction, so it
  arrives only on commit. Idle workers `LISTEN` and claim at once. A job
  backing off wakes them when it becomes due. A 5 s fallback poll covers a
  notification lost while a `LISTEN` connection was down, so a job never
  depends on a notification to run, only to run promptly.
- **At least once, made safe:** a job whose worker dies runs again. That is
  harmless because a backtest has no side effects before its fenced
  `complete()`, and it is deterministic: same request, same pinned prices,
  a seeded bootstrap, and BLAS/OpenMP pinned to one thread in the compute
  child, so a multithreaded reduction can't reorder floating-point sums.
  Whichever attempt commits writes the same numbers, bit for bit.

### Idempotency

`POST /backtests` takes an optional `Idempotency-Key` header, following the
[IETF draft](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/):

- **Same key, same request:** the original job comes back: 202, the same id,
  `Idempotent-Replayed: true`. Requests are compared after validation, so
  key order, omitted defaults, `5` vs `5.0` and ticker case don't matter.
- **Same key, different request:** **422** `idempotency-key-reused`. The
  draft keeps **409** for a retry that arrives while the original is still
  in flight. That state can't happen here: the job is created by one
  `INSERT`, and a concurrent retry waits on the unique index until the
  first transaction commits, then sees its job. So `POST /backtests` never
  returns 409. The API's only 409 is `GET /backtests/{id}/series` for a job
  that hasn't succeeded: the job exists, but its series doesn't yet.
- **Database down:** **503** with `Retry-After: 5`. The POST may or may not
  have created its job, and retrying with the same key is safe either way.

Every error is an RFC 9457 `application/problem+json` body with its own
`type`, so a client can tell "key reused" from "invalid body" although both
are 422.

### Price snapshots

yfinance's adjusted closes are rewritten after every dividend or split. So
prices are never updated in place. Each ingest stores an immutable
`price_series`, identified by a SHA-256 hash of its float64 contents
(re-ingesting identical data is a no-op), and each job pins the series it
was submitted against. A retry next week uses the same prices as the first
attempt. This was observed, not just assumed. Two smoke runs, each
downloading SPY afresh, got Sharpe 0.7312770 and 0.7312760 for the same
request. Two downloads of SPY 2015-2024 taken one minute apart then
differed on 1,989 of 2,516 adjusted closes, by up to 7×10⁻⁷ relative.
Yahoo's adjustment arithmetic isn't stable from one response to the next.
So in practice almost every ingest creates a new series. On any one
snapshot, repeated runs of a request agree bit for bit.

### Strategy versions

A strategy's behaviour is versioned the way the schema is migrated:
`qbt.strategies.STRATEGY_VERSIONS` gives each strategy a version, and a
published version never changes. The adaptive strategies, which the service
doesn't serve yet, are versioned the same way in
`qbt.adaptive.ADAPTIVE_STRATEGY_VERSIONS`. If a code change alters what a strategy
does, it gets a new version. `tests/test_strategy_versions.py` pins a
fingerprint of every version's positions and fails until that happens. Each
version is its own row in `strategies`, keyed by (name, version, parameter
schema). `GET /backtests/{id}` reports the `strategy_version` a job ran, and
`GET /strategies` the version that new jobs get.

When a version is retired (`ml_direction` 1 became 2 when it stopped
predicting its own training period), its stored backtests stay as they were:
same results, same metrics, still reported as version 1. Jobs still queued
under the old version fail with a message to resubmit, because this code
can't reproduce it. A job for a version newer than the worker's own is
retried with backoff, so a newer worker can take it. Each retry counts as an
attempt, so it fails after `max_attempts` if none does. An `Idempotency-Key` replay returns the original
job, whatever version it ran.

### Model serving

No monthly model beat its baselines
([`results/ml/c2_results.json`](results/ml/c2_results.json)), so the
service serves one anyway as an engineering demonstration, and the API
says so. The default model is C2's best: LightGBM predicting next
month's regime, with the HMM features. Every registry entry and every
`POST /predict` response carries the model's out-of-sample accuracy next
to the baseline it had to beat (for regime, the rules-regime lookup
table), with the paired bootstrap interval, the McNemar p-value and
`beats: false`.

- **Training** is a one-shot CLI, `python -m qbt_service.train` (Compose:
  `docker compose run --rm train …`). It pins a snapshot the way a
  backtest does: the newest stored series covering the range. It then
  runs the C2 nested purged walk-forward on that snapshot for the
  out-of-sample numbers. Finally it fits the served model with the same
  tuning on every labelled month
  ([`src/qbt/ml/train.py`](src/qbt/ml/train.py)).
- **The registry** (`model_registry`, migration 0004) keeps one immutable
  row per version, with the target, feature list, hyperparameters,
  training window and pinned snapshot. The out-of-sample accuracy, the
  baseline's name and accuracy, and `beats_baseline` are NOT NULL
  columns, so no model can be registered without them. The model is a
  `bytea` artifact with its SHA-256. The artifact is JSON plus LightGBM's
  own text format, never pickle, so loading it runs no code.
- **Reproducibility:** the same snapshot and seed give a **byte-identical
  artifact**, not just identical predictions. LightGBM runs
  deterministically (`deterministic`, `force_row_wise`, one thread, fixed
  seed), and its text format holds no timestamps. Training the same model
  again therefore finds the existing version instead of adding one. A
  different seed changes only the seed that LightGBM records in the
  artifact, because bagging is off. That makes a new version whose
  predictions are identical. Holds on the same library versions, which
  each row records.
- **Prediction:** `POST /predict` scores the latest version, or
  `{"version": n}`, on the latest complete month in that version's
  snapshot (`as_of`). It forecasts the month after `as_of`, whose label
  isn't in the snapshot, so the model never trained on it. The HMM
  features take tens of seconds to build, so they are computed once at
  training and stored with their SHA-256. A request then loads the model
  (cached per process after its artifact's SHA-256 is checked), scores
  the stored input after checking its hash, and logs the prediction in
  `predictions`. A hash mismatch is a 500 (`model-integrity-error`), and
  nothing is scored or logged.

```bash
docker compose run --rm ingest SPY --start 2008-01-01 --end 2025-01-01
docker compose run --rm train SPY --start 2008-01-01 --end 2025-01-01
curl -s -X POST localhost:8000/predict
```
```
{"model_version": 1, "as_of": "2024-12-31", "prediction": 0.0,
 "prediction_name": "Calm Uptrend", "probabilities": [...],
 "out_of_sample": {"accuracy": 0.808, "n_months": 104, ...},
 "baseline": {"name": "rules_regime_lookup", "accuracy": 0.827,
              "difference_ci": [-0.058, 0.019], "mcnemar_p": 0.69, "beats": false},
 "summary": "Out of sample (104 months, 2016-04-29 to 2024-11-29) this model was
             right 80.8% of the time; its baseline, rules_regime_lookup, was right
             82.7% on the same months. It did not beat that baseline: ...", ...}
```

(Rounded; the numbers are those of `results/ml/c2_results.json` for
regime / base+hmm / LightGBM. On a fresh Yahoo download they can differ
in the last digits, since every download is a new snapshot.)
`GET /models` lists the versions, and `GET /models/{version}` gives the
full evaluation. Errors are problem details: `no-model-registered` and
`unknown-model-version` (422), and `model-not-found` (404 on GET).

### How it's tested

- **Forced-ordering concurrency tests** (`tests/service/test_queue.py`): one
  transaction holds a row lock, a second is made to queue behind it (the
  test polls `pg_locks` until it is waiting), then the first commits. That
  runs each order of a race explicitly, instead of leaving it to the
  scheduler: a stale completion against the reaper, in both orders, and a
  claim skipping a row another claim holds. Free races check that only one
  side wins every time: 4 concurrent reapers, 8 concurrent enqueues, and
  16 concurrent POSTs with one key (`test_backtests_api.py`).
- **Fault injection with `pg_terminate_backend`** (`tests/service/test_worker.py`,
  `test_backtests_api.py`): no mocks. The test takes a lock the worker's next
  statement needs, waits until `pg_stat_activity` shows that backend
  blocked on exactly that statement, and kills it. That covers every
  database-error path in the worker, the API's 503s, and a dropped `LISTEN`
  connection.
- **A real SIGKILL** (`test_sigkilled_worker_job_is_finished_by_another_worker`):
  a real worker process group is killed mid-job. A second worker reaps the
  job, runs it as attempt 2, and its stored equity curve, positions and
  metrics equal those of an uninterrupted run.
- **Compose smoke test** ([`scripts/compose_smoke.py`](scripts/compose_smoke.py)):
  - The stack is built from scratch.
  - A backtest is POSTed, and its metrics are checked against a direct
    `qbt.runner` run.
  - A model is trained with the one-shot `train` service, and training
    again finds the same version. `POST /predict` is then checked: the
    response carries the out-of-sample accuracy and the baseline, the
    prediction is logged, and the features are rebuilt from the snapshot
    in a separate container, giving the same input hash and identical
    probabilities from the registered artifact.
  - A worker is stopped mid-job with SIGTERM, and the job's release and
    hand-over are checked.
  - CI runs everything but the worker stop on every push (the `e2e` job).

### Run it

Requires Docker with Compose v2.

```bash
docker compose up -d --build --wait     # db -> migrate -> api + 1 worker
docker compose up -d --scale worker=4   # optional: more workers
docker compose run --rm ingest SPY --start 2015-01-01 --end 2025-01-01

curl -s localhost:8000/strategies       # names and parameter schemas
```

Submit a backtest (`end` is exclusive, as in yfinance):

```bash
curl -si -X POST localhost:8000/backtests \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: "my-run-1"' \
  -d '{"strategy": "sma_crossover", "ticker": "SPY",
       "start": "2015-01-01", "end": "2025-01-01",
       "params": {"short_window": 20, "long_window": 100}}'
```
```
HTTP/1.1 202 Accepted
location: /backtests/e73d550b-b7c5-4094-9835-7ed8fc38fe5c

{"id":"e73d550b-…","status":"queued","href":"/backtests/e73d550b-…"}
```

Fetch it:

```bash
curl -s localhost:8000/backtests/e73d550b-b7c5-4094-9835-7ed8fc38fe5c
```
```
{"id": "e73d550b-…", "status": "succeeded", "attempts": 1, "max_attempts": 3,
 "price_series_id": 1,
 "metrics": {"sharpe_ratio": 0.6506, "sharpe_ci_low": 0.1406, "sharpe_ci_high": 1.1915,
             "cagr": 0.0754, "max_drawdown": -0.2199, "num_trades": 25.0, …},
 "result": {"attempt": 1, "bootstrap_impl": "cpp", "duration_ms": 71, "n_days": 2516, …}, …}
```

Its day-by-day series, once it has succeeded:

```bash
curl -s localhost:8000/backtests/e73d550b-b7c5-4094-9835-7ed8fc38fe5c/series
```

This returns `dates`, `equity`, `drawdown` and `position`, one entry per
trading day, plus `out_of_sample_from` for a learning strategy (see
[Known limitations](#known-limitations)). A job that hasn't succeeded yet
gets a **409** `job-not-succeeded`.

Errors are problem details, for example a range with no snapshot:

```
{"type": "/problems/no-price-data", "status": 422, "title": "No price data",
 "detail": "No price snapshot for QQQ covers [2015-01-01, 2025-01-01). Ingest one first: …"}
```

`docker compose down` stops the stack (add `-v` to drop the data).
`GET /docs` serves the OpenAPI UI.

### Results

Measured on one laptop: Intel Core 7 150U (2 P + 8 E cores, 12 threads,
15 W), 15.7 GiB, on AC power, with every repetition power-guarded.
Docker Desktop 29.9.3 runs on WSL2 (kernel 6.18), and its VM has 12 CPUs
and 7.6 GiB, **shared with the load generator**. The job mix is
`sma_crossover`, `momentum`, `mean_reversion` and `ml_direction`, round
robin, on SPY 2010-2025 with `n_boot` 2000. Cells are median [min-max].
Scripts and method are in [bench/README.md](bench/README.md). These runs used
`ml_direction` version 1. Version 2 (out of sample only) trains the same model
once but has not been re-measured.

**Service measurements** (3 runs each). The queue-latency and GET rows
were measured before the two fixes below:

| Measurement | Result | Source |
|---|---|---|
| Throughput, 1 / 2 / 4 / 8 workers | **472 / 740 / 926 / 1097 jobs/min** (1.00 / 1.57 / 1.96 / 2.33x). That is 62-73% of a compute-only control at each count. The host clock falls from 3.7 to 2.5 GHz as workers are added. | `bench/results/service/throughput.json` |
| Compute-only control, 1 / 2 / 4 / 8 containers, no API, queue or DB | 758 / 1152 / 1274 / 1606 backtests/min | `bench/results/service/control.json` |
| Queue latency, 4 workers, Poisson arrivals at 50% of capacity (7.7 jobs/s) | submit-to-start p50 345 [231-353] ms, p99 2521 [868-2859] ms; backlog stable | `bench/results/service/latency.json` |
| Queue latency, 75% of capacity (11.6 jobs/s) | **queue grows** in 2 of 3 runs, by +1.93 [0.01-2.38] jobs/s, so no p99 is reported | `bench/results/service/latency.json` |
| GET /backtests/{id} at 100 req/s, workers idle | p50 5.6 ms, p99 10.5 ms | `bench/results/service/get.json` |
| GET at 100 req/s, 4 workers draining a backlog | p50 3167 [9-4789] ms, p99 6557 [17-17777] ms | `bench/results/service/get.json` |
| POST /predict at 100 req/s for 60 s, workers idle (regime LightGBM+HMM, version 1) | p50 7.7 [7.7-8.0] ms, p99 14.0 [13.2-23.0] ms; cold first request after an API restart 46 ms; 0 errors, every prediction logged once | `bench/results/service/predict.json` |
| `docker kill` a worker mid-job (4 workers, 60% load) | requeued 33 s, finished 38 s after the kill (30 s lease + up to 15 s reaper interval). 0 jobs lost, 0 duplicated; killed jobs' metrics equal a direct run | `bench/results/service/kill.json` |

**Two fixes, before vs after** (2 runs each, interleaved as before, after,
after, before):
- **Claim latency:** a 1 s idle poll was replaced with LISTEN/NOTIFY.
- **The GET cliff:** one API process at the default CPU weight was replaced
  with 2 processes at `cpu_shares` 4096.

| Measurement | Before | After |
|---|---|---|
| Submit-to-start p50, 50% load, 4 workers | 236 [220-252] ms | **8 [8-8] ms** |
| Submit-to-start p99, same | 833 [832-834] ms | **363 [220-506] ms** |
| Submit-to-done p50 / p99 | 459 / 1022 ms | 214 / 668 ms |
| GET p50, 100 req/s, 4 workers busy | 8475 [4539-12411] ms | **17 [14-20] ms** |
| GET p99, same | 10010 [6110-13910] ms | 4023 [366-7680] ms |
| GET p50 / p99, workers idle (must not regress) | 5.5 / 169 [9-328] ms | 5.5 / 10.3 [10.0-10.5] ms |

Source: `bench/results/service/before_after.json`.

What the "before" numbers show: a single Python process tops out at about
1.4 CPUs because of the GIL, and under load the API ran at exactly 1.39-1.40
CPUs and queued requests for seconds. Two processes double that headroom.
`cpu_shares` 4096 (cgroup weight 303 against 100) gives the API priority
only under contention; it is not a cap.

**The GET p99 limitation, stated plainly:** the p50 under load is fixed, but
the p99 is not. One "after" run had a 366 ms p99 and the other 7.7 s.
During the bad run, the load generator itself fell 710 ms behind its own
schedule, the host clock dropped to 1.97 GHz, and the whole machine was
saturated. A CPU weight inside Docker's VM can't protect a process from
contention on the host. A trustworthy p99 under full saturation needs the
service and the load generator on separate machines. One 15 W laptop
running both can't provide it.

### Known limitations

- **For learning strategies, the service's metrics are full-period.**
  `ml_direction` holds no position before its first out-of-sample day
  (its training period, the first `train_frac` of the rows), but the
  metrics a job stores cover the whole requested window, those flat
  training days included. That dilutes return, Sharpe and the rest.
  `GET /backtests/{id}/series` returns `out_of_sample_from`, so a client
  can mark the training period and label the numbers as full-period.
  Metrics computed over the out-of-sample window only are not built yet.
  The research results in
  [`results/summary.md`](results/summary.md#out-of-sample-only-strategies-that-learn)
  are out-of-sample only.
- **No authentication.** Anyone who can reach the port can submit jobs. The
  ports are bound to 127.0.0.1.
- **The served model doesn't beat its baseline**, and each model version
  predicts one month: the month after its snapshot ends. A newer forecast
  needs a new snapshot and a new version.
- **Idempotency keys are global and never expire.** With no clients to scope
  them to, two callers who choose the same key share it.
- **Only the 4 base strategies** (`sma_crossover`, `momentum`,
  `mean_reversion`, `ml_direction`). The adaptive strategies take
  structured arguments, and cross-sectional momentum is a portfolio
  strategy, so neither is exposed yet.
- **Workers from before strategy versioning don't check versions.** In a
  rolling deploy, one could run a version-2 job on version-1 code. Stop the
  old workers before starting new ones.
- **A `docker kill`ed worker isn't restarted.** Under
  `restart: unless-stopped`, Docker restarts only containers that exit on
  their own. The dead worker's jobs are still requeued by the other
  workers' reapers, but capacity stays reduced until `docker compose up`
  runs again. An orchestrator (Kubernetes, Nomad, Swarm) would replace it.

## Deploying to AWS

The service has run on AWS twice, both times applied from `infra/` and
then destroyed. Nothing runs there between sessions.

- **D2 (2026-11-15):** measured from a laptop.
- **D3b (2026-11-15):** verified the deploy path end to end. publish.yml
  built commit `07f6f63` on main, and `scripts/deploy.sh --migrate`
  deployed it in 113.6 s. The api and worker then ran the published
  digest, and one backtest and one `/predict` succeeded.
  - The first run of `deploy.sh` failed at its own digest check: the copy
    into ECR had changed the digest. It was fixed and re-run.
  - Details:
    [`bench/results/aws/deploy_verify.json`](bench/results/aws/deploy_verify.json),
    summarised in [`bench/results/aws/README.md`](bench/results/aws/README.md#deploy-verified-d3b).

### Architecture

```
GitHub, push to main ──▶ publish.yml (GITHUB_TOKEN: contents read, packages write)
                          builds the image, pushes ghcr.io/nmhetre-unc/momentum-strategy:<sha>,
                          records its digest (job summary + artifact image-digest)

your machine, `aws login` ──▶ scripts/deploy.sh <sha>
   1. GHCR must serve the digest publish.yml recorded, or it stops
   2. pulls that digest, pushes it to ECR (qbt-service:<sha>), checks ECR's digest
   3. SSM Run Command on the instance: pull qbt-service@<digest> from ECR,
      qbt up, wait for GET /health = 200

 your IP ──:8000──▶ EC2 t3.small, us-east-2 (default VPC, public subnet)
                    ├─ api container ────┐
                    ├─ worker container ─┼──▶ RDS PostgreSQL 17, db.t4g.micro (private)
                    └─ one-shots: migrate, ingest, train
                    pulls only from ECR; DATABASE_URL from SSM Parameter Store
```

[`infra/README.md`](infra/README.md) covers the Terraform: what each
resource is, why Compose on one instance, and the Free Tier check.

### Why build and deploy are separate

The plan was a single GitHub Actions workflow: GitHub's OIDC token,
exchanged for a short-lived AWS role, would push to ECR and redeploy over
SSM. The account doesn't allow it. It's a project account under an AWS
Organizations SCP, and that SCP denies the GitHub OIDC identity provider:

- Listing providers is refused by the SCP itself:
  `iam:ListOpenIDConnectProviders on resource: arn:aws:iam::<account>:oidc-provider/* with an explicit deny in a service control policy`
  (policy `p-u3fekd1i`).
- The IAM policy simulator, run against the role I sign in with
  (`aws iam simulate-principal-policy`, `oidc-provider/token.actions.githubusercontent.com`),
  gives `iam:CreateOpenIDConnectProvider: explicitDeny`, with
  `AllowedByOrganizations: False`.
- The same simulation allows `iam:CreateRole` and `iam:PutRolePolicy` under
  `role/qbt/`. Only the identity provider is blocked.

Without the provider, nothing in GitHub can obtain AWS credentials, and
long-lived access keys are ruled out. So the work is split along that line:

- **Building is automatic.** GitHub builds every commit on main and
  publishes it to GHCR with `GITHUB_TOKEN`, which can't touch AWS.
- **Deploying is manual.** A person runs `scripts/deploy.sh` with the
  short-lived credentials `aws login` issues.
- **The digest connects the two.** `deploy.sh` deploys only an image whose
  GHCR digest equals the one publish.yml recorded for that commit on main.
  It copies that exact digest to ECR and checks it there, so the instance
  runs what CI built. The instance still pulls only from ECR, with its own
  role.

**On an account that allows OIDC,** `deploy.sh` would become a
`workflow_dispatch` workflow:

- **Trust:** a role trusting `token.actions.githubusercontent.com` with
  `aud = sts.amazonaws.com` and
  `sub = repo:nmhetre-unc/momentum-strategy:ref:refs/heads/main`.
  The default `sub` doesn't say which event started the run. Limiting the
  role to `workflow_dispatch` takes GitHub's OIDC subject customization to
  add `event_name` to `sub`.
- **Permissions:**
  - ECR push to `qbt-service` only (plus the unscopable `ecr:GetAuthorizationToken`);
  - `ec2:DescribeInstances`;
  - `ssm:SendCommand` only for `AWS-RunShellScript` and only on instances tagged `Project=qbt`;
  - `ssm:GetCommandInvocation`.
- **Cost:** IAM roles and OIDC providers are free, so $0.
- **What stays:** the GHCR hop and the digest check could go. Everything
  on the instance side would stay the same.

### Security choices

- **No long-lived AWS credentials anywhere.** Deploys use `aws login`
  sessions. The instance uses its role, through IMDSv2.
- **GitHub can build but can't deploy.** publish.yml has `contents: read`
  and `packages: write` only. Its actions are pinned to commit SHAs.
  `deploy.sh` accepts only images built from main whose digest matches the
  one recorded.
- **SSM instead of SSH.** No key pair and no port 22. Commands go through
  SSM Run Command, and shells through Session Manager.
- **Only your IP can reach the API.** The security group admits port 8000
  from one `/32` (`allowed_cidr`). The API is plain HTTP with no
  authentication, so this is its only protection.
- **TLS to the database is verified.** `sslmode=verify-full` against the
  RDS CA bundle, which is pinned by SHA-256 and mounted read-only. D2
  checked it against the real certificate: connecting by IP address, or
  with the system CA bundle, is refused.
- **Containers can't use the instance role.** IMDSv2 is required, with a
  hop limit of 1, so the instance metadata service is out of reach from
  inside Docker's network.
- **The database password is in neither state nor plan.** It's an
  ephemeral value, sent only through write-only arguments (`password_wo`,
  `value_wo`), and read at start-up from an SSM SecureString.
- **Output is scrubbed.** `deploy.sh` and the AWS benchmark pass their
  output through [`bench/scrub.py`](bench/scrub.py), after a psycopg error
  once printed the database URL during D2. That credential was destroyed
  with the database.

### Cost

From [`infra/README.md`](infra/README.md#cost), us-east-2 on-demand prices:

| | Cost |
|---|---|
| A 2-hour session (apply, use, destroy) | about $0.09 |
| The stack left running | about $33.80 a month |
| D2's 40 minutes | about $0.03 |
| publish.yml (public repo, so Actions minutes are free), deploy.sh | $0 |
| GHCR | $0 if the package is public. A private one counts against the free Packages storage, so check its visibility after the first publish |

The account is on the Free plan, so all of this comes out of its credits.
It can't be billed.

### Runbook

```bash
# 1. Infrastructure (~8 min, mostly RDS).
cd infra
printf 'allowed_cidr = "%s/32"\n' "$(curl -s https://checkip.amazonaws.com)" > terraform.tfvars
terraform init && terraform plan -out=qbt.tfplan && terraform apply qbt.tfplan
cd ..

# 2. Deploy a commit that publish.yml has built (its job summary prints this command).
aws login
scripts/deploy.sh <full commit sha> --migrate   # --migrate: first deploy, or a commit with a migration

# 3. First time only: data and model, on the instance (Session Manager, or SSM Run Command).
#    sudo qbt ingest SPY --start 2008-01-01 --end 2025-01-01
#    sudo qbt train  SPY --start 2008-01-01 --end 2025-01-01

# 4. Later builds: merge to main, wait for publish.yml, then
scripts/deploy.sh <full commit sha>

# 5. Teardown: everything, including ECR images and RDS backups.
cd infra && terraform destroy
```

After `terraform destroy`, D2 confirmed with the AWS CLI that nothing
billable was left: no instances, EBS volumes or snapshots, RDS instances or
snapshots, public IPs or ENIs, ECR repositories, or SSM parameters. The
commands are in [`bench/results/aws/teardown.json`](bench/results/aws/teardown.json).

**What `deploy.sh` stops on, each with its own message:**

- expired or missing AWS credentials;
- no running instance, more than one, or an SSM agent that isn't online;
- no ECR repository;
- no successful publish.yml run on main for the commit;
- a digest mismatch between GHCR, the recorded digest and ECR;
- `/health` not returning 200 within 180 s;
- an api or worker container running a different image.

The first three were run against the destroyed stack: a bad argument, no
credentials, and real credentials with no instance. Each stopped with its
message.

**The instance's compose file names one tag,** `image_tag` (default `v1`),
and ECR tags are immutable. `deploy.sh` therefore pulls the image by digest
and tags it locally under that name, so `qbt up` and the boot-time
`qbt.service` run it without changing `infra/`.

### Measured on AWS (D2)

Full results and caveats are in [`bench/results/aws/`](bench/results/aws/README.md).
The client was a laptop on a home connection, so every latency it measured
includes the network. The round trip (TCP connect) is shown next to each
one.

| Measurement | Result | Network round trip, p50 |
|---|---|---|
| `terraform apply` start to API healthy | 941.5 s (RDS 449 s of the 467 s apply; ~2.5 min of manual checks) | |
| GET /backtests/{id}, 20 req/s | p50 42.9 ms, p99 78.2 ms | 36.2 ms |
| POST /predict, 20 req/s | p50 46.1 ms, p99 124.0 ms | 37.4 ms |
| Throughput, 1 worker | 158 jobs/min | not included (database timestamps) |
| Throughput, 2 workers | 29 jobs/min (CPU-credit throttled) | not included |
| Kill a worker mid-job | passed on the 3rd attempt (2 of 2 reps): 0 lost, 0 duplicated, 0 completed twice; requeued in 31–36 s | not included |

**Caveats:**

- **The t3.small ran in standard credit mode, which can't be billed for
  bursts.**
  - Its credit balance stayed between 0.0 and 0.6, and CPU sat at the 20%
    baseline.
  - That throttling explains the 2-worker figure: 3.1 s per backtest
    against 0.22 s. Neither throughput number reflects the service.
  - The worker healthcheck's Python start-up also ran past its 5 s timeout.
    That check is now a shell file-age test, which takes 2 ms in the image
    against 254 ms for the Python check.
- **The kill test passed once on AWS, on its third attempt.** All three
  attempts are reported:
  - **Attempt 1:** no job was in flight to kill.
  - **Attempt 2:** two kills hit an idle worker, then the benchmark harness
    failed on that healthcheck timeout. Its integrity checks never ran, and
    its jobs were deleted, so lost, duplicated and completed-twice are
    unknown for it.
  - **Attempt 3:** passed.
- **Reduced load.** The load was reduced from the laptop benchmarks (step 9)
  to fit 2 vCPUs, with 1 repetition (2 for kill).

## Performance

**What was hot.** Profiling the full results run (`scripts/regenerate_results.py`:
11 strategies, walk-forward refits, 2000-replicate bootstraps) picked out the
stationary bootstrap. Measured on AC power, it takes **62% of the run**
[58-63%] (`benchmarks/results/full_run.json`). Its Python loop made about 4 tiny
NumPy calls per block, with roughly 214 blocks per replicate. The hand-written
HMM's forward and backward loops cost about 5 s, under 10%. So the
bootstrap was the one worth porting.

**Three implementations, one answer.** `qbt.bootstrap` has three
implementations, all producing bit-identical index arrays:
- `python`: the original loop, kept as the reference.
- `numpy`: the same scalar draws in the same order, with each replicate's
  indices built in one vectorized step.
- `cpp`: a pybind11 extension that draws from the caller's own
  `np.random.Generator` through NumPy's C routines in `libnpyrandom`. It holds
  the generator's lock for each replicate and writes into a preallocated
  buffer with no allocation inside the loop.

The default, `impl="auto"`, picks `cpp` when the extension is available.

**How identity is proven** (`tests/test_bootstrap.py`):
- **Pinned outputs:** Sharpe intervals and SHA-256 hashes of the index stream
  were captured from the original inline loops *before* the refactor and are
  compared with `==`.
- **Hypothesis tests** compare every implementation with the reference across
  `n` up to 5000, integer and float `mean_block` in `[1, n]`, and seeds up to
  2^63−1. Both the index arrays and the generator's state afterwards must match.
- **Generator-state checks:** stopping after k replicates must leave the
  generator exactly where the reference leaves it.
- **Mutation testing:** the equivalence tests were checked against deliberately
  broken copies of the NumPy and C++ versions (rotated output, one extra draw,
  swapped draw order). All three were caught. This was done by hand and isn't
  part of the suite.
- **Full regeneration:** `results/summary.md` regenerates identically under every
  implementation.

**NumPy-version guard.** NumPy doesn't promise the same random stream across
releases. So the extension records the NumPy it was built against, and
`qbt.bootstrap` only uses it when the installed NumPy is that same version.
- On a mismatch, or with no extension, `impl="auto"` falls back to `numpy` with
  a `BootstrapFallbackWarning`.
- An explicit `impl="cpp"` raises instead.
- NumPy is pinned exactly in both the build and runtime requirements.
- With no C++ compiler, the package installs as pure Python.

**Results.** Measured on an Intel Core 7 150U (12 logical CPUs) under WSL2
(Linux 6.18) on AC power, with Python 3.12.16, NumPy 2.5.4 and GNU 15.2.0 at
`-O3`. Workload: n=4278, mean_block=20, n_boot=2000. Times are medians; micro rows show the IQR,
and full-run rows show `[min-max]` of 3 runs. Speedups are paired within
each repeat, shown as median `[min-max]`.

| | python | numpy | cpp | cpp vs numpy | cpp vs python |
|---|---|---|---|---|---|
| One replicate (indices + Sharpe) | 680 µs (IQR 35) | 337 µs (IQR 16) | 36.7 µs (IQR 4.3) | 9.0x [6.7-12.3] | 18.5x [13.2-24.0] |
| `stationary_bootstrap_sharpe` call | 1306 ms (IQR 58) | 644 ms (IQR 26) | 68 ms (IQR 4) | 9.5x [8.6-10.2] | 19.4x [17.6-21.1] |
| Bootstrap time inside the full run | 31.8 s [29.1-35.3] | 16.3 s [14.0-17.4] | 2.35 s [2.16-2.69] | 6.5x [6.0-7.5] | 13.1x [12.4-14.7] |
| Bootstrap share of the full run | 62% | 45% | 11% | | |
| **Full run, end to end** | **55.1 s [47.0-55.7]** | **36.6 s [32.0-37.1]** | **20.0 s [19.8-25.5]** | **1.6x [1.5-1.8]** | **2.4x [2.2-2.8]** |

**Amdahl's law.** The other ~20 s of the run (HMM fitting, walk-forward
refits, sklearn) didn't change. So even an instant bootstrap would cap the
end-to-end speedup at about 55/20, roughly 2.7x, and 2.4x is most of
what was available. Within the full run the bootstrap gains 13x rather than
the call's 19x. That's because the script's paired bootstrap computes two
Sharpe ratios per replicate, and that part is still NumPy. The hottest code
now is the HMM's forward and backward loops, at 17% and 9% of the C++ run.

**Methodology.**
- **Power:** everything ran on AC power, which is recorded in every result.
  An earlier batch on battery was thrown out: throttling made one Python run
  beat every NumPy run. It's kept as
  `benchmarks/results/INVALID_on_battery_full_run.json`.
- **Ordering:** every repeat runs each implementation once, with the order
  rotating between repeats, and speedups are paired ratios within a repeat.
  That keeps CPU-speed drift from landing on one implementation.
- **Isolation:** each full run is a fresh subprocess, its output hash is
  checked, and the CPU used by other processes is recorded (0.06-0.23 CPUs
  throughout).
- **Residual noise:** the identical non-bootstrap work still varied 17.5-23.3 s
  across runs, which is why every number carries a spread.

Full details, scripts and raw JSON are in [benchmarks/](benchmarks/README.md).

## Findings

**No timing strategy beats buy-and-hold detectably.** Eleven strategies were
tested on SPY, and none is scored on data it learned from. The four that learn
nothing are scored over the full period. The seven that learn (a fitted model,
a regime model, or a choice selected on past performance) are scored from
their first out-of-sample day, 2019-11-22 or 2020-04-08, against buy-and-hold
over the same window. Every one produces a 90% confidence interval on the
Sharpe difference against buy-and-hold that contains zero. Over the full
period the closest is `volatility_targeted` at +0.22 [-0.08, +0.54], against
buy-and-hold's 0.607. Out of sample it is `regime_filtered` at +0.18 [-0.31,
+0.72], against buy-and-hold's 1.15. (`adaptive_ensemble` used to be closest,
at +0.324 [-0.045, +0.686], and `ml_direction` second, at +0.10 [-0.01, +0.20].
Both numbers counted days the strategies had learned from. See
[FINDINGS.md](FINDINGS.md) section 1.)

**The choice of benchmark inverts the conclusion.** Each strategy's own
Sharpe interval excludes zero — `sma_crossover` [0.35, 1.17] over the full
period, `adaptive_ensemble` [0.19, 1.71] out of sample — and read alone both
look like clear wins. But buy-and-hold's Sharpe over the same window (0.607,
1.15) falls inside both, so neither is distinguishable from holding the
index. The same inversion appears in the deflated Sharpe: against zero
`sma_crossover` stays above 0.90 at any trial count, while against the
benchmark it falls to 0.236 at five trials and `adaptive_ensemble` to 0.042.
Reporting only the against-zero number is what a backtest implicitly
does when it cites a Sharpe ratio and calls it significant.

**Volatility targeting is the one mechanism that survives.** It cut maximum
drawdown from -51.9% to -18.7% at a Sharpe of 0.832, holding 71.7% average
exposure. A constant 71.7% position achieves a Sharpe of exactly 0.607 —
identical to buy-and-hold, since Sharpe is invariant under constant rescaling
— and still draws down -39.7%. The improvement comes from when it sizes down,
not how much. This is the only result here whose mechanism does not require
forecasting returns: volatility clusters and is predictable where direction
is not.

**Cross-sectional momentum works long and fails short.** 12-1 momentum on the
S&P 100, decile long-short, monthly rebalance: combined Sharpe 0.001, max
drawdown -77.0%. The long leg alone has a Sharpe of 1.071 [0.727, 1.441], the
only interval in this project that clearly excludes zero. The short leg is
-0.882 [-1.281, -0.483] and exactly cancels it.

Equal dollar weights did not produce equal market exposure. The winner decile
skewed defensive, giving the long leg a beta of +0.789, while shorting the
loser decile gave a normal -1.003 — a net -0.214 against a universe that
returned +1059%. A log-space decomposition attributes 88.7% of the loss to
that beta mismatch and 11.3% to signal. Hedging the beta halves the drawdown
and improves 2009 from -63.6% to -12.4%, but the resulting Sharpe of 0.196
still has a confidence interval containing zero.

**Refitting per fold removes a measurable share of apparent edge.** The
rolling walk-forward previously fit each strategy once on the full series.
Refitting on each fold's own training window lowers mean fold Sharpe by 0.42
to 0.49 for strategies whose behaviour depends on a Sharpe-selected choice,
by about 0.20 for fitted classifiers, and not at all for strategies with no
fit step.

Full detail, including the regime analysis, is in [FINDINGS.md](FINDINGS.md).
Generated tables are in [results/summary.md](results/summary.md).

## Machine learning

The question: can a model predict next month's SPY direction (up or not)
or next month's regime (the most common daily `rules` regime: Calm
Uptrend, Quiet Range, Volatile Rally, Crisis / Selloff) better than a
simple rule? **No.** None of the models beats its bar, and the one
served by the API says so in every response.

### Leakage controls

Monthly samples have a decision date (features known, prediction made)
and a label end date (the next month end). They overlap in time, so
plain k-fold leaks. [`src/qbt/ml/splits.py`](src/qbt/ml/splits.py) has
the details.

- **Purging.** A training sample is dropped when its label interval
  touches the test block. In walk-forward this removes one boundary month
  per fold (`n_purged: 1` in each fold of
  [`results/ml/c1_baselines.json`](results/ml/c1_baselines.json)).
- **Embargo** (12 months, the longest fixed feature look-back). It drops
  the samples just *after* a test block, whose features were built from
  the test block's prices. Walk-forward trains only on samples before
  the test block, so there is nothing after it to drop. The embargo is a
  no-op there (`n_embargoed: 0` in every fold), kept so both splitters
  share one rule. It does real work in the purged k-fold used for inner
  tuning.
- **Nested tuning.** Hyperparameters are chosen on purged inner folds of
  each outer training set only. Every reported month is predicted by a
  model that never saw it, and never saw a choice made on it.
- **A lookahead canary for every feature.** `tests/test_ml_monthly.py`
  cuts the price history at several dates and requires each of the 14
  features (10 base, 4 HMM) to be unchanged on every day that survives
  the cut. The canary is proven against three deliberately leaky
  features (tomorrow's return, a full-sample z-score, a centred moving
  average), and it catches all three.

### The audit: strategies scored on data they learned from

Before this branch, the seven daily strategies that learn something (a
fitted classifier, an HMM regime model, or a choice picked by past
Sharpe) were scored over the full period. That included the years they
had learned from. Two audits measured this against the old code and the
fixed code:

| What was counted | Before | After | Source |
|---|---|---|---|
| `ml_direction`: days with a signal from its own training period | 2,982 of 4,279 (70%), long on 2,937 | 0 | [`daily_ml_audit.json`](results/ml/daily_ml_audit.json) |
| `ml_direction`, `ml_regime_conditional`: rolling walk-forward folds scored on training data | 18 of 27 each | 0 | [`learning_strategies_audit.json`](results/ml/learning_strategies_audit.json) |
| HMM fitted on days the ML model treats as out of sample | 87 days | 0 | [`daily_ml_audit.json`](results/ml/daily_ml_audit.json) |
| The five regime wrappers: in-sample days holding a position | 846 to 2,335 each | 0 | [`learning_strategies_audit.json`](results/ml/learning_strategies_audit.json) |
| The five regime wrappers: rolling folds scored on training data | 19 of 27 each | 0 | same |
| `regime_sized`: positions changed by rewriting *later* prices (lookahead) | 1,494 days | 0 | same |

The fix: every learning strategy marks its first out-of-sample day and
holds no position before it, and it is scored only from that day,
against buy-and-hold over the same window. The changed strategies got
new version numbers (v2 for `ml_direction`, the regime wrappers and
`ml_regime_conditional`), recorded on
every backtest the service runs and pinned by fingerprint in the tests.
See [Strategy versions](#strategy-versions).

What it changed, in plain terms: `adaptive_ensemble` had looked like the
best strategy. The deflated Sharpe against buy-and-hold estimates the
chance its true Sharpe beats the index after allowing for how many
strategies were tried. With the learning period counted, it was 0.906
at 1 trial, 0.549 at 5 and 0.380 at 11. Scored only on days it hadn't
learned from, it is **0.295, 0.042 and 0.015**
([`results/ml/deflated_sharpe_audit.json`](results/ml/deflated_sharpe_audit.json),
both sections measured by one script, `scripts/audit_deflated_sharpe.py`).
The apparent edge was mostly the strategy remembering its training data.

### Monthly models (C2) against the real bars

Logistic regression and LightGBM run in nested purged walk-forward on
SPY 2008-2025, with and without HMM features. Each is compared with its
**bar**, on the same months:

- **Direction:** majority class (always "up", i.e. buy-and-hold).
- **Regime:** `rules_regime_lookup`, the most frequent next-month regime
  among training months that had the same `rules` regime on their
  decision date. Regimes persist, so this simple table is hard to beat:
  it beats the majority class and "same as last month" too.

A model "beats" its bar only if the paired 95% stationary-bootstrap
interval of the accuracy difference is above zero and the exact McNemar
p < 0.05. All numbers are from
[`results/ml/c2_results.json`](results/ml/c2_results.json)
(`verdict.classification`).

| Target | Features | Model | Months | Accuracy | Bar | Bar accuracy | Difference [95% CI] | McNemar p | Beats |
|---|---|---|---|---|---|---|---|---|---|
| direction | base | logistic | 128 | 0.656 | majority class | 0.688 | -0.031 [-0.086, +0.008] | 0.22 | no |
| direction | base | LightGBM | 128 | 0.664 | majority class | 0.688 | -0.023 [-0.094, +0.031] | 0.51 | no |
| direction | base+hmm | logistic | 104 | 0.721 | majority class | 0.721 | 0.000 [0.000, 0.000] | 1.00 | no (always "up") |
| direction | base+hmm | LightGBM | 104 | 0.721 | majority class | 0.721 | 0.000 [0.000, 0.000] | 1.00 | no (always "up") |
| regime | base | logistic | 128 | 0.711 | rules_regime_lookup | 0.805 | -0.094 [-0.180, -0.016] | 0.012 | no (worse) |
| regime | base | LightGBM | 128 | 0.773 | rules_regime_lookup | 0.805 | -0.031 [-0.078, +0.023] | 0.39 | no |
| regime | base+hmm | logistic | 104 | 0.779 | rules_regime_lookup | 0.827 | -0.048 [-0.087, -0.010] | 0.18 | no |
| regime | base+hmm | LightGBM | 104 | 0.808 | rules_regime_lookup | 0.827 | -0.019 [-0.058, +0.019] | 0.69 | no |

(The HMM features start later, so the base+hmm rows cover 104 months,
2016-04 to 2024-11, rather than 128. The bar is recomputed on those
months, which is why it differs between the base and base+hmm rows.)

**The economic test.** A direction model trades: long when P(up) > 0.5,
flat otherwise, 5bps costs. Against buy-and-hold over the same 128
months, the base logistic model has a Sharpe of 0.833 against 0.910
(difference -0.077 [-0.202, +0.015]), and LightGBM 0.881 against 0.910
(-0.029 [-0.209, +0.111]). With HMM features both models are long every
month, so they *are* buy-and-hold. The deflated Sharpe against
buy-and-hold counts **116 trials**: every target, feature set, model and
hyperparameter combination tried. At that count it is 0.002 (logistic)
and 0.004 (LightGBM); even at a single trial it would be 0.41 and 0.46.

**Disclosure: a discarded first pass.** Before the grids were widened, a
first pass of 24 configurations (4 logistic and 8 LightGBM, both targets,
base features) was run and its out-of-sample results were seen. Those
configurations are also in the final grids, but that pass was a separate
selection, so it counts as separate trials: 92 + 24 = 116
(`trials.first_pass` in `c2_results.json`).

**Conclusion.** Nothing beats its bar. The best model, regime LightGBM
with HMM features, is right 80.8% of the time against the lookup table's
82.7%, a difference whose interval includes zero. Its errors aren't
detectably different from the table's. No direction model beats always
"up", and none trades better than buy-and-hold. With 104 to 128
out-of-sample months the intervals are a few points wide, so a small
real edge can't be ruled out either; this data can't establish one.

### Serving it anyway, honestly

The regime LightGBM with HMM features is served through the API as an
engineering demonstration ([Model serving](#model-serving)):

- **Registry.** Each version is an immutable row (migration 0004) with
  its training snapshot, window, features, hyperparameters and seed. Its
  out-of-sample accuracy, its bar and `beats_baseline` are required
  columns, so a model can't be registered without them.
- **Reproducible artifacts.** The same snapshot and seed give a
  byte-identical artifact (JSON wrapping LightGBM's text model, never
  pickle), stored with its SHA-256. The hash is checked on every load,
  and the served input is checked against its own hash. Retraining an
  identical model finds the existing version.
- **Honest `/predict`.** Every response carries the out-of-sample
  accuracy (0.808 over 104 months) next to `rules_regime_lookup`'s 0.827,
  the difference interval, the McNemar p, `beats: false`, and a sentence
  saying to treat the prediction as a demonstration. Every prediction is
  logged.
- **Latency.** At 100 requests/s for 60 s: p50 7.7 ms, p99 14.0 ms, 0
  errors, every prediction logged once
  ([`bench/results/service/predict.json`](bench/results/service/predict.json)).

## What property-based testing found

Two numeric bugs in the metrics code surfaced through the Hypothesis property
tests in `tests/test_properties.py`, and both are fixed in `src/qbt/analytics.py`:

- **A floating-point guard that missed zero.** A constant return series should
  have zero standard deviation, but mean-then-subtract arithmetic leaves about
  1e-17, so an exact `== 0` check let the Sharpe and Sortino ratios blow up.
  The guard is now `np.isclose(std, 0.0, atol=1e-12)`.
- **A float64 overflow in annualization.** Compounding a short or volatile
  window to a year (for example a 300x move over two days) overflowed
  `(1 + r) ** 252` to infinity. Annualized return is now computed in log space
  with the exponent clipped to +/-700, in `cagr()` and in the per-regime
  annualization.

## What's in the engine

- Backtest and portfolio backtest with explicit transaction costs and a
  shift-by-1 execution convention
- Risk metrics with stationary block bootstrap confidence intervals, deflated
  Sharpe, and Newey-West standard errors; the bootstrap has bit-identical
  Python, NumPy and C++ (pybind11) implementations
- Regime detection via k-means, Gaussian mixture, a hand-written Gaussian HMM
  with Baum-Welch, and rule-based thresholds, with causal fitting by default
- Walk-forward validation with per-fold refitting
- Cross-sectional momentum on an S&P 100 price panel
- A Streamlit dashboard over all of it

## Limitations

Single asset for the timing strategies, single 17-year window, long/flat
positions only. Costs are a flat 5bps one-way spread proxy with no market
impact, borrow, or financing.

The S&P 100 universe is current membership, so companies that left the index
between 2008 and 2025 are absent. Those are disproportionately the sustained
underperformers a short leg would have profited from, so the bias runs in the
strategy's favor and the negative result is if anything understated.

The seven learning strategies are scored out of sample only, over 1,191 to
1,284 days to the end of 2024. That is under a third of the full period and a
single, mostly rising market, so their intervals are wide. Their rows are
compared with buy-and-hold over their own window and can't be ranked
directly against the full-period rows. `results/ml/learning_strategies_audit.json`
lists what each strategy learns and which rows that makes in-sample.

With 17 years of daily data the standard error on a Sharpe estimate is
roughly 0.26. Establishing a 0.32 improvement at 90% confidence would require
about 45 years — more history than SPY has, and long enough that market
structure would have changed repeatedly. The question is close to
unanswerable at this frequency and sample size.

## Setup

Requires Python 3.12+. A C++17 compiler (GCC, Clang, or MSVC Build Tools on
Windows) is needed for the fast bootstrap. CMake, Ninja and pybind11 are
fetched automatically during the install. Without a compiler, the install still
succeeds as pure Python, and the bootstrap falls back to the NumPy
implementation with a warning.

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[dev,ui]"      # builds the C++ extension
python -c "import qbt.bootstrap as b; print('C++ bootstrap:', b.cpp_available())"
```

For the service, see [Running as a service](#running-as-a-service).

Extras: `ui` is the dashboard and the CLI's charts (Streamlit, Altair,
Matplotlib); `dev` is the test and lint tools; `api` is the backtest
service. A plain `pip install .` gives the library alone (backtests,
analytics, the C++ bootstrap) without any of them. `pip install -r
requirements.txt`, which is what Streamlit Community Cloud runs, installs
the package with `ui`.

After changing `src/cpp/` or upgrading NumPy, re-run `pip install -e .` to
rebuild the extension. NumPy is pinned exactly: the extension is only used
with the NumPy version it was built against.

Run the dashboard:

```bash
streamlit run app/app.py
```

Run the CLI:

```bash
python main.py --ticker SPY --strategy momentum --rolling
```

Run the tests (653 tests, 94% line coverage). The service's tests need
Postgres, which Compose provides:

```bash
pip install -e ".[dev,api,ui,ml]"   # ml: LightGBM, for the C2 monthly models
docker compose up -d db
export DATABASE_URL=postgresql+psycopg://qbt:qbt@localhost:5432/qbt
pytest --cov=src/qbt --cov=src/qbt_service
```

Without `DATABASE_URL`, the database tests are skipped.

Run the benchmarks (not part of the test run; see
[benchmarks/README.md](benchmarks/README.md)):

```bash
python benchmarks/micro_interleaved.py
python benchmarks/full_run.py --runs 3
python bench/service_bench.py throughput latency get kill   # the service; see bench/README.md
python bench/before_after.py
```

## References

Bailey DH, López de Prado M. 2014. The deflated Sharpe ratio. Journal of
Portfolio Management. 40:94-107.

Daniel K, Moskowitz TJ. 2016. Momentum crashes. Journal of Financial
Economics. 122:221-247.

Jegadeesh N, Titman S. 1993. Returns to buying winners and selling losers.
Journal of Finance. 48:65-91.

Politis DN, Romano JP. 1994. The stationary bootstrap. Journal of the
American Statistical Association. 89:1303-1313.