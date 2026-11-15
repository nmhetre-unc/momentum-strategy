# The service on AWS: one session, measured from a laptop

D2, 2026-11-15. `infra/` applied, measured with `bench/aws_bench.py`, destroyed. Every number below is from a JSON file in this directory.

**Setup** ([`environment.json`](environment.json)):

- **Server, us-east-2:**
  - EC2 t3.small (2 vCPU, 1 core × 2 threads, 1,909 MiB visible to the OS), standard CPU credits, us-east-2c. It ran the api (1 uvicorn process) and 1–2 workers with Docker Compose.
  - RDS PostgreSQL 17.9 on db.t4g.micro, 20 GB gp3, single AZ.
  - Image `qbt-service:v1`, built from commit 2fdad5c.
- **Client:** this laptop (Intel Core 7 150U, Windows + WSL2) over the public internet from a home connection, on AC power throughout. Every measurement passed the power guard.
- **Network round trip:** TCP connects to port 8000, measured before each measurement: p50 34.5–37.5 ms, min 30.7–31.3 ms. Every latency below that was measured at the laptop includes it; queue times read from database timestamps don't.

## Deploy: apply start to API healthy, 941.5 s (15 min 42 s)

[`deploy.json`](deploy.json) has the command outputs.

| s after apply start (when the step ended) | Step |
|---|---|
| 0.0 | terraform apply start |
| 185.2 | image push to ECR (during apply, once the repository existed) |
| 467.2 | terraform apply complete (15 added; RDS 7m29s, instance 13s) |
| 481.0 | instance Online in SSM (cloud-init already done at the first check, 21:08:46) |
| 713.1 | pull image |
| 720.6 | qbt migrate |
| 734.0 | qbt ingest |
| 867.3 | qbt train |
| 930.7 | qbt up |
| 941.5 | first GET /health = 200 from the laptop |

RDS creation was 449 of the 467 s of apply. The steps ran by hand, one after another. The gap between "SSM online" (481 s) and the image pull starting (645 s) was spent checking cloud-init and the instance by hand, not waiting on AWS. Training took 122 s on the instance against 47 s on the laptop because of the CPU credits (below).

## TLS to RDS: verify-full against the real certificate

[`tls.json`](tls.json):

- **Certificate chain:** the server certificate's CN is the endpoint, issued by "Amazon RDS us-east-2 Subordinate CA RSA2048 G1.A.4". That is issued by "Amazon RDS us-east-2 Root CA RSA2048 G1", which is in the pinned bundle.
- **openssl:** `openssl s_client -starttls postgres -verify_hostname <endpoint>` with that bundle returned `Verify return code: 0 (ok)` over TLSv1.3.
- **The service's connection:** from inside the api container, libpq ran with `sslmode verify-full` and `sslrootcert /etc/ssl/rds/global-bundle.pem`. `pg_stat_ssl` showed TLSv1.3, TLS_AES_256_GCM_SHA384, 256-bit.
- **Every connection:** all 4 of the service's connections (api pool, worker, the check itself) used TLS 1.3.
- **Negative checks:** connecting by the endpoint's IP was refused ("does not match host name"), and so was using the system CA bundle ("certificate verify failed"). So verification is enforced, not merely available.

## Load, reduced from step 9

| Measurement | Condition | Result | Network RTT | Source |
|---|---|---|---|---|
| Throughput | 1 worker, backlog of 60 | **158 jobs/min**, backtest median 220 ms | not in it (DB timestamps) | [`throughput.json`](throughput.json) |
| Throughput | 2 workers, backlog of 120 | **29 jobs/min**, backtest median 3118 ms: CPU-credit throttled | not in it | [`throughput.json`](throughput.json) |
| Queue latency | 2 workers, Poisson 0.238 jobs/s (50% of the 2-worker figure) for 90 s | submit-to-start p50 7.1 ms, p99 13.4 ms; submit-to-done p50 247 ms, p99 1036 ms; backlog stable | not in it (DB timestamps) | [`latency.json`](latency.json) |
| POST /backtests | same run, at the laptop | p50 71.5 ms, p99 109.7 ms | RTT p50 37.0 ms (min 30.9) | [`latency.json`](latency.json) |
| GET /backtests/{id} | 20 req/s for 30 s, workers idle | p50 42.9 ms, p99 78.2 ms, 0 errors | RTT p50 36.2 ms (min 31.0) | [`get.json`](get.json) |
| POST /predict | 20 req/s for 60 s, workers idle | **p50 46.1 ms, p99 124.0 ms**, max 206.4 ms; cold first request 260 ms; 0 errors; 1301 of 1301 200s logged, each once | RTT p50 37.4 ms (min 30.8) | [`predict.json`](predict.json) |
| Kill a worker (SIGKILL), attempt 3 | rep 1: 2 workers, 40-job backlog, then Poisson at 60% | job 557138eb requeued 36.1 s, reclaimed 41.1 s, finished 41.4 s after the kill. 73 jobs: lost 0, duplicated 0, completed twice 0 (results discarded), result from the wrong attempt 0, fencing rejections 0; only the killed job took a 2nd attempt | not in it (instance clock, polling the API on localhost) | [`kill.json`](kill.json) |
| Kill a worker (SIGKILL), attempt 3 | rep 2: 2 workers, 40-job backlog, then Poisson at 60% | job c1e79ac3 requeued 31.5 s, reclaimed 37.6 s, finished 39.5 s after the kill. 75 jobs: lost 0, duplicated 0, completed twice 0 (results discarded), result from the wrong attempt 0, fencing rejections 0; only the killed job took a 2nd attempt | not in it (instance clock, polling the API on localhost) | [`kill.json`](kill.json) |

Subtracting the network p50 from the laptop-side p50s leaves roughly the server's time:

- **GET:** about 7 ms.
- **/predict:** about 9 ms.
- **POST /backtests:** about 34 ms. That is two round trips, because each POST opens a fresh connection.

The laptop-measured /predict p50 of 7.7 ms ([`../service/predict.json`](../service/predict.json)) has no network in it, so it compares with the 9 ms server estimate here, not with the 46 ms.

**CPU credits decide everything on this instance** ([`cloudwatch.json`](cloudwatch.json)):

- The t3.small ran in `standard` credit mode, as planned, so it can't be billed for surplus. Its credit balance stayed between 0.0 and 0.6 for the whole session; T3 in standard mode gets no launch credits.
- CPU utilisation averaged 16.2–24.3% per 5 minutes, which is the 20% baseline.
- The 1-worker run came first and still got short bursts. The 2-worker run then ran at the baseline: 3.1 s per backtest instead of 0.22 s, so 2 workers were 5.5× slower than 1.
- Those two throughput figures measure the credit budget, not the service. They aren't comparable to each other, or to the laptop's 472 jobs/min with 1 worker ([`../service/throughput.json`](../service/throughput.json)).
- Unlimited credit mode or a non-burstable instance would be needed for a real throughput number. Either is a change to `infra/` that wasn't part of the approved plan.

**What was reduced, compared with step 9 ([`../service/`](../service/)):**

| Measurement | Step 9 (laptop) | This session |
|---|---|---|
| Repetitions | 3 of each | 1 (2 for kill) |
| Throughput | 1, 2, 4 and 8 workers, 300 jobs per worker | 1 and 2 workers, 60 jobs per worker |
| Queue latency | 4 workers, 50% and 75% load, 150 s | 2 workers, 50% load only, 90 s |
| GET | 100 req/s for 60 s, idle and under load | 20 req/s for 30 s, idle only |
| /predict | 100 req/s | 20 req/s |
| Kill | 4 workers, Poisson at 60% | 2 workers, a 40-job backlog then Poisson at 60%; killed jobs' metrics not re-checked against a direct run |
| Compute-only control | yes | none |

**Kill test: three attempts, all reported.** The table above shows only attempt 3. Console output for all three is in [`kill_console.log`](kill_console.log).

| Attempt | Started (UTC) | Outcome | Integrity checks | File |
|---|---|---|---|---|
| 1 | 21:25:12 | **No job in flight.** 2 reps × 5 kill attempts never found a worker holding a running job. | Ran: 33 and 35 jobs; 0 lost, 0 duplicated, 0 not succeeded, 0 discarded results. `all_checks_pass` is false only because there was no victim. | [`kill_first_attempt_no_job_in_flight.json`](kill_first_attempt_no_job_in_flight.json) |
| 2 | 21:30:31 | **Aborted by the harness.** Two SIGKILLs hit an idle worker. Then the benchmark's own rescale with `--wait` failed on a healthcheck timeout. | **Not run.** Lost, duplicated, stuck and completed-twice are unknown; see below. | [`kill_second_attempt_aborted.json`](kill_second_attempt_aborted.json) |
| 3 | 21:33:34 | **Passed**, 2 of 2 reps. | All pass; the numbers are in the table above. | [`kill.json`](kill.json) |

- **Attempt 1:** the load was 60% of the throttled capacity, with no backlog. Jobs of about 0.25 s left the workers idle almost all the time.
- **Attempt 2** added a 40-job backlog but kept attempt 1's way of choosing a victim:
  - It queried the database for a running job's lease owner, SIGKILLed that worker, then queried again. Each query is a `docker compose exec … python`, which took seconds at the CPU baseline.
  - Both times, worker `d82239faee54` had finished its job before the SIGKILL landed: the second query found no running job leased to it. A job held by a dead worker stays `running` until its 30 s lease expires, so that query would have seen one. Neither kill interrupted a job.
  - Before the third kill attempt, the script rescaled with `docker compose up --wait`. The worker that had never been killed failed its healthcheck: `python -m qbt_service.health worker`, 5 s timeout, exceeded at the baseline. Compose reported it unhealthy, and `aws_bench.py` raised and exited.
  - The diagnosis a minute later: both workers running and healthy again, 0 restarts. Their health logs show the timeouts followed by passes, and the 12 log lines shown are all attempt 1 and `succeeded`. The worker itself never failed; the healthcheck's Python start-up did.
- **What attempt 2 cannot tell us:**
  - Its checks never ran. Attempt 3 starts by deleting all jobs, which removed attempt 2's, including any still queued or running. The database no longer exists, so lost, duplicated, stuck and completed-twice can't be determined now.
  - What is known: no kill interrupted a job, so requeueing and fencing were never exercised. Attempt 3's checks count only its own jobs and its own logs, and found 0 extra jobs, 0 fencing rejections and 0 discarded results, so attempt 2 didn't leak into it.
- **Fixes before attempt 3:**
  - The kill is triggered by a worker's own `claimed job` log line. Reading the log needs no Python start-up, so the SIGKILL lands milliseconds after the claim. The job is then confirmed still `running`.
  - `scale()` waits for the containers to be running, not healthy.
  - Neither fix changed the service.

## Deploy verified (D3b)

[`deploy_verify.json`](deploy_verify.json) holds the timestamps and the
scrubbed output of every command. On 2026-11-15, `infra/` was applied again
(the same 15 resources; RDS took 10m32s of the 649.2 s apply), and
`scripts/deploy.sh 07f6f63… --migrate` was run the way a user would run it.
Then came the first-time one-shots, `qbt ingest` (8.9 s) and `qbt train`
(78.4 s, OOS 0.808 against 0.827 as before), then the checks, then
`terraform destroy`.

**Attempt 1 failed, at deploy.sh's own check.**

- **What happened:** the copy from GHCR into ECR used
  `docker buildx imagetools create`. publish.yml produces a
  single-platform manifest (`679d45…`). Given one, imagetools doesn't copy
  it as is: it wraps it in a new manifest list (`729dc0…`). The check
  "ECR must hold the published digest" stopped the run before anything
  reached the instance.
- **Why it wasn't caught earlier:** the local test before merge had copied
  `hello-world`, which is already a manifest list and so keeps its digest.
- **The fix:** `deploy.sh` now runs `docker pull` by digest, `docker tag`
  and `docker push`, with the same check. Before re-running, the wrong ECR
  image was deleted with `aws ecr batch-delete-image`, because ECR tags are
  immutable. That deletion was the only AWS change outside `deploy.sh`,
  Terraform and the one-shots.

**Attempt 2 succeeded: 113.6 s from start to "deployed".**

| Phase | s |
|---|---|
| Preflight: credentials, instance and SSM agent, ECR repository | 2.2 |
| Publish run and its recorded digest (`gh`) | 2.8 |
| GHCR serves that digest | 1.3 |
| ECR login | 2.0 |
| Pull from GHCR to the laptop (home connection) | 75.3 |
| Push to ECR and digest check | 1.8 |
| Instance: `docker pull` by digest from ECR | 10.0 |
| Instance: `qbt migrate` (0001 → 0004) | 3.2 |
| Instance: `qbt up` until api and worker are healthy | 11.5 |
| Instance: `/health` = 200 and image check | 0.2 |
| SSM round trips and summary | 3.5 |

The push is fast only because attempt 1 had already uploaded every layer.
A first push into an empty repository would upload about 177 MB; D2's took
8 s.

**Checks after the deploy:**

- **The containers run the published digest.** Both api and worker run
  image `2894092cd15b`, whose only repo digest is
  `qbt-service@sha256:679d45…`. That is the digest publish run 43769442391
  recorded and GHCR served. Locally it is tagged as the compose image name
  `qbt-service:v1` (the retag).
- **The healthchecks pass.** Both containers became healthy on the first
  `qbt up`; the new shell worker check passed every time it ran.
- **End to end, from the laptop:**
  - `GET /health` returned 200;
  - `POST /backtests` (sma_crossover, SPY 2010–2024) returned 202;
  - the job `succeeded` on attempt 1 with 12 metrics, finishing 0.27 s
    after it was created;
  - `POST /predict` returned 200 (model v1, as of 2024-12-31, OOS 0.808
    against the 0.827 baseline, `beats: false`).

**Teardown:** `terraform destroy` removed all 15 resources in 148.6 s. As
in D2, one automated RDS snapshot was still listed right after destroy, and
RDS deleted it by 22:57:58Z. The final check came back empty for:

- EC2 instances, apart from the terminated one;
- EBS volumes and snapshots;
- Elastic IPs and network interfaces;
- RDS instances, snapshots and retained backups;
- ECR repositories;
- SSM parameters.

Terraform state has 0 resources.

## Teardown

[`teardown.json`](teardown.json): `terraform destroy` removed all 15 resources in 2 min 19 s (2026-11-15T21:38:30.812Z to 2026-11-15T21:40:49.680Z). The checks with the AWS CLI in us-east-2 then came back empty for:

- EC2 instances other than the terminated one, which isn't billed;
- EBS volumes and snapshots;
- Elastic IPs and network interfaces;
- RDS instances, snapshots and retained automated backups;
- ECR repositories;
- SSM parameters.

Terraform's state is empty. One automated RDS snapshot, 20 GB, was still listed right after destroy. RDS deleted it on its own within 3 minutes, by 21:43:57Z.

**Cost:** the resources existed for about 40 minutes, 20:58 to 21:41 UTC. That's roughly $0.03 at the infra/README.md rates, drawn from Free plan credits.
