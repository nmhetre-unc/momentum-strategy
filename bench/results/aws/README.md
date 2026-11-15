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
| Kill a worker (SIGKILL) | rep 1: 2 workers, 40-job backlog, then Poisson at 60% | job 557138eb requeued 36.1 s, reclaimed 41.1 s, finished 41.4 s after the kill. 73 jobs: lost 0, duplicated 0, completed twice 0 (results discarded), result from the wrong attempt 0, fencing rejections 0; only the killed job took a 2nd attempt | not in it (instance clock, polling the API on localhost) | [`kill.json`](kill.json) |
| Kill a worker (SIGKILL) | rep 2: 2 workers, 40-job backlog, then Poisson at 60% | job c1e79ac3 requeued 31.5 s, reclaimed 37.6 s, finished 39.5 s after the kill. 75 jobs: lost 0, duplicated 0, completed twice 0 (results discarded), result from the wrong attempt 0, fencing rejections 0; only the killed job took a 2nd attempt | not in it (instance clock, polling the API on localhost) | [`kill.json`](kill.json) |

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

**Kill test, first attempt:** [`kill_first_attempt_no_job_in_flight.json`](kill_first_attempt_no_job_in_flight.json).

- At 60% of the throttled capacity alone, jobs of about 0.25 s left the workers idle almost all the time. Ten kill attempts found no job in flight. Its integrity checks passed anyway: 0 lost, 0 duplicated.
- A second attempt failed on the worker healthcheck. Its Python start-up exceeded the 5 s timeout at the CPU baseline, so Compose `--wait` reported the worker unhealthy while it was processing jobs.
- The fix: a backlog first, the kill triggered by a worker's own `claimed job` log line (milliseconds between seeing the claim and the SIGKILL), and scaling that waits for the containers to be running rather than healthy.

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
