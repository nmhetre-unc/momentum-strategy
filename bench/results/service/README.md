# Service measurements

Intel(R) Core(TM) 7 150U (10 cores / 12 threads, 2 P + 8 E cores, 15 W), 15.7 GiB, Windows power plan Balanced, on AC power (every repetition guarded). Docker Desktop 29.9.3 (WSL2 backend, kernel 6.18.33.2-microsoft-standard-WSL2); its VM: 12 CPUs, 7.6 GiB, shared with the load generator. Fixed mix: sma_crossover, momentum, mean_reversion, ml_direction round robin, SPY 2010-2025, n_boot 2000.

| Measurement | Condition | Result: median [min-max] of 3 runs | Source |
|---|---|---|---|
| Throughput | 1 worker | **472 [435-483] jobs/min**, 1.00x of 1 worker, 62% of the compute-only ceiling. Backtest 82 [81-88] ms, overhead 21 [20-22] ms/job, host clock 3.73 [2.85-3.82] GHz; CPUs busy: VM 1.3 [1.3-1.3] of 12, Postgres 0.09 [0.09-0.10], API 0.03 [0.03-0.04] | bench/results/service/throughput.json |
| Throughput | 2 workers | **740 [649-747] jobs/min**, 1.57x of 1 worker, 64% of the compute-only ceiling. Backtest 105 [104-118] ms, overhead 26 [26-30] ms/job, host clock 3.30 [3.02-3.34] GHz; CPUs busy: VM 2.5 [2.5-2.5] of 12, Postgres 0.17 [0.16-0.17], API 0.04 [0.04-0.05] | bench/results/service/throughput.json |
| Throughput | 4 workers | **926 [668-939] jobs/min**, 1.96x of 1 worker, 73% of the compute-only ceiling. Backtest 176 [172-244] ms, overhead 40 [39-54] ms/job, host clock 2.97 [2.37-3.02] GHz; CPUs busy: VM 4.8 [4.8-5.0] of 12, Postgres 0.30 [0.30-0.33], API 0.06 [0.05-0.07] | bench/results/service/throughput.json |
| Throughput | 8 workers | **1097 [1076-1111] jobs/min**, 2.33x of 1 worker, 68% of the compute-only ceiling. Backtest 313 [309-316] ms, overhead 57 [56-57] ms/job, host clock 2.53 [2.51-2.54] GHz; CPUs busy: VM 9.2 [9.1-9.2] of 12, Postgres 0.47 [0.46-0.49], API 0.08 [0.07-0.08] | bench/results/service/throughput.json |
| Compute-only control | 1 parallel container, no API/queue/DB | 758 [756-760] backtests/min, 77 [76-77] ms each, host clock 4.08 [3.96-4.14] GHz | bench/results/service/control.json |
| Compute-only control | 2 parallel containers, no API/queue/DB | 1152 [1134-1154] backtests/min, 99 [99-99] ms each, host clock 3.44 [3.44-3.45] GHz | bench/results/service/control.json |
| Compute-only control | 4 parallel containers, no API/queue/DB | 1274 [1248-1286] backtests/min, 184 [183-185] ms each, host clock 3.06 [3.02-3.06] GHz | bench/results/service/control.json |
| Compute-only control | 8 parallel containers, no API/queue/DB | 1606 [1578-1608] backtests/min, 289 [287-291] ms each, host clock 2.68 [2.67-2.69] GHz | bench/results/service/control.json |
| Queue latency | 4 workers, Poisson, 50% of 4-worker throughput (7.7 jobs/s) | submit-to-start p50 345 [231-353] ms, p99 2521 [868-2859] ms; submit-to-done p50 581 [388-640] ms, p99 2938 [1003-3250] ms (backlog stable: 0.029 [-0.002-0.031] jobs/s) | bench/results/service/latency.json |
| Queue latency | 4 workers, Poisson, 75% of 4-worker throughput (11.6 jobs/s) | **queue grows** in 2 of 3 runs: backlog +1.93 [0.01-2.38] jobs/s (arrivals 11.9 [11.8-12.0]/s, completions 10.4 [9.9-11.8]/s); no p99 reported | bench/results/service/latency.json |
| GET /backtests/{id} | 100 req/s for 60 s, workers idle | p50 5.6 [5.3-5.7] ms, p99 10.5 [9.5-12.0] ms | bench/results/service/get.json |
| GET /backtests/{id} | 100 req/s for 60 s, 4 workers draining a backlog | p50 3167.3 [9.3-4789.3] ms, p99 6556.7 [16.8-17776.9] ms | bench/results/service/get.json |
| POST /predict | 100 req/s for 60 s, workers idle, model v1 | p50 7.7 [7.7-8.0] ms, p99 14.0 [13.2-23.0] ms, max 86.9 [83.1-109.7] ms; cold first request 46 [45-47] ms; errors 0; every 200 logged once: True | bench/results/service/predict.json |
| Kill a worker (docker kill, SIGKILL) | 4 workers, Poisson at 60% | job requeued 33.1 [30.4-33.1] s, reclaimed 38.2 [35.5-38.2] s, finished 38.4 [35.7-38.5] s after the kill. Jobs lost 0, duplicated 0, without a result 0, fencing rejections 0; killed jobs' metrics equal a direct run: True | bench/results/service/kill.json |
