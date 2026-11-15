# Service benchmarks

Measurements of the backtest service (`src/qbt_service`) running under
Docker Compose. These are separate from `benchmarks/`, which times the
bootstrap implementations inside one process.

| File | What it is |
|---|---|
| `service_bench.py` | The orchestrator. It runs every measurement 3 times, checks for AC power before each run, and writes JSON plus the results table. The module docstring describes each method. |
| `loadgen.py` | Async open-loop load generators: Poisson POSTs, bursts, and fixed-rate GETs. Latency is measured from each request's scheduled send time. |
| `environment.py` | Records the host CPU, power state and Windows power plan, the Docker engine, and the CPUs and memory of the VM it runs in. Every result includes this. |
| `results/service/` | `throughput.json`, `control.json`, `latency.json`, `get.json`, `kill.json` and `README.md` (the table). |

```bash
pip install -e ".[api]"                      # httpx2 and psycopg for the load generator
python bench/service_bench.py all            # about 70 minutes, on AC power
python bench/service_bench.py report         # rebuild the table from the JSON
docker compose -p qbt-bench down -v          # remove the benchmark stack
```

The benchmark stack is its own Compose project (`qbt-bench`, ports 56432
and 58100). It never touches the development stack or its data.

The load generator runs in WSL, so it shares Docker Desktop's VM and
CPUs with the service. Requests reach the API through Docker Desktop's
port forwarding.
