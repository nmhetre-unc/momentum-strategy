# Benchmarks

Speed of the three bit-identical stationary-bootstrap implementations in
`qbt.bootstrap` (`python` reference, `numpy`, `cpp` extension), alone and
inside the full `scripts/regenerate_results.py` pipeline. Nothing here runs
in a bare `pytest` (`testpaths = ["tests"]`).

| File | What it measures | Result file |
|---|---|---|
| `micro_interleaved.py` | Per replicate, full `stationary_bootstrap_sharpe` call, batched vs per-replicate fill. Every repeat runs every implementation, the order rotates, and ratios are paired within each repeat. **Source of the micro numbers below.** | `results/micro_interleaved.json` |
| `bench_bootstrap.py` | The same, via pytest-benchmark. Each implementation's rounds run back to back, so speed drift between tests also moves the ratios. Kept as a cross-check. | `results/bootstrap_micro.json` |
| `full_run.py` | The full pipeline with every bootstrap call forced to one implementation. Each run is a fresh subprocess, the order is interleaved and rotated, and each run times its own bootstrap calls and checks its output hash. | `results/full_run.json` |
| `profile_summary.py` | One py-spy `--nonblocking` profile of a full `cpp` run. | `results/profile_cpp.json` |
| `machine.py` | CPU, OS, Python, NumPy, extension compiler, power state and git commit, recorded with every result. | |

```bash
python benchmarks/micro_interleaved.py
pytest benchmarks/bench_bootstrap.py --benchmark-only --benchmark-json=benchmarks/results/bootstrap_micro.json
python benchmarks/full_run.py --runs 3
python benchmarks/profile_summary.py --impl cpp
```

Run on AC power. On battery this laptop's CPU throttles enough that one
`python` full run beat every `numpy` run. That batch is kept as
`results/INVALID_on_battery_full_run.json`, not used.

## Results

Intel Core 7 150U (12 logical CPUs), WSL2 (Linux 6.18) on AC power,
Python 3.12.16, NumPy 2.5.4, extension built with GNU 15.2.0 at `-O3`.
n=4278 (SPY 2008-2025), mean_block=20, n_boot=2000.
Times are medians; micro rows show the IQR, full-run rows `[min-max]` of 3 runs.
Speedups are paired within each repeat, shown as median `[min-max]`.

| | python | numpy | cpp | cpp vs numpy | cpp vs python |
|---|---|---|---|---|---|
| One replicate (indices + Sharpe) | 680 µs (IQR 35) | 337 µs (IQR 16) | 36.7 µs (IQR 4.3) | 9.0× [6.7-12.3] | 18.5× [13.2-24.0] |
| `stationary_bootstrap_sharpe` call | 1306 ms (IQR 58) | 644 ms (IQR 26) | 68 ms (IQR 4) | 9.5× [8.6-10.2] | 19.4× [17.6-21.1] |
| Bootstrap time inside the full run (18 calls) | 31.8 s [29.1-35.3] | 16.3 s [14.0-17.4] | 2.35 s [2.16-2.69] | 6.5× [6.0-7.5] | 13.1× [12.4-14.7] |
| Bootstrap share of the full run | 62% [58-63] | 45% [44-47] | 11% [11-12] | | |
| **Full run, end to end** | **55.1 s [47.0-55.7]** | **36.6 s [32.0-37.1]** | **20.0 s [19.8-25.5]** | **1.6× [1.5-1.8]** | **2.4× [2.2-2.8]** |

Reading it:

- **End to end is the headline.** The bootstrap call is about 19× faster than
  the reference, but the pipeline is only about 2.4× faster: the other ~20 s
  (HMM fitting, walk-forward refits, sklearn) is untouched. Even an
  instantaneous bootstrap would cap the full run at roughly 55/20 ≈ 2.7×.
- **The full run's bootstrap gains less (13×) than the standalone call (19×).**
  The script's paired bootstrap computes two Sharpe ratios per replicate,
  and the Sharpe computation (in NumPy) didn't change. In the `cpp` micro
  numbers, index generation is about a third of a replicate and
  `values[idx]` plus `sharpe_ratio` the rest.
- **Some machine noise remains.** The non-bootstrap part of the run is identical work for every
  implementation, yet it measured 17.5-23.3 s across the 9 runs (±15%),
  which is why each speedup carries a range. Other processes averaged
  0.06-0.23 CPUs during those runs, so the noise is the CPU's own speed,
  not contention.
- **No batched mode.** The shipped one-call-per-replicate design costs
  2.2 ms per 2000 replicates (IQR 0.85) over a preallocated batched fill
  (20.2 vs 18.0 ms), about 3% of the 68 ms call. A batched fill that also
  allocates its (2000, 4278) buffer is slower (23.2 ms). Not worth giving up
  the lazy-draw guarantee for.
- **Hottest remaining code** (py-spy, `cpp` run): `GaussianHMM.fit` is 53%
  of samples inclusive. Its hand-written forward and backward recursions are
  17.1% and 8.8% self time, the two hottest functions in the run. sklearn's
  k-means inside the HMM's mixture-model initialisation is 15.4%. The
  bootstrap's remaining code is under 2% self. About 22% of py-spy samples
  errored in non-blocking mode, so treat the shares as approximate.
