"""
Before/after measurement of two fixes, with interleaved runs:

1. Claim latency: idle workers polled every 1 s (before) vs LISTEN/NOTIFY
   with a 5 s fallback poll (after). Submit-to-start p50/p99 at 50% load.
2. The GET cliff: one API process with default CPU weight (before) vs two
   uvicorn processes with cpu_shares 4096 (after). GET /backtests/{id}
   p50/p99 at 100 req/s with 4 workers busy, and with the workers idle
   (which must not regress).

    python bench/before_after.py
    python bench/before_after.py --summarize   # rebuild the summary from the saved runs

"before" is the committed tree (HEAD): its image is built from
`git archive HEAD` and its compose.yaml from `git show HEAD:compose.yaml`,
so the working tree is never touched. "after" is the working tree. Both
run as the qbt-bench project on the same database; switching recreates
the api and worker containers.

Order: before, after, after, before (ABBA), so slow drift (heat, the
battery charging) affects both variants alike. Each measurement is one
power-guarded repetition from bench/service_bench.py (latency_once,
get_once); results go to bench/results/service/before_after.json.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import service_bench as sb  # noqa: E402
from environment import KeepAwake, environment, on_ac_power  # noqa: E402

ROOT = sb.ROOT
ORDER = ("before", "after", "after", "before")
LOAD = 0.5


def build_images() -> None:
    sb.log("building qbt-service:before from git archive HEAD")
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                             check=True, capture_output=True).stdout
    subprocess.run(["docker", "build", "-q", "-t", "qbt-service:before", "-"], input=archive,
                   check=True, capture_output=True)
    sb.log("building qbt-service:after from the working tree")
    subprocess.run(["docker", "build", "-q", "-t", "qbt-service:after", "."], cwd=ROOT,
                   check=True, capture_output=True)


def compose_file(variant: str, directory: Path) -> Path:
    """The variant's compose.yaml, using its prebuilt image instead of building."""
    if variant == "before":
        text = subprocess.run(["git", "show", "HEAD:compose.yaml"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    else:
        text = (ROOT / "compose.yaml").read_text()
    assert "image: qbt-service:dev" in text and "  build: .\n" in text
    text = text.replace("image: qbt-service:dev", f"image: qbt-service:{variant}")
    text = text.replace("  build: .\n", "")
    path = directory / f"compose.{variant}.yaml"
    path.write_text(text)
    return path


def switch(variant: str, directory: Path) -> None:
    sb.COMPOSE_FILES = ["-f", str(compose_file(variant, directory))]
    sb.log(f"switching to {variant}")
    sb.compose("up", "-d", "--wait", "--wait-timeout", "180", "--force-recreate",
               "--scale", f"worker={sb.LATENCY_WORKERS}")


def api_processes() -> int:
    """Processes serving requests in the api container: uvicorn's spawned
    workers, or uvicorn itself when it runs alone."""
    (cid,) = sb.compose("ps", "-q", "api", capture=True).split()
    out = subprocess.run(["docker", "top", cid, "-o", "pid,args"], capture_output=True,
                         text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if "spawn_main" in line) or 1


def main() -> int:
    if "--summarize" in sys.argv[1:]:
        saved = json.loads((sb.RESULTS / "before_after.json").read_text())
        return finish(saved["runs"], LOAD * sb.capacity_per_second(sb.LATENCY_WORKERS))
    if not on_ac_power():
        sys.exit("not on AC power")
    rate = LOAD * sb.capacity_per_second(sb.LATENCY_WORKERS)
    runs: list[dict] = []
    with KeepAwake(), tempfile.TemporaryDirectory() as tmp:
        build_images()
        switch(ORDER[0], Path(tmp))
        if not sb.sql("SELECT 1 FROM price_series WHERE ticker = 'SPY'"):  # same data for both
            sb.compose("run", "--rm", "ingest", sb.loadgen.TICKER, "--start", sb.loadgen.START,
                       "--end", sb.loadgen.END)
        try:
            for index, variant in enumerate(ORDER):
                switch(variant, Path(tmp))
                rep = ORDER[:index + 1].count(variant) - 1
                processes = api_processes()
                sb.log(f"{variant} #{rep + 1}: latency at {LOAD:.0%} ({rate:.1f} jobs/s)")
                latency = sb.guarded(f"{variant} latency", sb.latency_once, rep, LOAD, rate)
                sb.log(f"  start p50/p99 {latency['submit_to_start_ms_p50']:.0f}/"
                       f"{latency['submit_to_start_ms_p99']:.0f} ms, growth "
                       f"{latency['backlog_growth_jobs_per_s']:+.3f} jobs/s")
                gets = {}
                for condition in ("under_load", "idle"):
                    sb.log(f"{variant} #{rep + 1}: GET {condition}")
                    gets[condition] = sb.guarded(f"{variant} get {condition}", sb.get_once,
                                                 rep, condition)
                    sb.log(f"  p50/p99 {gets[condition]['get_ms_p50']:.1f}/"
                           f"{gets[condition]['get_ms_p99']:.1f} ms")
                runs.append({"variant": variant, "order": index + 1, "rep": rep + 1,
                             "api_processes": processes, "latency": latency, "get": gets})
                sb.checkpoint("before_after", runs, {"order": ORDER})
        except sb.PowerChanged as exc:
            sb.log(f"STOPPED: {exc}")
            return 1
    return finish(runs, rate)


def finish(runs: list[dict], rate: float) -> int:
    def values(variant: str, pick) -> list[float]:
        # A CPU sample can lack the host clock (no reading in its window).
        return [v for r in runs if r["variant"] == variant and (v := pick(r)) is not None]

    metrics = {
        "submit_to_start_ms_p50": lambda r: r["latency"]["submit_to_start_ms_p50"],
        "submit_to_start_ms_p99": lambda r: r["latency"]["submit_to_start_ms_p99"],
        "submit_to_done_ms_p50": lambda r: r["latency"]["submit_to_done_ms_p50"],
        "submit_to_done_ms_p99": lambda r: r["latency"]["submit_to_done_ms_p99"],
        "backlog_growth_jobs_per_s": lambda r: r["latency"]["backlog_growth_jobs_per_s"],
        "get_under_load_ms_p50": lambda r: r["get"]["under_load"]["get_ms_p50"],
        "get_under_load_ms_p99": lambda r: r["get"]["under_load"]["get_ms_p99"],
        "get_under_load_rate_per_s": lambda r: r["get"]["under_load"]["achieved_rate_per_s"],
        "get_idle_ms_p50": lambda r: r["get"]["idle"]["get_ms_p50"],
        "get_idle_ms_p99": lambda r: r["get"]["idle"]["get_ms_p99"],
        "host_clock_ghz_latency": lambda r: r["latency"]["cpus_busy"].get("host_clock_ghz"),
        "host_clock_ghz_get_under_load":
            lambda r: r["get"]["under_load"]["cpus_busy"].get("host_clock_ghz"),
        "api_cpus_get_under_load": lambda r: r["get"]["under_load"]["cpus_busy"].get("api"),
    }
    summary = {v: {name: sb.summarize(vs) if (vs := values(v, pick)) else None
                   for name, pick in metrics.items()}
               for v in ("before", "after")}
    data = {"measurement": "before_after", "environment": environment(),
            "config": {"order": ORDER, "latency_load": LOAD, "latency_rate_per_s": rate,
                       "workers": sb.LATENCY_WORKERS, "get_rate_per_s": sb.GET_RATE,
                       "before": "HEAD: 1 s idle poll; 1 API process, default CPU weight",
                       "after": "LISTEN/NOTIFY + 5 s fallback poll; 2 API processes, "
                                "cpu_shares 4096"},
            "summary": summary, "runs": runs}
    sb.save("before_after", data)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
