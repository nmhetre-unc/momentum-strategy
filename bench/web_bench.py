"""
Measures the dashboard's production build as served by nginx in the Compose
stack, and writes bench/results/web/<name>.json:

    .venv/bin/python bench/web_bench.py                      everything -> web.json
    .venv/bin/python bench/web_bench.py --only bundle --name bundle_before_split

1. Builds every image fresh and starts its own Compose project (qbt-web-bench,
   via web/e2e/stack.mjs: up --build, ingest SPY, train), then submits one
   sma_crossover backtest with the dashboard's defaults and waits for it.
2. Bundle: every file under /assets/ fetched through nginx with
   Accept-Encoding: gzip, and its size on the wire (nginx gzip level 6, as
   served). "initial" is what index.html loads (scripts, modulepreloads,
   stylesheets); "lazy" is every other chunk, loaded when a page needs it.
3. Browser: web/scripts/measure-web.mjs in the Playwright image on the
   project's network: Lighthouse performance (3 runs each for the results
   page and the Run page, median and range) and time to first chart on the
   results page (10 cold runs after one unrecorded warm-up).
4. Removes the project (down -v).

Power guard (as bench/service_bench.py): runs only on AC power, keeps
Windows awake, and discards and repeats a measurement during which the
machine slept, woke or changed power source. The result records the
machine (bench/environment.py), the web image id and the git commit.
"""

import argparse
import gzip
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from environment import KeepAwake, environment, image_id, on_ac_power  # noqa: E402
from service_bench import PowerChanged, guarded, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RESULTS = ROOT / "bench" / "results" / "web"
PROJECT = "qbt-web-bench"
PORTS = {"QBT_DB_PORT": "56532", "QBT_API_PORT": "58200", "QBT_WEB_PORT": "58280"}
ENV = os.environ | PORTS | {"QBT_E2E_PROJECT": PROJECT}
WEB_URL = f"http://127.0.0.1:{PORTS['QBT_WEB_PORT']}"
API_URL = f"{WEB_URL}/api"
PLAYWRIGHT_IMAGE = ("mcr.microsoft.com/playwright:v1.64.1-noble@sha256:"
                    "2f05f8a734509a7b3f26a912639488ee08f58a7b292d14d62271bcff9038768f")
# The Run page's defaults (web/src/features/run/form.ts), so the results page
# measured is the one a first-time user sees.
BACKTEST = {"strategy": "sma_crossover", "ticker": "SPY", "start": "2015-01-01",
            "end": "2025-01-01", "cost_bps": 5, "n_boot": 2000, "seed": 0,
            "params": {"short_window": 50, "long_window": 200}}


def node(*args: str) -> None:
    command = f"source ~/.nvm/nvm.sh >/dev/null && cd {WEB} && node {' '.join(args)}"
    subprocess.run(["bash", "-c", command], env=ENV, check=True)


def http(method: str, url: str, body: dict | None = None,
         headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={
        **({"Content-Type": "application/json"} if data else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def submit_backtest() -> str:
    status, _, raw = http("POST", f"{API_URL}/backtests", BACKTEST)
    if status != 202:
        raise RuntimeError(f"POST /backtests: {status} {raw[:300]!r}")
    job_id = json.loads(raw)["id"]
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = json.loads(http("GET", f"{API_URL}/backtests/{job_id}")[2])
        if job["status"] == "succeeded":
            return job_id
        if job["status"] == "failed":
            raise RuntimeError(f"backtest failed: {job['last_error']}")
        time.sleep(0.5)
    raise RuntimeError("backtest did not finish in 300 s")


def asset_files() -> list[str]:
    out = subprocess.run(
        ["docker", "compose", "-p", PROJECT, "exec", "-T", "web",
         "ls", "/usr/share/nginx/html/assets"],
        env=ENV, capture_output=True, text=True, check=True).stdout
    return sorted(out.split())


def bundle() -> dict:
    """Every asset's gzipped size as nginx serves it, split into initial and lazy."""
    status, headers, raw = http("GET", f"{WEB_URL}/", headers={"Accept-Encoding": "gzip"})
    html = gzip.decompress(raw) if headers.get("Content-Encoding") == "gzip" else raw
    initial = set(re.findall(r'(?:src|href)="/assets/([^"]+)"', html.decode()))
    files = {}
    for name in asset_files():
        status, headers, raw = http("GET", f"{WEB_URL}/assets/{name}",
                                    headers={"Accept-Encoding": "gzip"})
        if status != 200:
            raise RuntimeError(f"/assets/{name}: {status}")
        files[name] = {
            "gzip_bytes": len(raw),  # urllib doesn't decompress: bytes on the wire
            "encoding": headers.get("Content-Encoding", "identity"),
            "raw_bytes": len(gzip.decompress(raw)) if headers.get("Content-Encoding") == "gzip"
            else len(raw),
            "initial": name in initial,
        }
    missing = initial - files.keys()
    if missing:
        raise RuntimeError(f"index.html loads assets nginx doesn't have: {missing}")

    def total(initial_flag: bool, kind: str) -> int:
        return sum(f[kind] for f in files.values() if f["initial"] == initial_flag)

    return {
        "initial_gzip_bytes": total(True, "gzip_bytes"),
        "initial_raw_bytes": total(True, "raw_bytes"),
        "lazy_gzip_bytes": total(False, "gzip_bytes"),
        "lazy_raw_bytes": total(False, "raw_bytes"),
        "files": files,
    }


def browser(job_id: str) -> dict:
    network = f"{PROJECT}_default"
    uid = f"{os.getuid()}:{os.getgid()}"
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", network, "--user", uid, "--ipc=host",
         "-e", "HOME=/tmp", "-v", f"{WEB}:/work", "-w", "/work", PLAYWRIGHT_IMAGE,
         "node", "scripts/measure-web.mjs", "--base", "http://web:8080", "--job", job_id,
         "--lighthouse-runs", "3", "--chart-runs", "10"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def summary(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=["bundle", "all"], default="all")
    parser.add_argument("--name", default="web", help="results file name (no .json)")
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    args = parser.parse_args()

    if not on_ac_power():
        sys.exit("not on AC power: measurements would not be comparable; plug in and rerun")

    with KeepAwake():
        log("stack: build fresh, start, ingest, train")
        node("e2e/stack.mjs", "up")
        try:
            env = environment()
            env["web_image"] = image_id("qbt-web:dev")
            result: dict = {"measurement": args.name, "environment": env,
                            "config": {"project": PROJECT, "nginx_gzip_level": 6,
                                       "backtest": BACKTEST}}
            result["bundle"] = guarded("bundle", bundle)
            log(f"bundle: initial {result['bundle']['initial_gzip_bytes']} B gzipped, "
                f"lazy {result['bundle']['lazy_gzip_bytes']} B")
            if args.only == "all":
                job_id = submit_backtest()
                result["config"]["job_id"] = job_id
                measured = guarded("browser", browser, job_id)
                result["browser"] = measured
                result["summary"] = {
                    "lighthouse_performance": {
                        page: summary([r["score"] for r in runs])
                        for page, runs in measured["lighthouse"].items()},
                    "first_chart_ms": summary(measured["first_chart_ms"]),
                    "initial_gzip_kb": round(result["bundle"]["initial_gzip_bytes"] / 1000, 1),
                    "lazy_gzip_kb": round(result["bundle"]["lazy_gzip_bytes"] / 1000, 1),
                }
                log(json.dumps(result["summary"]))
            RESULTS.mkdir(parents=True, exist_ok=True)
            path = RESULTS / f"{args.name}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            log(f"wrote {path.relative_to(ROOT)}")
        except PowerChanged as error:
            sys.exit(f"stopped: {error}")
        finally:
            if not args.keep:
                node("e2e/stack.mjs", "down")


if __name__ == "__main__":
    main()
