"""
Profiles one full scripts/regenerate_results.py run (impl forced, default
cpp) with py-spy in non-blocking mode, then writes the aggregated result
as JSON: inclusive share for every project function (qbt/ and the script),
and self (leaf-frame) share for the top functions overall.

    python benchmarks/profile_summary.py --impl cpp --out benchmarks/results/profile_cpp.json

Requires py-spy on PATH. Non-blocking mode samples without pausing the
process; blocking mode slowed this run 2.5x and skewed the shares.
"""

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAME = re.compile(r"(?P<name>.*?) \((?P<file>.*?)(?::(?P<line>\d+))?\)$")


def normalize(frame: str) -> str:
    m = FRAME.match(frame)
    if not m:
        return frame
    file = re.sub(r".*/site-packages/", "", m["file"])
    file = re.sub(r".*/momentum-strategy/", "", file)
    return f"{m['name']} ({file})"


def aggregate(raw: Path, top: int) -> dict:
    inclusive: collections.Counter = collections.Counter()
    self_time: collections.Counter = collections.Counter()
    total = 0
    for line in raw.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        n = int(count)
        total += n
        frames = [normalize(f) for f in stack.split(";")]
        for f in set(frames):
            inclusive[f] += n
        self_time[frames[-1]] += n

    def project(f: str) -> bool:
        return "qbt/" in f or "regenerate_results" in f

    return {
        "total_samples": total,
        "inclusive_project": [{"function": f, "share": c / total}
                              for f, c in inclusive.most_common() if project(f)],
        "self_top": [{"function": f, "share": c / total} for f, c in self_time.most_common(top)],
    }


def main(impl: str, out: Path, rate: int, top: int) -> None:
    if shutil.which("py-spy") is None:
        sys.exit("py-spy not found on PATH")
    version = subprocess.run(["py-spy", "--version"], capture_output=True, text=True).stdout
    info = machine.machine_info()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "profile.raw"
        proc = subprocess.run(
            ["py-spy", "record", "--nonblocking", "-r", str(rate), "-f", "raw", "-o", str(raw),
             "--", sys.executable, str(HERE / "full_run.py"), "--child", impl, tmp],
            capture_output=True, text=True, check=True,
        )
        # py-spy writes its own "py-spy> ..." status lines to the same stdout.
        run = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
        samples = re.search(r"Samples: (\d+) Errors: (\d+)", proc.stdout + proc.stderr)
        doc = {
            "benchmark": f"py-spy --nonblocking profile of one full run, impl={impl}",
            "py_spy": version.strip(),
            "rate_hz": rate,
            "samples_reported": int(samples[1]) if samples else None,
            "sample_errors": int(samples[2]) if samples else None,
            "run": run,
            "machine": info,
            **aggregate(raw, top),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"run main {run['main_s']:.2f}s, bootstrap {run['bootstrap_s']:.2f}s "
          f"({run['bootstrap_share_of_main']:.1%}); samples {doc['total_samples']}, "
          f"errors {doc['sample_errors']}")
    print("inclusive (project):")
    for row in doc["inclusive_project"][:25]:
        print(f"  {row['share']:6.1%}  {row['function']}")
    print("self (leaf):")
    for row in doc["self_top"][:15]:
        print(f"  {row['share']:6.1%}  {row['function']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="cpp", choices=("python", "numpy", "cpp"))
    ap.add_argument("--out", type=Path, default=HERE / "results/profile_cpp.json")
    ap.add_argument("--rate", type=int, default=100)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    main(args.impl, args.out, args.rate, args.top)
