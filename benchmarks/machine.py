"""Machine and build provenance recorded with every benchmark result."""

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _power_state() -> dict:
    """AC/battery state where the kernel exposes it. On battery, laptop CPUs
    (and WSL2 guests on them) throttle enough to swamp the differences being
    measured, so results taken on battery shouldn't be trusted."""
    state: dict = {}
    root = Path("/sys/class/power_supply")
    try:
        for supply in sorted(root.iterdir()):
            for name in ("online", "status", "capacity"):
                f = supply / name
                if f.exists():
                    state[f"{supply.name}.{name}"] = f.read_text().strip()
    except OSError:
        pass
    return state


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{head}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_info() -> dict:
    from qbt import bootstrap

    ext = bootstrap._EXT
    try:
        load = os.getloadavg()
    except OSError:
        load = (float("nan"),) * 3
    return {
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "extension_available": ext is not None,
        "extension_numpy_build_version": getattr(ext, "NUMPY_BUILD_VERSION", None),
        "extension_compiler": getattr(ext, "COMPILER", None),
        "loadavg_1_5_15_at_start": list(load),
        "power": _power_state(),
        "git_commit": _git_commit(),
    }
