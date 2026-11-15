"""
The worker's compute subprocess: runs backtests with qbt.runner and
nothing else. No database connection and no network, so the supervisor
can kill and replace it at any moment (lease lost, timeout, shutdown)
without leaving anything half-written.

Protocol over a multiprocessing Pipe:

    child -> parent  ("ready", Handshake)            once, at startup
    parent -> child  Task                            one at a time
    child -> parent  ("ok", payload dict)            the result
                     ("invalid", message)            ValueError: bad input
                     ("error", traceback)            any other exception
    parent -> child  None                            exit

Both failure kinds are deterministic (the same input fails the same way),
so the supervisor fails the job without retrying it. A child that dies
(killed, out of memory, crashed) is different: that is retried.

The child is started with the "spawn" method, a fresh interpreter, after
the supervisor has set the BLAS/OpenMP thread variables to 1. That keeps
a worker to one core per backtest and makes floating-point reductions
run in one fixed order.
"""

import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import pandas as pd

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class Handshake:
    pid: int
    numpy_version: str
    cpp_available: bool
    cpp_unavailable_reason: str | None
    thread_settings: dict[str, str | None]


@dataclass(frozen=True)
class Task:
    frame: pd.DataFrame
    strategy: str
    params: dict[str, Any]
    cost_bps: float
    n_boot: int
    seed: int


def compute(task: Task, delay_seconds: float = 0.0) -> tuple[str, Any]:
    """Runs one backtest and returns ("ok", payload), ("invalid", msg) or ("error", tb)."""
    import numpy as np

    from qbt.runner import run

    try:
        if delay_seconds:
            time.sleep(delay_seconds)
        started = time.perf_counter()
        out = run(task.frame, task.strategy, task.params, cost_bps=task.cost_bps,
                  n_boot=task.n_boot, seed=task.seed)
        duration_ms = round((time.perf_counter() - started) * 1000)
    except ValueError as exc:
        return "invalid", str(exc)
    except Exception:
        return "error", traceback.format_exc()

    result = out.result
    return "ok", {
        "metrics": out.metrics,
        "bootstrap_impl": out.bootstrap_impl,
        "numpy_version": np.__version__,
        "duration_ms": duration_ms,
        "dates": [ts.date() for ts in pd.DatetimeIndex(result.index)],
        "equity": result["equity"].astype(float).tolist(),
        "position": result["position"].astype(float).tolist(),
    }


def child_main(conn: Connection, delay_seconds: float) -> None:
    """Entry point of the compute process."""
    import numpy as np

    import qbt.bootstrap as qb

    conn.send(("ready", Handshake(
        pid=os.getpid(), numpy_version=np.__version__, cpp_available=qb.cpp_available(),
        cpp_unavailable_reason=qb._EXT_UNAVAILABLE,
        thread_settings={name: os.environ.get(name) for name in THREAD_VARIABLES},
    )))
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):  # the supervisor is gone: exit, don't linger
            return
        if task is None:
            return
        conn.send(compute(task, delay_seconds))
