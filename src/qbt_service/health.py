"""
Container healthchecks. Exit status 0 is healthy, 1 unhealthy, with the
reason on stderr (Docker keeps it in `docker inspect`).

    python -m qbt_service.health api [--url URL]
    python -m qbt_service.health worker [--max-age SECONDS]

api: GET /health, which answers 200 only if the database does.

worker: the age of QBT_HEALTH_FILE, which the worker touches after each
successful claim attempt (at least every QBT_POLL_SECONDS when idle) and each lease
heartbeat (every QBT_HEARTBEAT_SECONDS while running a job). Both need
the database, so a fresh file means the loop is turning and the database
answers. The default limit is the lease length: a worker silent for that
long could not have kept a job anyway.

Standard library only (plus Settings), so a check costs a few ms of
interpreter start-up, not a pandas import.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

from qbt_service.settings import Settings

DEFAULT_API_URL = "http://127.0.0.1:8000/health"


def check_api(url: str, timeout: float) -> str | None:
    """None if healthy, else the reason."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            if response.status == 200:
                return None
            return f"{url} answered {response.status}"
    except urllib.error.HTTPError as exc:
        return f"{url} answered {exc.code}: {exc.read(500).decode(errors='replace')}"
    except (urllib.error.URLError, OSError) as exc:
        return f"{url} unreachable: {exc}"


def check_worker(path: str | None, max_age: float, now: float | None = None) -> str | None:
    """None if healthy, else the reason."""
    if not path:
        return "QBT_HEALTH_FILE is not set"
    try:
        modified = os.stat(path).st_mtime
    except FileNotFoundError:
        return f"{path} does not exist yet (worker starting, or never reached the database)"
    age = (time.time() if now is None else now) - modified
    if age > max_age:
        return f"{path} is {age:.0f}s old (limit {max_age:.0f}s): worker stuck or database down"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qbt_service.health", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="role", required=True)
    api = sub.add_parser("api")
    api.add_argument("--url", default=DEFAULT_API_URL)
    api.add_argument("--timeout", type=float, default=3.0)
    worker = sub.add_parser("worker")
    worker.add_argument("--max-age", type=float, default=None,
                        help="seconds (default: QBT_LEASE_SECONDS)")
    args = parser.parse_args(argv)

    if args.role == "api":
        problem = check_api(args.url, args.timeout)
    else:
        settings = Settings.from_env()
        max_age = settings.lease_seconds if args.max_age is None else args.max_age
        problem = check_worker(settings.health_file, max_age)
    if problem is not None:
        print(f"unhealthy: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
