"""The worker's container healthcheck: a shell check of the heartbeat file's age.

compose.yaml and the instance's compose file (infra/templates/compose.yaml.tftpl)
carry the same command. It is run here with /bin/sh, the way Docker runs a
CMD-SHELL check, against a fresh, a stale and a missing file.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qbt_service.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = [ROOT / "compose.yaml", ROOT / "infra" / "templates" / "compose.yaml.tftpl"]


def worker_check(path: Path) -> list[str]:
    """The healthcheck `test:` of the service that sets QBT_HEALTH_FILE, as written."""
    tests = [json.loads(m) for m in re.findall(r'^\s*test: (\[.*\])\s*$', path.read_text(), re.M)]
    worker = [t for t in tests if "QBT_HEALTH_FILE" in t[-1]]
    assert len(worker) == 1, f"{path}: expected one worker healthcheck, found {len(worker)}"
    return worker[0]


def as_docker_runs_it(check: list[str]) -> str:
    # Compose turns $$ into a literal $; Terraform's templatefile leaves "$$"
    # alone and only interpolates "${", which the command must not contain.
    assert check[0] == "CMD-SHELL"
    assert "${" not in check[1]
    return check[1].replace("$$", "$")


def test_both_compose_files_carry_the_same_check():
    first, second = (worker_check(p) for p in COMPOSE_FILES)
    assert first == second


def test_limit_is_the_lease_length():
    command = as_docker_runs_it(worker_check(COMPOSE_FILES[0]))
    (limit,) = {int(n) for n in re.findall(r"-le (\d+)", command)}
    assert limit == Settings(database_url="unused").lease_seconds


def test_no_python_in_the_check():
    # The point of the shell check: no interpreter start-up (D2's timeouts).
    assert "python" not in worker_check(COMPOSE_FILES[0])[1]


needs_gnu_sh = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("stat") is None,
    reason="the check uses GNU stat (-c %Y), as in the Debian-based image")


def run_check(env_file: str | None) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"]}
    if env_file is not None:
        env["QBT_HEALTH_FILE"] = env_file
    command = as_docker_runs_it(worker_check(COMPOSE_FILES[0]))
    return subprocess.run(["/bin/sh", "-c", command], env=env, capture_output=True, text=True,
                          timeout=10)


@needs_gnu_sh
def test_fresh_file_is_healthy(tmp_path):
    f = tmp_path / "worker-alive"
    f.touch()
    result = run_check(str(f))
    assert result.returncode == 0, result.stderr


@needs_gnu_sh
def test_file_at_the_limit_is_healthy(tmp_path):
    f = tmp_path / "worker-alive"
    f.touch()
    t = time.time() - 29
    os.utime(f, (t, t))
    assert run_check(str(f)).returncode == 0


@needs_gnu_sh
def test_stale_file_is_unhealthy(tmp_path):
    f = tmp_path / "worker-alive"
    f.touch()
    t = time.time() - 45
    os.utime(f, (t, t))
    result = run_check(str(f))
    assert result.returncode == 1
    assert "s old (limit 30 s = lease)" in result.stderr


@needs_gnu_sh
def test_missing_file_is_unhealthy(tmp_path):
    result = run_check(str(tmp_path / "never-written"))
    assert result.returncode == 1
    assert "no heartbeat file at" in result.stderr


@needs_gnu_sh
def test_unset_variable_is_unhealthy():
    result = run_check(None)
    assert result.returncode == 1
    assert "no heartbeat file at" in result.stderr
