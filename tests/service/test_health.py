"""qbt_service.health: the container healthcheck commands."""

import http.server
import os
import threading
import time

import pytest

from qbt_service.health import check_api, check_worker, main


def test_worker_fresh_file_is_healthy(tmp_path):
    path = tmp_path / "alive"
    path.touch()
    assert check_worker(str(path), max_age=30) is None


def test_worker_stale_file_is_unhealthy(tmp_path):
    path = tmp_path / "alive"
    path.touch()
    old = time.time() - 31
    os.utime(path, (old, old))
    assert "31s old (limit 30s)" in check_worker(str(path), max_age=30)


def test_worker_missing_or_unconfigured_file_is_unhealthy(tmp_path):
    assert "does not exist" in check_worker(str(tmp_path / "nope"), max_age=30)
    assert "not set" in check_worker(None, max_age=30)


def test_worker_main_uses_the_lease_as_the_default_limit(tmp_path, monkeypatch, capsys):
    path = tmp_path / "alive"
    path.touch()
    old = time.time() - 20
    os.utime(path, (old, old))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x@localhost/x")
    monkeypatch.setenv("QBT_HEALTH_FILE", str(path))
    monkeypatch.setenv("QBT_LEASE_SECONDS", "30")
    assert main(["worker"]) == 0
    monkeypatch.setenv("QBT_LEASE_SECONDS", "15")
    monkeypatch.setenv("QBT_HEARTBEAT_SECONDS", "5")
    assert main(["worker"]) == 1
    assert "unhealthy" in capsys.readouterr().err
    assert main(["worker", "--max-age", "60"]) == 0


@pytest.fixture
def http_server():
    """A local server answering /ok with 200 and /down with 503."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            code = 200 if self.path == "/ok" else 503
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b'{"detail": "db down"}')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_api_check(http_server):
    assert check_api(f"{http_server}/ok", timeout=3) is None
    assert "answered 503" in check_api(f"{http_server}/down", timeout=3)
    assert main(["api", "--url", f"{http_server}/ok"]) == 0
    assert main(["api", "--url", f"{http_server}/down"]) == 1


def test_api_check_unreachable():
    assert "unreachable" in check_api("http://127.0.0.1:1/health", timeout=1)
