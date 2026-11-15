"""qbt_service.settings: environment parsing and the lease/heartbeat invariant."""

import os
import socket

import pytest

from qbt_service.settings import Settings

URL = "postgresql+psycopg://qbt:qbt@localhost:5432/qbt"


def test_defaults():
    s = Settings.from_env({"DATABASE_URL": URL})
    assert s.database_url == URL
    assert (s.lease_seconds, s.heartbeat_seconds, s.reaper_seconds, s.poll_seconds) == (
        30.0, 10.0, 15.0, 1.0,
    )
    assert s.max_attempts == 3
    assert s.allow_numpy_fallback is False
    assert s.worker_id == f"{socket.gethostname()}:{os.getpid()}"


def test_overrides():
    s = Settings.from_env({
        "DATABASE_URL": URL,
        "QBT_LEASE_SECONDS": "6",
        "QBT_HEARTBEAT_SECONDS": "2",
        "QBT_REAPER_SECONDS": "3",
        "QBT_POLL_SECONDS": "0.2",
        "QBT_MAX_ATTEMPTS": "5",
        "QBT_WORKER_ID": "w1",
        "QBT_ALLOW_NUMPY_FALLBACK": "yes",
    })
    assert (s.lease_seconds, s.heartbeat_seconds, s.reaper_seconds, s.poll_seconds) == (
        6.0, 2.0, 3.0, 0.2,
    )
    assert s.max_attempts == 5 and isinstance(s.max_attempts, int)
    assert s.worker_id == "w1"
    assert s.allow_numpy_fallback is True


def test_worker_settings():
    s = Settings.from_env({"DATABASE_URL": URL})
    assert (s.retry_backoff_seconds, s.retry_backoff_max_seconds) == (5.0, 300.0)
    assert (s.job_timeout_seconds, s.shutdown_grace_seconds) == (600.0, 8.0)
    assert (s.statement_timeout_seconds, s.test_compute_delay_seconds) == (60.0, 0.0)
    s = Settings.from_env({
        "DATABASE_URL": URL, "QBT_RETRY_BACKOFF_SECONDS": "0",
        "QBT_RETRY_BACKOFF_MAX_SECONDS": "60", "QBT_JOB_TIMEOUT_SECONDS": "30",
        "QBT_SHUTDOWN_GRACE_SECONDS": "0", "QBT_STATEMENT_TIMEOUT_SECONDS": "5",
        "QBT_TEST_COMPUTE_DELAY_SECONDS": "2.5",
    })
    assert (s.retry_backoff_seconds, s.retry_backoff_max_seconds) == (0.0, 60.0)
    assert (s.job_timeout_seconds, s.shutdown_grace_seconds) == (30.0, 0.0)
    assert (s.statement_timeout_seconds, s.test_compute_delay_seconds) == (5.0, 2.5)


def test_health_file():
    assert Settings.from_env({"DATABASE_URL": URL}).health_file is None
    assert Settings.from_env({"DATABASE_URL": URL, "QBT_HEALTH_FILE": ""}).health_file is None
    s = Settings.from_env({"DATABASE_URL": URL, "QBT_HEALTH_FILE": "/tmp/alive"})
    assert s.health_file == "/tmp/alive"


def test_empty_worker_id_falls_back_to_default():
    assert Settings.from_env({"DATABASE_URL": URL, "QBT_WORKER_ID": ""}).worker_id


def test_database_url_is_required():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings.from_env({})


def test_lease_must_outlive_two_missed_heartbeats():
    with pytest.raises(ValueError, match="at least 3x"):
        Settings.from_env({"DATABASE_URL": URL, "QBT_LEASE_SECONDS": "29"})
    Settings.from_env({"DATABASE_URL": URL, "QBT_LEASE_SECONDS": "30"})


@pytest.mark.parametrize(
    ("name", "raw", "message"),
    [
        ("QBT_LEASE_SECONDS", "soon", "must be a number"),
        ("QBT_POLL_SECONDS", "0", "must be > 0"),
        ("QBT_MAX_ATTEMPTS", "2.5", "must be an integer"),
        ("QBT_MAX_ATTEMPTS", "-1", "must be > 0"),
        ("QBT_ALLOW_NUMPY_FALLBACK", "maybe", "must be one of"),
        ("QBT_SHUTDOWN_GRACE_SECONDS", "-1", "must be >= 0"),
        ("QBT_RETRY_BACKOFF_SECONDS", "nan", "must be >= 0"),
        ("QBT_TEST_COMPUTE_DELAY_SECONDS", "later", "must be a number"),
        ("QBT_JOB_TIMEOUT_SECONDS", "0", "must be > 0"),
    ],
)
def test_bad_values(name, raw, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env({"DATABASE_URL": URL, name: raw})
