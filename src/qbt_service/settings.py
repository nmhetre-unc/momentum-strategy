"""
Service configuration, read from environment variables only (no config
files, so the same image runs in Compose, CI and tests unchanged).

    DATABASE_URL                  required, e.g. postgresql+psycopg://qbt:qbt@localhost:5432/qbt
    QBT_LEASE_SECONDS             job lease length (default 30)
    QBT_HEARTBEAT_SECONDS         lease renewal interval (default 10)
    QBT_REAPER_SECONDS            expired-lease sweep interval (default 15)
    QBT_POLL_SECONDS              idle worker poll interval (default 1)
    QBT_MAX_ATTEMPTS              claims before a job is failed (default 3)
    QBT_WORKER_ID                 default "<hostname>:<pid>"
    QBT_ALLOW_NUMPY_FALLBACK      let a worker start without the C++ extension (default 0)
"""

import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE | (_FALSE - {''}))}, got {raw!r}")


def _parse_positive(name: str, raw: str, kind: type[int] | type[float]) -> float:
    try:
        value = kind(raw)
    except ValueError:
        noun = "an integer" if kind is int else "a number"
        raise ValueError(f"{name} must be {noun}, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    reaper_seconds: float = 15.0
    poll_seconds: float = 1.0
    max_attempts: int = 3
    worker_id: str = field(default_factory=_default_worker_id)
    allow_numpy_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        # A lease must outlive two missed heartbeats, or a single slow
        # renewal (GC pause, busy database) would hand a healthy job to
        # another worker.
        if self.lease_seconds < 3 * self.heartbeat_seconds:
            raise ValueError(
                f"QBT_LEASE_SECONDS ({self.lease_seconds}) must be at least 3x "
                f"QBT_HEARTBEAT_SECONDS ({self.heartbeat_seconds})"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        kwargs: dict = {"database_url": env.get("DATABASE_URL", "")}
        floats = {
            "lease_seconds": "QBT_LEASE_SECONDS",
            "heartbeat_seconds": "QBT_HEARTBEAT_SECONDS",
            "reaper_seconds": "QBT_REAPER_SECONDS",
            "poll_seconds": "QBT_POLL_SECONDS",
        }
        for attr, name in floats.items():
            if name in env:
                kwargs[attr] = _parse_positive(name, env[name], float)
        if "QBT_MAX_ATTEMPTS" in env:
            kwargs["max_attempts"] = int(
                _parse_positive("QBT_MAX_ATTEMPTS", env["QBT_MAX_ATTEMPTS"], int)
            )
        if env.get("QBT_WORKER_ID"):
            kwargs["worker_id"] = env["QBT_WORKER_ID"]
        if "QBT_ALLOW_NUMPY_FALLBACK" in env:
            kwargs["allow_numpy_fallback"] = _parse_bool(
                "QBT_ALLOW_NUMPY_FALLBACK", env["QBT_ALLOW_NUMPY_FALLBACK"]
            )
        return cls(**kwargs)
