"""bench/scrub.py: the AWS benchmark's output never carries the database password."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "bench_scrub", Path(__file__).resolve().parents[1] / "bench" / "scrub.py")
assert _spec is not None and _spec.loader is not None
scrub_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrub_module)
scrub = scrub_module.scrub

PASSWORD = "Xq7pLm2Rt9vB4nK8sW1cZ6yH3jD5fG0aE2uI4oP7"  # 40 letters and digits, like infra's
HOST = "qbt-db.example.us-east-2.rds.amazonaws.com"
DATABASE_URL = (f"postgresql+psycopg://qbt:{PASSWORD}@{HOST}:5432/qbt"
                "?sslmode=verify-full&sslrootcert=/etc/ssl/rds/global-bundle.pem")


def test_database_url():
    out = scrub(DATABASE_URL)
    assert PASSWORD not in out
    assert out == (f"postgresql+psycopg://qbt:***@{HOST}:5432/qbt"
                   "?sslmode=verify-full&sslrootcert=/etc/ssl/rds/global-bundle.pem")


def test_env_file_line():
    assert PASSWORD not in scrub(f"DATABASE_URL={DATABASE_URL}\n")


def test_psycopg_error_message():
    # The message psycopg raised in D2 when handed the SQLAlchemy URL.
    error = ('Traceback (most recent call last):\n'
             '  File "<string>", line 9, in <module>\n'
             'psycopg.ProgrammingError: invalid connection option '
             f'"{DATABASE_URL.split("=")[0]}"\n\nfailed to run commands: exit status 1')
    out = scrub(error)
    assert PASSWORD not in out
    assert f'"postgresql+psycopg://qbt:***@{HOST}:5432/qbt?sslmode"' in out


def test_psycopg_operational_error_with_conninfo():
    error = (f"psycopg.OperationalError: connection failed: host={HOST} user=qbt "
             f"password={PASSWORD} dbname=qbt sslmode=verify-full")
    out = scrub(error)
    assert PASSWORD not in out
    assert "password=*** dbname=qbt" in out


def test_quoted_conninfo_password():
    assert "s3cr et" not in scrub("host=h password='s3cr et' dbname=qbt")


@pytest.mark.parametrize("password", ["p@ss:w0rd", "a:b", "x@y@z"])
def test_password_with_url_delimiters(password):
    out = scrub(f"postgresql://qbt:{password}@{HOST}/qbt")
    assert password not in out
    assert out == f"postgresql://qbt:***@{HOST}/qbt"


def test_explicit_secret_anywhere():
    # A bare password (say, echoed by some tool without its URL) is removed when passed in.
    assert scrub(f"token {PASSWORD} end", PASSWORD) == "token *** end"


def test_leaves_other_text_alone():
    text = ("connected: ssl_in_use True | libpq sslmode verify-full | host " + HOST + "\n"
            "http://127.0.0.1:8000/backtests/abc and https://example.com/a:b")
    assert scrub(text) == text
