"""
Removes database passwords from text before it is printed or saved.

bench/aws_bench.py passes everything that comes back from the instance
(command output, and the stderr inside a failed command's error) through
scrub(). The case that leaked once in D2: psycopg given a SQLAlchemy URL
raised

    psycopg.ProgrammingError: invalid connection option
    "postgresql+psycopg://qbt:<password>@host:5432/qbt?sslmode"

so the full DATABASE_URL, password included, ended up in the error text.
"""

import re

MASK = "***"

# scheme://user:password@ -- the password is everything up to the last "@"
# before the host, so a password containing "@" or ":" is still covered.
_URL_PASSWORD = re.compile(
    r"(?P<head>\b[a-z][a-z0-9+.-]*://[^:/@\s\"']*:)(?P<password>[^\s\"']*)@",
    re.IGNORECASE,
)
# libpq key/value conninfo: password=secret or password='se cret'
_CONNINFO_PASSWORD = re.compile(r"(?P<head>\bpassword\s*=\s*)(?P<password>'(?:\\.|[^'])*'|\S+)",
                                re.IGNORECASE)


def scrub(text: str, *secrets: str) -> str:
    """`text` with URL and conninfo passwords, and each of `secrets`, replaced by ***."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, MASK)
    text = _URL_PASSWORD.sub(lambda m: f"{m['head']}{MASK}@", text)
    return _CONNINFO_PASSWORD.sub(lambda m: f"{m['head']}{MASK}", text)
