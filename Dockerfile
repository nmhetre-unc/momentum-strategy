# The backtest service image: API, worker, migrations and ingest, one image.
#
#   docker build -t qbt-service .        (or: docker compose build)
#
# build:   compiles the C++ bootstrap extension (QBT_REQUIRE_EXTENSION=1:
#          no compiler or a failed compile fails the build instead of
#          silently shipping the 9x slower NumPy fallback) and installs the
#          service's dependencies into a venv.
# runtime: the same Python base without a compiler; just the venv and
#          alembic.ini, run as a non-root user.
#
# Both stages use the same base image, so the venv's interpreter symlinks
# and the extension's ABI match. Bump the tag and digest together.
ARG PYTHON_IMAGE=python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv

WORKDIR /src

# Dependencies first, from pyproject.toml alone, so editing code doesn't
# reinstall them: [project.dependencies] plus the [api] extra, minus httpx2
# (only the tests and bench/ use it). The dashboard's packages live in the
# [ui] extra, which the image never installs. The list is derived here
# rather than copied, so a new dependency reaches the image automatically;
# the assert catches a rename of the excluded one.
COPY pyproject.toml ./
RUN python - > /tmp/service-requirements.txt <<'EOF'
import re
import tomllib

project = tomllib.load(open("pyproject.toml", "rb"))["project"]
not_in_image = {"httpx2"}
requirements = project["dependencies"] + project["optional-dependencies"]["api"]
names = {re.match(r"[A-Za-z0-9._-]+", r).group(0).lower(): r for r in requirements}
missing = not_in_image - names.keys()
assert not missing, f"excluded but not in pyproject.toml any more: {missing}"
for name, requirement in names.items():
    if name not in not_in_image:
        print(requirement)
EOF
RUN cat /tmp/service-requirements.txt \
    && pip install -r /tmp/service-requirements.txt

# Then qbt itself. pip builds the wheel in an isolated environment from
# [build-system].requires, i.e. against numpy==2.5.4, the same pin the
# venv just installed; qbt.bootstrap refuses the extension on any mismatch.
COPY CMakeLists.txt ./
COPY src ./src
RUN QBT_REQUIRE_EXTENSION=1 pip wheel --no-deps --wheel-dir /wheels . \
    && pip install --no-deps /wheels/qbt-*.whl \
    && python -c "import qbt.bootstrap as b; assert b.cpp_available(), b._EXT_UNAVAILABLE" \
    && pip check \
    && pip uninstall -y pip \
    && find /opt/venv -depth -type d -name tests -path "*/site-packages/*" -exec rm -rf {} +
# (Last line: the dependencies' own test suites, about 105 MB that nothing
# imports at run time. The runtime stage's check and the smoke test cover it.)

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# A fixed uid/gid, so volume permissions don't depend on the build host.
RUN groupadd --system --gid 10001 qbt \
    && useradd --system --uid 10001 --gid qbt --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin qbt

COPY --from=build /opt/venv /opt/venv
COPY alembic.ini /app/alembic.ini

# XDG_CACHE_HOME: yfinance (ingest) keeps a timezone cache under the user
# cache dir; /tmp is writable even when the root filesystem is read-only.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app
USER qbt

# Fail the build unless the image can do what the worker's startup check
# demands: the extension loads, under the numpy it was built against.
RUN python - <<'EOF'
import importlib.metadata
import importlib.util

import numpy

import qbt.bootstrap as b
from qbt import _bootstrap_ext

pinned = next(r.split("==")[1] for r in importlib.metadata.requires("qbt")
              if r.startswith("numpy=="))
assert b.cpp_available(), b._EXT_UNAVAILABLE
assert _bootstrap_ext.NUMPY_BUILD_VERSION == numpy.__version__ == pinned, (
    _bootstrap_ext.NUMPY_BUILD_VERSION, numpy.__version__, pinned)
for name in ("streamlit", "altair", "matplotlib"):
    assert importlib.util.find_spec(name) is None, f"{name} should not be in the image"
import qbt_service.api, qbt_service.health, qbt_service.ingest, qbt_service.worker  # noqa
print("ok: C++ extension built against numpy", _bootstrap_ext.NUMPY_BUILD_VERSION)
EOF

EXPOSE 8000
# The API by default; compose.yaml runs the worker, migrations and ingest
# from the same image with other commands.
CMD ["uvicorn", "--factory", "qbt_service.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
