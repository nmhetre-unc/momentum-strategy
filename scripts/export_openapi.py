"""
Writes the API's OpenAPI schema as JSON, without a database or a server.

    python scripts/export_openapi.py                 # -> web/openapi.json
    python scripts/export_openapi.py --out -         # -> stdout

create_app() only connects to the database in its lifespan (on startup),
so building the app and asking for its schema needs no DATABASE_URL. The
URL passed below is never used. The web client is generated from this file
(web/README.md, "The API client"); `npm run api:check` in web/ fails when
the file or the generated types fall behind the API.
"""

import argparse
import json
import sys
from pathlib import Path

from qbt_service.api import create_app
from qbt_service.settings import Settings

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "web" / "openapi.json"


def schema_json() -> str:
    app = create_app(Settings(database_url="postgresql+psycopg://unused@localhost/unused"))
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="file to write, or - for stdout")
    args = parser.parse_args()
    text = schema_json()
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
