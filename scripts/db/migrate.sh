#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${FL_DB_URL:-}" ]; then
    echo "ERROR: FL_DB_URL is not set. Define it in .env or export it manually." >&2
    exit 1
fi

"$PROJECT_ROOT/.venv/bin/alembic" "$@"
