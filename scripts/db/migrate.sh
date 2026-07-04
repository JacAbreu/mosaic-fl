#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Preserva FL_DB_URL já exportado no shell (ex.: para migrar um banco diferente
# do padrão, como um segundo Postgres do servidor num cenário multi-máquina) —
# .env só preenche o que ainda não estiver definido, nunca sobrescreve.
_FL_DB_URL_PRE="${FL_DB_URL:-}"
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
if [ -n "$_FL_DB_URL_PRE" ]; then
    FL_DB_URL="$_FL_DB_URL_PRE"
fi

if [ -z "${FL_DB_URL:-}" ]; then
    echo "ERROR: FL_DB_URL is not set. Define it in .env or export it manually." >&2
    exit 1
fi

"$PROJECT_ROOT/.venv/bin/alembic" "$@"
