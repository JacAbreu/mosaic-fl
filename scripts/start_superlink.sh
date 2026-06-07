#!/usr/bin/env bash
# start_superlink.sh — Inicia o flower-superlink com TLS obrigatório.
#
# Variáveis de ambiente:
#   FL_TLS_CERT_DIR   Diretório com ca.crt, server.crt, server.key  (obrigatório)
#   FL_FLEET_API      Endereço da Fleet API (default: 0.0.0.0:9091)
#   FL_APPIO_API      Endereço da ServerApp I/O API (default: 0.0.0.0:9092)
#   FL_SUPERLINK_DB   Caminho do banco SQLite de estado (default: superlink.db)
#
# Uso:
#   FL_TLS_CERT_DIR=/certs ./scripts/start_superlink.sh
set -euo pipefail

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR não definido. Execute scripts/gen_certs.sh primeiro.}"

FL_FLEET_API="${FL_FLEET_API:-0.0.0.0:9091}"
FL_APPIO_API="${FL_APPIO_API:-0.0.0.0:9092}"
FL_SUPERLINK_DB="${FL_SUPERLINK_DB:-superlink.db}"

exec flower-superlink \
    --ssl-certfile    "${FL_TLS_CERT_DIR}/server.crt" \
    --ssl-keyfile     "${FL_TLS_CERT_DIR}/server.key" \
    --ssl-ca-certfile "${FL_TLS_CERT_DIR}/ca.crt" \
    --database        "${FL_SUPERLINK_DB}" \
    --fleet-api-address "${FL_FLEET_API}" \
    --serverappio-api-address "${FL_APPIO_API}" \
    "$@"
