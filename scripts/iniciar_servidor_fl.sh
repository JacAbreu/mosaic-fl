#!/usr/bin/env bash
# iniciar_servidor_fl.sh — Inicia o coordenador central do aprendizado federado (flower-superlink) com TLS.
#
# Variaveis de ambiente:
#   FL_TLS_CERT_DIR   Diretorio com ca.crt, server.crt, server.key  (obrigatorio)
#   FL_FLEET_API      Endereco da Fleet API (default: 0.0.0.0:9091)
#   FL_APPIO_API      Endereco da ServerApp I/O API (default: 0.0.0.0:9092)
#   FL_SUPERLINK_DB   Caminho do banco SQLite de estado (default: superlink.db)
#
# Uso:
#   FL_TLS_CERT_DIR=/certs bash scripts/iniciar_servidor_fl.sh
#   ou: make superlink
set -euo pipefail

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR nao definido. Execute: bash scripts/gerar_certs_tls.sh}"

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
