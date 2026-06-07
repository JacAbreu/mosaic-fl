#!/usr/bin/env bash
# start_supernode.sh — Inicia um flower-supernode (cliente FL) para um hospital.
#
# Variáveis de ambiente:
#   FL_TLS_CERT_DIR       Diretório com ca.crt  (obrigatório)
#   FL_CLIENT_ID          ID único deste hospital, ex: hospital_1  (obrigatório)
#   FL_DATA_SOURCE        Fonte de dados: simulated | sgbd | csv  (default: simulated)
#   FL_SUPERLINK_ADDRESS  Endereço Fleet API do SuperLink  (default: localhost:9091)
#   FL_MAX_RETRIES        Tentativas de reconexão (default: 20, vazio = infinito)
#
# Uso:
#   FL_TLS_CERT_DIR=/certs FL_CLIENT_ID=hospital_1 FL_DATA_SOURCE=sgbd \
#       ./scripts/start_supernode.sh
set -euo pipefail

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR não definido. Execute scripts/gen_certs.sh primeiro.}"
: "${FL_CLIENT_ID:?FL_CLIENT_ID não definido. Ex: FL_CLIENT_ID=hospital_1}"

FL_DATA_SOURCE="${FL_DATA_SOURCE:-simulated}"
FL_SUPERLINK_ADDRESS="${FL_SUPERLINK_ADDRESS:-localhost:9091}"
FL_MAX_RETRIES="${FL_MAX_RETRIES:-20}"

exec flower-supernode \
    --root-certificates "${FL_TLS_CERT_DIR}/ca.crt" \
    --superlink "${FL_SUPERLINK_ADDRESS}" \
    --node-config "client-id=${FL_CLIENT_ID},data-source=${FL_DATA_SOURCE}" \
    --max-retries "${FL_MAX_RETRIES}" \
    "$@"
