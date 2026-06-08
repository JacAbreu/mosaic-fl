#!/usr/bin/env bash
# iniciar_cliente_fl.sh — Inicia um no cliente do aprendizado federado (hospital) conectando ao servidor.
#
# Variaveis de ambiente:
#   FL_TLS_CERT_DIR       Diretorio com ca.crt  (obrigatorio)
#   FL_CLIENT_ID          ID unico deste hospital, ex: hospital_1  (obrigatorio)
#   FL_DATA_SOURCE        Fonte de dados: simulated | sgbd | csv  (default: simulated)
#   FL_SUPERLINK_ADDRESS  Endereco do servidor FL  (default: localhost:9091)
#   FL_MAX_RETRIES        Tentativas de reconexao (default: 20, vazio = infinito)
#
# Uso:
#   FL_TLS_CERT_DIR=/certs FL_CLIENT_ID=hospital_1 FL_DATA_SOURCE=sgbd \
#       bash scripts/iniciar_cliente_fl.sh
#   ou: make supernode FL_CLIENT_ID=hospital_1
set -euo pipefail

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR nao definido. Execute: bash scripts/gerar_certs_tls.sh}"
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
