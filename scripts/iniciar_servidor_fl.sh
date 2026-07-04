#!/usr/bin/env bash
# iniciar_servidor_fl.sh — Inicia o coordenador central do aprendizado federado (flower-superlink) com TLS.
#
# Variaveis de ambiente:
#   FL_TLS_CERT_DIR   Diretorio com ca.crt, server.crt, server.key  (obrigatorio)
#   FL_FLEET_API      Endereco da Fleet API — SuperNodes/clientes conectam aqui (default: 0.0.0.0:9091)
#   FL_APPIO_API      Endereco da ServerApp I/O API — uso interno do SuperLink (default: 0.0.0.0:9092)
#   FL_CONTROL_API    Endereco da Control API — "flwr run" submete rodadas aqui (default: 0.0.0.0:9093)
#   FL_SUPERLINK_DB   Caminho do banco SQLite de estado (default: superlink.db)
#
# Uso:
#   FL_TLS_CERT_DIR=/certs bash scripts/iniciar_servidor_fl.sh
#   ou: make superlink
#
# Rede real (desktop + notebook): libere a porta da Fleet API (padrão 9091) no
# firewall do desktop para o SuperNode do notebook conseguir conectar — mesma
# lógica do Caminho A (README, seção "Rede Federada Real"), porta diferente.
set -euo pipefail

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR nao definido. Execute: bash scripts/gerar_certs_tls.sh}"

FL_FLEET_API="${FL_FLEET_API:-0.0.0.0:9091}"
FL_APPIO_API="${FL_APPIO_API:-0.0.0.0:9092}"
FL_CONTROL_API="${FL_CONTROL_API:-0.0.0.0:9093}"
FL_SUPERLINK_DB="${FL_SUPERLINK_DB:-superlink.db}"

LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "=================================================================="
echo "  SuperLink iniciando — Fleet API em ${FL_FLEET_API}"
echo "=================================================================="
if [ -n "${LOCAL_IP:-}" ]; then
    echo "  IP deste desktop na rede local: ${LOCAL_IP}"
    echo ""
    echo "  No notebook, use este IP para o SuperNode conectar:"
    echo "    make supernode FL_CLIENT_ID=HSL FL_SUPERLINK_ADDRESS=${LOCAL_IP}:9091"
    echo ""
    echo "  Certifique-se de que o certificado (scripts/gerar_certs_tls.sh) foi"
    echo "  gerado com este IP como segundo argumento, ex.:"
    echo "    bash scripts/gerar_certs_tls.sh certs ${LOCAL_IP}"
fi
echo "=================================================================="
echo ""

exec flower-superlink \
    --ssl-certfile    "${FL_TLS_CERT_DIR}/server.crt" \
    --ssl-keyfile     "${FL_TLS_CERT_DIR}/server.key" \
    --ssl-ca-certfile "${FL_TLS_CERT_DIR}/ca.crt" \
    --database        "${FL_SUPERLINK_DB}" \
    --fleet-api-address "${FL_FLEET_API}" \
    --serverappio-api-address "${FL_APPIO_API}" \
    --control-api-address "${FL_CONTROL_API}" \
    "$@"
