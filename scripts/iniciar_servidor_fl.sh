#!/usr/bin/env bash
# iniciar_servidor_fl.sh — Inicia o coordenador central do aprendizado federado (flower-superlink) com TLS.
#
# Variaveis de ambiente:
#   FL_TLS_CERT_DIR   Diretorio com ca.crt, server.crt, server.key  (obrigatorio)
#   FL_FLEET_API      Endereco da Fleet API — SuperNodes/clientes conectam aqui (default: 0.0.0.0:9091)
#   FL_APPIO_API      Endereco da ServerApp I/O API — uso interno do SuperLink (default: 0.0.0.0:9092)
#   FL_CONTROL_API    Endereco da Control API — "flwr run" submete rodadas aqui (default: 0.0.0.0:9093)
#   FL_SUPERLINK_DB   Caminho do banco SQLite de estado (default: superlink.db)
#   FL_LOG_FILE       Caminho do arquivo de log (default: experiments/logs/superlink_<timestamp>.log).
#                      flower-superlink grava nativamente aqui — a saída também continua no terminal.
#
# Uso:
#   FL_TLS_CERT_DIR=/certs bash scripts/iniciar_servidor_fl.sh
#   ou: make superlink
#
# Rede real (desktop + notebook): libere a porta da Fleet API (padrão 9091) no
# firewall do desktop para o SuperNode do notebook conseguir conectar — mesma
# lógica do Caminho A (README, seção "Rede Federada Real"), porta diferente.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# flower-superlink mora no .venv do projeto — não confiar em estar no PATH do
# shell (o binário não é instalado globalmente por padrão).
FLOWER_SUPERLINK_BIN="$PROJECT_ROOT/.venv/bin/flower-superlink"
if [ ! -x "$FLOWER_SUPERLINK_BIN" ]; then
    FLOWER_SUPERLINK_BIN="flower-superlink"   # fallback: tenta o PATH mesmo assim
fi
# O SuperLink dispara subprocessos internos (ex.: flower-superexec) via
# subprocess.Popen, que resolvem o executável pelo PATH herdado — só apontar
# para o binário do .venv não é suficiente, o .venv/bin precisa estar no PATH
# para esses subprocessos filhos também serem encontrados.
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR nao definido. Execute: bash scripts/gerar_certs_tls.sh}"

FL_FLEET_API="${FL_FLEET_API:-0.0.0.0:9091}"
FL_APPIO_API="${FL_APPIO_API:-0.0.0.0:9092}"
FL_CONTROL_API="${FL_CONTROL_API:-0.0.0.0:9093}"
FL_SUPERLINK_DB="${FL_SUPERLINK_DB:-superlink.db}"
mkdir -p "$PROJECT_ROOT/experiments/logs"
FL_LOG_FILE="${FL_LOG_FILE:-$PROJECT_ROOT/experiments/logs/superlink_$(date +%Y%m%d_%H%M%S).log}"

# Valida que as 3 portas estão livres antes de tentar subir — se alguma já
# estiver em uso (ex.: uma tentativa anterior que travou), o erro do próprio
# flower-superlink não indica o PID. Aqui sim, para o usuário decidir se mata
# o processo (não fazemos isso automaticamente).
_check_port_free() {
    local addr="$1" label="$2"
    local port="${addr##*:}"
    local hit
    hit="$(ss -tlnp 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print}')"
    if [ -n "$hit" ]; then
        local pid
        pid="$(echo "$hit" | grep -oP 'pid=\K[0-9]+' | head -1)"
        echo "ERRO: porta $port ($label) já está em uso." >&2
        echo "  $hit" >&2
        if [ -n "$pid" ]; then
            echo "  PID ocupando a porta: $pid" >&2
            echo "  Para investigar: ps -p $pid -o pid,cmd" >&2
            echo "  Para encerrar (se for um processo travado seu): kill $pid" >&2
        fi
        exit 1
    fi
}
_check_port_free "$FL_FLEET_API"   "Fleet API"
_check_port_free "$FL_APPIO_API"   "ServerAppIo API"
_check_port_free "$FL_CONTROL_API" "Control API"

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
echo "  Log gravado em: ${FL_LOG_FILE}"
echo "=================================================================="
echo ""

exec "$FLOWER_SUPERLINK_BIN" \
    --ssl-certfile    "${FL_TLS_CERT_DIR}/server.crt" \
    --ssl-keyfile     "${FL_TLS_CERT_DIR}/server.key" \
    --ssl-ca-certfile "${FL_TLS_CERT_DIR}/ca.crt" \
    --database        "${FL_SUPERLINK_DB}" \
    --fleet-api-address "${FL_FLEET_API}" \
    --serverappio-api-address "${FL_APPIO_API}" \
    --control-api-address "${FL_CONTROL_API}" \
    --log-file "${FL_LOG_FILE}" \
    "$@"
