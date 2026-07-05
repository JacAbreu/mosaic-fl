#!/usr/bin/env bash
# iniciar_cliente_fl.sh — Inicia um no cliente do aprendizado federado (hospital) conectando ao servidor.
#
# Variaveis de ambiente:
#   FL_TLS_CERT_DIR       Diretorio com ca.crt  (obrigatorio)
#   FL_CLIENT_ID          ID unico deste hospital, ex: hospital_1  (obrigatorio)
#   FL_DATA_SOURCE        Fonte de dados: simulated | sgbd | csv  (default: simulated)
#   FL_SUPERLINK_ADDRESS  Endereco do servidor FL  (default: localhost:9091)
#   FL_MAX_RETRIES        Tentativas de reconexao (default: 20, vazio = infinito)
#   FL_CLIENTAPPIO_API    Endereco da ClientAppIo API local (default: 0.0.0.0:9094)
#   FL_LOG_FILE           Caminho do arquivo de log (default: experiments/logs/supernode_<client-id>_<timestamp>.log).
#                          flower-supernode não tem --log-file nativo — a saída é espelhada
#                          com "tee" (grava no arquivo E continua aparecendo no terminal).
#
# Uso:
#   FL_TLS_CERT_DIR=/certs FL_CLIENT_ID=hospital_1 FL_DATA_SOURCE=sgbd \
#       bash scripts/iniciar_cliente_fl.sh
#   ou: make supernode FL_CLIENT_ID=hospital_1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# flower-supernode mora no .venv do projeto — não confiar em estar no PATH do
# shell (o binário não é instalado globalmente por padrão).
FLOWER_SUPERNODE_BIN="$PROJECT_ROOT/.venv/bin/flower-supernode"
if [ ! -x "$FLOWER_SUPERNODE_BIN" ]; then
    FLOWER_SUPERNODE_BIN="flower-supernode"   # fallback: tenta o PATH mesmo assim
fi
# O SuperNode também pode disparar subprocessos internos via subprocess.Popen,
# que resolvem o executável pelo PATH herdado — mesma razão do iniciar_servidor_fl.sh.
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

: "${FL_TLS_CERT_DIR:?FL_TLS_CERT_DIR nao definido. Execute: bash scripts/gerar_certs_tls.sh}"
: "${FL_CLIENT_ID:?FL_CLIENT_ID não definido. Ex: FL_CLIENT_ID=hospital_1}"

FL_DATA_SOURCE="${FL_DATA_SOURCE:-simulated}"
FL_SUPERLINK_ADDRESS="${FL_SUPERLINK_ADDRESS:-localhost:9091}"
FL_MAX_RETRIES="${FL_MAX_RETRIES:-20}"
FL_CLIENTAPPIO_API="${FL_CLIENTAPPIO_API:-0.0.0.0:9094}"
mkdir -p "$PROJECT_ROOT/experiments/logs"
FL_LOG_FILE="${FL_LOG_FILE:-$PROJECT_ROOT/experiments/logs/supernode_${FL_CLIENT_ID}_$(date +%Y%m%d_%H%M%S).log}"

# Valida que a porta local do SuperNode (ClientAppIo API) está livre antes de
# tentar subir — mesma lógica do iniciar_servidor_fl.sh: se estiver em uso (ex.:
# tentativa anterior travada), aponta o PID em vez de deixar o erro genérico do
# próprio flower-supernode. Não mata nada automaticamente.
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
_check_port_free "$FL_CLIENTAPPIO_API" "ClientAppIo API"

echo "Log gravado em: ${FL_LOG_FILE}"

# Sem "exec" aqui de propósito — o "tee" precisa do shell vivo para gerenciar o
# pipe (grava no arquivo e mantém a saída ao vivo no terminal ao mesmo tempo).
"$FLOWER_SUPERNODE_BIN" \
    --root-certificates "${FL_TLS_CERT_DIR}/ca.crt" \
    --superlink "${FL_SUPERLINK_ADDRESS}" \
    --node-config "client-id=\"${FL_CLIENT_ID}\" data-source=\"${FL_DATA_SOURCE}\"" \
    --max-retries "${FL_MAX_RETRIES}" \
    --clientappio-api-address "${FL_CLIENTAPPIO_API}" \
    "$@" 2>&1 | tee "${FL_LOG_FILE}"
