#!/bin/bash
# gerar_certs_tls.sh — Gera certificados TLS autoassinados para desenvolvimento/rede local
#
# Uso:
#   bash scripts/gerar_certs_tls.sh              # gera em ./certs/
#   bash scripts/gerar_certs_tls.sh /tmp/mycerts # gera no diretorio especificado
#
# Saída:
#   $OUT_DIR/ca.crt        ← CA raiz (distribuir para todos os clientes)
#   $OUT_DIR/ca.key        ← chave privada da CA (guardar em segurança, nunca distribuir)
#   $OUT_DIR/server.crt    ← certificado do servidor (apenas o servidor)
#   $OUT_DIR/server.key    ← chave privada do servidor (apenas o servidor)
#
# Uso com o MOSAIC-FL:
#   export FL_TLS_CERT_DIR="$OUT_DIR"
#   python infrastructure/mosaicfl_server/runner.py --address 0.0.0.0:8080
#
#   # Em cada máquina cliente (copiar só ca.crt):
#   export FL_TLS_CERT_DIR="/path/to/dir/com/ca.crt"
#   python infrastructure/mosaicfl_client/runner.py --server SERVIDOR_IP:8080
#
# Validade: 365 dias (desenvolvimento). Para produção, usar CA institucional.

set -euo pipefail

OUT_DIR="${1:-certs}"
mkdir -p "$OUT_DIR"

COUNTRY="BR"
STATE="SP"
CITY="Sao Paulo"
ORG="MOSAIC-FL"
CA_CN="MOSAIC-FL CA"
SERVER_CN="${2:-localhost}"   # use o IP/hostname real em produção

echo "Gerando certificados TLS em: $OUT_DIR/"
echo "  CN do servidor: $SERVER_CN"
echo ""

# ── 1. CA raiz ────────────────────────────────────────────────────────────────
echo "[1/3] Gerando CA raiz..."
openssl req -x509 -newkey rsa:4096 \
  -keyout "$OUT_DIR/ca.key" \
  -out    "$OUT_DIR/ca.crt" \
  -days 365 -nodes \
  -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=$CA_CN" \
  2>/dev/null
echo "  ca.crt e ca.key gerados."

# ── 2. Par de chaves do servidor ──────────────────────────────────────────────
echo "[2/3] Gerando chave e CSR do servidor..."
openssl req -newkey rsa:4096 \
  -keyout "$OUT_DIR/server.key" \
  -out    "$OUT_DIR/server.csr" \
  -nodes \
  -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=$SERVER_CN" \
  2>/dev/null

# SAN (Subject Alternative Name) — necessário para que clientes modernos aceitem.
# Se SERVER_CN for um endereço IPv4 literal (ex.: 192.168.1.100), precisa entrar
# como "IP:", não "DNS:" — a validação TLS de clientes gRPC distingue os dois tipos
# de entrada, e rejeita a conexão se o IP usado para conectar só existir como DNS SAN.
if [[ "$SERVER_CN" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
  SERVER_CN_SAN="IP:$SERVER_CN"
else
  SERVER_CN_SAN="DNS:$SERVER_CN"
fi
cat > "$OUT_DIR/server_ext.cnf" <<EOF
[SAN]
subjectAltName=IP:127.0.0.1,DNS:localhost,$SERVER_CN_SAN
EOF

# ── 3. Assinar certificado do servidor com a CA ───────────────────────────────
echo "[3/3] Assinando certificado do servidor com a CA..."
openssl x509 -req \
  -in "$OUT_DIR/server.csr" \
  -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" \
  -CAcreateserial \
  -out "$OUT_DIR/server.crt" \
  -days 365 \
  -extfile "$OUT_DIR/server_ext.cnf" \
  -extensions SAN \
  2>/dev/null

# Limpeza
rm -f "$OUT_DIR/server.csr" "$OUT_DIR/server_ext.cnf"

# ── Permissões ────────────────────────────────────────────────────────────────
chmod 600 "$OUT_DIR/ca.key" "$OUT_DIR/server.key"
chmod 644 "$OUT_DIR/ca.crt" "$OUT_DIR/server.crt"

echo ""
echo "========================================"
echo "  Certificados gerados com sucesso!"
echo "========================================"
echo ""
echo "Arquivos em $OUT_DIR/:"
ls -lh "$OUT_DIR/"
echo ""
echo "Para usar com o MOSAIC-FL:"
echo ""
echo "  # Servidor (precisa de ca.crt + server.crt + server.key):"
echo "  export FL_TLS_CERT_DIR=\"$(realpath "$OUT_DIR")\""
echo "  python infrastructure/mosaicfl_server/runner.py"
echo ""
echo "  # Cliente (precisa apenas de ca.crt):"
echo "  # Copie ca.crt para a maquina do cliente e aponte:"
echo "  export FL_TLS_CERT_DIR=\"/caminho/para/pasta/com/ca.crt\""
echo "  python infrastructure/mosaicfl_client/runner.py --server IP_SERVIDOR:8080"
echo ""
echo "AVISO: ca.key e server.key sao chaves privadas."
echo "Nunca os envie pelo repositorio ou por canais inseguros."
echo ""
