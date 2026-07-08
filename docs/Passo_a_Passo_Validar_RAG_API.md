# Passo a passo — Validar RAG + API de Inferência

Registrado em 2026-07-07 (madrugada), pra retomar quando der. Não executei nada disso —
é só o roteiro.

## 1. Ollama + gemma3:4b (backend do RAG)

```bash
make ollama-check
```
Confirma se o Ollama está rodando e se o modelo (`FL_LLM_MODEL`, default gemma3:4b) está
disponível. Se falhar:
```bash
make ollama-setup
ollama pull gemma3:4b   # se ollama-check disser que o modelo não foi encontrado
```

## 2. Banco com um checkpoint treinado

A API carrega o **melhor checkpoint direto do banco** — precisa de pelo menos um
treinamento concluído (`fl_checkpoints` com alguma linha). Confirmar:
```bash
docker exec mosaicfl-db psql -U mosaicfl -d mosaicfl -c "SELECT id, round, accuracy, training_id FROM metrics.fl_checkpoints ORDER BY created_at DESC LIMIT 3;"
```
Se estiver vazio nesse banco, apontar pro banco que tem checkpoint (ex.: `mosaicfl-db-bpsp`,
`training_id=7`, validado ontem) via `FL_DB_URL`.

## 3. Subir a API

**Importante:** `FL_LLM_BACKEND` e `FL_LLM_MODEL` têm default `huggingface`/`distilgpt2`
no código — sem declarar as duas, o RAG usa silenciosamente o modelo fraco (texto em
inglês, mais truncado), não o gemma3:4b via Ollama.

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:PORTA/BANCO"  # ajustar porta/banco
export FL_LLM_BACKEND="ollama"
export FL_LLM_MODEL="gemma3:4b"
make api
```
Por padrão sobe em `localhost:8000` (`FL_API_PORT`).

## 3b. Autenticação (achado em 2026-07-07 — a API exige token por padrão)

Sem header de autorização, o request falha com `403 Token de autorização ausente`.
Não existe senha/token fixo no código — duas formas de contornar em teste local:

**Opção 1 (mais simples): desativar autenticação pra esse teste**
```bash
FL_AUTH_REQUIRED=false make api
```

**Opção 2: passar qualquer token** — sem `FL_JWT_SECRET` configurado, a validação de
JWT é pulada e **qualquer string não-vazia** passa como token (só exige que o header
exista, não valida o conteúdo):
```bash
-H "Authorization: Bearer qualquer-coisa"
```
(adicionar esse header no curl do passo 4, junto com `Content-Type`)

## 4. Testar `/api/predict` — validado em 2026-07-07, funcionando

Schema real (achado ao testar — o exemplo do README estava desatualizado): o campo é
`exams` (não `records`), e cada item usa `exam_name` (não `exam`):

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer qualquer-coisa" \
  -d '{"patient_id": "TEST-001", "exams": [
        {"exam_name": "LEUCOCITOS", "date": "2020-04-01", "value": 12.5, "unit": "10^3/uL"},
        {"exam_name": "PCR", "date": "2020-04-01", "value": 48.0, "unit": "mg/L"}
      ]}' | python3 -m json.tool
```

**Confirmado, funcionando:** retorna `risk_score`, `class_probabilities` (5 classes, com
incerteza via MC Dropout), `predicted_class`/`predicted_label`, `model_metadata`.

**RAG integrado ao `/api/predict` em 2026-07-07** — inicialmente não estava (achado
nesta validação), implementado na mesma sessão a pedido da autora. Resposta agora
inclui `rag_explanation` (`justificativa`, `fontes`, `alucinacao_detectada`, `confiavel`,
`llm_backend`, `llm_model_used`). Padrão: RAG **sempre ativo** — use `?explain=false`
na URL pra resposta rápida sem justificativa, quando a latência do LLM não for aceitável.
Se o RAG falhar (Ollama fora do ar etc.), a predição principal continua funcionando —
só `rag_explanation.erro` vem preenchido, não derruba o request inteiro.

Exemplo com explicação desativada:
```bash
curl -s -X POST "http://localhost:8000/api/predict?explain=false" \
  -H "Content-Type: application/json" -H "Authorization: Bearer qualquer-coisa" \
  -d '{"patient_id": "TEST-001", "exams": [...]}'
```

## 5. Interface — existe, e é mais completa do que eu tinha achado antes

**Correção**: há um painel web completo em `infrastructure/mosaicfl_api/static/index.html`
("MOSAIC-FL — Painel Clínico"), servido automaticamente na raiz da API (montado em
`app.py`, via `StaticFiles`). Bootstrap + Chart.js — lista de pacientes, formulário de
ingestão de exames, gráfico, status do FL.

```bash
# com a API rodando (passo 3):
xdg-open http://localhost:8000/
```

## 6. Se passar, o resto é "melhorias" (nas suas palavras) — não bloqueante pro TCC.

---
