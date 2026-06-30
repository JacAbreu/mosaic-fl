# Sumário de Treinamento — Parte 2

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico  
**Autora:** Jacqueline Abreu | ICMC/USP  
**Continuação de:** `docs/Sumario_Treinamento.md` (Exp 1–16)  
**Iniciado em:** 2026-06-29

---

## Sessão 2026-06-29 — O que foi acordado e implementado

### Contexto da sessão

A sessão começou com a validação do estado da RAG após a troca do backend LLM para Ollama. Foram identificados e corrigidos 4 bugs críticos, implementados 2 features novas, e corrigido o nome do modelo Ollama. Tudo implementado antes do Exp 17.

### Bugs encontrados e corrigidos

| Bug | Arquivo | Problema | Fix |
|---|---|---|---|
| Special tokens na knowledge base | `interpretability.py` | `[PAD]`, `[CLS]`, `[SEP]` apareciam como top attention tokens (alta atenção por construção, não por sinal clínico); contaminavam os perfis da KB | `_SPECIAL_TOKENS` frozenset + `_is_clinical_token()` — filtra qualquer token que começa com `[` ou `<` |
| `replace("", "adulto")` corrompia KB | `rag.py` `build_knowledge_base()` | `str(p.get("idade_exacta", ""))` retornava `""` quando ausente; `text.replace("", "adulto")` insere "adulto" entre **cada caractere** do texto em Python | Guard `if idade_exacta:` antes do `replace()` |
| `tokenizer.encode()` antes do dispatch de backend | `rag.py` `generate_justification()` | Primeiras linhas chamavam `self.tokenizer.encode()` sempre, antes de checar o backend; com Ollama, `self.tokenizer=None` → `AttributeError` | Dispatch (`if self._llm_backend == "ollama":`) movido para o topo da função |
| Fallback para HF usava nome do modelo Ollama | `rag.py` `__init__()` | Ao cair no fallback HuggingFace, tentava `AutoTokenizer.from_pretrained("gemma3:4b")` — inválido (`:` não é permitido em repo IDs do HF) | Adicionado `llm_hf_model` em `RuntimeConfig` (`FL_LLM_HF_MODEL`, padrão `distilgpt2`); fallback usa esse campo |
| Nome do modelo Ollama errado | `setup.sh`, `Makefile`, `.env.example`, `rag.py`, `docs/` | `gemma4:4b` não existe no registry do Ollama | Corrigido para `gemma3:4b` em todos os arquivos |

### Features implementadas

**1. Ollama com fallback automático para HuggingFace**
- `_check_ollama_available()`: GET `/api/tags` com timeout 5s no `__init__` — detecta indisponibilidade antes do primeiro uso
- Se Ollama offline: loga WARNING, usa `RUNTIME_CFG.llm_hf_model` (distilgpt2) automaticamente — sem intervenção manual
- `make ollama-setup`: instala Ollama + faz pull do `gemma3:4b` (~3,3 GB) — standalone e integrado ao `make setup` (steps 5+6 de `setup.sh`)
- `make ollama-check`: valida se Ollama está rodando e modelo disponível

**2. Seeding determinístico por rodada × cliente** (`client.py`)
- `torch.manual_seed(FED_CFG.random_seed + current_round * FED_CFG.num_clients + self.client_id)` no início de cada `fit()`
- `current_round` vem do `config` dict do servidor (já existia no protocolo Flower)
- Garante que runs independentes com mesmos hiperparâmetros produzam resultados idênticos
- Impacto esperado na acurácia: **negligenciável** — só muda ordem dos batches, não o gradiente médio

**3. DP-FedAvg (McMahan et al. 2018)** — dois níveis:
- **Cliente** (`client.py`): clipa update Δ = w_final − w_global à norma S (`dp_max_grad_norm`) antes de retornar ao servidor
- **Servidor** (`fl_core.py` → `apply_dp_noise()`): após agregação FedNova, adiciona N(0, (σ·S/n)²) ao estado global
- Desabilitado por padrão (`FL_DP_NOISE=0.0`) — sem overhead nos Exps anteriores
- Variáveis de configuração: `FL_DP_NOISE` (σ) e `FL_DP_CLIP` (S) — via env var ou Makefile

### Decisões tomadas

| Decisão | Justificativa |
|---|---|
| Seeding + DP no mesmo treinamento (Exp 17) | Impactos separáveis: seeding afeta apenas ordem de batches (efeito negligenciável); qualquer degradação observada é DP. Não confunde a análise. |
| DP sem Opacus | Opacus não está instalado; implementação manual do DP-FedAvg é suficiente para o TCC e evita nova dependência |
| Cota de ε com mecanismo gaussiano simples | Suficiente para ilustrar trade-off no TCC; cota é conservadora (RDP daria menor). Documentado como limitação. |
| `gemma3:4b` como modelo padrão | `gemma4` não existe no Ollama registry; `gemma3:4b` (~3,3 GB) é o modelo mais recente disponível na família Gemma 4B |
| Fallback obrigatório para HF | Requisito da sessão: sistema não pode falhar por indisponibilidade do Ollama; fallback automático com WARNING é a solução correta |

---

## Estado do Projeto ao Iniciar Esta Parte

### Melhor resultado histórico (referência para todos os próximos experimentos)

| Métrica | Valor | Experimento |
|---|---|---|
| Accuracy | **69,59%** | Exp 15 — FL Federado FedNova, R79 |
| Macro AUC | 0,8181 | Exp 15 |
| Macro F1 | 0,4946 | Exp 15 |
| ECE isotônica | 0,0149 | Exp 15 |
| RF Centralizado | 68,41% | Exp 15 (baseline) |
| BEHRT Pooled B | 68,68% | Exp 16 (120 épocas, budget equiv.) |

**Conclusão do Exp 15/16:** custo de privacidade da federação é **negativo** — FL FedNova supera todos os baselines centralizados com budget equivalente (120 rodadas = 120 épocas).

### Pipeline atual (`make training-full`)

4 fases em sequência:
1. BPSP-only (Exp 13 ref.: 64,86%)
2. HSL-only (Exp 14 ref.: 40,05%)
3. **Federado BPSP+HSL** ← foco dos próximos experimentos
4. BEHRT Pooled baseline (Exp 16 ref.: 68,68%)

Duração total: ~9h43min (583 min) na máquina de desenvolvimento.

### Implementações concluídas (prontas para uso nos próximos treinamentos)

| Componente | Status | Ativar com |
|---|---|---|
| FedNova | ✓ ativo por padrão | `use_fednova=True` em `config.py` |
| Checkpoint guloso + scoping por `training_id` | ✓ ativo | automático |
| Calibração isotônica OvR | ✓ ativa | automático |
| Gradient clipping (max_norm=1,0) | ✓ ativo | automático |
| Seeding determinístico por rodada × cliente | ✓ ativo | automático |
| DP-FedAvg (McMahan et al. 2018) | ✓ implementado, **desabilitado por padrão** | `FL_DP_NOISE=σ` |
| RAG com Ollama (gemma3:4b) | ✓ implementado | `FL_LLM_BACKEND=ollama` (padrão) |
| RAG bugs corrigidos (special tokens + replace) | ✓ corrigido | automático |
| Fallback automático HF se Ollama offline | ✓ implementado | automático |

### Modelo Ollama

```bash
# Verificar se está disponível
make ollama-check

# Instalar / baixar modelo (se necessário)
make ollama-setup          # gemma3:4b (~3,3 GB)
```

---

## Experimento 17 — DP-FedAvg + Seeding Fix

**Status:** Pronto para executar  
**Comando:**
```bash
FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full
```

### O que muda em relação ao Exp 15

| Mudança | Impacto esperado |
|---|---|
| Seeding determinístico (`torch.manual_seed` por round × client) | Negligenciável na acurácia; elimina variância espúria entre runs |
| DP-FedAvg σ=1,0, S=1,0 | Degradação de acurácia a medir (estimativa: −2 a −8 p.p.) |

### Parâmetros DP

| Parâmetro | Valor |
|---|---|
| `FL_DP_NOISE` (σ — multiplicador de ruído) | 1,0 |
| `FL_DP_CLIP` (S — sensitivity / norma máxima do update) | 1,0 |
| ruído_std por parâmetro | σ·S/n = 1,0·1,0/2 = **0,50** |
| δ (probabilidade de falha da garantia DP) | 1e-5 |
| ε por rodada (cota solta Gaussiana) | √(2·ln(1,25/1e-5)) / 1,0 ≈ **3,52** |
| ε acumulado (120 rodadas, cota solta) | ≈ **422** |

> A cota ε≈422 é conservadora (mecanismo gaussiano simples). RDP/moments accountant produziria valor menor. Suficiente para o TCC ilustrar o trade-off.

### O que registrar após o treinamento

Copiar os resultados abaixo após `make training-full` completar:

```
### Resultados Exp 17

**Data:** ___  
**Log:** experiments/logs/___  
**σ usado:** 1,0 | **S:** 1,0

#### Federado (fase 3/4)

| Métrica | Exp 17 (σ=1,0) | Exp 15 (σ=0) | Δ |
|---|---|---|---|
| Accuracy | ___ | 69,59% | ___ |
| Macro AUC | ___ | 0,8181 | ___ |
| Macro F1 | ___ | 0,4946 | ___ |
| ECE isotônica | ___ | 0,0149 | ___ |
| Melhor rodada | ___ | R79 | — |

#### DP metrics (extrair do log)
- dp_update_norm médio por cliente: ___
- Fração de updates clipados (scale < 1): ___
- ε acumulado reportado pelo logger: ___

#### BPSP-only (fase 1/4)
| Métrica | Valor |
|---|---|
| Accuracy | ___ |
| Melhor rodada | ___ |

#### HSL-only (fase 2/4)
| Métrica | Valor |
|---|---|
| Accuracy | ___ |
| Melhor rodada | ___ |

#### BEHRT Pooled (fase 4/4)
| Config | Accuracy | Macro F1 |
|---|---|---|
| Pooled A (sem demo) | ___ | ___ |
| Pooled B (late fusion) | ___ | ___ |

#### RAG Precision@3 (federado)
| Classe | P@3 |
|---|---|
| curado_pronto | ___ |
| curado_internado | ___ |
| melhora_pronto | ___ |
| melhora_internado_breve | ___ |
| melhora_internado_grave | ___ |
| **Macro P@3** | ___ |

#### Duração
| Fase | Duração |
|---|---|
| 1/4 BPSP-only | ___ |
| 2/4 HSL-only | ___ |
| 3/4 Federado | ___ |
| 4/4 BEHRT Pooled | ___ |
| **Total** | ___ |
```

---

## Série DP planejada (após Exp 17)

| Exp | σ | S | ε_acum (cota solta, 120 rounds) | Objetivo |
|---|---|---|---|---|
| 15 | 0,0 | — | ∞ | referência (sem DP) |
| **17** | **1,0** | **1,0** | **≈422** | **primeiro ponto da curva** |
| 18 | 0,5 | 1,0 | ≈845 | menos ruído, avaliar degradação menor |
| 19 | 2,0 | 1,0 | ≈211 | mais ruído, avaliar degradação maior |

Comando genérico:
```bash
FL_DP_NOISE=<σ> FL_DP_CLIP=<S> make training-full
```

---

## Roadmap Geral — Estado 2026-06-29

| Fase | Descrição | Status |
|---|---|---|
| **1 — Treinamento FL** | FedNova + calibração isotônica + checkpoint guloso | ✓ Concluído (Exp 15: 69,59%) |
| **2 — DP** | DP-FedAvg implementado; Exp 17/18/19 para medir curva Acc × ε | ✓ Implementado — **aguardando execução** |
| **3 — Distribuído** | Desktop como servidor + notebook como cliente; comunicação real entre nós | Pendente (pós-Exp 19) |
| **4 — API de Inferência** | REST endpoint para prognóstico de novo paciente com o modelo federado | ✓ **Concluído — ver seção abaixo** |
| **Análise erros clínicos** | Tipologia dos 4 tipos de erro + implicações para a defesa | ✓ Concluído (`docs/analise_erros_clinicos.md`) |
| **Tabela unificada Exp 1–19** | Tabela comparativa de todos os experimentos com slots para DP | Pendente — aguardar Exp 17/18/19 |
| **Diagrama de execução** | Fluxo do `make training-full` (4 fases) para a defesa | Pendente |
| **Refactoring MVP** | Modularização + configurações em banco; executar APÓS todas as fases | Pendente |
| **Fusão 3 classes** | Avaliar com orientadora se clinicamente justificável | A definir com orientadora |

---

## API de Inferência — Concluída em 2026-06-29

### O que foi implementado

O gap entre o pipeline de treinamento e a API de inferência estava em: o treinamento salva checkpoints no banco (`CheckpointStore` → PostgreSQL/SQLite), mas a API lia apenas arquivos `.pt` em disco — que nunca existiam. Sem conexão entre os dois.

**3 mudanças para fechar o gap:**

1. **`InferenceEngine.load_from_store(checkpoint: dict)`** (`infrastructure/mosaicfl_api/inference_engine.py`)  
   Carrega pesos + vocab + metadados diretamente do dict retornado por `CheckpointStore.load_best()`.

2. **Fallback em `state._get_engine()`** (`infrastructure/mosaicfl_api/state.py`)  
   Sequência: procura `round_*.pt` em `FL_CHECKPOINT_DIR` → se não encontrar, tenta `CheckpointStore.load_best(FL_DB_URL)` → se banco vazio, sobe sem modelo com WARNING → nunca trava.

3. **`make api` + `make export-checkpoint`** (`Makefile` + `scripts/export_checkpoint.py`)  
   `make api` sobe o banco e inicia o servidor na porta 8000. `make export-checkpoint` extrai o melhor checkpoint do banco para `checkpoints/best_model.pt` (para deploy offline).

### Verificado ao vivo

```
POST /api/predict HTTP/1.1 → 200 OK

startup: checkpoint carregado do PostgreSQL — round=79, acc=0.6959, vocab=648, version=c71727ce4b53
resposta: predicted_label=melhora_pronto, risk_score=0.39, mc_samples=50
          curado_pronto: 0.3423 ± 0.1167
          melhora_pronto: 0.3703 ± 0.1082
          melhora_internado_breve: 0.1998 ± 0.0794
          curado_internado: 0.0435 ± 0.0205
          melhora_internado_grave: 0.0441 ± 0.0282
```

### Testes adicionados

10 novos testes em `tests/unit/test_inference_engine_store.py`:
- `TestLoadFromStore` (6): vocab, temperatura, metadados, erros de validação, pesos realmente carregados no modelo
- `TestGetEngineFallback` (4): usa store quando sem `.pt`, pula store quando `.pt` existe, banco vazio não trava, erro de conexão não trava

**Suite completa:** 41 testes integração (API) + 10 novos unit = todos passando.

### Como usar

```bash
# Iniciar a API (modelo carregado automaticamente do banco)
make api

# Para desenvolvimento sem autenticação:
FL_AUTH_REQUIRED=false FL_ENV=development make api

# Exportar checkpoint para arquivo (deploy offline / inspeção):
make export-checkpoint
make export-checkpoint FL_TRAINING_ID=5   # treinamento específico

# Swagger UI (documentação interativa):
# http://localhost:8000/docs
```

### O que já estava pronto (não foi alterado)

A API já tinha toda a estrutura hexagonal implementada: endpoints `/api/predict`, `/api/exams/ingest`, `/api/patients`, `/api/fl/status`, `/api/fl/reload`, segurança JWT/API Key, rate limiting, pseudonimização LGPD (HMAC-SHA256), `PatientDB` PostgreSQL/SQLite, `outcome_feedback` (ground truth tardio), exportação FHIR R4, ClinicalPath, UI Bootstrap em `static/index.html`. O único gap era o carregamento do modelo.

---

## Referência Rápida — Arquivos Chave

### Treinamento e modelo

| Arquivo | Responsabilidade |
|---|---|
| `src/mosaicfl/core/config.py` | Todos os hiperparâmetros (`FedConfig`, `RuntimeConfig`) |
| `src/mosaicfl/core/client.py` | Treino local, FedProx, DP clipping, seeding |
| `experiments/training/fl_core.py` | Agregação FedNova, `apply_dp_noise()`, loop federado |
| `src/mosaicfl/core/rag.py` | RAG: backends Ollama/HF, KB building, geração |
| `src/mosaicfl/core/interpretability.py` | Extração de tokens de atenção, filtro de special tokens |
| `experiments/run_training.py` | Entrypoint de cada fase do `make training-full` |
| `infrastructure/shared/checkpoint_store.py` | Persistência de checkpoints (SQLite + PostgreSQL) |

### API de Inferência

| Arquivo | Responsabilidade |
|---|---|
| `infrastructure/mosaicfl_api/inference_engine.py` | Motor de predição: tokenização idêntica ao treino, MC Dropout, `load_from_store()` |
| `infrastructure/mosaicfl_api/state.py` | Singleton do engine com fallback ao `CheckpointStore` |
| `infrastructure/mosaicfl_api/routers/prediction.py` | `POST /api/predict` e `POST /api/exams/ingest` |
| `infrastructure/mosaicfl_api/routers/admin.py` | `GET /api/fl/status` e `POST /api/fl/reload` |
| `infrastructure/mosaicfl_api/db.py` | `PatientDB`: histórico de risco, exames, ground truth tardio |
| `infrastructure/mosaicfl_api/schemas.py` | Pydantic: `PredictRequest/Response`, `IngestRequest/Response` |
| `scripts/export_checkpoint.py` | Exporta melhor checkpoint do banco para `checkpoints/best_model.pt` |

### Documentação e análise

| Arquivo | Responsabilidade |
|---|---|
| `docs/Sumario_Treinamento.md` | Histórico completo Exp 1–16 |
| `docs/analise_erros_clinicos.md` | Tipologia dos 4 erros por classe + implicações para defesa |
| `docs/avaliacao_metodologia_mosaicfl.md` | Estado MVP + qualidade científica + lacunas para mestrado |

### Make targets relevantes nesta fase

| Target | Uso |
|---|---|
| `make training-full` | Executa as 4 fases do pipeline |
| `FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full` | Exp 17 (DP σ=1,0) |
| `make api` | Inicia a API de inferência na porta 8000 |
| `make export-checkpoint` | Exporta melhor checkpoint do banco para arquivo `.pt` |
| `make ollama-setup` | Instala Ollama + baixa gemma3:4b |
| `make ollama-check` | Verifica se Ollama está rodando |
