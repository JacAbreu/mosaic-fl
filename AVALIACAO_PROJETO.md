# Avaliação do Projeto MosaicFL

> Este arquivo registra avaliações periódicas do projeto nos critérios definidos abaixo.
> Cada seção de avaliação inclui a data, o prompt exato utilizado e as notas com justificativas.
> O objetivo é permitir reavaliações com os mesmos critérios à medida que o projeto evolui.

---

## Critérios de avaliação (fixos — não alterar entre avaliações)

### AVALIAÇÃO ACADÊMICA
Avalia o projeto sob a ótica das melhores práticas e do estado da arte acadêmico:
- Alinhamento com literatura de referência (FedProx, BEHRT, RAG, FL clínico)
- Contribuição original em relação ao que já existe
- Rigor metodológico (labels, baseline, validação, ablation)
- Qualidade da arquitetura e do código sob perspectiva de engenharia de software
- Cobertura e organização de testes
- Proposta evolutiva documentada

### AVALIAÇÃO DE PRODUÇÃO CLÍNICA
Avalia o projeto sob a ótica das melhores práticas da indústria para sistemas clínicos:
- Segurança e privacidade (TLS, autenticação, pseudonimização, LGPD)
- Interoperabilidade (FHIR R4, LOINC, ClinicalPath)
- Robustez operacional (recovery, scheduler, fallback, observabilidade)
- Viabilidade clínica (generalização do modelo, validação por especialistas, audit trail)
- Bloqueadores identificados para deploy real

### Padrão de avaliação
Alto — perspectiva de engenheiro sênior de software, arquiteto e cientista de ML clínico.
Justificativas baseadas em fontes de dados observáveis no repositório (não em afirmações do README).

---

## Nota metodológica — diferença entre avaliações

As avaliações neste arquivo usam **metodologias distintas** conforme a sessão em que foram realizadas. Para comparar resultados entre datas, observe a metodologia declarada em cada entrada:

| Metodologia | Foco | Critérios de engenharia de software | Critérios de rigor científico |
|---|---|---|---|
| **Engenharia (2026-06-07)** | Qualidade do código, infraestrutura, testes, CI/CD | ✓ Principal | Não avaliado |
| **Holística (2026-06-24)** | Alinhamento com estado da arte + viabilidade clínica real | ✓ Componente | ✓ Principal |

Isso explica por que a nota acadêmica da avaliação de 2026-06-07 (8.2–9.18) é substancialmente maior que a de 2026-06-24 (6.5): a primeira mede qualidade de engenharia de FL; a segunda mede adicionalmente rigor metodológico de pesquisa (qualidade dos labels, baseline comparativo, generalização do modelo, avaliação de calibração). Ambas são válidas para fins diferentes.

---

## Avaliação histórica — 2026-06-07 (metodologia: Engenharia de Software)

> Avaliação realizada em sessão anterior, incluída aqui como baseline histórico.
> Metodologia: critérios ponderados de engenharia de software. Não inclui avaliação de rigor científico/acadêmico.
> Reproduzida integralmente para permitir reavaliação com os mesmos pesos futuramente.

### Prompt utilizado (2026-06-07)

Não registrado na sessão original. Critérios inferidos da estrutura da avaliação: qualidade do código Python, padrões arquiteturais, infraestrutura, testes, documentação, CI/CD, segurança, corretude funcional, confiabilidade, observabilidade e operabilidade.

### Tabela de evolução — 2026-06-07 (5 reavaliações na mesma sessão)

```
ACADÊMICA (TCC)
Critério                 | Peso | 1ª   | 2ª   | 3ª   | 4ª   | 5ª
-------------------------|------|------|------|------|------|------
Arquitetura e Design     | 25%  | 8.5  | 9.0  | 9.2  | 9.2  | 9.5
Qualidade de Código      | 20%  | 7.5  | 8.5  | 8.8  | 8.8  | 9.0
Infraestrutura           | 20%  | 8.0  | 8.5  | 9.0  | 9.0  | 9.5
Testes                   | 20%  | 8.5  | 9.0  | 9.2  | 9.3  | 9.3
Documentação             | 10%  | 9.0  | 9.0  | 9.2  | 9.2  | 9.2
CI/CD                    |  5%  | 6.5  | 6.5  | 6.5  | 6.5  | 6.5
MÉDIA FINAL              |100%  | 8.2  | 8.7  | 8.9  | 8.96 | 9.18

PRODUÇÃO CLÍNICA
Critério                      | Peso | 1ª  | 2ª  | 3ª  | 4ª  | 5ª
------------------------------|------|-----|-----|-----|-----|-----
Segurança e Privacidade       | 25%  | 4.0 | 4.0 | 4.5 | 5.5 | 5.5
Corretude Funcional           | 20%  | 5.5 | 7.0 | 7.0 | 7.0 | 7.5
Confiabilidade e Resiliência  | 20%  | 5.0 | 5.5 | 7.5 | 7.5 | 8.0
Observabilidade               | 15%  | 6.5 | 6.5 | 6.5 | 7.0 | 7.0
Qualidade de Código           | 10%  | 7.5 | 8.5 | 8.8 | 8.8 | 9.0
Operabilidade                 | 10%  | 5.0 | 5.0 | 5.5 | 5.5 | 6.0
MÉDIA FINAL                   |100%  | 5.4 | 5.8 | 6.4 | 6.8 | 7.0
```

### Estado do projeto na 1ª avaliação (2026-06-07 — antes das correções)

Problemas identificados que fundamentaram as notas iniciais mais baixas:
- `fit()` ignorava `config["local_epochs"]` e `config["proximal_mu"]` — servidor não controlava hiperparâmetros
- `_proximal_loss()` usava global `FED_CFG.proximal_mu` em vez do valor enviado pelo servidor
- `get_parameters()` sem `.copy()` — aliasing de memória entre numpy arrays e tensores
- TLS opcional: ausência de `FL_TLS_CERT_DIR` gerava `logger.warning`, não `raise`
- Fallback silencioso para dados sintéticos quando SGBD falhava
- `except Exception: continue` em `fit()` silenciando erros de hardware
- Single point of failure no servidor Flower (sem SuperLink)
- Sem recovery de sessão após crash do servidor
- Sem audit trail LGPD
- Backend SQLite sem escalabilidade

### Correções aplicadas ao longo da sessão de 2026-06-07

| Reavaliação | Principais correções | Δ TCC | Δ Clínico |
|---|---|---|---|
| 1ª → 2ª | config dict propagado ao cliente; _proximal_loss com mu explícito; .copy() em get_parameters(); fallback silencioso removido | +0.5 | +0.4 |
| 2ª → 3ª | Flower SuperLink (elimina SPOF); recovery de sessão (TrainingStateStore); TLS obrigatório (raise); watchdog de round | +0.2 | +0.6 |
| 3ª → 4ª | Audit trail LGPD Art. 37; PII removida do log de aplicação; modelo de auth corrigido (sem API key fixa) | +0.1 | +0.4 |
| 4ª → 5ª | PostgreSQL + TimescaleDB + pgvector; migrate_sqlite.py idempotente; except→raise em fit(); total_samples correto | +0.2 | +0.2 |

### Gaps remanescentes ao final da sessão de 2026-06-07

Para nota TCC acima de 9.5: pipeline CI com GitHub Actions e `--cov-fail-under`.

Para nota clínica acima de 8.0:
- Backoff exponencial com jitter no reconect do cliente legado
- CORS allowlist explícita (remover default `"*"`)
- Validação de range clínico no ingest da API
- Criptografia em repouso (`mosaicfl_api.db` e checkpoints `.pt`)
- **Privacidade diferencial (DP-SGD)** — único gap que impede uso com dados reais sem risco de vazamento por gradientes

---

## Avaliação 1 — 2026-06-24

### Prompt utilizado

```
agora, com o readme atualizado, faça uma avaliacao do projeto nos criterios AVALIACAO ACADEMICA
e AVALIACAO DE PRODUCAO CLINICA, levando em consideracao as melhores praticas da academia e da
industria. Tenha um padrão alto de avaliação no ponto de vista de engenharia de software,
arquitetura de software e avaliacao do estado da arte academico, ponderando também se o projeto
tem alguma proposta evolutiva no que já existe. Justifique as avaliações com as fontes de dados
utilizadas.
```

### Estado do projeto na data da avaliação

- **Branch:** main
- **Commits relevantes:** 5b62d6f (separação por hospital), 526cfa1 (rede local), de12b2e (calibração), c89e393 (checkpoints SQLite/Postgres), 2ad6c6a (RAG + tensores)
- **Linhas de código Python:** ~16.300 (src + infrastructure + integration + experiments)
- **Arquivos de teste:** 39
- **Labels do modelo:** 4 classes de prognóstico (alta, internacao_prolongada, uti, obito) — migrado de 5 faixas de duração nesta sessão

### Fontes consultadas para a avaliação

| Fonte | O que revelou |
|---|---|
| `README.md` (1.194 linhas) | Arquitetura, decisões de design, interoperabilidade, segurança |
| `docs/FLUXO_APRENDIZADO_FEDERADO.md` | Pipeline detalhado: SQL → tokenização → BEHRT → ClinicalPath |
| `docs/TODO.md` | Bloqueadores de produção identificados pelo próprio projeto |
| `src/mosaicfl/core/model.py` | Arquitetura SimplifiedBEHRT: embed_dim=64, num_layers=2, num_heads=4 |
| `src/mosaicfl/core/client.py` | FedProx com proximal term + class weights por hospital |
| `src/mosaicfl/core/preprocessor.py` | Mapeamento outcome_class FAPESP → 4 classes; label `internacao_prolongada` = "em atendimento" |
| `src/mosaicfl/core/evaluation.py` | ECE, AUC-ROC OVR, F1 por classe implementados |
| `infrastructure/mosaicfl_api/inference_engine.py` | MC Dropout (50 passes), temperature scaling, thread-safe |
| `integration/clinical-path/models.py` | ProbabilityEstimate, RiskPrediction, FL_PROB_* constants |
| `integration/clinical-path/exporter.py` | Exportação distribuição completa + backward-compat histórico |

---

## AVALIAÇÃO ACADÊMICA — 6,5 / 10

### Fundamentação da nota

#### Pontos que sustentam a nota

**Estado da arte — alinhamento correto com a literatura**

O projeto cita e implementa corretamente as referências principais:
- **FedProx** (Li et al., MLSys 2020) — algoritmo certo para dados não-IID hospitalares; proximal term com μ configurável implementado; class weights para desbalanceamento no cliente FL (poucos trabalhos de FL clínico fazem isso).
- **BEHRT** (Rasmy et al., npj Digital Medicine 2021) — CLS token pooling com `nn.Parameter` + `trunc_normal_(std=0.02)` fidedigno ao paper; token de padding correto.
- **Temperature scaling** (Guo et al., ICML 2017) — calibração pós-treinamento via LBFGS minimizando NLL, implementada em `calibration.py`; T persistido no checkpoint; ECE como métrica.
- **RAG** (Lewis et al., NeurIPS 2020) — referência correta para explicabilidade.
- **Flower SuperLink** (Beutel et al., 2020, v1.8+) — estado da arte para separação plano de dados / plano de controle em FL.

Isso é acima da média de TCCs na área: a maioria usa apenas FedAvg com modelos densos simples.

**Contribuição técnica incremental genuína**

- Exportar **distribuição completa de probabilidade por desfecho com incerteza MC-Dropout** como exames sintéticos no ClinicalPath (`FL_PROB_*` + `FL_PROB_*_INCERTEZA`) não aparece em trabalhos similares de integração FL + visualização clínica.
- Separação `PatientExport` (ClinicalPath) vs. `InferenceOutput` (FHIR) como contratos arquiteturalmente independentes — o módulo FHIR é estruturalmente incapaz de vazar dados clínicos.
- `correlation_token` efêmero para resolver o `subject` obrigatório do FHIR R4 sem vazar identidade — design elegante e correto.
- Vocabulário canônico único distribuído antes do treinamento (`scripts/build_standard_vocab.py`) — sem esse mecanismo a agregação FedAvg seria semanticamente inválida entre hospitais com nomes de analitos diferentes.

**Engenharia de software consistentemente boa**
- Arquitetura hexagonal: `mosaicfl.core` (domínio puro) isolado de `infrastructure/` e `experiments/`
- 39 arquivos de teste em três camadas (unit / integration / e2e)
- `test_fl_cycle_explained.py`: documentação executável do protocolo FL
- Structured logging JSON, health endpoints liveness/readiness separados
- Recovery de sessão documentado e implementado

---

#### Pontos que penalizam a nota

**1. Problema central de label — maior risco acadêmico (−1,5)**

O label de treino é `outcome_class` do FAPESP COVID-19, mapeado em 4 classes. Há três problemas sérios:

- **Alta (0+1)**: `curado` e `melhora` são clinicamente diferentes. Ambos mapeados para `alta` injeta ruído na classe mais numerosa.
- **`internacao_prolongada` = "em atendimento" (4)**: esse valor FAPESP significa que o registro foi capturado antes de haver desfecho. É um **estado de censura**, não um prognóstico. Tratá-lo como classe positiva contamina o modelo com dados censurados sem tratamento de sobrevivência (Cox / Kaplan-Meier). Um modelo treinado assim aprende a identificar pacientes cujos dados foram colhidos cedo, não pacientes que terão internação prolongada.
- **Dataset de domínio único**: FAPESP é COVID-19 (HSL + BPSP, São Paulo). Generalização para outras patologias ou outros hospitais não avaliada — limita fortemente a contribuição científica.

Fonte: `src/mosaicfl/core/preprocessor.py`, `_OUTCOME_TO_PROGNOSIS`, mapeamento `4 → internacao_prolongada`.

**2. Arquitetura BEHRT muito simplificada para o claim (−0,5)**

| Parâmetro | MosaicFL | BEHRT original | Med-BERT |
|---|---|---|---|
| embed_dim | 64 | 288 | 768 |
| num_layers | 2 | 6 | 12 |
| num_heads | 4 | 12 | 12 |
| age embeddings | ✗ | ✓ | ✓ |
| visit embeddings | ✗ | ✓ | parcial |
| segment embeddings | ✗ | ✓ | ✓ |

Sem embeddings de idade e visita, o modelo perde as principais vantagens do BEHRT sobre um Transformer genérico. Ausência de ablation study mostrando que as simplificações não degradam resultado neste dataset.

Fonte: `src/mosaicfl/core/model.py`, `src/mosaicfl/core/config.py`.

**3. Ausência de baseline comparativo formal (−0,5)**

Sem experimento comparando SimplifiedBEHRT com logistic regression, random forest ou LSTM. Não é possível afirmar que BEHRT agrega valor sobre alternativas mais simples nos dados disponíveis.

**4. RAG com DistilGPT-2 — qualidade de justificativa não avaliada (−0,5)**

DistilGPT-2 (82M parâmetros) tem qualidade de geração muito inferior a modelos modernos. A detecção de alucinação (`confiavel: bool`) não tem avaliação formal — sem ROUGE, BERTScore ou avaliação humana por especialista clínico. Para um claim de "justificativa diagnóstica interpretável", a evidência é insuficiente.

**5. Differential Privacy ausente (informativo — não penaliza a nota)**

Documentado como roadmap em `docs/TODO.md` (Gaussian mechanism, ε-δ DP). Deve ser declarado explicitamente como limitação na monografia com referência a Geyer et al. (2017) e McMahan et al. (2018).

---

## AVALIAÇÃO DE PRODUÇÃO CLÍNICA — 7,0 / 10

### Fundamentação da nota

#### Pontos que sustentam a nota

**Segurança em profundidade**

- TLS obrigatório em todas as vias FL — `EnvironmentError` se `FL_TLS_CERT_DIR` ausente (fonte: `infrastructure/shared/tls.py`)
- JWT HS256/RS256 na API
- Rate limiting sliding window (120 req/min geral, 30 req/min ingest)
- Per-patient asyncio locks — evita race condition em inferências paralelas
- Path traversal bloqueado em `output_dir` — rejeita `..` em qualquer parte do caminho
- SHA-256 em checkpoints — integridade verificável
- HMAC-SHA256 para pseudonimização — irreversível sem o secret; `FL_ENV=production` bloqueia subida sem o secret configurado
- `ExamInput.value` rejeita NaN, ±∞ e negativos

**Arquitetura operacional sólida**

- Flower SuperLink como plano de controle separado: ServerApp pode reiniciar sem derrubar SuperNodes
- Recovery de sessão em `logs/training_state.json`: retoma do checkpoint após falha sem reiniciar contagem de convergência
- Scheduler com APScheduler + cron + `--once` — flexibilidade real de implantação (CronJob Kubernetes às 2h)
- Verificação de quórum de hospitais antes de cada round
- DataLoader cache no cliente: sem re-query ao banco a cada round
- Exponential backoff com jitter na reconexão do cliente
- Dados nunca saem do hospital — implementado por construção arquitetural, não por política

**Observabilidade**

- Structured logging JSON com campos tipados (fonte: `integration/clinical-path/exporter.py` → `logger.info("patient_exported", extra={...})`)
- Health endpoints liveness/readiness separados na porta 8081
- `model_metadata.trained: bool` na resposta da API — consumidor sabe se está recebendo predição de modelo treinado ou ruído

---

#### Pontos que penalizam a nota

**1. Rate limiter in-process — bloqueador real de produção (−0,8)**

`_SlidingWindowLimiter` é por processo Python. Com Gunicorn + 4 workers, limite efetivo = 480 req/min em vez de 120. O projeto identifica o bloqueador em `docs/TODO.md`: "instalar `fastapi-limiter` + Redis". No código atual, o controle de abuso não funciona em deploy multi-worker.

**2. Generalização clínica muito limitada (−0,7)**

Modelo treinado exclusivamente em COVID-19 de 2 hospitais de São Paulo (2020-2021). Em produção hospitalar real, a primeira questão de qualquer comitê de ética ou diretor médico seria: "em que população esse modelo foi validado?" Ausência de validação externa é um bloqueador clínico real independente de engenharia.

**3. ~~FHIR exporter não implementado~~ — CORRIGIDO (avaliação inicial incorreta)**

O README marcava o módulo como "a implementar", mas o código estava completo: `FHIRExporter.to_risk_assessment()` implementado em `mapper.py`, `InferenceOutput` com validação em `models.py`, integrado em `service.py` (o `IngestResponse` retorna `fhir_risk_assessment`), e 33 testes passando. O README foi corrigido em 2026-06-24. Esta penalização foi removida da nota; a nota real é +0,5 em relação ao publicado nesta data.

**4. Prometheus ausente (−0,3)**

Métricas `fl_round_total`, `fl_accuracy`, `fl_loss`, `fl_clients_active` no TODO mas não implementadas. Sem counters e gauges não é possível configurar alertas automáticos de degradação de modelo ou queda de clientes. Fonte: `docs/TODO.md`, seção "Observabilidade".

**5. Audit trail LGPD ausente (−0,5)**

LGPD (Lei 13.709/2018) exige rastreabilidade de acesso a dados de saúde. O sistema tem pseudonimização correta mas não tem log imutável de "token HMAC X teve predição gerada em Y às Z por requisição do sistema W". Fonte: `docs/TODO.md`, seção "Dependências de Produção".

**6. Validação clínica ausente (informativo — não penaliza a nota)**

Nenhum especialista clínico avaliou: se as classes de prognóstico fazem sentido médico, se σ > 0.15 como limiar de incerteza é clinicamente interpretável, ou se `FL_RISK_SCORE > 0.3` como alerta é uma threshold razoável. Esperado num TCC; deve ser declarado como limitação.

---

## Síntese

| Dimensão | Nota | Principal fortaleza | Principal limitação |
|---|---|---|---|
| **Acadêmica** | **6,5 / 10** | FedProx + BEHRT + MC Dropout + calibração corretamente integrados; contribuição original na exportação de distribuição de probabilidade para ClinicalPath | Label `internacao_prolongada` = registros censurados; dataset COVID-19 único domínio; sem baseline |
| **Produção clínica** | **7,5 / 10** | Segurança em profundidade (TLS/JWT/HMAC/locks/path traversal); recovery; SuperLink; FHIR R4 implementado e integrado na API | Rate limiter quebra com múltiplos workers; sem audit trail LGPD completo; CORS default `"*"` |

O projeto é **acima da média para um TCC**. A proposta mais original — usar FL para predizer prognóstico clínico e injetar a distribuição completa de probabilidade por desfecho (com incerteza MC-Dropout) na linha do tempo visual do ClinicalPath — não foi encontrada em trabalhos similares. A nota acadêmica está limitada principalmente pela qualidade dos dados de treino e pela ausência de validação comparativa, não pela engenharia.

---

*Próxima reavaliação recomendada: após (a) resolução do label `internacao_prolongada`; ou (b) implementação do FHIR exporter; ou (c) validação com especialista clínico.*

---

## Avaliação 2 — 2026-06-24 (reavaliação com correções)

### Prompt utilizado

Mesmo da Avaliação 1 (2026-06-24). Reavaliação motivada por duas penalidades incorretas identificadas na análise anterior.

### Estado do projeto na data da reavaliação

- **Branch:** main, commit `e466e0a` (mesmo da avaliação anterior)
- **Testes:** 541 passando, 6 deselected
- **Linhas Python:** ~11.950 (src + infrastructure + integration)
- **Novo artefato:** `experiment_results.json` com 5 experimentos (dados sintéticos)

### Fontes consultadas

| Fonte | O que revelou |
|---|---|
| `infrastructure/mosaicfl_api/audit.py` | `log_access()` implementado com JSON estruturado, SHA-256 pseudonimização, propagate=False |
| `infrastructure/mosaicfl_api/service.py:592,642,682,713,771` | `audit.log_access()` chamada em todos os 5 endpoints com dados de paciente |
| `logs/audit.log` (149KB) | Entradas reais com timestamp, token_fingerprint, patient_id_hash — audit trail está em produção |
| `infrastructure/mosaicfl_api/service.py:270–293` | `FL_CORS_ORIGINS='*'` bloqueia startup com `ValueError` em `FL_ENV=production` |
| `experiment_results.json` | exp5: acurácia 0.4667 constante por 11 rodadas — platô desde a rodada 1, sem aprendizado incremental |

---

## AVALIAÇÃO ACADÊMICA — 6,5 / 10 *(sem alteração)*

Nenhum commit novo com conteúdo acadêmico desde a avaliação anterior. Penalidades inalteradas:

| Penalidade | Situação |
|---|---|
| Label `internacao_prolongada` = dado censurado (−1,5) | `preprocessor.py:_OUTCOME_TO_PROGNOSIS[4] = "internacao_prolongada"` — inalterado |
| SimplifiedBEHRT sem justificativa formal (−0,5) | Late fusion não implementada; ablation ausente |
| Ausência de baseline comparativo (−0,5) | `experiment_results.json` tem exp2 com ganho FL vs. local, mas `"simulado": true` e sem baseline (LR/RF) |
| RAG DistilGPT-2 sem avaliação formal (−0,5) | ROUGE/BERTScore ausentes |

**Observação — exp5:** Acurácia 0.4667 e loss 0.7161 constantes por 11 rodadas. O `ConvergenceTracker` confunde platô estático com convergência real (nenhuma melhora desde a rodada 1). Dado sintético, não afeta a nota, mas deve ser investigado antes de apresentar resultados.

---

## AVALIAÇÃO DE PRODUÇÃO CLÍNICA — 8,0 / 10 *(+0,5 em relação à avaliação anterior)*

### Correções

**Correção 1 — Audit trail LGPD (penalidade −0,5 → −0,1)**

A avaliação anterior afirmou "Audit trail LGPD ausente". Incorreto. O audit trail está completamente implementado:
- `audit.py`: `log_access()` com JSON estruturado, `token_fingerprint` (SHA-256[:12]), `patient_id_hash` (SHA-256[:16]), `propagate=False`.
- Chamado em `service.py` nos endpoints `predict`, `ingest`, `patient_list`, `patient_read`, `model_reload`.
- `logs/audit.log` (149KB) contém entradas reais com estrutura correta.

Gap residual: `RotatingFileHandler` (50MB × 10) permite sobrescrita. Para LGPD estrita, o registro deveria ser append-only ou WORM. Penalidade: −0,1 (imutabilidade), não ausência.

**Correção 2 — CORS `"*"` (penalidade removida)**

`service.py:270` levanta `ValueError` ao iniciar com `FL_CORS_ORIGINS='*'` em produção. Controle adequado — não é gap de produção.

### Penalidades remanescentes

| Penalidade | Fonte | Peso |
|---|---|---|
| Rate limiter in-process | `service.py:155 _SlidingWindowLimiter` — por processo Python; 4 workers = 480 req/min | −0,8 |
| Generalização clínica | COVID-19 / 2 hospitais SP / 2020–2021, sem validação externa | −0,7 |
| Prometheus ausente | TODO documentado, não implementado | −0,3 |
| Audit trail — imutabilidade | `RotatingFileHandler` não é WORM | −0,1 |

---

## Síntese comparativa

| Dimensão | Avaliação 1 | **Avaliação 2** | Δ | Razão |
|---|---|---|---|---|
| **Acadêmica** | 6,5 / 10 | **6,5 / 10** | 0 | Nenhum commit novo com conteúdo acadêmico |
| **Produção clínica** | 7,5 / 10 | **8,0 / 10** | +0,5 | Audit trail estava implementado; CORS bloqueado em produção |

**Para subir a nota acadêmica:** late fusion demográfica + baseline formal (ambos documentados no TODO com proposta concreta, sem implementação).

**Para subir a nota clínica:** substituir `_SlidingWindowLimiter` por Redis + fastapi-limiter — único bloqueador que impede homologação.

---

*Próxima reavaliação recomendada: após implementação de (a) late fusion + baseline, ou (b) Redis rate limiter.*

---

## Avaliação 3 — 2026-06-24 (avaliação holística do ciclo de vida completo)

### Prompt utilizado

```
avalie novamente o projeto, tendo como base o avaliacao_projeto.md - dessa vez,
não veja só as últimas alteracoes, veja o projeto inteiro em todo seu tempo de vida
```

### Metodologia desta avaliação

Avaliação de ciclo de vida completo: considera todos os 50 commits (2026-05-31 a 2026-06-24),
a trajetória de evolução do projeto, artefatos produzidos ao longo das sessões, e o estado atual
do repositório. Critérios idênticos às avaliações anteriores (seção "Critérios de avaliação").

### Estado do projeto na data desta avaliação

- **Branch:** main — 50 commits, 24 dias de desenvolvimento
- **Período:** 2026-05-31 (commit inicial) → 2026-06-24 (baseline RF + correção class_weights)
- **Testes:** 541 passando, 6 deselected (e2e sem ambiente disponível)
- **Arquivos Python:** 68 fonte + 39 teste = 107 arquivos, 13.818 linhas
- **Camadas arquiteturais:** `src/mosaicfl/core/` (domínio) · `infrastructure/` (API, server, shared) · `integration/` (FHIR, ClinicalPath) · `experiments/`

### Fontes consultadas para a avaliação

| Fonte | O que revelou |
|---|---|
| `git log --oneline` (50 commits) | Trajetória completa: 7 sessões de desenvolvimento com arcos temáticos distintos |
| `src/mosaicfl/core/model.py` | SimplifiedBEHRT sem embeddings de idade/visita; CLS token correto; ~712K parâmetros |
| `src/mosaicfl/core/client.py` | Correção class_weights nesta sessão; proximal term correto após sessão 2 |
| `src/mosaicfl/core/preprocessor.py` | `outcome_class=4 → internacao_prolongada` — label de censura, não prognóstico |
| `src/mosaicfl/core/calibration.py` | Temperature scaling LBFGS, T persistido em checkpoint |
| `src/mosaicfl/core/evaluation.py` | ECE (15 bins), AUC-ROC OVR, F1 macro, matriz de confusão — pipeline completo |
| `src/mosaicfl/core/rag.py` | DistilGPT-2 + MiniLM; confiabilidade por heurística textual, sem métrica formal |
| `integration/fhir/` (models.py, mapper.py, loinc_map.py) | FHIR R4 RiskAssessment completo; `correlation_token` efêmero |
| `integration/clinical-path/exporter.py` | 5 arquivos ClinicalPath exportados; FL_PROB_* como exames sintéticos |
| `infrastructure/mosaicfl_api/audit.py` | log_access JSON + SHA-256 + RotatingFileHandler (não-WORM) |
| `infrastructure/mosaicfl_api/service.py` | TLS obrigatório, JWT, rate limiter in-process, CORS bloqueado em produção |
| `infrastructure/shared/tls.py` | EnvironmentError se FL_TLS_CERT_DIR ausente — TLS obrigatório por construção |
| `experiments/training_runner/run_experiments_simulation.py` | 5 experimentos + baseline RF (exp6) adicionado nesta sessão |
| `experiment_results.json` | exp1-5 com dados sintéticos; exp5 acurácia corrigida após bug class_weights |
| `docs/TODO.md` | 11 bloqueadores de produção abertos; DP no roadmap |
| `tests/unit/` (34 arquivos) | Cobertura unit em quase todos os módulos; sem e2e real |

---

## Trajetória do projeto — visão de ciclo de vida

O projeto evoluiu em 7 sessões com arcos temáticos distintos:

| Sessão | Data | Commits | Arco temático |
|---|---|---|---|
| 1 | 2026-05-31 | 2 | Bootstrap: FL sintético, SQLite, ChromaDB |
| 2 | 2026-06-02/03 | 7 | Algoritmo: FedProx, BEHRT v1, RAG v1 (DistilGPT-2), Ray |
| 3 | 2026-06-04/05 | 8 | Produção v1: daemons, convergência, watchdog, JSON logging |
| 4 | 2026-06-06/07 | 14 | Produção v2: SuperLink, recovery, TLS obrigatório, LGPD, PostgreSQL |
| 5 | 2026-06-15 | 5 | Refinamento: CheckpointStore ABC, RAG real, hospital_id |
| 6 | 2026-06-23/24 | 7 | Interoperabilidade: FHIR R4 completo (33 testes), evaluate() no simulador |
| 7 | 2026-06-24 | 7 | Avaliação crítica: RF baseline, correção class_weights, documentação |

**O que a trajetória revela:** o projeto seguiu uma ordem racional — algoritmo → produção → interoperabilidade → avaliação crítica. Não houve reescrita arquitetural ou descarte de artefatos; cada sessão adicionou camadas ao mesmo núcleo. Isso é consistência metodológica rara em TCC.

**O que a trajetória esconde:** todos os experimentos (exp1-exp6) usam dados sintéticos. O dado real FAPESP nunca foi usado para validação. Isso não é um detalhe tardio — a decisão de usar sintético está presente desde a sessão 1 e nunca foi revisitada. O projeto construiu infraestrutura de produção sobre uma base de dados que ainda não validou a hipótese central.

---

## AVALIAÇÃO ACADÊMICA — 7,0 / 10 *(+0,5 em relação à Avaliação 2)*

### Fontes de força que a visão holística confirma

**1. Alinhamento teórico consistente ao longo de todo o desenvolvimento**

O projeto implementa corretamente os quatro pilares do estado da arte em FL clínico:
- **FedProx** (Li et al., MLSys 2020): proximal term com μ configurável via `config` dict do servidor, class weights por hospital para não-IID. Correção do bug de propagação de μ documentada na sessão 4. Em `client.py:103-104`.
- **BEHRT** (Rasmy et al., npj Digital Medicine 2021): CLS token como `nn.Parameter` + `trunc_normal_(std=0.02)`, positional encoding sinusoidal, `padding_idx=0`. Em `model.py:27-38, 98-117`.
- **Temperature Scaling** (Guo et al., ICML 2017): LBFGS sobre NLL, T persistido no checkpoint, ECE como métrica primária de calibração. Em `calibration.py`.
- **MC Dropout** (Gal & Ghahramani, 2016): 50 forward passes, thread-safe via `threading.Lock()`, incerteza exportada como exame ClinicalPath. Em `inference_engine.py`.

Isso é acima da média de TCCs: a maioria implementa FedAvg + MLP sem calibração ou mecanismo de incerteza.

**2. Contribuições originais verificáveis no repositório**

- **Vocabulário canônico distribuído** (`scripts/build_standard_vocab.py`): sem esse artefato, FedAvg entre hospitais com nomes de analitos divergentes seria semanticamente inválido. Nenhum trabalho de FL clínico revisado menciona esse passo como artefato explícito.
- **Exportação de distribuição completa de probabilidade com incerteza** para ClinicalPath (`FL_PROB_*` + `FL_PROB_*_INCERTEZA` como exames sintéticos): injeta a epistemologia bayesiana na linha do tempo visual. Fonte: `integration/clinical-path/exporter.py`, `models.py:RiskPrediction`.
- **`correlation_token` efêmero para FHIR**: resolve o campo obrigatório `subject` do R4 sem armazenar mapeamento identidade → token. Isolamento arquitetural verificado por teste: `test_no_infrastructure_import`, `test_no_patient_export_import`. Fonte: `integration/fhir/models.py`.

**3. Engenharia de software consistentemente disciplinada**

Ao longo dos 50 commits:
- Arquitetura hexagonal mantida: `core/` nunca importa `infrastructure/`, `integration/` nunca importa `infrastructure/`. Verificável via testes de isolamento.
- 541 testes com 34 suites unitárias cobrindo todos os subsistemas principais.
- Cada bug corrigido foi documentado (proximal_mu: sessão 4; class_weights: sessão 7; except→raise: sessão 4).
- `docs/TODO.md` como roadmap honesto de limitações — não como lista de boas intenções, mas como item rastreável por sessão.

---

### Penalidades — visão de ciclo de vida completo

**1. Problema central de label — permanece o maior risco acadêmico (−1,5)**

`outcome_class=4 → "internacao_prolongada"` permanece inalterado desde o commit inicial.
Esse mapeamento é um **dado censurado** (paciente ainda internado no momento do snapshot),
não um prognóstico de desfecho. O modelo aprende a identificar pacientes cujo registro foi
capturado antes do desfecho, não pacientes que terão internação prolongada. Isso contamina a
classe mais clinicamente relevante para predição de risco.

Fonte: `src/mosaicfl/core/preprocessor.py:_OUTCOME_TO_PROGNOSIS[4]`.
A penalidade não é reduzida porque o problema não foi nem reconhecido formalmente no código
(nenhum `# WARNING: censored label` ou equivalente), nem mitigado com análise de sobrevivência.

**2. SimplifiedBEHRT sem ablation — mesmo gap (−0,5)**

A comparação com BEHRT original (`embed_dim=288`, age/visit/segment embeddings) continua ausente.
A justificativa de "escala hospitalar, CPU-only" está documentada em `docs/TODO.md` mas não
como ablation study executável — é uma afirmação não verificada no repositório.
Late fusion de demográficos (proposta documentada no TODO) não implementada.

**3. Baseline comparativo — parcialmente resolvido nesta sessão (−0,5 → −0,2)**

Random Forest com Bag-of-Tokens implementado em dois modos (centralizado + por hospital)
com `n_estimators=200, class_weight="balanced"`, reordenamento de colunas por classe,
métricas completas (accuracy, macro_F1, macro_AUC, ECE). Fonte: `experiments/training_runner/run_experiments_simulation.py:run_baseline_rf()`.

Gap residual: comparação executada exclusivamente com dados sintéticos (2 classes
de 4 presentes, AUC retorna NaN para classes ausentes). Sem validação com dados FAPESP reais,
não é possível afirmar que SimplifiedBEHRT supera RF no domínio alvo. A penalidade parcial
(−0,2) permanece pela ausência de dados reais.

**4. RAG DistilGPT-2 sem avaliação formal — mesmo gap (−0,5)**

Os resultados do exp4 (`100% úteis, 0% alucinadas, 4.7/5`) são auto-avaliação em 50 amostras
sintéticas. DistilGPT-2 (82M parâmetros, pré-treinado em inglês) gerando justificativas clínicas
em português é uma limitação fundamental não quantificada. Detecção de alucinação por heurística
textual (`"certeza" in justification.lower()`) não tem validade clínica.
ROUGE, BERTScore ou avaliação por especialista clínico ausentes.

**5. Validação exclusivamente sintética — limitação transversal (observacional — não penaliza adicionalmente)**

Nenhum dos 6 experimentos foi executado com dados FAPESP reais. Isso não gera penalidade
adicional porque é esperado num TCC com acesso restrito a dados clínicos, mas deve ser declarado
explicitamente como limitação principal na monografia. A nota atual assume que os resultados
sintéticos demonstram que a arquitetura funciona; a afirmação de que ela funciona *neste domínio clínico*
permanece não verificada.

### Cálculo da nota acadêmica

| Item | Valor |
|---|---|
| Nota base | 10,0 |
| Label censurado (`internacao_prolongada = 4`) | −1,5 |
| SimplifiedBEHRT sem ablation | −0,5 |
| Baseline parcial (RF sintético, sem real) | −0,2 |
| RAG sem avaliação formal | −0,5 |
| **Subtotal** | **7,3** |
| Ajuste holístico (−): experimentos 100% sintéticos como limitação transversal | −0,3 |
| **NOTA FINAL** | **7,0 / 10** |

---

## AVALIAÇÃO DE PRODUÇÃO CLÍNICA — 8,0 / 10 *(sem alteração)*

### O que a visão de ciclo de vida confirma

**Evolução de segurança em profundidade (verificado commit a commit)**

A sessão 4 (2026-06-06/07) representou um salto qualitativo em produção:
- TLS: `logger.warning` → `EnvironmentError` (commit de segurança explícito)
- Audit trail LGPD: implementado como artefato independente (`audit.py`) com logging estruturado
- SuperLink: eliminou SPOF com separação de plano de dados / plano de controle
- TrainingStateStore → CheckpointStore ABC: recovery de sessão com verificação SHA-256
- `except Exception: continue` → `raise`: sem silenciamento de falhas de hardware

Esses não foram correções pontuais — foram decisões arquiteturais com impacto estrutural. O código
atual não tem nenhum ponto onde falhas críticas são silenciadas silenciosamente.

**Interoperabilidade completa (confirmada na sessão 6)**

FHIR R4 (`integration/fhir/`): `FHIRExporter.to_risk_assessment()` completo, 33 testes
unitários verificando isolamento arquitetural, mapeamento LOINC de 22 analitos, correlation_token.
ClinicalPath (`integration/clinical-path/`): 5 arquivos exportados, distribuição completa de
probabilidade com incerteza injetada como exame clínico. 70% completo — bloqueio externo
(autorização Prof. Claudio para FL_PROB_* em `list_exams.txt`).

**Padrão operacional acima do esperado para TCC**

- Scheduler APScheduler + cron + `--once` (deploy como CronJob Kubernetes)
- Quórum de hospitais verificado antes de cada round
- DataLoader cache por cliente (sem re-query)
- Exponential backoff com jitter na reconexão
- `model_metadata.trained: bool` na resposta da API (consumidor sabe se recebe predição válida)
- Health endpoints liveness/readiness em porta separada

### Penalidades remanescentes — confirmadas pela visão holística

| Penalidade | Evidência | Peso |
|---|---|---|
| Rate limiter in-process | `service.py:_SlidingWindowLimiter` — por processo; 4 workers = 480 req/min efetivos | −0,8 |
| Generalização clínica | COVID-19 / 2 hospitais SP / 2020-2021 / dados sintéticos nos experimentos | −0,7 |
| Prometheus ausente | TODO documentado desde sessão 3, sem implementação ao longo de todo o ciclo | −0,3 |
| Audit trail imutabilidade | `RotatingFileHandler` permite sobrescrita — não é WORM (LGPD estrita) | −0,1 |

**Gaps documentados mas não bloqueadores para TCC:**
- MC Dropout sequencial sem timeout (risco com alta carga, documentado no TODO)
- Rotação de `FL_PATIENT_ID_SECRET` sem estratégia de key versioning
- Sem DP (Gaussian mechanism) — bloqueador para dados reais, mas fora do escopo do TCC com dados sintéticos

---

## Síntese holística — ciclo de vida completo

### O que o projeto entregou em 24 dias (50 commits)

| Artefato | Status | Testes |
|---|---|---|
| FedProx com proximal term + class weights | Completo, bug corrigido | `test_fedprox_client.py` |
| SimplifiedBEHRT com CLS token | Completo | `test_simplified_behrt.py` |
| Temperature scaling + ECE | Completo, T no checkpoint | `test_training_state_store.py` |
| MC Dropout com thread safety | Completo | (in `inference_engine.py`) |
| FHIR R4 RiskAssessment | Completo, isolado | 33 testes |
| ClinicalPath exporter | 70% (bloqueio externo) | sem teste unitário |
| LGPD audit trail | Completo (não-WORM) | `test_audit.py` |
| SuperLink + recovery SHA-256 | Completo | `test_training_state_store.py` |
| Baseline RF (2 modos) | Completo, sintético | (em `run_experiments_simulation.py`) |
| Validação com dados FAPESP reais | **Ausente** | — |

### Notas finais comparativas

| Dimensão | Av. 1 | Av. 2 | **Av. 3 (holística)** | Δ total |
|---|---|---|---|---|
| **Acadêmica** | 6,5 / 10 | 6,5 / 10 | **7,0 / 10** | +0,5 |
| **Produção clínica** | 7,5 / 10 | 8,0 / 10 | **8,0 / 10** | +0,5 |

### Avaliação qualitativa do ciclo de vida

**O que o projeto fez bem (perspectiva de ciclo de vida):**
O MosaicFL demonstrou consistência metodológica: cada sessão teve um arco temático claro,
os artefatos se acumularam sem reescrita, os bugs encontrados foram corrigidos com documentação
explícita do que estava errado. A decisão de separar `integration/fhir/` com testes de isolamento
arquitetural (`test_no_infrastructure_import`) é o tipo de detalhe que distingue código produzido
com intenção de código produzido por acumulação.

**O que o projeto deixou em aberto (perspectiva de ciclo de vida):**
A hipótese central — que FL com SimplifiedBEHRT prediz prognóstico clínico de forma superior a
alternativas mais simples — permanece não testada com dados reais. Todos os experimentos usam
dados sintéticos com distribuição não representativa do dataset FAPESP real. O label mais
clinicamente relevante (`internacao_prolongada`) é um dado censurado. Esses gaps não surgiram
tarde no projeto — estão presentes desde o commit inicial e atravessaram 24 dias de desenvolvimento
sem revisão. Isso é o principal risco para a defesa do TCC.

**Veredicto geral:**
Acima da média para um TCC em engenharia de software e arquitetura de sistemas clínicos.
Abaixo do que seria necessário para uma publicação acadêmica (rigor metodológico dos experimentos).
O caminho mais curto para elevar a nota acadêmica é documentar explicitamente o label
`internacao_prolongada` como dado censurado na monografia e executar o baseline RF com dados FAPESP reais.

---

*Próxima reavaliação recomendada: após (a) execução dos experimentos com dados FAPESP reais, ou (b) resolução formal do label `internacao_prolongada` com análise de sobrevivência ou reclassificação.*

---

## Pontos para Alinhar com a Orientadora — 2026-06-25

> Esta seção registra as questões abertas que requerem validação acadêmica ou clínica externa.
> Atualizar após cada reunião com a orientadora.

### Questões metodológicas que precisam de resposta

**Q1 — Threshold de 10 dias para "internação grave" tem respaldo na literatura?**

O label `melhora_internado_grave` é definido como internação com duração > 10 dias. Esse valor foi escolhido por critério técnico (distribuição do dataset), não por evidência clínica. Em COVID-19 há referências ao percentil 75 de tempo de internação como marcador de gravidade? O SOFA score ou algum critério de consenso define internação prolongada? Sem referência, o threshold é uma limitação metodológica a declarar na monografia.

**Q2 — Incluir atendimentos de pronto-socorro muda a pergunta clínica do modelo?**

O pipeline foi expandido de internados-only (~14k) para todos os atendimentos (~33k), incluindo pronto-socorro. Com essa mudança, o modelo passa a responder duas perguntas distintas ao mesmo tempo:
- Para pacientes de pronto: *"este paciente vai precisar internar, e qual será o desfecho?"*
- Para pacientes internados: *"qual será a evolução desta internação?"*

Perguntar à orientadora: faz sentido misturar esses dois contextos clínicos num único modelo de prognóstico para o TCC? Ou é melhor manter dois modelos separados, ou restringir ao contexto de admissão hospitalar?

**Q3 — O ablation de late fusion demográfica deve entrar nos resultados ou é suficiente como trabalho futuro?**

A late fusion (idade + sexo concatenados ao CLS antes do classifier head) está implementada em `model.py` com parâmetro `demo_dim`. O experimento comparativo `demo_dim=0` vs `demo_dim=2` não foi rodado. Em COVID-19, idade é o preditor dominante de mortalidade — se o ablation mostrar AUC +5%, é um resultado com valor para a defesa. Perguntar: vale rodar antes da entrega, ou declarar como trabalho futuro com a justificativa de que o sinal demográfico está arquiteturalmente preparado?

### Resultados para apresentar na reunião

**R1 — Distribuição não-IID entre hospitais (evidência empírica de necessidade de FL)**

Do dry-run com dados FAPESP reais:

| Hospital | N pacientes | Classe dominante | % |
|---|---|---|---|
| BPSP | 28.599 | curado_pronto | 55,6% |
| HSL | 5.174 | melhora_pronto | 61,5% |

Esses dois hospitais têm populações completamente diferentes. BPSP atende majoritariamente casos leves de pronto-socorro; HSL tem maior proporção de internações com melhora. Num modelo centralizado, a distribuição do BPSP (5,5x maior) dominaria o treino e apagaria o perfil do HSL. O FedProx com class weights por hospital preserva as distribuições locais. Este é o argumento empírico mais forte para a escolha de FL.

**R2 — Resultados da simulação de 20 rodadas** *(pendente — simulação em andamento)*

Incluir após conclusão: accuracy, macro_F1, AUC por classe, ECE pré/pós-calibração.

### Status de alinhamento

| Questão | Status |
|---|---|
| Q1 — Threshold 10 dias | Aberta — pendente resposta da orientadora |
| Q2 — Escopo pronto vs internado | Aberta — pendente resposta da orientadora |
| Q3 — Ablation late fusion | Aberta — pendente decisão da orientadora |
| R1 — Não-IID BPSP/HSL | Pronto para apresentar |
| R2 — Resultados simulação | Pendente conclusão da simulação |

---

## Avaliação 4 — 2026-06-25 (pós-redesign de labels e expansão de escopo)

### Prompt utilizado

Mesmo das avaliações anteriores (2026-06-24). Reavaliação motivada pelo redesign completo do esquema de labels e expansão do dataset de internados-only para todos os atendimentos.

### Estado do projeto na data desta avaliação

- **Branch:** main — commits da sessão de 2026-06-24/25 não commitados ainda
- **Labels:** 5 classes de prognóstico (redesign desta sessão)
- **Dataset:** todos os atendimentos FAPESP (~33.773 pacientes: BPSP 28.599 + HSL 5.174)
- **Simulação:** em andamento (20 rodadas, dados reais FAPESP)
- **Testes:** 541 passando (sem alteração desde Avaliação 3)
- **Arquivos modificados (não commitados):** `preprocessor.py`, `config.py`, `model.py`, docs

### Fontes consultadas para a avaliação

| Fonte | O que revelou |
|---|---|
| `src/mosaicfl/core/preprocessor.py` | `_SQL_ATENDIMENTOS` (sem filtro de internação), `_map_outcome(outcome_class, duration_days, attendance_type)` → 5 classes, exclusão de outcome_class=4 |
| `src/mosaicfl/core/config.py` | `_DEFAULT_CLASS_LABELS` = 5 classes; `FED_CFG` lendo env vars (fix do FL_NUM_ROUNDS) |
| `src/mosaicfl/core/model.py` | `demo_dim` parâmetro para late fusion; classifier head condicional |
| `experiments/logs/dryrun.log` | Distribuição não-IID: BPSP 55,6% curado_pronto; HSL 61,5% melhora_pronto |
| `experiments/logs/evaluation_round_1.json` | Round 1 da simulação em andamento: acc=0.513, macro_AUC=0.755, ECE=0.079 |
| `experiments/training_runner/run_experiments_simulation.py` | Baseline RF real implementado; ablation demográfica documentada |
| `AVALIACAO_PROJETO.md` (Avaliação 3) | Penalidades da avaliação holística anterior como baseline de comparação |

---

## AVALIAÇÃO ACADÊMICA — 8,3 / 10 *(+1,3 em relação à Avaliação 3)*

### Mudanças que elevaram a nota

**1. Resolução do problema central de label (penalidade −1,5 → −0,4)**

O label `internacao_prolongada = outcome_class 4` (dado censurado) foi completamente eliminado.
O novo esquema crosses três dimensões clinicamente observáveis no momento do desfecho:
`outcome_class` (curado/melhora) × `attendance_type` (internado/pronto) × `duration_days` (≤10d / >10d).

Resultado: 5 classes com critérios determinísticos a partir de campos FAPESP observados — sem dados censurados, sem imputação, sem ambiguidade sobre o que o modelo está aprendendo.

Fonte: `preprocessor.py:_map_outcome()` — função pura de 3 argumentos sem estado.

Penalidade residual (−0,4): o threshold de 10 dias para separar `melhora_internado_breve` de `melhora_internado_grave` é tecnicamente arbitrário. Nenhuma referência clínica foi citada no código ou na documentação. Além disso, a mistura de atendimentos de pronto-socorro e internados no mesmo modelo levanta a questão de qual pergunta clínica está sendo respondida — dois contextos de admissão distintos tratados como um problema único.

**2. Baseline com dados reais FAPESP (penalidade −0,2 → −0,1)**

O Random Forest em `run_experiments_simulation.py:run_baseline_rf()` roda com os mesmos
27.018 amostras de treino reais que o FL. A comparação SimplifiedBEHRT vs RF com dados FAPESP
reais estará disponível ao final da simulação em andamento.

Penalidade residual (−0,1): comparação numérica pendente — a simulação ainda está em andamento
no momento desta avaliação.

**3. Evidência empírica de não-IID (novo resultado positivo)**

O dry-run revelou distribuição fortemente não-IID entre BPSP e HSL. Isso é um resultado
acadêmico relevante: demonstra empiricamente — com dados hospitalares reais — que o cenário
que justifica FL (heterogeneidade de dados entre instituições) está presente neste dataset.
O FedProx com class weights por hospital é a resposta metodologicamente correta para este achado.

Fonte: `dryrun.log` — distribuições `{0: 15892, 1: 318, 2: 120, 3: 9448, 4: 2821}` (BPSP) vs
`{0: 67, 1: 45, 2: 3182, 3: 1280, 4: 600}` (HSL).

---

### Penalidades remanescentes

**1. Label scheme — threshold clínico sem referência (−0,4)**

O corte de 10 dias em `_map_outcome()` não tem referência citada. Em COVID-19, a literatura
de gravidade usa critérios como SOFA ≥ 2, necessidade de UTI ou ventilação mecânica —
não diretamente dias de internação. O threshold de 10 dias pode ser correto empiricamente
mas precisa de justificativa clínica ou deve ser declarado como limitação. Adicionalmente,
a mistura de pronto-socorro e internação num mesmo modelo levanta questão de escopo clínico
ainda não respondida.

**2. SimplifiedBEHRT — late fusion implementada, ablation não executado (−0,4)**

`model.py` tem `demo_dim` parâmetro funcional. O classifier head usa `embed_dim + demo_dim`
quando demográficos são passados. Mas sem o experimento `demo_dim=0` vs `demo_dim=2`
com dados reais, não há evidência quantitativa de que a inclusão de idade e sexo melhora
o modelo. A justificativa de simplificação (dataset pequeno, CPU-only, internação única vs
multi-visita) está documentada no TODO mas não em ablation executável.

**3. Baseline comparativo — resultado pendente (−0,1)**

Implementação correta, dados reais carregados, comparação numericamente pendente.

**4. RAG DistilGPT-2 — sem avaliação formal (−0,5)**

Inalterado desde a Avaliação 3. DistilGPT-2 (82M parâmetros, pré-treinado em inglês)
gerando justificativas clínicas em português não tem ROUGE, BERTScore ou avaliação por
especialista. Fonte: `src/mosaicfl/core/rag.py`.

### Cálculo da nota acadêmica

| Item | Valor |
|---|---|
| Nota base | 10,0 |
| Label scheme — threshold sem referência + escopo misto | −0,4 |
| SimplifiedBEHRT — ablation não executado | −0,4 |
| Baseline — resultado numérico pendente | −0,1 |
| RAG sem avaliação formal | −0,5 |
| Ajuste positivo: evidência empírica não-IID com dados reais | +0,3 |
| **NOTA FINAL** | **8,3 / 10** |

Nota: o ajuste +0,3 reflete que o projeto agora demonstra com dados reais o fenômeno central que justifica FL. Nenhuma avaliação anterior tinha esse resultado.

---

## AVALIAÇÃO DE PRODUÇÃO CLÍNICA — 8,0 / 10 *(sem alteração)*

Nenhuma mudança nos módulos de infraestrutura, segurança ou observabilidade nesta sessão.
As penalidades remanescentes da Avaliação 3 permanecem inalteradas:

| Penalidade | Peso |
|---|---|
| Rate limiter in-process (4 workers = 480 req/min efetivos) | −0,8 |
| Generalização clínica (COVID-19 / 2 hospitais SP / 2020-2021) | −0,7 |
| Prometheus ausente | −0,3 |
| Audit trail — imutabilidade (RotatingFileHandler, não WORM) | −0,1 |

---

## Síntese comparativa — evolução das avaliações

| Dimensão | Av. 1 | Av. 2 | Av. 3 (holística) | **Av. 4** | Δ total |
|---|---|---|---|---|---|
| **Acadêmica** | 6,5 | 6,5 | 7,0 | **8,3** | +1,8 |
| **Produção clínica** | 7,5 | 8,0 | 8,0 | **8,0** | +0,5 |

**Principal alavanca desta sessão:** eliminação do dado censurado e redesign para 5 classes com critérios observáveis. Essa mudança sozinha foi responsável por +1,1 pontos acadêmicos.

**Próxima alavanca disponível (maior retorno sobre esforço):** executar o ablation `demo_dim=0` vs `demo_dim=2` após a simulação atual. Se late fusion mostrar AUC ≥ +3%, justifica +0,3 adicional e resolve parcialmente a penalidade de SimplifiedBEHRT.

---

*Próxima reavaliação recomendada: após conclusão da simulação de 20 rodadas com dados FAPESP reais e alinhamento com a orientadora sobre Q1 (threshold 10 dias) e Q2 (escopo pronto vs internado).*
