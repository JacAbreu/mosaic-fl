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
- Vocabulário canônico único distribuído antes do treinamento (`build_standard_vocab.py`) — sem esse mecanismo a agregação FedAvg seria semanticamente inválida entre hospitais com nomes de analitos diferentes.

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
