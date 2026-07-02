# Linha do Tempo do Projeto MOSAIC-FL

**Propósito deste documento:** reconstrução cronológica factual de tudo o que foi implementado, testado, medido e decidido desde o início do projeto (registro em `docs/CHANGELOG.md` de v0.1.0) até 2026-07-01. É material de apoio para a redação do texto de defesa — não é, em si, o texto da monografia. Cada seção indica a fonte primária de onde a informação foi extraída, para que qualquer afirmação aqui possa ser verificada de volta ao documento ou commit original.

**Fontes cruzadas:**
- `git log` (67 commits, 2026-05-31 → 2026-07-01) — datas e mensagens de commit
- `docs/CHANGELOG.md` — versões v0.1.0/v0.2.0
- `AVALIACAO_PROJETO.md` — 4 avaliações formais com nota acadêmica e de produção clínica, mais a "trajetória do projeto" por sessão
- `docs/documentacao_etapas_legadas.md` — análise dos dados FAPESP, roadmap LGPD, adequações ClinicalPath
- `docs/Sumario_Treinamento.md` — Experimentos 1–17 (dados reais, pré-correção do split)
- `docs/Sumario_Treinamento_Parte2.md` — retrospectiva do Bloco 1 (renomeado T1–T16), Bloco 2, GPU, modularização
- `docs/avaliacao_metodologia_mosaicfl.md` — estado MVP e lacunas para rigor de mestrado
- `docs/TODO.md` — roadmap, discussão SCAFFOLD vs FedNova, instrução da autora sobre padrão de qualidade

**Convenção de nomenclatura ao longo do tempo:** os treinamentos foram chamados de "Experimento N" em `Sumario_Treinamento.md` (Exp 1–17) e depois renomeados retrospectivamente para "Treinamento N" (T1–T16) em `Sumario_Treinamento_Parte2.md`, com a justificativa de que "Treinamento" é mais preciso porque múltiplos parâmetros mudam juntos entre execuções — não há controle estrito de variável única como o termo "Experimento" sugeriria em uma dissertação. Este documento usa a nomenclatura do documento-fonte em cada seção, com a equivalência indicada.

**Correção de data:** `docs/CHANGELOG.md` registra `[0.1.0] — 2026-05-01`, mas essa data está incorreta — confirmado pela autora: o projeto começou no último sábado de maio, em torno do dia 31, coincidindo com o primeiro commit deste repositório ("Initial commit", 2026-05-31). A v0.1.0 (implementação inteiramente sintética, descrita na Parte 0 abaixo) corresponde, portanto, aos primeiros dias de vida do repositório — não a um mês antes. O CHANGELOG deveria ser corrigido para refletir isso.

---

## Parte 1 — Início do projeto: bootstrap e v0.1.0 (2026-05-31 → 2026-06-07)

### 2026-05-31 — Bootstrap do repositório e primeira implementação (v0.1.0)

**Fonte:** git log + `AVALIACAO_PROJETO.md` ("Sessão 1: Bootstrap") + `docs/CHANGELOG.md`

Data confirmada de início do projeto (último sábado de maio). `docs/CHANGELOG.md` registra esta primeira implementação como `[0.1.0]`, mas com a data `2026-05-01` — **incorreta**; corrigir para 2026-05-31.

| Hora | Commit | O que foi feito |
|---|---|---|
| 19:03 | `6ebe313` | Initial commit |
| 19:06 | `e85076b` | feat: initial commit MOSAIC-FL |
| 22:57 | `0d48ff8` | First mosaic-fl implementation. Next steps: tests with database |

**Conteúdo da v0.1.0** (inteiramente com dados sintéticos, fonte `docs/CHANGELOG.md`):
- BEHRT com mean pooling (não CLS token — viria só na v0.2.0)
- FedProx básico
- RAG v1 com ChromaDB
- `run_experiments.py` / `run_experiments_v2.py`
- 5 experimentos do TCC nesta versão: preprocessamento, efeito equalizador do FL, heterogeneidade non-IID, RAG, eficiência (nenhum resultado numérico documentado para esta fase)

Estado ao final do dia: FL sintético, SQLite, ChromaDB.

### 2026-06-02 → 2026-06-03 — Algoritmo (FedProx, BEHRT v1, RAG v1, Ray)

**Fonte:** git log + `AVALIACAO_PROJETO.md` ("Sessão 2: Algoritmo") + `docs/CHANGELOG.md` (v0.2.0, 2026-06-03)

7 commits nesta janela: correções diversas, melhora do `.gitignore`, ajustes de experimentos, e principalmente a evolução para permitir execução paralela via **Ray** (`eb75834`, 06-03 23:07).

**v0.2.0 (2026-06-03) — mudanças acumuladas:**
- `SimplifiedBEHRT` v2 com **CLS token pooling** (troca do mean pooling da v1) e `BEHRTEncoderLayer` customizado que expõe pesos de atenção (necessário para o RAG extrair padrões depois)
- `FedProxClient` v2 com termo proximal corrigido e tratamento de exceção por batch
- `ConvergenceTracker` com contagem incremental de estabilidade
- `EHRPreprocessor` v2 com normalização de unidades médicas (kg/lb, anos/meses)
- `data_loader.py` reescrito com **Strategy Pattern**: SGBD → CSV → sintético (este padrão persiste até a modularização de 2026-07-01)
- `RAGSystem` v2 com ChromaDB + **DistilGPT-2** e truncagem de prompt
- Daemons de produção criados: `server_daemon.py`, `client_daemon.py`, `scheduler_daemon.py`
- `ProductionFedProxStrategy` com checkpoint e exportação de métricas JSON
- Suítes `test_mosaicfl.py` e `test_v2_core.py`

### 2026-06-04 → 2026-06-05 — Primeira produção (v1)

**Fonte:** git log + `AVALIACAO_PROJETO.md` ("Sessão 3: Produção v1")

8 commits: estrutura schedule/server/client (`5731582`, criação da separação que mais tarde vira `infrastructure/mosaicfl_server`, `infrastructure/mosaicfl_client`, `infrastructure/mosaicfl_scheduler`), teste explicativo passo a passo (`3d8c888` — germe do futuro `test_fl_cycle_explained.py`), correção do `fit_metrics_aggregation_fn` (perda de loss na agregação), ajuste do cálculo de `communication_mb` (de valor fixo para calculado), `CONTRIBUTING.md`, cobertura de testes, redefinição de logs.

### 2026-06-06 → 2026-06-07 — Segunda produção (v2): SuperLink, TLS, LGPD, PostgreSQL

**Fonte:** git log + `AVALIACAO_PROJETO.md` ("Sessão 4: Produção v2", seção "Correções aplicadas ao longo da sessão de 2026-06-07")

14 commits nesta janela — a mais densa até então. Estrutura `wire-production` criada (`5eb49f5`) para validar comportamento esperado de produção em ambiente local. Commit `498afd8` ("Ajustes diversos para o projeto ter mais maturidade tecnica") é o mesmo commit em que o `src/data_loader.py` órfão (removido em 2026-07-01) teve seu último toque.

**Correções específicas de 2026-06-07** (bugs corrigidos na mesma sessão):

| Bug | Correção |
|---|---|
| `config` dict não propagado ao cliente — `_proximal_loss()` usava `FED_CFG.proximal_mu` global em vez do valor enviado pelo servidor | μ explícito passado via config |
| `get_parameters()` sem `.copy()` — aliasing de memória entre arrays numpy e tensores do modelo | `.copy()` adicionado |
| TLS ausente causava apenas `logger.warning` | Corrigido para `EnvironmentError` — TLS passa a ser obrigatório |
| Fallback silencioso para dados sintéticos quando o SGBD falhava | Removido — falha agora propaga |
| `except Exception: continue` em `fit()` mascarava falhas de hardware | Trocado para `raise` |
| Servidor Flower como único ponto de falha | **Flower SuperLink** implementado (upgrade Flower 1.5→1.8, commit `f27cc90`) |
| Sem trilha de auditoria LGPD | `audit.py` implementado (auditoria simplificada, `060cf64`) |
| Sem persistência robusta | **PostgreSQL + TimescaleDB + pgvector** adotados; `migrate_sqlite.py` idempotente criado |

Outros marcos da mesma janela: isolamento do `mosaicfl.core` do resto do projeto (`84b49e4`, "garantir o funcionamento com o mesmo resultado, não importando a origem da execução") — esta é a origem da estrutura hexagonal que persiste até hoje; testes e2e criados (`9480319`); estrutura definitiva do banco (`f53093e`, `f88dcaa`).

### 2026-06-07 — Primeira avaliação formal (5 reavaliações no mesmo dia)

**Fonte:** `AVALIACAO_PROJETO.md`, seção "Avaliação histórica — 2026-06-07"

Primeira aplicação de um critério de avaliação formal ao projeto, com metodologia **"Engenharia de Software"** — sem exigência de rigor científico (esse critério viria depois, em 06-24). A nota evoluiu 5 vezes na mesma sessão, à medida que correções eram aplicadas e reavaliadas:

| Reavaliação | Nota Acadêmica | Nota Produção Clínica |
|---|---|---|
| 1ª | 8,2 | 5,4 |
| 2ª | 8,7 | 5,8 |
| 3ª | 8,9 | 6,4 |
| 4ª | 8,96 | 6,8 |
| 5ª | **9,18** | **7,0** |

**Gaps residuais identificados para nota clínica > 8,0** (não resolvidos nesta sessão): backoff exponencial com jitter, CORS allowlist explícita, validação de range clínico, criptografia em repouso, e — o mais importante para o restante do projeto — **Differential Privacy (DP-SGD) identificado já em 2026-06-07 como o único gap que impede uso com dados reais sem risco de vazamento por gradientes** (ataques de inversão de gradiente). Este item só seria implementado ~3 semanas depois (ver Parte 6) e ainda está com os experimentos formais (Exp 17/18/19) pendentes de execução em 2026-07-01.

### 2026-06-07 — Roadmap de conformidade LGPD

**Fonte:** `docs/documentacao_etapas_legadas.md`, seção 2

8 itens mapeados por artigo da LGPD, todos pendentes na data (projeto ainda usava dados sintéticos, sem PII real):

| # | Item | Artigo LGPD |
|---|---|---|
| 1 | Pseudonimização | Art. 13 |
| 2 | Minimização de dados | Art. 6 |
| 3 | Auditoria | Art. 37 |
| 4 | Differential Privacy | Art. 46 |
| 5 | Consentimento | Art. 7 |
| 6 | Retenção | Art. 15/16 |
| 7 | Controle de acesso (mTLS/RBAC) | Art. 46 |
| 8 | Notificação de incidentes | Art. 48 |

Ordem de implementação recomendada na época: 1→3→4→5→6→7→8 (consentimento antes de retenção). Confirmado em revisão posterior no mesmo documento: os itens 1 (pseudonimização HMAC-SHA256), 3 (audit.py), 4 (parcial — DP implementado mas não executado) e 7 (parcial — TLS obrigatório, RBAC não confirmado) foram implementados nas sessões seguintes.

---

## Parte 2 — Dados reais FAPESP e integração ClinicalPath (2026-06-08 → 2026-06-15)

### 2026-06-08 — Carga e análise dos dados FAPESP

**Fonte:** `docs/documentacao_etapas_legadas.md`, seção 1 + git log (`5428196`, `b39790d`, `1b9e0c5`)

Carregamento dos 5 hospitais do dataset FAPESP COVID-19 (janela Jan–Ago/2021) via migrations 001–005:

| Hospital | Pacientes | Exames | Outcomes registrados |
|---|---|---|---|
| HSL | 8.971 | 1.346.802 | 42.598 |
| BPSP | 39.000 | 5.838.999 | 217.157 |
| HEI | 79.863 | 3.029.830 | — (sem outcomes) |
| HCSP | 3.751 | 2.320.739 | — (sem outcomes) |
| HFL | 470.967 | 17.097.334 | — (sem outcomes) |
| **Total** | **602.552** | **29.633.704** | — |

Distribuição de exames por paciente: mediana=32, p90=313, p95=621, máximo=15.599. **44,3% dos pacientes têm menos de 10 exames** — implicação direta para o `max_seq_len=128` adotado depois. Apenas HSL e BPSP fornecem arquivos de desfecho — por isso o projeto usa exclusivamente esses dois hospitais como clientes FL do início ao fim.

**Limitação crítica identificada nesta data e nunca resolvida:** ausência confirmada de óbitos e admissões em UTI em toda a base FAPESP (versão Jan/2021). O arquivo de desfechos registra o resultado de cada *visita*, não do paciente — um paciente que evoluiu a óbito aparece com "Alta" em atendimentos anteriores no mesmo arquivo. Esta limitação é estrutural do dataset, não do pipeline, e deve ser declarada explicitamente como limitação de escopo no texto de defesa.

**Decisão de vocabulário:** tokenizar por analito individual (`DE_ANALITO`), aceitando redundância do hemograma (12 tokens por coleta), em vez de agrupar por painel. Vocabulário construído do zero a partir dos dados FAPESP (sem pré-treino externo) — 649 tokens no vocabulário original desta fase, depois estabilizado em **648 tokens** nos experimentos com dados reais (Parte 4 em diante).

O esquema de labels vigente nesta data ainda não era o de 5 classes atual — ver evolução completa na Parte 4.

### 2026-06-10 — Sequência temporal e organização de documentação

**Fonte:** git log (`d4afdbe`, `1019e67`)

Definição inicial da sequência temporal dos exames (viria a se tornar o `dia_relativo`, adicionado ao modelo só em 2026-06-25 — ver Parte 4) e dos scripts de carregamento de dados. Reorganização dos arquivos de documentação.

### 2026-06-11 — Documentação da API (esquema de labels já obsoleto na época) e adequação ClinicalPath

**Fonte:** `docs/documentacao_etapas_legadas.md`, seções 4 e 5 + git log (`20bb0f7`)

A documentação do endpoint `POST /api/predict` nesta data descrevia um esquema de **5 classes de duração de internação** (1–3, 4–7, 8–14, 15–30, >30 dias) — esquema completamente diferente do atual (que cruza outcome × tipo de atendimento × duração) e substituído em 2026-06-24/25 (Parte 4).

**Dois bugs de inferência corrigidos nesta data:**

| Bug | Causa | Correção |
|---|---|---|
| Tokenização incompatível | `InferenceEngine` usava hash MD5 do nome do analito; modelo treinado com tokens `{analito}_{classificação}` — vocabulários incompatíveis | Vocabulário canônico via `scripts/build_standard_vocab.py`, carregado junto ao checkpoint |
| `risk_score` incorreto | Código retornava `probs[0,1]` (probabilidade da classe de índice 1) em vez de probabilidade acumulada por gravidade | `risk_score = Σ prob × linspace(0,1,n_classes)` |

**Adequações ao formato ClinicalPath v2.0** — comparação entre o que o `ClinicalPathExporter` gerava e o formato esperado:

| Item | Problema | Status na data |
|---|---|---|
| Coluna 3 do export | MOSAIC-FL colocava fase clínica (`OUTPATIENT=-2`...) em vez de categoria de resultado laboratorial (-2 muito baixo...+2 muito alto) | **Corrigido** com `_result_category()` |
| `node-inline-time-complete.txt` | 8 colunas geradas vs 6 esperadas (`sex_ref_low`/`sex_ref_high` extras) | **Corrigido** |
| `network.txt` (lista de exam_ids por paciente) | Ausente — bloqueava carregamento no ClinicalPath | **Pendente** |
| `time-metadata.txt` | Esquema de IDs incompatível (formato JGraphX) | **Bloqueado** — requer código-fonte Java não disponível |
| `patient-metadata.txt`, `list_exams.txt` | Ausentes | **Pendente**; `list_exams.txt` com campos `FL_PROB_*` bloqueado por autorização pendente do Prof. Claudio Linhares (autor do ClinicalPath original — email enviado, resposta não registrada nos documentos) |

### 2026-06-14 → 2026-06-15 — Correções e RAG real

**Fonte:** git log (`b15a65d`, `2ad6c6a`, `c89e393`) + `AVALIACAO_PROJETO.md` ("Sessão 5: Refinamento")

Correção de vários bugs, atualização do `TODO.md`. `CheckpointStore` como ABC introduzida. RAG passa a integrar tensores **reais** armazenados no banco (antes usava apenas contexto sintético). Pesos de experimentos gravados no SQLite; pesos de homologação/produção gravados no PostgreSQL — esta distinção de backend por ambiente persiste (hoje: `SQLiteCheckpointStore` vs `PostgreSQLCheckpointStore`, ambos atrás da mesma interface ABC, modularizados em pacote próprio em 2026-07-01). `hospital_id` explícito adicionado ao `DataSourceFactory`.

---

## Parte 3 — Interoperabilidade e primeira avaliação com rigor científico (2026-06-23 → 2026-06-24)

### 2026-06-23 → 2026-06-24 — FHIR R4, split por hospital, calibração

**Fonte:** git log (`de12b2e`, `526cfa1`, `5b62d6f`, `e466e0a`) + `AVALIACAO_PROJETO.md` ("Sessão 6: Interoperabilidade")

Após um hiato de ~8 dias sem commits (06-15 → 06-23), retomada com: evolução de calibração e rastreabilidade (`de12b2e`); separação da estrutura para simular servidor/cliente em rede local, simulando hospitais fisicamente distintos (`526cfa1` — este é o antecedente direto do modo "Rede Federada Real" documentado no README); carregadores de dados ajustados para separar a carga por `hospital_id` (`5b62d6f`); implementação do **FHIR R4** completo (`e466e0a`, 33 testes) — `RiskAssessment` com `correlation_token` efêmero para satisfazer o requisito obrigatório de `subject` do FHIR sem expor identidade de paciente.

`evaluate()` integrado ao simulador com relatório pré/pós-calibração.

### 2026-06-24 — BEHRT vs Random Forest e modularização do service.py

**Fonte:** git log (`de1df1e`, `1e8f8e0`) + `AVALIACAO_PROJETO.md` ("Sessão 7: Avaliação crítica")

Criada a comparação formal entre BEHRT e Random Forest (`de1df1e`) — justificativa para manter BEHRT como transformer do projeto, mesmo que o RF viesse a superá-lo em quase todos os experimentos até o T15 (ver Parte 4/5). `service.py` separado em classes para maior modularização (`1e8f8e0`) — primeiro precedente de refatoração por responsabilidade única no projeto, ~1 semana antes da modularização em massa de 2026-07-01. Correção de bug em `class_weights`.

### 2026-06-24 — Avaliações 1, 2 e 3: rigor científico aplicado pela primeira vez

**Fonte:** `AVALIACAO_PROJETO.md`, "Avaliação 1", "Avaliação 2", "Avaliação 3"

Primeira aplicação de critérios de **rigor científico** (além de qualidade de engenharia) — nota cai de 9,18 (critério anterior) para um novo patamar, medido com régua diferente:

| Avaliação | Nota Acadêmica | Nota Prod. Clínica | O que mudou |
|---|---|---|---|
| 1 (manhã) | 6,5/10 | 7,0/10 (revisado de 7,5→7,0 no próprio doc) | Primeira aplicação do critério de rigor científico |
| 2 (mesma data) | 6,5/10 (inalterada) | 8,0/10 (+0,5) | 2 penalidades incorretas corrigidas |
| 3 (mesma data, holística) | **7,0/10** (+0,5) | 8,0/10 (inalterada) | Considerado todo o histórico (50 commits, 24 dias) |

**Penalidades da Avaliação 1** (a mais severa da série):
- **−1,5 — o maior risco acadêmico identificado no projeto até então:** `outcome_class=4 → "internacao_prolongada"` no FAPESP significa "em atendimento no momento do snapshot" — é um **estado de censura**, não um prognóstico real. Um modelo treinado com esse label aprende a identificar registros capturados cedo, não pacientes com internação de fato prolongada. Esta penalidade motivou diretamente o redesign completo do esquema de labels (Parte 4).
- −0,5 — arquitetura BEHRT simplificada demais (`embed_dim=64` vs 288 do BEHRT original / 768 do Med-BERT; `num_layers=2` vs 6; sem embeddings de idade/visita/segmento)
- −0,5 — ausência de baseline comparativo formal
- −0,5 — RAG com DistilGPT-2 sem avaliação formal de qualidade

**Correções na Avaliação 2:** audit trail LGPD já estava completamente implementado (penalidade reduzida de −0,5 para −0,1, resíduo = falta de imutabilidade WORM no `RotatingFileHandler`); CORS `"*"` já era bloqueado por `ValueError` em produção (penalidade removida). Observação anotada, não penalizante: `experiment_results.json` sintético mostrava `exp5` com acurácia constante 0,4667 por 11 rodadas — indício de que o `ConvergenceTracker` confundia platô estático com convergência real (dado sintético — não afetou a nota, mas ficou registrado como algo a observar quando dados reais entrassem em cena).

**Achado crítico transversal da Avaliação 3, holística:** até esta data, **todos os 6 experimentos do projeto haviam usado dados sintéticos** — a hipótese central do projeto (FL clínico com BEHRT em dados reais) nunca havia sido validada empiricamente. Isso muda completamente na semana seguinte (Parte 4). Baseline RF penalidade reduzida de −0,5 para −0,2 (implementado, mas só com dados sintéticos).

---

## Parte 4 — Redesign de labels e primeiros treinos com dados reais (2026-06-24 → 2026-06-25)

### 2026-06-24/25 — Eliminação do label censurado, esquema de 5 classes

**Fonte:** `docs/documentacao_etapas_legadas.md` (nota de atualização, seção 4) + `AVALIACAO_PROJETO.md` "Avaliação 4" + git log (`6193d45`, 06-24 23:43)

Marco técnico central desta fase: eliminação do label `internacao_prolongada` (censurado, ver Parte 3) e adoção de um esquema que cruza 3 dimensões observáveis no momento real do desfecho — `outcome_class` (curado/melhora) × `attendance_type` (internado/pronto-socorro) × `duration_days` (limiar de 10 dias). Resultado: as **5 classes que persistem até hoje**, implementadas em `preprocessor.py:_map_outcome()` (hoje `preprocessor/outcomes.py`):

| Classe | Critério |
|---|---|
| 0 — curado_pronto | outcome=0 (curado), não internado |
| 1 — curado_internado | outcome=0, internado |
| 2 — melhora_pronto | outcome=1 (melhora), não internado |
| 3 — melhora_internado_breve | outcome=1, internado ≤ 10 dias |
| 4 — melhora_internado_grave | outcome=1, internado > 10 dias |

Casos administrativos, de transferência e evasão são excluídos. O **limiar de 10 dias foi escolhido por critério técnico (distribuição observada nos dados), sem referência clínica formal citada** — isto ficou registrado como questão aberta para a orientadora (ver "Questões em aberto" ao final deste documento).

### 2026-06-25 — Avaliação 4: salto de nota com o redesign de labels

**Fonte:** `AVALIACAO_PROJETO.md`, "Avaliação 4"

| Métrica | Valor |
|---|---|
| Nota Acadêmica | **8,3/10** (+1,3 vs Avaliação 3 — "principal alavanca desta sessão") |
| Nota Produção Clínica | 8,0/10 (inalterada) |

Escopo expandido de "apenas internados" (~14k pacientes) para "todos os atendimentos" (33.773 pacientes: BPSP 28.599 + HSL 5.174). Evidência empírica de não-IID apresentada nesta avaliação (dry-run com dados reais, antes do primeiro treinamento oficial):

| Hospital | Classe dominante | % |
|---|---|---|
| BPSP (28.599 atend.) | curado_pronto | 55,6% |
| HSL (5.174 atend.) | melhora_pronto | 61,5% |

Esta assimetria de classe dominante entre hospitais é a origem direta da discussão sobre FedAvg vs FedNova que ocupa boa parte da Parte 5.

Penalidade de label reduzida de −1,5 para −0,4 (limiar de 10 dias ainda sem referência citada). Baseline RF: −0,2 → −0,1 (dados reais carregados, resultado ainda pendente). Ajuste positivo de +0,3 pela evidência empírica de non-IID.

**Três questões abertas registradas nesta avaliação para a orientadora** (sem resposta localizada nos documentos lidos até 2026-07-01):
1. O limiar de 10 dias entre "breve" e "grave" tem respaldo na literatura clínica?
2. Misturar pronto-socorro e internados no mesmo modelo muda a pergunta clínica que está sendo respondida?
3. A ablação de late fusion demográfica deve entrar nos resultados finais da defesa?

---

## Parte 5 — Primeiros treinamentos com dados reais: Experimentos 1–9 (2026-06-25 → 2026-06-28)

Config comum a todos os experimentos desta parte: 2 clientes (BPSP=28.599 atendimentos, HSL=5.174), vocabulário 648 tokens, `max_seq_len=128`, `SimplifiedBEHRT` (embed_dim=64, num_layers=2, num_heads=4). Split 70/10/10/10 (treino/val/cal/teste) a partir do Exp3 — Exp1 e Exp2 usaram 80/10/10, sem conjunto de calibração isolado.

**Fonte principal desta parte:** `docs/Sumario_Treinamento.md`, cruzado com `docs/Sumario_Treinamento_Parte2.md` (que renomeia estes mesmos experimentos para T1–T9 na sua tabela retrospectiva) e git log.

### Experimento 1 (2026-06-25, 06:18–07:16) — primeiro treino com dados reais

Primeiro treinamento FL com dados reais FAPESP. 20 rodadas, FedProx μ=0,01, batch=16, épocas locais=2.

**Resultado:** Acc=58,0% (R20), Macro AUC=0,740, Macro F1=0,359, ECE pré=0,059, ECE pós=0,098 (temperature scaling **piorou** — primeira ocorrência de um padrão que se repetiria em todos os experimentos seguintes). F1 `melhora_pronto`=0,083.

**Bug encontrado:** crash pós-simulação — `AttributeError: 'list' object has no attribute 'get'` em `run_experiments_simulation.py` (formato de retorno de `history` incompatível com o esperado pelo caller). RAG e RF não executaram nesta run. Corrigido antes do Exp2.

### Experimento 2 (2026-06-25, 07:51–08:12)

Mesmos hiperparâmetros do Exp1, após correção do crash.

**Resultado:** **convergência atingida na rodada 7** — única convergência real registrada nesta fase inicial do projeto. Acc=52,5%, Macro AUC=0,767, Macro F1=0,287. ECE pré=0,061, pós=0,064. RAG P@3 macro=0,134. RF Centralizado: Acc=68,1%, AUC=0,786, F1=0,504 — já 15,6 p.p. acima do FL nesta fase.

### Experimento 3 (2026-06-25, ~08:40–09:54) — split 70/10/10/10 adotado

Primeira execução com split 70/10/10/10 e conjunto de calibração isolado (3.376 amostras). Marcadores estruturados `FL_TRAINING_COMPLETE`/`TREINAMENTO_COMPLETO` adicionados aos logs, permitindo parsing automatizado depois.

**Resultado:** Acc=55,8% (R20, não convergiu), Macro AUC=0,755, Macro F1=**0,398** (pico do F1 `melhora_pronto`=0,397 até este ponto do projeto). ECE pré=0,087, pós=0,102 (piora, mas MCE — erro máximo de calibração — melhorou de 0,445 para 0,229). RF Centralizado: Acc=68,0%.

**Ablação late fusion demográfica (primeira execução do projeto):** Config A (sem demográficos)=54,5% Acc/0,398 F1; Config B (late fusion idade+sexo)=**67,3%** Acc/0,449 F1. Δ=+12,7 p.p. — ganho substancial nesta primeira medição.

**Decisão:** split 70/10/10/10 adotado permanentemente — necessário para que a calibração seja metodologicamente correta (conjunto de calibração independente do de teste).

### Experimento 4 (2026-06-25, 12:48–13:58)

Primeiro uso do pipeline refatorado via `make training-full`. Hiperparâmetros idênticos ao Exp3.

**Resultado:** Acc=54,75% (R20), Macro AUC=0,7616, Macro F1=0,3661. ECE pré=0,041, pós=0,087. Ablação: Config A=62,8% (anormalmente alto), Config B=69,6% — Δ=+6,8 p.p., mas o valor de A é tratado como anomalia atribuída à variância estocástica de seed=42, não a um efeito real.

**Bug encontrado:** `behrt-pooled` falhou com `AttributeError: 'EvaluationReport' object has no attribute 'per_class_f1'` em `ablation.py`. Corrigido antes do Exp5. Também: `evaluation_round_20.json` foi sobrescrito pelo Exp5 (nome de arquivo fixo, sem distinção por experimento) — perda parcial de métricas de precisão/recall por classe. Este episódio motivaria, meses depois, a migração 012 (`evaluation_json JSONB` no banco).

### Experimento 5 (2026-06-25, 14:47–15:57)

`make training-full` após a correção do bug de `ablation.py`.

**Resultado:** Acc=56,55% (R20), Macro AUC=0,7217, Macro F1=0,3336. ECE pré=0,046, pós=0,069.

**Achado crítico:** `melhora_pronto` **colapsou** — F1=0,025 (vs 0,397 no Exp3). 252 das 321 amostras dessa classe foram classificadas erroneamente como `curado_pronto` — o modelo priorizou a classe majoritária do BPSP. Este colapso, junto com a alta variância entre execuções com hiperparâmetros idênticos, foi identificado como o problema estrutural que motivaria a introdução do `DiaRelativoEmbedding` no experimento seguinte.

**Ablação:** Config A=57,4%, Config B=69,1% — Δ=+11,7 p.p. (confirma a magnitude do Exp3, e por contraste classifica o resultado do Exp4 como anomalia).

### BEHRT Pooled Baseline (2026-06-25, 15:57–17:14) — artefato de pesquisa após o Exp5

BEHRT treinado com acesso ao pool BPSP+HSL sem privacidade, 40 épocas (= 20 rodadas × 2 épocas locais) — para isolar o custo real de privacidade da federação nesta fase inicial.

**Resultado:** Pooled A (sem demográficos)=63,5% Acc/0,496 F1/0,826 AUC; Pooled B (late fusion)=63,6% Acc.

**Custo de privacidade calculado nesta fase:** FL Exp5 (56,6%) vs Pooled B (63,6%) = **−7,0 p.p.** — a federação custava acurácia significativa neste ponto do projeto. RF Centralizado (68,4%) ainda superior a ambos. Esta é a referência de "custo de privacidade positivo" (federação prejudica) que seria completamente revertida no Exp15 (Parte 6).

### Experimento 6 (2026-06-25, ~20:10) — `dia_relativo` embedding: maior ganho arquitetural do projeto

Adição do `DiaRelativoEmbedding` — `nn.Embedding(62, embed_dim, padding_idx=0)`, somado ao token embedding antes do positional encoding, para representar o dia relativo à admissão de cada exame. O campo `dia_relativo` já era calculado na query SQL desde 2026-06-10, mas era descartado em `_build_tensors` até este experimento.

**Resultado:** Acc=59,63% — **+3,08 p.p. sobre o Exp5, o maior ganho de uma única alteração arquitetural em todo o projeto**. Macro AUC=0,7456, Macro F1=0,3515. MCE pré-calibração despencou de 0,736 para 0,180. F1 `melhora_pronto`: 0,025 → 0,112 (+4,5×).

**Decisão:** `DiaRelativoEmbedding` mantido permanentemente — componente arquitetural fixo a partir daqui. Motivação clínica: a progressão do quadro (ex.: PCR crescente ao longo de 48h) tem significado prognóstico diferente do valor isolado de um exame — sem essa informação temporal, o BEHRT não consegue distinguir uma trajetória em piora de uma medida pontual.

**Bug encontrado:** RAG falhou com `too many values to unpack (expected 2)` em `rag.py:90` — a mudança do formato de dados para incluir `dia_relativo` (agora uma 3-tupla) não havia sido propagada a essa função. Corrigido para o Exp7.

**Achado com sinal invertido:** ablação com late fusion deu Δ=−0,98 p.p. — a única vez, até este ponto, em que a demografia prejudicou em vez de ajudar. Atribuído a instabilidade em apenas 10 épocas de treino da ablação, não invalida o achado geral do Exp3/4/5.

### Experimento 7 (2026-06-25 22:53 → 2026-06-26 03:17) — 120 rodadas, madrugada de treino contínuo

Múltiplas mudanças simultâneas: μ do FedProx 0,01 → **0,10** (reduzir client drift, seguindo Li et al. 2020); teto de rodadas 20 → **120**; `min_rounds=20` como warm-up; correção do bug RAG da 3-tupla; `generation_config` unificada; `clean_up_tokenization_spaces=False` (GPT-2 usa BPE). **Checkpoint ainda não é guloso nesta run** — o store passa a registrar a melhor rodada como metadado, mas a avaliação final continua sendo feita na última rodada.

Ambiente: notebook Dell Inspiron 5402 (i7-1165G7), **CPU-only, sem GPU dedicada** — a run rodou 4,4h ininterruptas durante a madrugada, com aquecimento térmico notado mas sem falha.

**Resultado:** melhor rodada real foi **R89 (63,29%)**, mas a avaliação oficial foi feita na R120 (59,36%) — gap de 3,93 p.p. perdido por falta de checkpoint guloso. Este gap é exatamente a evidência que motiva o Exp8. Macro AUC=0,7703 (+0,025 vs Exp6). Macro F1=0,3837. ECE pré=**0,0326** — melhora drástica de calibração nativa vs 0,105 do Exp6. **`melhora_pronto` AUC: 0,654 → 0,836 (+0,182 — o maior salto de AUC em uma única classe de todo o projeto).** RAG funcionou pela primeira vez sem erro (P@3 macro=0,110).

### Experimento 8 (2026-06-26, 13:05–17:30) — checkpoint guloso

Implementação do checkpoint guloso: `save()` no PostgreSQL sempre que `acc_global > best_accuracy`; `load_best()` restaura o melhor checkpoint antes da avaliação final. Correção paralela de um bug de calibração: parametrização em log-space (`T = exp(log_T)`) para impedir que o otimizador LBFGS encontre uma temperatura negativa.

**Resultado:** **novo recorde do projeto — Acc=66,61% (R91)**, restaurado via `load_best()`, contra 58,27% na última rodada (gap de 8,34 p.p. recuperado só pelo checkpoint guloso). Macro AUC=0,8096 (+0,039), Macro F1=0,4812 (+0,098). **`melhora_pronto` F1: 0,249 → 0,619 (+0,370 — a maior evolução de qualquer métrica de classe em todo o projeto)**, AUC 0,836→0,920.

**Bug crítico encontrado:** sem o log-space, o LBFGS havia saltado para **T=−8,9997** em uma tentativa anterior, destruindo completamente a calibração pós-treino (ECE inválido=0,3335). Corrigido em `calibration.py`. `make recalibrate` executado em 2026-06-26 19:23 confirmou o fix (T=1,0849, positivo) — mas revelou que **temperature scaling continuou piorando a calibração** mesmo corrigido (ECE pós=0,1066 > ECE pré=0,0859), confirmando que o problema é estrutural, não um bug de implementação. Recomendação registrada na época: usar T=1,0 (sem calibração) para o checkpoint R91 em produção — esta recomendação seria substituída pela calibração isotônica a partir do Exp13.

**Outro bug:** BEHRT Pooled omitido neste experimento por `POOLED_EPOCHS` desacoplado (240 em vez de 120 épocas equivalentes). Corrigido (`pooled_epochs=120` fixado em `FedConfig`), retorna no Exp9.

Custo de privacidade nesta fase: FL (66,61%) vs RF Centralizado (68,20%) = gap de 1,59 p.p. — o menor gap até então, sinalizando que o FL estava se aproximando do RF pela primeira vez.

### Experimento 9 (2026-06-28, 07:46–12:02) — adoção do FedNova (resultado inválido por bug)

**Fonte cruzada:** `Sumario_Treinamento.md` (Exp9) + `Sumario_Treinamento_Parte2.md` ("T8 → T9 — cross-contamination")

Substituição da agregação FedAvg/FedProx por **FedNova** (Wang et al. 2020, NeurIPS, arXiv:2007.14481) — normaliza os updates de cada cliente pelos passos efetivos τᵢ antes de agregar. Motivação: mesmo com μ=0,1 e 120 rodadas, o Exp7/8 continuava oscilando ±8-12 p.p. sem estabilizar. FedAvg pondera updates por número de amostras, mas BPSP (~2.502 batches/rodada) e HSL (~453 batches/rodada) — razão 5,5× — produzem updates de magnitudes completamente diferentes mesmo após essa ponderação.

**SCAFFOLD foi avaliado e explicitamente descartado** (discussão preservada em `docs/TODO.md`, não datada explicitamente mas anterior a este experimento): risco identificado de que a variável de controle do BPSP corrigisse o HSL na direção errada, dado apenas 2 clientes e heterogeneidade extrema de classes (não apenas de volume). FedNova foi preferido por não introduzir hiperparâmetro novo nem estado adicional por cliente.

**Resultado nominal:** R33 (melhor do próprio Exp9)=63,86%. Avaliação oficial registrada: 66,73%.

**Bug crítico — cross-contamination de checkpoint:** o PostgreSQL checkpoint store é compartilhado entre experimentos e não havia sido isolado por experimento. `load_best()` sem filtro retornou o checkpoint do **Exp8** (R91=0,6661, superior ao R33=0,6386 do próprio Exp9). O log mostra `checkpoint_best_loaded_postgres round=91` seguido, de forma inconsistente, por "Modelo restaurado da rodada 33". **Conclusão: o Exp9 não produziu uma avaliação válida do FedNova.** Aspecto positivo medido antes da descoberta do bug: velocidade de ~113s/rodada vs ~133s/rodada do Exp8 — FedNova ~15% mais rápido por rodada, independente da questão de correção do resultado.

**Ação corretiva decidida:** implementar isolamento por `training_id` — entregue no Exp12 (migration 011, `metrics.fl_trainings`).

---

## Parte 6 — MVP e o marco histórico: Experimentos 12–16 (2026-06-28 → 2026-06-29)

**Fonte:** `docs/Sumario_Treinamento.md` (Exp 12–17) cruzado com `docs/Sumario_Treinamento_Parte2.md` (T12–T16, "Bloco 1").

### Experimento 12 (2026-06-28 18:27 → 2026-06-29 02:07) — primeira avaliação válida do FedNova

Migration 011 aplicada (`metrics.fl_trainings` + FK `training_id` em `metrics.fl_checkpoints`, índice UNIQUE parcial). `register_training()` chamado antes do loop; UPSERT com `ON CONFLICT (training_id) WHERE training_id IS NOT NULL`; `load_best(training_id)` agora filtra explicitamente.

**Resultado:** **Acc=67,44% (R115) — novo recorde do projeto, sem cross-contamination** (confirmado por log: `checkpoint_best_loaded_postgres round=115 accuracy=0.6744 training_id=2`). Macro AUC=0,8015, Macro F1=0,4840, ECE (temperature)=0,1086. **`melhora_pronto` AUC=0,9553 — melhor resultado histórico do projeto para essa métrica.** `curado_internado` F1=0,0323 — pior histórico (N=28, classe extremamente rara).

Custo de privacidade nesta fase: FL (67,44%) vs BEHRT Pooled B (69,12%) = **−1,68 p.p.** — tratado na época como "argumento central para o TCC" (custo de privacidade existe mas é pequeno). Este resultado seria completamente superado (e invertido) no Exp15, três dias depois.

**Decisão:** FedNova adotado permanentemente como algoritmo de agregação a partir daqui — confirmado ganho de +0,83 p.p. sobre FedAvg (T8: 66,61%) com o mesmo budget e sem hiperparâmetro adicional.

### Experimento 13 — BPSP-only, leave-one-client-out (2026-06-29, 07:45–10:28, training_id=3)

Primeira execução da fase 1/4 do `make training-full` já reestruturado em pipeline de 4 fases. Conjunto completo de melhorias entra simultaneamente pela primeira vez:

| Melhoria | Motivação |
|---|---|
| `FL_INCLUDE_HOSPITALS` (leave-one-client-out) | Isolar a contribuição de cada hospital |
| DataLoader determinístico (generator por cliente) | Reprodutibilidade de shuffle |
| `FL_CLASS_LABELS` parametrizável | Flexibilidade de esquema de classes |
| **Épocas locais 2 → 1** | Reduzir divergência entre clientes (Li et al. 2020) |
| **Class weight clipping** `clamp(max=15.0)` | Peso bruto de `melhora_pronto` no BPSP chegaria a ~47–117× — causava explosão de gradiente em tentativa anterior |
| **Gradient clipping** `max_norm=1.0` | Estabilidade numérica |
| **`IsotonicCalibrator`** (Zadrozny & Elkan, 2002) adicionado ao lado do `TemperatureScaler` | Ver diagnóstico de calibração abaixo |
| Determinismo CUDA (`cudnn.deterministic=True`) | Reprodutibilidade — mas note-se: só fecha parcialmente a lacuna CPU↔GPU, ver Parte 9 |
| Ablação multi-seed (k=3: seeds 42, 7, 123) | Reduzir variância da medição de ablação |

**Resultado:** Acc=64,86% (R118), Macro F1=0,3302, Macro AUC=0,7065. **Achado central:** `melhora_pronto` F1=0,000, AUC=0,5149 (equivalente a aleatório) — **confirmação empírica direta de que o BPSP isolado (0,4% de `melhora_pronto` no treino) nunca aprende essa classe** — este é o argumento quantitativo mais direto do projeto para "a federação é clinicamente necessária", não apenas uma boa prática de privacidade.

**Calibração:** primeira vez que a isotônica supera o temperature scaling — ECE pré=0,0447; Temperature (T=1,5266, maior valor já registrado no projeto)=0,0921 (piora, 9º experimento consecutivo em que isso acontece); **Isotônica=0,0237 (−47% vs pré)**.

**Decisão:** calibração isotônica OvR torna-se o calibrador de referência do projeto a partir deste experimento.

### Experimento 14 — HSL-only (2026-06-29, 10:28–10:57, training_id=4)

Segunda fase — treino apenas com HSL, avaliado no test set global (dominado por BPSP).

**Resultado:** Acc=40,05% (R100) — muito abaixo do BPSP-only. Macro F1=0,2853, Macro AUC=0,6572. Regressão severa na R120 (24,16%) — sinal de instabilidade. Duração de apenas 18,9 min (HSL tem 226 batches/rodada vs 1.252 do BPSP — 6,7× mais rápido). ECE pré=0,2997 (altíssimo — o modelo treinado só com HSL está completamente desalinhado com a distribuição global dominada pelo BPSP); T=1,9887 (maior temperatura do projeto); Isotônica=0,0466 (ainda a melhor, mesmo neste cenário adverso).

**Achado com sinal invertido:** ablação com late fusion **piorou** a acurácia em −4,06 p.p. — a única vez no projeto em que isso aconteceu de forma consistente (não anômala como no Exp6). Hipótese registrada: os demográficos criam um viés específico do perfil HSL que não generaliza para o BPSP quando não há dados do BPSP no treino para balancear.

### Experimento 15 — Federado BPSP+HSL com pipeline MVP completo (2026-06-29, 10:57–14:06, training_id=5) — MARCO PRINCIPAL DO PROJETO

Terceira fase — federado completo com 2 clientes, todas as melhorias MVP simultâneas pela primeira vez em conjunto (local_epochs=1, grad clipping, class weight clipping, DataLoader determinístico, isotônica OvR, FedNova com scoping).

**Resultado:** **Acc=69,59% (R79) — novo recorde absoluto do projeto**. Macro AUC=0,8181, Macro F1=0,4946. **ECE isotônica=0,0149 — a melhor calibração de todo o projeto** (temperature scaling piorou mais uma vez: 0,0575→0,0849, T=1,1322 — 10º experimento consecutivo confirmando o padrão).

**Marco histórico do projeto:** primeira vez que o BEHRT-FL federado supera o baseline RF centralizado — RF=68,41% vs FL=69,59% (+1,18 p.p.).

**Reversão do sinal do custo de privacidade:** FL (69,59%) vs BEHRT Pooled B com budget equivalente (68,68%) = **+0,91 p.p. — o FL supera o modelo centralizado.** Isso inverte completamente a narrativa observada no Exp5 (−7,0 p.p., federação prejudicava) para "privacidade tem benefício líquido" neste dataset. Hipótese de mecanismo, registrada no Sumário: o FedNova atua como regularizador implícito diante do non-IID, evitando que o BPSP (5,5× maior) suprima o sinal do HSL — o mesmo mecanismo que resolve o viés de agregação parece também melhorar a generalização.

**Ablação:** Δ=−15,03 p.p. — a maior penalização negativa de todo o projeto, com alta variância (±9,34%). Atribuída a instabilidade do ramo demográfico em apenas 10 épocas de ablação, com distribuições demográficas conflitantes entre BPSP e HSL — não deve ser lida como conclusão isolada sobre demografia.

### Experimento 16 — BEHRT Pooled, budget equivalente (2026-06-29, 14:06–17:28) — quarta fase

BEHRT centralizado (pool BPSP+HSL, sem privacidade) com 120 épocas — budget metodologicamente equivalente ao FL de 120 rodadas.

**Resultado:** Pooled A (sem demográficos)=68,29%, Pooled B (late fusion)=**68,68%**. RF Centralizado (nesta fase)=68,88%.

**Conclusão registrada como marco do TCC:** "pela primeira vez o BEHRT-FL federado (69,59%) supera todos os baselines centralizados com budget equivalente." Custo de privacidade da federação declarado negativo.

**Duração total do primeiro `make training-full` completo (Exp13–16):** 583 min (9h43min) em CPU.

---

## Parte 7 — Sessão de 2026-06-29 (tarde/noite): RAG Ollama, seeding, API de inferência

**Fonte:** `docs/Sumario_Treinamento_Parte2.md`, seções "Sessão 2026-06-29" e "API de Inferência"

### Migração do backend RAG: DistilGPT-2 → Ollama/gemma3:4b

DistilGPT-2 havia sido criticado formalmente na Avaliação 1 e 3 (`AVALIACAO_PROJETO.md`, penalidade −0,5, "qualidade de justificativa não avaliada"). O backend do RAG foi tornado configurável (`FL_LLM_BACKEND`, `FL_LLM_MODEL`), com **Ollama + gemma3:4b** como padrão operacional — motivação registrada: maior suporte à língua portuguesa do Brasil que o DistilGPT-2 (treinado majoritariamente em inglês), e isolamento de processo (modelo de 4B parâmetros roda fora do processo Python de treino, evitando pressão de memória durante `make training-full`). `distilgpt2` permanece como padrão do *código* (não do ambiente operacional) para que `make test` funcione sem exigir Ollama instalado.

**4 bugs corrigidos nesta sessão** (detalhe completo já registrado em `Sumario_Treinamento_Parte2.md`, resumo aqui):

| Bug | Sintoma | Fix |
|---|---|---|
| Special tokens (`[PAD]`, `[CLS]`, `[SEP]`) contaminavam a knowledge base do RAG | Apareciam como top attention tokens por construção, não por sinal clínico | `_is_clinical_token()` filtra qualquer token iniciado por `[` ou `<` |
| `str(...).replace("", "adulto")` corrompia a KB inteira | Em Python, `"texto".replace("", "x")` insere `"x"` entre **cada caractere** quando a string a substituir é vazia | Guard `if idade_exacta:` antes do `replace()` |
| `tokenizer.encode()` chamado antes do dispatch de backend | Com Ollama ativo, `self.tokenizer=None` → `AttributeError` | Dispatch movido para o topo de `generate_justification()` |
| Fallback HF tentava carregar `"gemma3:4b"` como repo ID do HuggingFace | `:` não é válido em nomes de repositório HF | Campo `llm_hf_model` (padrão `distilgpt2`) adicionado a `RuntimeConfig` |

### Seeding determinístico por rodada × cliente

`torch.manual_seed(FED_CFG.random_seed + current_round * FED_CFG.num_clients + client_id)` adicionado ao início de cada `fit()` do cliente — elimina a variância residual de runs independentes com os mesmos hiperparâmetros produzindo resultados ligeiramente diferentes. Implementado junto com DP-FedAvg para poder separar os dois efeitos na análise (ver Parte 8).

### API de Inferência — fechamento do gap treino↔produção

Até esta data, o treinamento salvava checkpoints no `CheckpointStore` (PostgreSQL/SQLite), mas a API só lia arquivos `.pt` em disco — que nunca existiam. Três mudanças fecharam o gap:

1. `InferenceEngine.load_from_store(checkpoint: dict)` — carrega pesos, vocabulário e metadados diretamente do dict retornado por `CheckpointStore.load_best()`
2. Fallback em `state._get_engine()`: procura `.pt` em disco → tenta `CheckpointStore.load_best()` → sobe sem modelo com WARNING (nunca trava)
3. `make api` (sobe banco + API na porta 8000) e `make export-checkpoint` (extrai checkpoint do banco para arquivo, para deploy offline)

Verificado ao vivo: `POST /api/predict` retornando 200 OK com checkpoint carregado do PostgreSQL (round=79, acc=0,6959 — o checkpoint do Exp15). 10 novos testes em `test_inference_engine_store.py`; suíte total (41 integração + 10 novos unit) passando.

---

## Parte 8 — Correção do split, gaps de observabilidade e o Bloco 2 (2026-06-29 noite → 2026-06-30 tarde)

**Fonte:** `docs/Sumario_Treinamento_Parte2.md`, seções "Sessão 2026-06-30 — Revisão de partições e seeds" e "Correção de gaps"

### Investigação da regressão de −5 p.p. no HSL (motivada pela pergunta da autora)

Run de Validação de 2026-06-29/30 (seeding fix + RAG fix, sem correção de split ainda — ver detalhe completo já registrado em `Sumario_Treinamento_Parte2.md`) atingiu 70,19% (novo recorde do Bloco 1), mas o HSL isolado regrediu de 40,05% para 35,05%. A autora questionou se isso vinha da partição usada no treinamento anterior — pergunta que motivou uma auditoria completa de como o projeto gerencia seeds e partições em todos os pontos de inicialização de dados.

**Conclusão da auditoria:** o split é determinístico e consistente entre todos os experimentos (FL, BEHRT pooled, RF, ablação recebem os mesmos `client_loaders` de uma única chamada) — mas foi encontrado e corrigido um bug real:

**Bug encontrado e corrigido — RNG compartilhado entre hospitais:** em `dataloaders.py`, um único gerador `rng` (seed=`RANDOM_SEED`) era compartilhado sequencialmente entre hospitais — a permutação do HSL dependia implicitamente de quantas amostras o BPSP tinha processado antes (o RNG "avança" `n_BPSP` passos antes de gerar a permutação do HSL). Qualquer mudança no tamanho do dataset BPSP mudaria o split do HSL, mesmo sem alterar nenhuma seed.

**Fix:** gerador independente por hospital — `torch.Generator().manual_seed(RANDOM_SEED + 1000 + cid)`. BPSP → seed 1042, HSL → seed 1043.

**A hipótese original da autora (interferência do `torch.manual_seed` por rodada no shuffle do DataLoader) foi investigada e refutada** — o DataLoader usa um `generator` explícito, que o PyTorch prioriza sobre o estado global de `torch.manual_seed`. A causa real da regressão do HSL foi a mudança de trajetória do **dropout** (que é controlado pelo estado global, não pelo `generator` do DataLoader) — datasets pequenos como o HSL (3.621 amostras) são sensíveis a essa mudança; o BPSP (20.019 amostras) dilui o efeito estatisticamente. Não foi um bug: é sensibilidade esperada de dataset pequeno a mudanças de aleatoriedade do dropout.

**Consequência metodológica:** todos os treinamentos de T1 a T8 (Bloco 1, incluindo o recorde de 70,19%) usaram o RNG compartilhado — **não são diretamente comparáveis** aos treinamentos a partir daqui (Bloco 2).

### Correção de 5 gaps de observabilidade (2026-06-30)

Sessão dedicada a corrigir riscos de incorreção latente antes de iniciar o Bloco 2:

| Gap | Problema | Fix |
|---|---|---|
| Gap 3 | Nome de arquivo `evaluation_round_{round_num}.json` ambíguo — `round_num` era a última rodada, não a melhor | Renomeado para `evaluation_best_r{best_round}_of_{round_num}.json`; payload com campos explícitos `best_round`/`total_rounds` |
| Gap 4 | API podia servir o checkpoint errado — `load_best()` sem `training_id` retornava o de maior accuracy global, não necessariamente o federado | Resolução em 2 camadas: `FL_TRAINING_ID` (env) → `last_federated_training_id.txt` (gravado automaticamente pelo orquestrador) → `None` |
| Gap 5 | `tau_eff` do FedNova era logado em texto mas descartado — análise pós-hoc exigia re-parsing de log | Migration 013 (`fl_round_history`), coluna `tau_eff REAL`, persistido por rodada |
| Gap 6 | Critério de checkpoint (accuracy) favorecia a classe majoritária — ver discussão de trade-off abaixo | `f1_macro` adotado como critério padrão do Bloco 2, parametrizável via `FL_CHECKPOINT_CRITERION` |
| Gap 7 | Edge case: se nenhuma rodada melhorasse acima de 0,0, `best_round=0` causava acesso ao último elemento em vez do primeiro (`history[-1]`) | Guarda centralizada após o loop |

**Discussão do Gap 6 (a mais substancial da sessão):** os dados do Run de Validação (70,19% Acc, F1 macro=0,4994, F1 `curado_internado`=0,000) mostraram um gap de 20 p.p. entre accuracy e F1 macro, sustentado quase inteiramente por `curado_pronto` (48% do dataset, F1=0,83). **Decisão: rastrear os dois critérios simultaneamente e usar `f1_macro` como critério padrão**, em vez de trocar para F1 de uma classe específica — com 28 amostras de `curado_internado` no teste, um critério baseado nessa classe isolada oscilaria por ruído estatístico, não por melhora real do modelo. Migration 014 registra qual critério foi usado em cada `training_id`, tornando comparações entre runs metodologicamente verificáveis.

### Bloco 2, Treinamento 1 — CPU (2026-06-30 manhã, também referenciado como "CPU 2026-06-30", training_ids 9-11)

Primeiro treinamento após a correção do split e a mudança de critério de checkpoint. Log: `experiments/logs/run_complete_20260630_091435.log`. Referenciado também como o commit `5c50a8d` ("Dados treinamento manha dia 30 de junho de 2026").

**Resultado da fase federada (training_id=11, componente principal):** Acc=**65,90%** (R77), F1 macro=**0,4905**, Macro AUC=**0,8105**, ECE isotônica=**0,0311**. **Não convergiu em 120 rodadas** — primeiro run do projeto sem convergência (exceto runs com DP severo, que não haviam sido executados ainda). τ_eff constante=1.095,0 em todas as 120 rodadas (valor correto, calculado a partir do número real de batches — a expectativa prévia de 40-80 estava simplesmente mal estimada, sem considerar passos locais reais).

Per-class F1 melhorou em **todas** as classes com amostras suficientes em relação ao Bloco 1, apesar de accuracy global menor (65,90% vs 70,19%) — evidência direta de que o critério F1 macro produz um modelo mais equilibrado, sacrificando um pouco da classe majoritária.

**Inversão em relação ao Bloco 1:** BEHRT Pooled A (69,51%) supera o FL federado (65,90%) em 3,61 p.p. — no Bloco 1 (T15) era o oposto. Isso é interpretado no próprio documento como resultado **mais honesto metodologicamente** (o split anterior pode ter favorecido artificialmente o FL) e não invalida o projeto: o custo de privacidade volta a existir, mas de forma mensurável corretamente.

**Recursos computacionais (baseline CPU, ponto de referência para GPU e pós-refactoring):**

| Fase | Duração | Peak RAM | CPU médio |
|---|---|---|---|
| BPSP-only | 67 min | 2.445 MB | 2.353% (~23 núcleos) |
| HSL-only | 7 min | 2.299 MB | 2.368% |
| Federado | 121 min | 2.295 MB | 2.358% |
| BEHRT Pooled A+B | ~186 min | — | — |
| **Total** | **~420 min (7h)** | **pico 2.445 MB** | **~23 núcleos** |

### Decisões de sequenciamento tomadas nesta sessão

A decisão de **não** trocar o critério de checkpoint para F1 de uma classe específica antes da refatoração, e **não** experimentar ajustes de `class_weights` antes da GPU, foi justificada por custo: cada ciclo de treino levava ~7h em CPU — caro demais para iteração de hiperparâmetro. **Sequência definida nesta sessão** (a decisão de refatorar *depois* da GPU, e não antes, foi da própria pesquisadora — argumento: ter os 3 pontos de comparação, CPU atual / GPU atual / GPU refatorado, torna o capítulo de resultados mais sólido):

```
1. Instalar driver NVIDIA (RTX 4070 Ti)
2. Verificar GPU operacional
3. make training-full em GPU — Bloco 2 Treinamento 1 na GPU
4. Comparar recursos CPU vs GPU
5. Refactoring MVP (modularização)
6. Confirmar resultados equivalentes pós-refactoring
7. Simulação distribuída (desktop servidor + notebook cliente)
```

---

## Parte 9 — GPU: bugs de device e a questão da reprodutibilidade CPU↔GPU (2026-06-30 noite)

**Fonte:** `docs/Sumario_Treinamento_Parte2.md`, seção "Sessão 2026-06-30 (noite)"

### Correção de uma afirmação anterior

A sessão da tarde havia registrado que "o DEVICE já é detectado automaticamente via `torch.cuda.is_available()`" — **isso estava incorreto**. `config.py` usa `torch.device(os.getenv("FL_DEVICE", "cpu"))`: o padrão é sempre CPU, sem auto-detecção. `FL_DEVICE=cuda` precisa ser exportado explicitamente. Target `make training-full-cuda` criado no Makefile para embutir essa variável nas 4 fases.

### Dois bugs de device corrigidos (nunca haviam sido exercitados, porque `DEVICE` sempre fora `cpu` até esta sessão)

| # | Arquivo | Sintoma | Causa | Fix |
|---|---|---|---|---|
| 1 | `fl_core.py:258` | `RuntimeError` em `aggregate_fednova`, 1ª rodada | Parâmetros do cliente reconstruídos com `torch.tensor(v)` sem `device=`, sempre em CPU, enquanto o modelo global estava em CUDA | `torch.tensor(v, device=DEVICE)` |
| 2 | `checkpoint_store.py:68` | `TypeError` ao salvar o primeiro checkpoint | `_model_version()` chamava `.numpy()` direto em tensor CUDA | `.cpu().numpy()` |

Um **terceiro e quarto bug** foram encontrados em uma sessão de continuação (mesma noite/madrugada seguinte, ver Bloco 2 Treinamento 2 abaixo): o RAG falhava silenciosamente (capturado por `try/except` no orquestrador) por tensores de batch não movidos para `DEVICE` antes do forward em `interpretability.py`.

### Fase BPSP-only na GPU: velocidade e a questão de convergência

Primeiro run completo da fase 1: **~9,5× mais rápido por rodada** que o baseline CPU. `nvidia-smi` confirmou o processo ativo na GPU (338 MiB, 47% de utilização, 49°C, 74W/285W).

**Pergunta da autora:** "ou seja, no ultimo treinamento teve convergencia e nessa nao? isso nao seria derivado da partition escolhida no treinamento anterior?"

**Investigação:** a partição foi descartada como causa — logs de ambos os runs mostram split idêntico (o split é sempre calculado em CPU, independente de `DEVICE`). **Causa real: não-reprodutibilidade numérica entre CPU e GPU no PyTorch — limitação documentada da própria biblioteca** ([PyTorch Reproducibility Notes](https://pytorch.org/docs/stable/notes/randomness.html)), não um bug do projeto. Dois motivos técnicos: (1) CPU usa RNG derivado de Mersenne Twister, CUDA usa Philox — mesma seed produz sequências diferentes, afetando o dropout desde a rodada 1; (2) `cudnn.deterministic=True` cobre principalmente seleção de algoritmos de convolução — o modelo não tem camadas conv (é Transformer), então não fecha a lacuna de não-associatividade de ponto flutuante em reduções paralelas de GPU (softmax da atenção, acumulação de gradiente).

**Implicação metodológica registrada para o TCC:** comparar CPU vs GPU com um único run de cada isola apenas o efeito de *velocidade* — a trajetória de treino (e portanto accuracy/F1/convergência) muda por um motivo independente da qualidade do código. Recomendação: se accuracy/F1/convergência entrarem na comparação (não só tempo), rodar múltiplas seeds em cada device (`ABLATION_SEEDS` já suporta isso) em vez de comparar dois runs pontualmente. Para comparação de velocidade pura, um único run por device é válido.

### Bloco 2, Treinamento 2 — GPU com bug de RAG (training_ids 16-18, "GPU 2026-06-30 com bug RAG", excluído da comparação final)

Duração total do pipeline: ~33,3 min contra ~420 min em CPU — **~12,6× mais rápido**. O quarto bug de device (RAG falhando silenciosamente por tensores não movidos para `DEVICE` em `interpretability.py`) invalidou o Precision@3 do RAG nesta run — corrigido, mas só validado no próximo run.

Resultado da fase federada (training_id=18): Acc=68,03% (R73) — +2,13 p.p. vs CPU (65,90%); F1 macro=0,4988; ECE=0,0198 (melhor que CPU). **Não deve ser lido como "GPU produz modelos melhores"** — é a trajetória estocástica específica desta run, conforme a discussão de reprodutibilidade acima.

### Bloco 2, Treinamento 3 — GPU, run válido (training_ids 19-21, "GPU 2026-06-30", referência final)

Com os 4 bugs de device corrigidos (incluindo o do RAG), este é o run de referência para a comparação CPU×GPU do Bloco 2 — **não usar o Treinamento 2**, que teve o bug do RAG.

**Comparação final "CPU 2026-06-30" (ids 9-11) vs "GPU 2026-06-30" (ids 19-21):**

| Métrica (fase Federado) | CPU (id=11) | GPU (id=21) | Δ |
|---|---|---|---|
| Accuracy (best) | 65,90% (R77) | **66,73%** (R57) | +0,83 p.p. |
| F1 macro | 0,4905 | **0,5175** | +0,0270 |
| Macro AUC (pós-cal) | 0,8105 | 0,8141 | +0,0036 |
| ECE isotônica | 0,0311 | **0,0293** | −0,0018 |
| Duração | 121,0 min | **11,95 min** | ~10,1× mais rápido |

| Componente | CPU total | GPU total | Speedup |
|---|---|---|---|
| Pipeline completo (4 fases) | ~420 min (7h) | ~38,7 min | **~10,9×** |

**Conclusões registradas para o TCC:** ~10,9× mais rápido no pipeline completo é a comparação de referência. Componente federado: GPU levemente melhor em Acc/F1/ECE, e revelou pela primeira vez sinal em `curado_internado` (F1 0,000→0,1176) — mas tratado como observação de uma única trajetória estocástica, não prova de superioridade sistemática de GPU. BPSP-only e HSL-only ficaram próximos entre CPU e GPU (~1-2 p.p.), reforçando que a GPU muda a velocidade, não sistematicamente a qualidade.

---

## Parte 10 — Modularização em massa (2026-07-01)

**Fonte:** conversa direta desta sessão + `docs/Sumario_Treinamento_Parte2.md` + git log (`0dc13f2`, `c67231e`)

### Motivação e sequenciamento

A decisão de modularizar havia sido explicitamente adiada (registrada em memória de sessão) até que o MVP estivesse validado — "vamos modularizar todas as classes, pois elas estão enormes", instrução dada no início desta sessão. Ordem de execução acordada com a autora, em 3 etapas: **(1) `training_runner`** (agrupar scripts `run_*.py` em namespace próprio) **→ (2) `training/core`** (mover a mecânica FL para um subnamespace, deixando `federated_training.py` e `experiment_server.py` no nível externo) **→ (3) modularização dos arquivos grandes**.

### 15 arquivos convertidos de módulo único para pacote

Todos preservando a API pública externa (nenhum import de consumidor mudou, exceto quando um teste fazia `patch()`/`monkeypatch` direto em uma constante de módulo — nesse caso, o alvo do patch foi atualizado, não o comportamento):

| # | Arquivo original | Linhas | Pacote resultante |
|---|---|---|---|
| 1 | `preprocessor.py` | 846 | `preprocessor/` (5 arquivos) |
| 2 | `rag.py` (core) | 368 | `rag/` (3 arquivos) |
| 3 | `data_loader.py` (core) | 962 | `data_loader/` (7 arquivos) |
| 4 | `fl_core.py` | 684 | `fl_core/` (5 arquivos) |
| 5 | `checkpoint_store.py` | 553 | `checkpoint_store/` (5 arquivos) |
| 6 | `db.py` (mosaicfl_api) | 689 | `db/` (7 arquivos, mixins) |
| 7 | `datasource.py` (client) | 356 | `datasource/` (5 arquivos) |
| 8 | `runner.py` (server) | 472 | `runner/` (6 arquivos) |
| 9 | `term_manager.py` | 404 | `term_manager/` (3 arquivos) |
| 10 | `strategy.py` | 417 | `strategy/` (5 arquivos, mixins) |
| 11 | `inference_engine.py` | 377 | `inference_engine/` (4 arquivos) |
| 12 | `runner.py` (client) | 292 | `runner/` (5 arquivos) |
| 13 | `exams_extract.py` | 351 | `exams_extract/` (4 arquivos) |
| 14 | `scheduler_daemon.py` | 331 | `scheduler_daemon/` (5 arquivos, mixins) |
| 15 | `metrics_store.py` | 274 | `metrics_store/` (5 arquivos) |

Mais reorganização estrutural de `experiments/` (`training_runner/` + `training/core/`), remoção do `src/data_loader.py` órfão e quebrado (referenciava um `src/config.py` inexistente, zero importadores), e realocação de `benchmark.py`/`build_standard_vocab.py`/`datasource.py` (órfãos na raiz) para `scripts/`.

**Decisões explícitas de não modularizar:** `orchestrator.py` (204 linhas, classe `FederatedTraining`) e `client.py` (206 linhas, classe `FedProxClient`) — avaliados como já pequenos e coesos, uma classe cada com métodos de responsabilidade única; dividir seria abstração prematura. Confirmado pela autora via decisão explícita.

**Restrição técnica recorrente:** `unittest.mock.patch()` só afeta o módulo exato de onde um nome é lido como variável livre no momento da chamada ("patch where used, not where defined"). Isso forçou manter, no mesmo arquivo: constantes de módulo lidas por funções que os testes fazem patch (`DEFAULT_CONNECTION_STRING` em `data_loader/sources.py`; `CHECKPOINT_DIR`/`LOG_DIR` em `strategy/core.py`; as 4 classes instanciadas em `FederatedScheduler.__init__` de `scheduler_daemon/core.py`); e a classe `ClinicalRAG` inteira em `rag/__init__.py` (não em submódulo), porque os testes fazem patch em `mosaicfl.core.rag.sa`/`.SentenceTransformer`. Métodos de classe (ex.: `DataSourceFactory.create`) são seguros para mover livremente, porque o patch muta o objeto de classe compartilhado, não uma referência de módulo.

**Validação:** 545 testes (unit + integration) passando após cada etapa da modularização, sem nenhum import externo quebrado.

### Validação funcional pós-refactoring (não é comparação formal)

`make training-full-cuda` executado após a modularização (training_ids 22-24), com pedido explícito da autora de que este run **não** entre na comparação formal CPU×GPU pós-refactoring — é validação de que o pipeline continua funcionando, não um dado de comparação. Resultado: ~36,3 min (vs ~38,7 min do "GPU 2026-06-30" pré-refactor), mesma faixa de valores e mesmo padrão estrutural em todas as 4 fases, nenhum erro — os 4 bugs de device continuam corrigidos nos novos caminhos de arquivo. Registrado em `docs/Sumario_Treinamento_Parte2.md`.

README.md atualizado no mesmo dia para refletir a nova estrutura de diretórios, o target `make training-full-cuda`, e a documentação de que `scripts/benchmark.py` está atualmente quebrado (referencia módulos `_v2` de uma reestruturação anterior que não existem mais).

---

## Estado do projeto em 2026-07-01 (fechamento deste documento)

### O que está concluído e validado

- Pipeline completo `make training-full` / `make training-full-cuda`, 4 fases, com FedNova, checkpoint guloso escopado por `training_id`, calibração isotônica OvR, seeding determinístico, class/gradient clipping
- Melhor resultado formal do Bloco 2 (split corrigido, metodologicamente válido para comparação): Acc=65,90% CPU / 66,73% GPU (fase federada)
- RAG com Ollama/gemma3:4b, fallback automático para HuggingFace, 4 bugs de KB corrigidos
- API de inferência conectada ao `CheckpointStore`, MC Dropout, exportação FHIR R4 e ClinicalPath
- DP-FedAvg implementado (clipping + ruído gaussiano), desabilitado por padrão
- Suporte GPU funcional, ~10,9× de speedup medido e validado
- Modularização completa de 15 arquivos grandes em pacotes, 545 testes passando
- Auditoria LGPD parcial (pseudonimização, audit trail, TLS obrigatório)

### O que está pendente

| Item | Status | Bloqueia |
|---|---|---|
| Experimentos DP (Exp 17/18/19, σ=1,0/0,5/2,0) | Implementado, **nunca executado** em nenhum documento lido | Curva Acc×ε — argumento central da seção de privacidade do TCC |
| Comparação formal CPU×GPU pós-refactoring | Só validação funcional feita; comparação formal pendente | Confirmação de não-regressão do refactoring como dado citável |
| Validação qualitativa do gemma3:4b | Não executada | Argumento de qualidade da justificativa do RAG |
| Análise clínica formal dos erros críticos | Matriz de confusão existe (ex.: 67/338 casos de `melhora_internado_grave` classificados como `curado_pronto` no Exp12), mas sem análise redigida | Discussão de risco clínico na defesa |
| `network.txt`, `time-metadata.txt`, `patient-metadata.txt` do ClinicalPath | Pendentes/bloqueados desde 2026-06-11 | Integração completa com ClinicalPath (não bloqueia o argumento central do TCC) |
| `scripts/benchmark.py` | Confirmado quebrado (imports `_v2` inexistentes) | Nada crítico — `make training-full[-cuda]` mede desempenho real |

### Questões em aberto para a orientadora (extraídas de `AVALIACAO_PROJETO.md`, Avaliação 4, 2026-06-25)

1. O limiar de 10 dias entre `melhora_internado_breve` e `melhora_internado_grave` tem respaldo na literatura clínica, ou é puramente um corte de distribuição dos dados?
2. Misturar atendimentos de pronto-socorro e internações no mesmo modelo muda a pergunta clínica que está sendo respondida?
3. A ablação de late fusion demográfica (com resultados de alta variância, ex.: Δ=−15,03 p.p. no Exp15) deve entrar nos resultados finais da defesa, ou ser tratada como estudo à parte pela instabilidade?

### Trajetória da nota de avaliação formal (para referência no texto de metodologia)

| Data | Critério | Nota Acadêmica | Nota Produção Clínica |
|---|---|---|---|
| 2026-06-07 (5ª rodada) | Engenharia de software | 9,18 | 7,0 |
| 2026-06-24 (Avaliação 1) | + rigor científico | 6,5 | 7,0 |
| 2026-06-24 (Avaliação 3, holística) | idem | 7,0 | 8,0 |
| 2026-06-25 (Avaliação 4, pós-redesign de labels) | idem | **8,3** | 8,0 |

Nenhuma avaliação formal posterior a 2026-06-25 foi localizada nos documentos lidos — ou seja, os marcos mais importantes do projeto (Exp15, Bloco 2, GPU, modularização) ainda não foram formalmente reavaliados com este critério.

---

## Fechamento da Fase de Ajuste e início da Fase de Experimentos Formais (decisão de 2026-07-01)

**Decisão tomada pela autora em 2026-07-01, após revisão desta linha do tempo e do `docs/Comparativo_Metodologia_Planejada_vs_Executada.md`:**

> Tudo o que está documentado neste arquivo — Experimentos 1–17/T1–T16 (`Sumario_Treinamento.md`), Bloco 1 e Bloco 2 CPU/GPU (`Sumario_Treinamento_Parte2.md`) e a modularização de 2026-07-01 — passa a ser classificado retroativamente como **"Treinamentos de Ajuste"**, não como resultados finais comparáveis entre si.

**Justificativa registrada pela autora:** ao longo de todo esse período, bugs estavam sendo ativamente corrigidos (cross-contamination de checkpoint, bugs de RAG, bug do RNG compartilhado entre hospitais, bug de calibração em log-space, 4 bugs de device na GPU, entre outros — ver Partes 5–9 acima), o esquema de labels foi redesenhado 3 vezes, o critério de checkpoint mudou de accuracy para F1 macro, e o próprio código passou por uma reestruturação completa (15 arquivos modularizados). **Comparar resultados numéricos entre pontos diferentes desse período não é justo nem metodologicamente válido** — cada "treinamento" media, em parte, o efeito de um bug corrigido ou de uma mudança estrutural, não apenas o efeito do hiperparâmetro ou algoritmo em teste.

**Marco de corte:** a validação funcional pós-modularização executada na manhã de 2026-07-01 (`make training-full-cuda`, training_ids 22-24 — ver Parte 10) é o ponto em que a autora considera "todas as arestas ajustadas" confirmadas — pipeline rodando sem erro, comportamento estrutural consistente com o run de referência pré-refactor. **A partir deste ponto, e não antes, começam os "Treinamentos Reais"** — a serem usados como resultados formais e comparáveis no texto da defesa.

**Ressalva explícita da autora:** este corte vale **"a menos que a gente perceba outra oportunidade"** — ou seja, se um novo bug ou lacuna estrutural for identificado depois deste ponto, o marco de início dos "Treinamentos Reais" desloca para depois da correção desse novo achado. O critério não é a data em si, é a ausência de arestas soltas conhecidas.

**Consequências práticas desta decisão:**

1. Nenhum número do Bloco 1 ou Bloco 2 (T1–T16, incluindo o marco "FL supera todos os baselines" do T15/Exp15, e a comparação CPU×GPU de 2026-06-30) deve ser apresentado no capítulo de resultados do TCC como resultado final — eles continuam válidos como **narrativa de desenvolvimento metodológico** (o "como chegamos até aqui"), mas não como a tabela de resultados que responde à pergunta de pesquisa.
2. A comparação formal CPU×GPU pós-refactoring (já listada como pendente na tabela acima) deixa de ser "pós-refactoring vs. pré-refactoring" e passa a ser, junto com todo o resto, o **primeiro Treinamento Real**.
3. Os experimentos formais (revisão do que restou do plano original — Seção 10 e 16 do `Comparativo_Metodologia_Planejada_vs_Executada.md`, incluindo os nunca executados Experimento 3 e Experimento 4) devem ser desenhados e executados a partir daqui, sobre o código estável e modularizado.
4. Um novo documento de sumário (`docs/Sumario_Treinamento_Parte3.md` ou nome a definir) deve ser iniciado quando o primeiro Treinamento Real for executado, para não misturar essa fase com o histórico de ajuste já registrado em `Sumario_Treinamento.md` e `Sumario_Treinamento_Parte2.md`.

**Pendência imediata:** o desenho exato do(s) primeiro(s) Treinamento(s) Real(is) ainda não foi definido nesta sessão — fica para quando a autora retomar o trabalho.

---

## Preparação do primeiro Treinamento Real — AUC-ROC no banco + Fase 5 (Experimento 3) (2026-07-01, mesma sessão)

Antes de disparar o primeiro `make training-full-cuda` da Fase de Experimentos Formais, duas lacunas identificadas na revisão do `Comparativo_Metodologia_Planejada_vs_Executada.md` foram fechadas.

### AUC-ROC passa a ser gravado no banco

**Problema encontrado:** o Macro AUC já era calculado ao final de cada treino federado (pré e pós-calibração), mas nunca chegava a `metrics.fl_trainings` — `orchestrator.py` gravava `macro_auc: None` fixo em `metrics_store.fl_metrics`, porque o valor calculado em `manual_loop.py` nunca era devolvido no `history`. O baseline RF/Pooled já gravava o AUC real corretamente; só a fase de treino FL em si (BPSP-only/HSL-only/Federado) ficava com `None`.

**Correção:**
- Migration 016: `macro_auc`, `macro_f1`, `ece` adicionados a `metrics.fl_trainings` (aplicada).
- `manual_loop.py`: captura `report_cal.macro_auc` (com fallback para `report_raw.macro_auc`) e grava via novo método `checkpoint_store.update_evaluation_metrics()`.
- `orchestrator.py`: `metrics_store.save()` passa a ler os valores reais de `self.history` em vez de `None` fixo — corrigido o mesmo bug para `macro_f1` e `ece` (mesma causa raiz).
- SQLite recebeu o método por paridade de interface, mas só loga — o schema local do SQLite já estava atrasado em relação às migrations do Postgres antes disso (`checkpoint_criterion` e métricas de recurso da migration 015 também nunca foram persistidas lá). Não corrigido — os treinos reais usam Postgres.

**Motivação da autora:** ter o AUC-ROC disponível para todo treino futuro, para poder trazê-lo depois e justificar comparativamente a escolha de F1 macro como critério de checkpoint (em vez de accuracy ou AUC) com dado, não só com afirmação.

### Fase 5 — contraste non-IID real vs. IID simulado (fecha o Experimento 3 do plano original)

**Decisão da autora:** não criar um treino "non-IID real" separado — a Fase 3 (Federado, BPSP+HSL reais) já é isso, roda toda vez. Só a Fase 5 (IID simulado) precisava ser construída. As duas fases saem do mesmo `make training-full[-cuda]`, com seed de inicialização, algoritmo e hiperparâmetros idênticos — só a origem dos dados de cada cliente muda.

**Implementação:**
- `dataloaders.py`: novo `FL_PARTITION_MODE` (`natural` [padrão] | `iid_simulado`). Em `iid_simulado`, nova função `_build_iid_simulated_hospital_data()` agrupa todos os hospitais num pool único, embaralha com seed dedicada (`RANDOM_SEED + 2000` — namespace novo, não colide com o `+1000` do split natural), e refatia em N clientes virtuais (`IID_0`, `IID_1`, ...). Testado com dados sintéticos na razão real (~4,35×): os dois clientes virtuais saíram com distribuição de classe quase idêntica entre si e a proporção de origem hospitalar preservada nos dois — a heterogeneidade desaparece por construção, sem perda de amostras.
- Avaliação por subgrupo de origem hospitalar (`_evaluate_subgroups()`) — uma única passagem sobre o checkpoint final (não por rodada, decisão de custo já registrada anteriormente), disponível nos dois modos de partição via um novo `test_loader_origin` retornado por `prepare_dataloaders_from_db()`. A Fase 3 (natural) ganha essa métrica de brinde, já que o mecanismo é o mesmo.
- Migration 017: `partition_mode` em `fl_trainings` (aplicada).
- Makefile: `training-iid-contrast[-cuda]` como alvos standalone, e a Fase 5 encadeada dentro de `training-full`/`training-full-cuda` — agora pipelines de **5 fases** (`1/5` a `5/5`), não mais 4.

**Achado colateral, não corrigido:** bug pré-existente em `SQLiteCheckpointStore.save()` (`ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint`) — reproduzido na `main` sem nenhuma das mudanças desta sessão. Não bloqueia nada porque os treinos reais usam Postgres; registrado para o caso de o caminho SQLite (testes locais sem banco) vir a ser usado.

**Validação:** 545 testes (unit + integration) passando após cada etapa; smoke test funcional isolado de `_build_iid_simulated_hospital_data()` e de `run_federated_learning_manual()` de ponta a ponta (com stub contornando o bug do SQLite) confirmando que `partition_mode`, `macro_auc`/`macro_f1`/`ece` reais e a avaliação por subgrupo chegam corretamente no fluxo.

### Estado ao final desta sessão

**Cronograma confirmado pela autora (2026-07-01, quarta-feira):**

- **Quarta-feira (2026-07-01, hoje):** `make training-full-cuda` disparado como **validação das alterações desta sessão** (Fase 5 + gravação de AUC-ROC) — mesmo padrão já usado pela manhã para validar a modularização. **Ainda não é o Treinamento Real.**
- **Quinta-feira (2026-07-02):** se a validação de hoje estiver OK, as rodadas de **Treinamento Real** acontecem — CUDA e CPU, 5 fases cada (BPSP-only, HSL-only, Federado non-IID real, BEHRT Pooled, Federado IID simulado). É aqui que os Experimentos 1, 2, 3 e 5 do plano original ficam formalmente cobertos.
- **Sexta-feira (2026-07-03):** simulação/avaliação do RAG — Experimento 4 (avaliação Likert humana).

### Validação de quarta-feira — resultado: implementação sem erros, pronta para o Treinamento Real de quinta

**Log:** `experiments/logs/run_complete_cuda_20260701_211304.log` | **Duração:** 21:13→21:41 (~28,5 min) | **training_ids:** 25 (BPSP-only), 26 (HSL-only), 27 (Federado non-IID real), 28 (Federado IID simulado — Fase 5).

**Checklist de validação — tudo confirmado:**
- Nenhum erro em nenhuma das 5 fases (`TREINAMENTO_COMPLETO status=ok fl_rounds=44 rag_ok=True baseline_rf_ok=True ablation_ok=True`).
- Fase 5 (`FL_PARTITION_MODE=iid_simulado`) rodou de fato: pool de 33.773 amostras (BPSP+HSL) redividido corretamente em 2 clientes virtuais de ~16.886 cada.
- `partition_mode` chega correto ao banco: `natural` nas fases 1-3, `iid_simulado` na fase 5 — confirmado tanto no log quanto em query SQL direta (`SELECT partition_mode FROM metrics.fl_trainings WHERE id IN (25,26,27,28)`).
- `macro_auc`/`macro_f1`/`ece` gravados com valores reais nas 4 fases FL (não mais `None`) — confirmado via `training_evaluation_metrics_saved` no log e via query SQL direta na tabela.
- `subgroup_metrics` (avaliação por origem hospitalar) calculado corretamente nas fases 3 e 5, com rótulos legíveis (`BPSP`/`HSL`, não índices numéricos).

**Resultado desta validação (ids 25-28) — só para checar plausibilidade, NÃO é dado formal do TCC:**

| id | Fase | partition_mode | Acc | AUC | F1 | ECE | Convergiu |
|---|---|---|---|---|---|---|---|
| 25 | BPSP-only | natural | 62,79% | 0,7326 | 0,3619 | 0,0621 | Não (120) |
| 26 | HSL-only | natural | 30,41% | 0,6709 | 0,2206 | 0,1237 | Sim (R27) |
| 27 | Federado (non-IID real) | natural | 64,71% | 0,8177 | 0,4918 | 0,1305 | Sim (R28) |
| 28 | Federado (IID simulado) | iid_simulado | **71,52%** | **0,8509** | **0,5298** | 0,0876 | Sim (R39) |

Achado preliminar (não formal): o cenário IID simulado (id=28) superou o non-IID real (id=27) em todas as métricas — Acc +6,81 p.p., F1 +0,038, AUC +0,033, ECE melhor, convergência mais rápida (R39 vs. R53) — primeira evidência direta e comparável de que a heterogeneidade non-IID prejudica o treinamento, com tudo mais (algoritmo, hiperparâmetros, seed) mantido idêntico entre as duas fases. A confirmar (ou refutar) com o Treinamento Real de quinta-feira.

### Pendências identificadas antes do Treinamento Real de 2026-07-02

Antes de disparar o Treinamento Real de quinta-feira, dois itens foram identificados como **inclusões simples de registro no banco** que devem ser feitas para que os logs oficiais contenham as informações necessárias ao texto de defesa. Registradas aqui para que não passem despercebidas no início da sessão de 2026-07-02.

#### Pendência A — ECE pré-calibração nos logs de avaliação

**Problema:** os arquivos de avaliação gerados por `manual_loop.py` (ex.: `evaluation_best_r39_of_44.json`) registram `ece_isotonic` (ECE pós-calibração isotônica OvR), mas **não registram `ece_pre`** (ECE antes de qualquer calibração — saída bruta do modelo). Para o documento de metodologia, o par antes/depois é necessário para evidenciar o ganho da calibração isotônica com uma comparação direta em dado real (não depender de experimentos anteriores onde o ECE pré estava num contexto diferente).

**O que fazer:** em `manual_loop.py`, no bloco de avaliação final, adicionar `"ece_pre": report_raw.ece` (ou o campo equivalente) ao dict de avaliação salvo em JSON e ao banco via `update_evaluation_metrics()`. Migration 016 já tem a coluna `ece` — verificar se aponta para o ECE pré ou pós; se for pós, renomear ou duplicar o campo para deixar inequívoco.

**Impacto:** alteração de 1–3 linhas em `manual_loop.py` e/ou `orchestrator.py`; nenhuma mudança de schema (coluna já existe).

#### Pendência B — Custo energético com GPU nos logs

**Problema:** os logs registram `peak_ram` e `avg_cpu` (via `resource_summary`), mas **não registram consumo de energia**. `nvidia-smi` já foi executado manualmente numa sessão anterior (confirmou 74 W/285 W durante BPSP-only, ver Parte 9), mas esse valor não é capturado automaticamente nem persistido por treino. Para a seção de análise de viabilidade do TCC (custo operacional da infraestrutura federada), o consumo de energia por experimento é necessário.

**O que fazer:** em `resource_summary` (provavelmente em `orchestrator.py` ou `fl_core/orchestrator.py`), adicionar captura de energia via `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` (coleta por amostragem durante o loop) ou via `pynvml`. Persistir a energia total estimada em Wh (potência média × duração em horas) no banco junto com `peak_ram`/`avg_cpu`. Migration 015 já tem colunas de recurso em `fl_trainings` — verificar se há espaço ali ou se é necessária uma migration 018.

**Impacto:** adição de um coletor por polling durante o loop + 1–2 colunas no banco; nenhum impacto no resultado do treinamento.

---

**Conclusão revista:** implementação da sessão de 2026-07-01 (AUC-ROC no banco + Fase 5) validada sem erros. **As duas pendências acima (ECE pré-calibração + custo energético) devem ser resolvidas no início de 2026-07-02 antes de disparar o Treinamento Real** — são adições de registro, não mudanças de comportamento, e não exigem novo ciclo de validação funcional completa.
