# Avaliação do Projeto MOSAIC-FL

Registro evolutivo das sessões de desenvolvimento. Cada avaliação captura o estado do projeto, o que foi feito, e o que ainda falta.

---

## Avaliação 7 — 2026-07-01

> **Resultados desta avaliação são pré-oficiais.** Os treinamentos registrados aqui foram executados durante e antes do fechamento da implementação (bug fix no checkpoint_store, mudança de métrica primária para F1-macro, migrations 016/017). Não servem como comparativo definitivo. Os treinamentos oficiais para comparação serão rodados em 2026-07-02 (quinta-feira), após consolidação de todas as correções.

### O que foi feito nesta sessão

#### GPU (CUDA) disponível — redução de tempo de treinamento em ~96%

O desktop passou a ter GPU disponível. A inferência por tempo de treinamento caiu de ~4 horas para 5–10 minutos por pipeline completo (`make training-full-cuda`). Logs confirmam:

- `run_complete_cuda_20260701_090530.log` — 5 experimentos completos em ~8 min (inclui BPSP-only, HSL-only, federado, RF baseline, ablação)
- `run_complete_cuda_20260701_211304.log` — 4 experimentos de heterogeneidade em ~28 min

Parâmetro `FL_DEVICE=cuda` passado via variável de ambiente; os loaders de dado continuam em CPU (`peak_ram ≈ 2450 MB`, `avg_cpu ≈ 104–133%`).

#### Gemma 3:4b substitui DistilGPT-2 no RAG

O backend de geração do RAG foi trocado de DistilGPT-2 para **Gemma 3:4b** (commit 4c822d2). Justificativa registrada no commit: "maior suporte para a língua portuguesa do Brasil". DistilGPT-2 é encoder-only convertido e gera texto em inglês por padrão; Gemma 3:4b é multilingual com treinamento explícito em PT-BR. A interface `ClinicalRAG` não mudou — a substituição é transparente para os módulos que a chamam.

Nota: a nota no documento de metodologia que mencionava BERTimbau como candidato para substituir DistilGPT-2 estava incorreta (BERTimbau é encoder-only, incapaz de geração). O candidato correto — e agora implementado — é Gemma 3:4b.

#### F1-macro como métrica primária (substituiu accuracy)

Commit 6eee3c2 mudou o critério de seleção do melhor checkpoint de `accuracy` para `f1_macro`. A mudança é justificada pelo desbalanceamento de classes real nos dados FAPESP:

| Classe | BPSP (n=28.599) | HSL (n=5.174) |
|--------|-----------------|---------------|
| curado_pronto (0) | 15.892 (55,6%) | 67 (1,3%) |
| curado_internado (1) | 318 (1,1%) | 45 (0,9%) |
| melhora_pronto (2) | 120 (0,4%) | 3.182 (61,5%) |
| melhora_internado_breve (3) | 9.448 (33,0%) | 1.280 (24,7%) |
| melhora_internado_grave (4) | 2.821 (9,9%) | 600 (11,6%) |

Accuracy favorece a classe majoritária de cada hospital. F1-macro trata as 5 classes igualmente, sendo mais informativo clinicamente. A mudança impacta qual rodada é salva como "melhor checkpoint" e portanto quais números constam nos logs.

#### Migration 016 e 017 — métricas e partition_mode no banco

- **016** (`fl_trainings_evaluation_metrics`): adiciona colunas `best_f1_macro`, `best_auc_roc`, `ece_pre`, `ece_post_isotonic` em `fl_trainings`. Permite consultar métricas de avaliação sem ler os arquivos JSON.
- **017** (`fl_trainings_partition_mode`): adiciona coluna `partition_mode` (`natural` | `iid_simulado`). Registrado em `training_registered_postgres` no início de cada treino.

Total de migrações Alembic: 17 (001–017).

#### Bug fix no checkpoint_store — training_id ausente no save final

`infrastructure/shared/checkpoint_store/postgres_store.py` não passava `training_id` no save final do melhor checkpoint (após restauração). O checkpoint era gravado sem escopo de treinamento, potencialmente causando contaminação cruzada entre experimentos (reprodução do incidente Exp 9). Corrigido com a passagem explícita do `training_id` na chamada final.

#### Experimento de heterogeneidade non-IID — 4 cenários

O maior avanço desta sessão. Implementado `FL_PARTITION_MODE` e `FL_INCLUDE_HOSPITALS` como variáveis de ambiente para controlar quais hospitais participam do treino e se a partição é real ou simulada. Makefile recebeu targets `training-bpsp-only`, `training-hsl-only`, `training-iid-contrast`, `training-iid-contrast-cuda` e `training-full-cuda` (que encadeia os 5 cenários sequencialmente).

**Resultados registrados em log e arquivos JSON de avaliação (pré-oficiais — ver nota no topo desta avaliação):**

| Cenário | training_id | Rounds (melhor) | F1-macro | Acc | AUC | ECE_iso | partition_mode |
|---------|-------------|-----------------|----------|-----|-----|---------|----------------|
| BPSP-only (tarde) | 25 | 120 (r94) — não convergiu | 0,362 | 0,628 | — | 0,031 | natural |
| HSL-only (tarde) | 26 | 46 (r27) — convergiu | 0,221 | 0,304 | — | 0,026 | natural |
| Federado 2-clientes | 27 | 53 (r28) — convergiu | 0,492 | 0,647 | 0,817 | 0,028 | natural |
| Federado IID sim | 28 | 44 (r39) — convergiu | 0,530 | 0,715 | 0,851 | 0,024 | iid_simulado |

Avaliação separada por hospital no conjunto de teste global (subgroup_metrics):

| Cenário | BPSP (n≈2.862) Acc / F1 | HSL (n≈519) Acc / F1 |
|---------|--------------------------|----------------------|
| BPSP-only (id=25) | 70,0% / 0,379 | **23,1% / 0,151** (colapso) |
| HSL-only (id=26) | **24,2% / 0,137** (colapso) | 64,4% / 0,348 |
| Federado natural (id=27) | 64,1% / 0,344 | 68,2% / 0,354 |
| Federado IID sim (id=28) | 71,8% / 0,365 | 69,7% / 0,289 |

**Interpretação**:
- Um hospital treinado isoladamente não generaliza para o outro: colapso de ~23–24% de accuracy (próximo ao acaso para 5 classes). Isso evidencia a necessidade da aprendizagem federada.
- A federação com partição real (non-IID natural) equilibra o desempenho: ambos os hospitais ficam em torno de 64–68% de accuracy, sem colapso.
- O gap entre partição não-IID real e IID simulado é Δ0,038 em F1-macro (0,492 vs 0,530) e Δ0,068 em accuracy (0,647 vs 0,715). Isso quantifica o custo da heterogeneidade de distribuição entre BPSP e HSL.
- FedNova apresenta comportamento diferente nos dois cenários: converge em ~28–39 rodadas no IID simulado e ~28 no federado natural, mas não converge em 120 rodadas no BPSP-only isolado (o modelo cresce lentamente sem o contrabalanceamento do HSL).

#### ECE isotônica medida — gap da Avaliação 6 fechado

O item pendente mais crítico da Avaliação 6 (ECE com dados reais) está fechado. `ece_isotonic` está sendo calculado e salvo em todos os arquivos de avaliação. Resultados:

| Experimento | ECE isotônica | Comparativo |
|-------------|---------------|-------------|
| Federado IID sim (r39) | **0,0242** | melhor calibração |
| Federado natural (r28) | 0,0275 | |
| BPSP-only (r94) | 0,0308 | |
| HSL-only (r27) | 0,0261 | |
| Experimentos sem partition_mode | 0,020–0,049 | |
| ECE pós Temperature Scaling (Exp 12, sessão anterior) | 0,1086 | pior (scaling piorou) |

A calibração isotônica OvR (implementada em sessão anterior mas sem medição com dados reais) produz ECE consistentemente abaixo de 0,05 em todos os cenários federados. Temperature Scaling havia piorado o ECE (de 0,0935 para 0,1086 no Exp 12); isotônica inverte esse resultado.

Nota: os arquivos de avaliação registram `ece_isotonic` mas não explicitam `ece_pre` (antes da calibração). Os valores pré-calibração dos experimentos da tarde não constam nos JSONs de avaliação desta sessão. Item ainda pendente para documentação completa.

#### Documentação nova

- `docs/Linha_do_Tempo_MOSAIC-FL.md` — cronologia do projeto
- `docs/Comparativo_Metodologia_MOSAIC-FL.md` — comparativo com trabalhos relacionados
- `docs/diagramas_c4_uml.md` — diagramas C4 e UML da arquitetura

### Estado atual

```
Testes:          417 passando, 4 falhando (pré-existentes — PermissionError em /app
                 no scheduler, problema de ambiente de CI, não do código)
Training IDs:    22–28 (17 experimentos registrados no PostgreSQL)
Migrações:       001–017 aplicadas
GPU:             disponível no desktop (CUDA)
RAG:             Gemma 3:4b ativo
Métrica primária: F1-macro
Calibração:      Isotônica OvR (ECE_iso ≈ 0,024–0,049 em todos os cenários)
```

### O que ainda falta (pré-defesa)

- **Rodar treinamentos oficiais em 2026-07-02**: com implementação fechada (bug fix aplicado, F1-macro como critério, migrations 016/017 ativas). Os resultados de 2026-07-01 não são comparativos válidos.
- **ECE pré-calibração com dados reais**: os JSONs de avaliação desta sessão não registram `ece_pre` (apenas `ece_isotonic`). Para o documento de metodologia, é necessário o par antes/depois para evidenciar o ganho da calibração isotônica.
- **BEHRT Pooled com 120 épocas**: resultado anterior (`behrt_pooled_20260625_223649.json`) usou 10 épocas locais. Comparativo justo com 120 rodadas FL está pendente.
- **Resposta do Prof. Claudio** para integração ClinicalPath (network.txt, time-metadata.txt, list_exams.txt).
- **Definir janela temporal da predição** com a orientadora.
- **Validação clínica**: rótulos de desfecho, fórmula do risk score, mapeamento LOINC.
- **Custo energético com GPU**: os logs registram `peak_ram` e `avg_cpu` mas não consumo de energia. Pendente para a seção de análise de viabilidade.

---

## Avaliação 6 — 2026-06-23/24

### O que foi feito nesta sessão

#### Interoperabilidade FHIR R4 — módulo completo

Criado `integration/fhir/` com três arquivos:

- **`models.py`** — `InferenceOutput`: contrato para exportação FHIR. Carrega apenas probabilidades por classe, `model_round`, `temperature`, `ece` e `correlation_token`. Nenhum dado clínico, nenhuma identidade do paciente. Validação em `__post_init__`: predictions não vazio, temperature > 0, soma das probabilidades ≈ 1,0 ±0,01.
- **`mapper.py`** — `FHIRExporter.to_risk_assessment()`: gera `RiskAssessment` FHIR R4 válido. `subject.identifier.system = "urn:mosaicfl:correlation"`, `subject.identifier.value = correlation_token`. Decisão arquitetural: probabilities pertencem ao quadro clínico, não ao indivíduo — o módulo não carrega `PatientExport`, não importa `infrastructure/`, não usa `sqlalchemy`.
- **`loinc_map.py`** — 22 analitos mapeados com LOINC codes + display names em português. `fl_risk_score` usa namespace próprio `urn:mosaicfl:analyte`. `lookup(name)` case-insensitive com aliases FAPESP.

`service.py` integrado: `IngestRequest` aceita `correlation_token`, `IngestResponse` retorna `fhir_risk_assessment`. 33 testes unitários criados em `tests/unit/test_fhir_exporter.py` incluindo testes de isolamento que inspecionam código-fonte para verificar ausência de imports proibidos.

#### Ajustes necessários para o FHIR funcionar

- **`temperature` adicionado ao retorno de `predict_proba()`** — `inference_engine.py` não incluía `temperature` nos dicts retornados; adicionado em ambos os branches (vazio e normal).
- **`hospital_id` explícito no DataSourceFactory** — `SimulatedDataSource` e `CSVDataSource` passaram a aceitar o parâmetro; cache key em `runner.py` mudou para tupla `(source_type, hospital_id)`.
- **Bug `scaler.T` fora de escopo em `strategy.py`** — `_save_evaluation_report()` referenciava `scaler.T` mas `scaler` era variável local de `_run_calibration()`. Corrigido removendo as linhas fora de escopo.

#### `evaluate()` integrado na simulação

`run_experiments_simulation.py` agora executa `evaluate()` de `mosaicfl.core.evaluation` antes e depois do temperature scaling, em ambos os modos (manual e Ray). Relatórios com ECE, AUC-ROC, F1 e matriz de confusão gravados em JSON em `experiments/logs/`.

#### Documentação e comunicação

- `docs/email_orientadora.md` — 8 decisões abertas para a defesa (rótulos de desfecho, janela temporal, calibração com dados reais, fórmula do risk score, escopo federado com 2 hospitais, integração ClinicalPath, custo energético, LGPD).
- `docs/email_claudio.md` — solicitação ao Prof. Claudio Linhares (ClinicalPath) para autorização de integração e documentação das 5 incompatibilidades identificadas.
- `README.md` — seção de Interoperabilidade reescrita com o padrão FHIR R4 + LOINC, tabela de contratos, exemplo de `RiskAssessment`, decisão de isolamento arquitetural.

### Estado atual

```
Testes:         ~541 passando (503 anteriores + 33 FHIR + ajustes nos de integração)
Módulos novos:  integration/fhir/ (3 arquivos + __init__)
Arquivos não commitados: integration/fhir/, tests/unit/test_fhir_exporter.py,
                         docs/email_*.md, alterações em service.py, inference_engine.py,
                         integration/clinical-path/*, src/mosaicfl/core/config.py, etc.
Dados reais:    carregamento FAPESP em andamento (PostgreSQL no desktop)
```

### O que ainda falta (pré-defesa)

- Medir ECE com dados reais do FAPESP (ECE sintético = 0,44 → não confiável)
- Resposta do Prof. Claudio para completar integração ClinicalPath (network.txt, time-metadata.txt, list_exams.txt)
- Definir janela temporal da predição com orientadora
- Validar clinicamente: rótulos de desfecho, fórmula do risk score, mapeamento LOINC
- Avaliar custo computacional/energético

---

## Avaliação 5 — 2026-06-07

### O que foi feito nesta sessão

#### Migração completa para PostgreSQL

O projeto saiu de uma arquitetura SQLite + ChromaDB para um banco de dados PostgreSQL único com três schemas semânticos:

| Schema | Tecnologia | Conteúdo |
|--------|-----------|----------|
| `clinical` | PostgreSQL puro | Pacientes, export_paths, config FL |
| `metrics` | TimescaleDB (hypertables) | Histórico de risco, registros de exames |
| `knowledge` | pgvector (HNSW) | Perfis clínicos prototípicos para RAG |

Arquivos criados/modificados:
- `scripts/db/init.sql` — DDL completo com extensões, schemas, índices HNSW
- `docker-compose.db.yml` — `timescale/timescaledb-ha:pg16` (PostgreSQL + TimescaleDB + pgvector em imagem única)
- `scripts/db/migrate_sqlite.py` — migração idempotente SQLite → PostgreSQL
- `infrastructure/mosaicfl_api/db.py` — reescrito com SQLAlchemy, dual-backend (SQLite dev / PostgreSQL prod)
- `infrastructure/mosaicfl_server/config_loader.py` — `PostgreSQLConfigLoader` adicionado; default mudou de `chroma` para `postgres`
- `src/mosaicfl/core/rag.py` — `ClinicalRAG` migrado do ChromaDB para `_PostgreSQLStore` (pgvector); interface ChromaDB-compatível preservada para que mocks de testes não mudem

#### LGPD Art. 37 — Trilha de auditoria

- `infrastructure/mosaicfl_api/audit.py` criado com:
  - `pseudonymize(patient_id)` — SHA-256[:16], irreversível
  - `token_fingerprint(token)` — SHA-256[:12], permite correlação com IAM sem armazenar credenciais
  - `log_access(operation, token_fp, patient_id, **kwargs)` — grava em `logs/audit.log` com `propagate=False` (isolado do log da aplicação)
- Todos os endpoints de paciente auditados: `predict`, `ingest`, `patient_list`, `patient_read`, `model_reload`

#### Autenticação por presença de token

- `_API_KEY` fixo removido; substituído por `_AUTH_REQUIRED` booleano
- Aceita `X-API-Key` ou `Authorization: Bearer <token>`; validação de identidade é responsabilidade do IAM upstream
- `FL_AUTH_REQUIRED=false` desativa a exigência para dev/testes

#### Confiabilidade do `fit()`

- `except Exception: continue` removido — erros de batch agora são logados estruturadamente e re-lançados para que o Flower marque o cliente como falho
- `len(self.train_loader.dataset)` → `total_samples` — reporta amostras realmente treinadas

#### Correções de schema

Bugs que teriam impedido o funcionamento com PostgreSQL real:
- `id SERIAL` adicionado a `metrics.risk_history` e `metrics.exam_records` no `init.sql` (SQLAlchemy faz `ORDER BY id` em ambas)
- Colunas `date` mudadas de `sa.Text` para `sa.Date` — psycopg2 retorna `datetime.date`, não string
- `service.py` corrigido para não chamar `date.fromisoformat()` em valores que já são `datetime.date`

#### Experimentos funcionando

- `sys.path` corrigido em `run_experiments_v2.py` via `Path(__file__).resolve().parent.parent`
- `makefile` legado (minúsculo) removido — estava sobrepondo o `Makefile` e ocultando o target `experiment`
- `make experiment` funcionando: 5 clientes FL, convergência em 4 rodadas, acurácia 71% (dados sintéticos)

### Estado atual

```
Testes:         503 passando, 6 deselected (e2e)
Arquivos .py:   97 arquivos, ~15.000 linhas
Cobertura:      unit + integration; e2e requer infraestrutura real
Banco:          PostgreSQL pronto (docker-compose.db.yml); SQLite em dev/testes
Auditoria LGPD: ativa em todos os endpoints de paciente
Experimentos:   make experiment funcional com dados sintéticos
```

### O que ainda falta

**Para dados reais:**
- Subir o container: `docker compose -f docker-compose.db.yml up -d`
- Configurar `FL_DB_URL=postgresql://mosaicfl:SENHA@localhost:5432/mosaicfl` no `.env`
- Executar migração se houver dados em SQLite: `python scripts/db/migrate_sqlite.py`

**Pendências técnicas:**
- `ClinicalRAG` nos experimentos usa `FL_DB_URL` vazia → pipeline RAG falha (erro "Could not parse SQLAlchemy URL"). Para experimentos acadêmicos, o RAG pode usar ChromaDB local ou ser desacoplado da URL de produção
- `SchedulerStateStore` ainda usa SQLite (há comentário explícito de migração futura)
- Métricas de rounds FL (accuracy/loss por rodada) são salvas em JSON; poderiam ir para uma hypertable `metrics.fl_rounds` no TimescaleDB
- Differential Privacy nos pesos ainda não implementado (roadmap LGPD)

---

## Avaliação 4 — 2026-06-03 (sessão anterior)

### O que foi feito

- Exportador ClinicalPath com 38 testes de contrato
- `ProductionFedProxStrategy` com checkpoint a cada round e exportação de métricas JSON
- `TrainingStateStore` — recuperação após crash (status `interrupted` detectado no próximo start)
- `ConvergenceTracker` com `stable_count` incremental e `patience` configurável
- `ConfigLoader` com backends `chroma` e `file` selecionáveis por `FL_CONFIG_BACKEND`
- Structured logging JSON via `python-json-logger` em todos os daemons
- Round timeout com watchdog thread; rounds que ultrapassam o limite são registrados em `timed_out_rounds`
- `weighted_average_accuracy` e `weighted_average_loss` — funções públicas separadas (bug da chave `"accuracy"` em `fit_metrics_aggregation_fn` corrigido)
- `_MODEL_SIZE_MB` calculado do `state_dict` real; `communication_mb` agora é `len(results) * size * 2`
- Testes de contrato para `fit()` e `evaluate()` (tipos Python exatos, shapes invariantes)
- `httpx` deprecation warning corrigido via `httpx2`

### Estado ao final

```
Testes: ~490 passando
Backend: SQLite + ChromaDB
Auth: _API_KEY fixo
```

---

## Avaliação 3 — 2026-05-20

### O que foi feito

- `FedProxClient` v2 com CLS token pooling, `BEHRTEncoderLayer` customizado
- `EHRPreprocessor` v2 com normalização de unidades médicas (kg/lb, anos/meses)
- `data_loader.py` Strategy Pattern: SGBD → CSV → sintético com fallback automático
- Daemons de produção: `server_daemon.py`, `client_daemon.py`, `scheduler_daemon.py`
- Suite de testes `test_v2_core.py` e `test_infrastructure.py`
- `.env.example`, `CHANGELOG.md`, `CONTRIBUTING.md`
- `make lint`, `make fmt`, `.pre-commit-config.yaml`

### Estado ao final

```
Testes: ~300 passando
Backend: SQLite (db.py original)
Experimentos: funcionando com dados sintéticos
```

---

## Avaliação 2 — 2026-05-10

### O que foi feito

- BEHRT v1 com mean pooling
- FedProx básico
- RAG v1 com ChromaDB + DistilGPT-2
- 5 experimentos do TCC implementados
- `run_experiments.py` e `run_experiments_v2.py`

---

## Avaliação 1 — 2026-05-01

### Estado inicial

- Implementação v1 com dados sintéticos
- Sem testes, sem infraestrutura de produção
- ChromaDB para embeddings clínicos
- SQLite simples para persistência
