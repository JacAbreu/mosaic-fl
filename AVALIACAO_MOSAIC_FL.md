# Avaliação do Projeto MOSAIC-FL

Registro evolutivo das sessões de desenvolvimento. Cada avaliação captura o estado do projeto, o que foi feito, e o que ainda falta.

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
