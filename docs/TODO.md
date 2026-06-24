# TODO — MOSAIC-FL

Dividido em três partes:
- **Qualidade profissional** — padrão de engenharia de software, independente de deploy real
- **Dependências de produção** — o que impede deploy em ambiente hospitalar real
- **Roadmap de produção** — funcionalidades para uso clínico completo

---

## Qualidade Profissional

### Consistência de código

- [x] ~~**Corrigir docstring de `set_parameters` em `client.py`**~~
- [x] ~~**Eliminar `from .config import *` em todos os módulos v2**~~
- [x] ~~**Unificar os dois `ConvergenceTracker`**~~
- [x] ~~**Corrigir `fit_metrics_aggregation_fn=weighted_average`**~~
- [x] ~~**Corrigir `communication_mb` no histórico**~~
- [x] ~~**CLS token: usar `self.cls_token` (nn.Parameter) diretamente no `forward()`**~~

  O parâmetro existia mas nunca era usado — o CLS real era `embedding[9999]`, que não recebia gradiente dedicado. Corrigido: `self.cls_token` agora é prefixado nos embeddings antes do `pos_encoder`, com inicialização `trunc_normal_(std=0.02)`. O buffer morto `cls_token_id` foi removido.

- [x] ~~**`num_classes` e `class_labels` configuráveis via env var**~~

  `FL_NUM_CLASSES` e `FL_CLASS_LABELS` (comma-separated) lidos em `MODEL_CFG`. `ModelConfig.__post_init__` valida que `len(class_labels) == num_classes` — se divergirem, o processo não sobe.

- [x] ~~**Class weights no treinamento para tratar desbalanceamento**~~

  `FedProxClient` agora computa pesos inversamente proporcionais à frequência de cada classe no loader local e passa para `CrossEntropyLoss(weight=...)`. Avaliação (`evaluate`) usa critério sem peso para comparabilidade entre rounds.

- [x] ~~**Implementar `_save_checkpoint` de verdade em `experiment_server.py`**~~

  `CheckpointStore` (ABC) em `infrastructure/shared/checkpoint_store.py` com dois backends:
  `SQLiteCheckpointStore` (experimentos, sem servidor) e `PostgreSQLCheckpointStore`
  (`metrics.fl_checkpoints` BYTEA, para produção/hml). Seleção automática via
  `get_checkpoint_store(FL_DB_URL)`: URL vazia → SQLite, URL configurada → PostgreSQL.
  Integridade via SHA-256 em ambos. `_save_checkpoint` virou no-op — cada rodada
  persiste em `aggregate_fit`. Migração futura para S3: substitui apenas os bytes,
  a interface do store não muda.

- [x] ~~**Integrar RAG com tensores reais (modo banco)**~~

  `run_rag_pipeline()` não depende mais de `df_raw`. Desfechos derivados dos labels
  reais do `test_loader`; `patient_data` construído do vocabulário inverso da amostra.
  `ClinicalRAG` usa `_InMemoryStore` (numpy cosine similarity) quando `FL_DB_URL` está
  vazio — sem dependência de PostgreSQL em experimentos. Labels dos perfis extraídos de
  `MODEL_CFG.class_labels` (5 classes de duração), configuráveis via `FL_CLASS_LABELS`.
  `main()` executa o RAG em ambos os modos (banco e CSV/sintético).

### Qualidade de código estático

- [x] ~~**Adicionar type hints completos**~~
- [x] ~~**Adicionar linting com ruff ao `make` e ao CI**~~
- [x] ~~**Configurar pre-commit hooks**~~

### Testes

- [x] ~~**Adicionar testes de contrato para `fit()` e `evaluate()`**~~
- [x] ~~**Testes de integração da API (503/503 passando)**~~

  `TestTokenizer` foi reescrito como `TestRecordsToTokens` com a assinatura atual de `records_to_tokens`. Mock de `predict_proba` corrigido. `PatientListResponse` (dict com `patients`) refletido nos testes. Namespace collision entre `tests/integration/` e `integration/` resolvida removendo `__init__.py` dos diretórios de teste. `BigInteger` → `with_variant(Integer, "sqlite")` para autoincrement em SQLite.

- [ ] **Teste de integração end-to-end real (sem mocks)**

  Sobe servidor + cliente em threads/processos separados, executa 1 round real e verifica que o modelo foi atualizado.

### Observabilidade

- [x] ~~**Structured logging em JSON**~~
- [x] ~~**Health check endpoint nos daemons (`/healthz`)**~~
- [ ] **Métricas Prometheus**

  Adicionar `prometheus_client` nos daemons: `fl_round_total`, `fl_accuracy`, `fl_loss`, `fl_clients_active`.

### Segurança

- [x] ~~**Pseudonimização LGPD: HMAC-SHA256 do `patient_id`**~~

  `FL_PATIENT_ID_SECRET` → HMAC-SHA256 antes de qualquer persistência. Em `FL_ENV=production`, o processo não sobe sem o secret configurado.

- [x] ~~**JWT validation (HS256 / RS256)**~~
- [x] ~~**Per-patient asyncio locks (substituiu lock global)**~~
- [x] ~~**Rate limiter sliding window (120 req/min geral, 30 req/min ingest)**~~
- [x] ~~**Integridade de checkpoints via SHA-256**~~
- [x] ~~**`ExamInput.value` rejeita NaN, infinito e negativos**~~
- [x] ~~**`output_dir` protegido contra path traversal**~~

  Rejeita `..` em qualquer parte do caminho. Em `FL_ENV=production`, restringe ao diretório base `FL_CLINICALPATH_OUTPUT`.

### Modelo e Inferência

- [x] ~~**Vocabulário padrão distribuído (`build_standard_vocab.py`)**~~

  Vocabulário canônico construído de `knowledge.term_dictionary` + `knowledge.analyte_references` sem dados de pacientes, distribuído a todos os clientes FL antes do treinamento. Garante que token IDs idênticos entre hospitais — sem isso, a agregação FedAvg é semanticamente inválida.

- [x] ~~**Probabilidades por classe com incerteza via MC Dropout**~~

  `InferenceEngine.predict_proba()` executa 50 passes com `model.train()` (dropout ativo) e retorna por classe: `value` (média) e `uncertainty` (desvio padrão). Thread-safe via `threading.Lock()`. `model_metadata` inclui `trained`, `calibrated: false`, método e número de amostras.

- [x] ~~**`trained: bool` na resposta da API**~~

  `model_metadata.trained` reflete se há checkpoint carregado. Quando `False`, as probabilidades são ruído aleatório de modelo não treinado — o consumidor da API é informado explicitamente.

- [x] ~~**Exponential backoff com jitter no reconect do cliente FL**~~
- [x] ~~**DataLoader cache no cliente (evita re-query ao banco a cada round)**~~
- [x] ~~**RotatingFileHandler (20MB / 5 backups)**~~

### Configuração e documentação

- [x] ~~**Arquivo `.env.example`**~~
- [x] ~~**CHANGELOG.md**~~
- [x] ~~**CONTRIBUTING.md**~~
- [ ] **Docstrings completas nos módulos públicos**

---

## Dependências de Produção

> **Estas issues bloqueiam deploy em ambiente hospitalar real.**
> Nenhuma delas impede o TCC ou a simulação local — mas sem elas, o sistema não pode receber dados de pacientes reais.

### BLOQUEADOR: Rate limiter não funciona com múltiplos workers

- [ ] **Substituir `_SlidingWindowLimiter` in-process por Redis + `fastapi-limiter`**

  O limitador atual é por processo Python. Com Gunicorn e N workers, cada worker tem seu próprio contador — um cliente pode fazer `120 × N` req/min sem ser bloqueado. Em produção com 4 workers, o limite efetivo é 480 req/min.

  **O que fazer:** instalar `fastapi-limiter` + `redis-py`, configurar `FL_REDIS_URL`, e substituir `_rate_check()` por `RateLimiter` do fastapi-limiter. O Redis pode ser o mesmo usado para session storage.

### BLOQUEADOR: Modelo retorna probabilidades sem calibração

- [x] ~~**Temperature scaling pós-treinamento**~~

  `TemperatureScaler` em `src/mosaicfl/core/calibration.py` aprende escalar T via LBFGS
  minimizando NLL no conjunto de calibração (Guo et al., ICML 2017). Executado
  automaticamente ao final de `run_federated_learning_manual()` e `run_federated_learning_ray()`
  em `run_experiments_v2.py`. T é persistido no checkpoint junto com `model_state` e `vocab`.
  `InferenceEngine` carrega T do checkpoint e divide os logits por T antes do softmax em
  todos os passes MC. `predict_proba()` retorna `calibrated: bool` e `service.py` propaga
  para `ModelMetadata.calibrated`. Ressalva: em simulação acadêmica o test_loader é
  reutilizado como calibration set — em produção deve-se reservar holdout separado.

### BLOQUEADOR: Sem conjunto de teste global no servidor

- [x] ~~**Implementar `_load_test_data()` em `FederatedServer`**~~

  `_load_test_data()` em `infrastructure/mosaicfl_server/runner.py` carrega holdout via
  `SequencePipeline.build_per_hospital()` quando `FL_DB_URL` está configurado. Fração
  configurável via `FL_TEST_HOLDOUT_FRACTION` (padrão 0.1). Retorna `None`
  silenciosamente quando sem banco — `evaluate_fn` não é ativado. Loga
  `_load_test_data_ready n=... hospitals=...` ao subir.

### BLOQUEADOR: `predict_proba` bloqueia sob carga

- [x] ~~**`FL_MC_SAMPLES` via env var**~~

  `_DEFAULT_MC_SAMPLES = int(os.getenv("FL_MC_SAMPLES", "50"))` em `inference_engine.py`.
  `predict_proba(mc_samples=_DEFAULT_MC_SAMPLES)` — reduzir para 20 em produção sem rebuild.

- [ ] **Timeout e circuit breaker no MC Dropout**

  Passes sequenciais com lock por request — sob carga alta, requests enfileiram antes do timeout HTTP.

  **O que fazer:** adicionar timeout interno e considerar batches paralelos com `torch.vmap`.

### BLOQUEADOR: Auto-discovery de checkpoint no startup da API

- [x] ~~**API carrega checkpoint automaticamente ao reiniciar**~~

  `_lifespan(app)` em `service.py` substitui `@app.on_event("startup")` (deprecated).
  Chama `_get_engine()` antes de aceitar tráfego — o engine já faz auto-discovery via
  `CheckpointStore.load_latest()` na construção. Se não há checkpoint, sobe com
  `trained: false` (comportamento explícito, não silencioso).

### BLOQUEADOR: `_run_ingest` não é atômica

- [x] ~~**Transação única abrangendo exames + risco + exportação**~~

  `_run_ingest` em `service.py` envolve todos os passos em `with _db.begin() as conn:`.
  `PatientDB.begin()` (contextlib.contextmanager) abre uma transação SQLAlchemy e repassa
  `conn` para os métodos `_tx` (`upsert_patient_tx`, `add_exams_tx`, `get_exams_tx`,
  `add_risk_tx`, `get_risk_history_tx`, `get_patient_tx`, `set_export_path_tx`).
  Se qualquer passo falhar (incluindo a exportação de arquivo), tudo reverte — sem estado parcial.

### Importante: Rastreabilidade de versão do modelo

- [x] ~~**Versão do modelo nas respostas da API**~~

  `checkpoint_round`, `checkpoint_at`, `model_version` (SHA-256 12 hex chars dos pesos)
  gravados em `_serialize()` de `checkpoint_store.py`. `InferenceEngine._load()` lê os
  três campos. `predict_proba()` e `ModelMetadata` os propagam nas respostas da API.
  `_run_calibration()` no servidor federado re-grava o checkpoint com T calibrado,
  preservando `checkpoint_round` e atualizando `checkpoint_at` + `model_version`.

### Importante: Rotação de chave HMAC

- [ ] **Estratégia de rotação para `FL_PATIENT_ID_SECRET`**

  Se o secret for comprometido ou precisar ser rotacionado, todos os hashes existentes ficam desvinculados dos registros originais — o banco perde a capacidade de identificar histórico do paciente.

  **O que fazer:** armazenar `key_version` junto com cada hash no banco (`patient_id_hash`, `key_version`). Na rotação, re-hash os registros existentes com a nova chave mantendo o `key_version` antigo como fallback temporário.

### Importante: `on_event("startup")` deprecated

- [x] ~~**Migrar para `lifespan` context manager (FastAPI)**~~

  `@app.on_event("startup")` removido. `_lifespan` com `@asynccontextmanager` passado para
  `FastAPI(lifespan=_lifespan)`. Zero `DeprecationWarning` nos testes.

---

## Roadmap de Produção

### Dados e Integração

- [x] ~~**Exportador ClinicalPath**~~
- [ ] Integração HL7 FHIR com EPR dos hospitais
- [ ] Conector genérico para prontuários eletrônicos brasileiros (MV, Tasy, Soul MV)
- [ ] Detecção de out-of-distribution: rejeitar exames com valores fisiologicamente impossíveis além do `value >= 0` atual

### Segurança e Privacidade

- [ ] TLS mútuo (mTLS) entre servidor e clientes Flower
- [ ] Differential Privacy nos pesos (Gaussian mechanism, ε-δ DP)
- [ ] Auditoria de acesso e rastreabilidade conforme LGPD Art. 37
- [ ] Consentimento informado e designação de DPO

### Modelo

- [ ] Temperature scaling (ver seção Dependências de Produção)
- [ ] Avaliação com AUC-ROC, sensibilidade e especificidade em estudo retrospectivo
- [ ] Fine-tuning em corpus clínico brasileiro (MIMIC-BR ou equivalente)
- [ ] Substituir DistilGPT-2 por LLM em português (Maritaca, Llama-PT) no módulo RAG

### Infraestrutura

- [x] ~~**Ambiente wire-production (Docker Compose)**~~
- [ ] Redis para rate limiting (ver seção Dependências de Produção)
- [ ] Monitoramento com Prometheus + Grafana para métricas de treino federado
- [ ] Message broker (RabbitMQ ou Redis) para orquestração de rounds
- [ ] Chamadas gRPC diretas do scheduler para o servidor Flower

### Regulatório

- [ ] Submissão ANVISA como Software como Dispositivo Médico (SaMD) — classificação provável Classe III (RDC 657/2022)
- [ ] Validação clínica prospectiva com parecer de comitê de ética (CEP/CONEP)
- [ ] Documentação técnica conforme Resolução CFM 2.227/2018 (sistemas de suporte à decisão clínica)
