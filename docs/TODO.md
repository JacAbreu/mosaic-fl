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

- [ ] **Implementar `_save_checkpoint` de verdade em `experiment_server.py`**

  Hoje o método apenas registra um caminho no histórico sem escrever nada em disco.

- [ ] **Integrar RAG com tensores reais (modo banco)**

  `run_rag_pipeline()` depende de `df_raw` que não existe no modo banco. Adaptar para construir `patient_data` a partir dos tensores/vocab do SequencePipeline.

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

- [ ] **Temperature scaling pós-treinamento**

  MC Dropout mede incerteza epistêmica mas não resolve a calibração. Um modelo pode ter `uncertainty = 0.01` e estar sistematicamente errado. Temperature scaling requer:
  1. Um conjunto de calibração holdout (não visto durante treino nem validação)
  2. Otimizar o parâmetro T que minimiza NLL no holdout
  3. Dividir todos os logits por T antes do softmax

  Sem isso, `model_metadata.calibrated` permanece `False` para sempre — e o `note` na resposta da API precisa ser levado a sério pelos integradores.

### BLOQUEADOR: Sem conjunto de teste global no servidor

- [ ] **Implementar `_load_test_data()` em `FederatedServer`**

  Hoje retorna `None`. Sem dados de teste holdout no servidor, não há como medir generalização do modelo global. O `evaluate_fn` nunca é chamado. É possível treinar N rounds, atingir "convergência" e ter acurácia aleatória em pacientes novos — sem nenhum sinal de alerta.

  **O que fazer:** reservar uma fração dos dados antes da partição por hospital, manter centralmente no servidor, e passar para `get_evaluate_fn()`.

### BLOQUEADOR: `predict_proba` bloqueia sob carga

- [ ] **Timeout e circuit breaker no MC Dropout**

  50 forward passes sequenciais por request. Com requisições concorrentes e o lock em `_mc_lock`, requests enfileiram e podem timeout na camada HTTP antes de receber resposta.

  **O que fazer:** configurar `mc_samples` via env var `FL_MC_SAMPLES` (padrão 50, reduzir para 20 em produção), adicionar timeout interno, e considerar executar os passes em batches paralelos com `torch.vmap` quando disponível.

### BLOQUEADOR: Auto-discovery de checkpoint no startup da API

- [ ] **API não carrega checkpoint automaticamente ao reiniciar**

  Se o processo da API reinicia depois de checkpoints existirem em disco, o `InferenceEngine` sobe com modelo aleatório (`trained: false`) até que `POST /api/fl/reload` seja chamado manualmente ou pelo scheduler.

  **O que fazer:** no startup da API, verificar `FL_CHECKPOINT_DIR` para o checkpoint mais recente (`round_*.pt`) e carregá-lo automaticamente antes de aceitar tráfego.

### BLOQUEADOR: `_run_ingest` não é atômica

- [ ] **Transação abrangendo exames + risco + exportação**

  Hoje: salva exames → lê histórico → prediz → salva risco → exporta. Se o processo morrer entre passo 3 e 4, o exame foi salvo mas o risco não. O estado fica inconsistente sem possibilidade de replay.

  **O que fazer:** envolver os passos de persistência em uma transação do SQLAlchemy. A exportação de arquivo pode ficar fora (não-transacional) mas deve ser idempotente (sobrescrever se já existir).

### Importante: Rastreabilidade de versão do modelo

- [ ] **Versão do modelo nas respostas da API**

  A resposta retorna probabilidades mas não qual round gerou o modelo, quando foi treinado, com quantos hospitais participaram. Se um checkpoint ruim é carregado, não há rastreabilidade para incidentes clínicos.

  **O que fazer:** incluir em `model_metadata`: `checkpoint_round`, `checkpoint_timestamp`, `participating_hospitals` (N, sem identificar quais), `model_version`. Gravar esses metadados no checkpoint junto com `model_state` e `vocab`.

### Importante: Rotação de chave HMAC

- [ ] **Estratégia de rotação para `FL_PATIENT_ID_SECRET`**

  Se o secret for comprometido ou precisar ser rotacionado, todos os hashes existentes ficam desvinculados dos registros originais — o banco perde a capacidade de identificar histórico do paciente.

  **O que fazer:** armazenar `key_version` junto com cada hash no banco (`patient_id_hash`, `key_version`). Na rotação, re-hash os registros existentes com a nova chave mantendo o `key_version` antigo como fallback temporário.

### Importante: `on_event("startup")` deprecated

- [ ] **Migrar para `lifespan` context manager (FastAPI)**

  `@app.on_event("startup")` gera `DeprecationWarning` em cada teste e será removido em versão futura do FastAPI. Migrar para o padrão `lifespan` com `asynccontextmanager`.

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
