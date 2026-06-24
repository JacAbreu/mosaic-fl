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
  `MODEL_CFG.class_labels` (4 classes de prognóstico: alta, internacao_prolongada, uti, obito), configuráveis via `FL_CLASS_LABELS`.
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

- [x] ~~**`hospital_id` explícito no DataSourceFactory e runner**~~

  `SimulatedDataSource` e `CSVDataSource` passaram a aceitar `hospital_id: Optional[str] = None`. `_client_fn()` em `runner.py` lê `hospital-id` de `context.node_config` (ou `FL_HOSPITAL_ID`), repassa para `DataSourceFactory.create()`. Cache key mudou de `str` para `(source_type, hospital_id)` — evita colisão quando dois clientes carregam datasources diferentes do mesmo tipo.

- [x] ~~**`temperature` no retorno de `predict_proba()`**~~

  Ambos os returns de `predict_proba()` em `inference_engine.py` incluem `"temperature": self._temperature`. Necessário para que `service.py` construa `InferenceOutput` para exportação FHIR com o valor de T real do checkpoint.

- [x] ~~**`evaluate()` integrado no script de simulação com relatório pré/pós-calibração**~~

  `run_federated_learning_manual()` e `run_federated_learning_ray()` em `run_experiments_simulation.py` executam `evaluate()` antes e depois do temperature scaling. Métricas incluem ECE, AUC-ROC, F1 e matriz de confusão. Resultados gravados em JSON em `experiments/logs/`. Bug corrigido: `scaler.T` estava sendo referenciado fora de escopo em `_save_evaluation_report()` na `strategy.py`.

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

- [x] ~~**Módulo FHIR R4 — `integration/fhir/` implementado e isolado**~~

  `InferenceOutput` (contrato sem dados clínicos, apenas probabilidades + `correlation_token`), `FHIRExporter.to_risk_assessment()` (gera `RiskAssessment` FHIR R4 válido), `loinc_map.py` (22 analitos mapeados). Isolamento arquitetural verificado por testes: o módulo não importa `infrastructure/`, `PatientExport` nem `sqlalchemy`. `correlation_token` é gerado pelo hospital chamador e ecoado no campo `subject.identifier` — MOSAIC-FL nunca armazena o mapeamento. `service.py` integrado: `IngestRequest` aceita `correlation_token`, `IngestResponse` retorna `fhir_risk_assessment`. 33 testes unitários.

- [ ] **ClinicalPath: adicionar `FL_PROB_*` ao `list_exams.txt`**

  Exames sintéticos injetados pelo exportador (`FL_PROB_ALTA`, `FL_PROB_INTERNACAO_PROLONGADA`, `FL_PROB_UTI`, `FL_PROB_OBITO`, variantes `_INCERTEZA`) não estão no `list_exams.txt` do ClinicalPath. Pendente autorização do Prof. Claudio Linhares (email enviado em `docs/email_claudio.md`).

- [ ] **ClinicalPath: implementar geração do `network.txt` no exportador**

  O exportador atual não gera esse arquivo, que o ClinicalPath requer para carregar o paciente. A lógica de geração já foi mapeada — é simples implementar do nosso lado sem alterar o ClinicalPath.

- [ ] **ClinicalPath: esquema de IDs do `time-metadata.txt` (JGraphX)**

  O ClinicalPath usa `exam_id × num_timestamps + timestamp_id` para nós internos do grafo JGraphX. Sem acesso ao código-fonte do `.jar`, não é possível determinar a fórmula exata. Bloqueado até resposta do Prof. Claudio.

- [ ] Integração FHIR com EPR dos hospitais (Tasy, MV, Soul MV)
- [ ] Conector genérico para prontuários eletrônicos brasileiros
- [ ] Detecção de out-of-distribution: rejeitar exames com valores fisiologicamente impossíveis além do `value >= 0` atual

### Segurança e Privacidade

- [ ] TLS mútuo (mTLS) entre servidor e clientes Flower
- [ ] Differential Privacy nos pesos (Gaussian mechanism, ε-δ DP)
- [ ] Auditoria de acesso e rastreabilidade conforme LGPD Art. 37
- [ ] Consentimento informado e designação de DPO

### Modelo

- [x] ~~**Temperature scaling pós-treinamento (ver seção Dependências de Produção)**~~
- [ ] **Medir ECE com dados reais do FAPESP**

  Em simulação com dados sintéticos o ECE foi 0,44 — colapso para classe majoritária, probabilidades não confiáveis. Medição com dados reais do FAPESP está pendente do carregamento completo do dataset no PostgreSQL. Limiar aceitável para defesa: ECE < 0,10 (idealmente < 0,05).

- [ ] **[DESKTOP] Confirmar colunas de magnitude e demográficos no dataset bruto FAPESP antes de implementar**

  > Prompt original: *"o dataset que temos permite processar magnitudes? você citou isso duas vezes e admito do meu conhecimento do tema, essa poderia ser um fator que traria força para o treinamento. Outro ponto também é que apesar de não conhecermos os pacientes, a idade e o genero se forem informados devem fazer parte da busca das probabilidades. Devemos ter 2 tipos de treinamento: um genérico com a média que foi justamente o ponto que você relatou que fazemos e um especifico em que temos a idade e o genero do paciente. Avalie os ganhos da minha afirmação e a viabilidade da implementação no estado atual do mosaicfl"*

  **Contexto arquitetural já analisado (retomar no desktop):**

  O schema garante que `metrics.exam_records.value` (REAL) e `metrics.exam_records.ref_low/ref_high` existem, e que `clinical.attendances.age` (REAL) e `.sex` (TEXT) existem. Porém a migration 009 adicionou colunas via ALTER TABLE e é possível que os CSVs FAPESP não tenham preenchido `value` ou que `age`/`sex` estejam nulos na maioria dos registros.

  **O que confirmar no desktop (com acesso ao PostgreSQL com dados reais):**
  ```sql
  -- 1. Magnitude: quantos registros têm value não-nulo e não-sentinela?
  SELECT COUNT(*) FILTER (WHERE value IS NOT NULL AND value >= 0) AS com_magnitude,
         COUNT(*) AS total
  FROM metrics.exam_records;

  -- 2. Referências canônicas: cobertura por analito
  SELECT analyte, COUNT(*) FILTER (WHERE canonical_ref_low IS NOT NULL) AS com_ref,
         COUNT(*) AS total
  FROM metrics.exam_records GROUP BY analyte ORDER BY total DESC LIMIT 20;

  -- 3. Demográficos: cobertura de age e sex nas internações
  SELECT COUNT(*) FILTER (WHERE age IS NOT NULL AND age > 0) AS com_idade,
         COUNT(*) FILTER (WHERE sex IN ('M','F')) AS com_sexo,
         COUNT(*) AS total
  FROM clinical.attendances;
  ```

  **Proposta de implementação (pendente confirmação de cobertura):**
  - **Magnitudes**: bucketing fino dentro de HIGH/NORMAL/LOW usando `(value - ref_low) / (ref_high - ref_low)` → 3 tiers por classe (ex.: `PCR_HIGH_MILD`, `PCR_HIGH_SEVERE`, `PCR_HIGH_CRITICAL`). Mantém arquitetura de tokens. Impacto esperado: AUC +2–5% nos analitos com gradiente clínico relevante (PCR, D-dimer, LDH, ferritina).
  - **Dois modos de inferência**: late fusion no classifier head — `Linear(embed_dim + 2, 64)` para modo específico (age normalizado + sex binário), `Linear(embed_dim, 64)` para modo genérico (atual). Sem alteração no Transformer. Impacto esperado: AUC +10–15% em UTI/óbito dado que idade é o preditor dominante em COVID-19.
  - **Arquivos a modificar**: `preprocessor.py` (SQL + `_build_tensors()`), `model.py` (classifier head + argumento `demographics`), `inference_engine.py` (passar demographics opcional), `service.py` (aceitar `age`/`sex` no request).

---

- [ ] **[DESKTOP / TCC] Resolver penalidade acadêmica: "Arquitetura BEHRT muito simplificada para o claim (−0,5)"**

  Esta tarefa nasceu da avaliação ACADÊMICA do projeto (score 6.5/10 em 2026-06-24) e condensa toda a discussão travada sobre arquitetura do `SimplifiedBEHRT`. **Não é necessariamente um bug — é uma lacuna de justificativa na tese.** O objetivo é ou (a) justificar formalmente que a simplificação é a escolha correta, ou (b) estender a arquitetura onde há ganho real.

  ### Arquitetura atual do SimplifiedBEHRT (`src/mosaicfl/core/model.py`)

  - `embed_dim=64`, `num_layers=2`, `num_heads=4`, `ff_dim=128`, ~712K parâmetros
  - **Sem** age embedding, **sem** visit embedding, **sem** segment embedding
  - Positional encoding **sinusoidal** (não aprendido) — codifica posição dos exames na sequência temporal
  - CLS token como `nn.Parameter` com `trunc_normal_(std=0.02)`, prefixado antes do encoder
  - `BEHRTEncoderLayer` customizado que expõe pesos de atenção por cabeça (`average_attn_weights=False`)
  - Classifier head: `Linear(64, 64) → ReLU → Dropout → Linear(64, 4)`
  - Cada attention head opera em subespaço de **16 dimensões** (64 / 4 heads) — muito estreito

  ### Por que foi simplificado (argumentos para a tese)

  1. **Tamanho do dataset**: dataset FAPESP tem escala hospitalar única (uma internação por paciente, sem histórico longitudinal longo). O BEHRT original (Rao et al., 2020) foi pré-treinado em 1,6M de pacientes UK Biobank. Modelos mais profundos com `embed_dim` maior convergem para o mesmo mínimo local quando os dados são escassos — ou pior, overfitam.
  2. **Hardware CPU-only**: sem GPU nos clientes FL (hospitais reais), treinamento com modelos profundos é inviável em rounds frequentes. O SimplifiedBEHRT treina em ~3min/round em CPU.
  3. **Estrutura das internações FAPESP**: cada paciente tem uma única internação COVID-19. O BEHRT original foi projetado para sequências multi-visita ao longo de anos. Com internação única, visitas ordenadas no tempo são exames dentro do mesmo episódio — a distinção semântica entre "visit embedding" e "positional encoding temporal" colapsa.
  4. **Ausência de pré-treinamento**: o BEHRT original usa MLM (Masked Language Model) sobre sequências de CID-10. Sem corpus pré-treinado em português/FAPESP, adicionar camadas extras sem pré-treino apenas adiciona parâmetros aleatórios a serem aprendidos do zero com dados escassos.

  ### Por que NÃO adicionar mais camadas (discutido e concluído)

  A intuição "mais camadas = mais capacidade" não se aplica aqui porque o **gargalo é `embed_dim=64`, não a profundidade**. Cada camada adicional processa o mesmo espaço residual de 64 dimensões (16-dim por cabeça). Mais camadas iterando sobre 16-dim não aumentam capacidade representacional — apenas aumentam computação. Para ganho real, precisaria aumentar `embed_dim` (ex.: 128 ou 256), o que triplica os parâmetros e o tempo de treino.

  ### O que é defensável na tese vs. o que precisa de extensão

  **Defensável sem mudança de código:**
  - Justificar SimplifiedBEHRT pela escala do dataset e ausência de pré-treinamento (itens 1–4 acima)
  - Citar que o positional encoding sinusoidal é equivalente ao aprendido em sequências curtas (Vaswani et al., 2017 — o paper original do Transformer)
  - Adicionar comparação formal com baseline simples (logistic regression ou XGBoost nos mesmos tokens) para demonstrar que o Transformer agrega valor mesmo simplificado

  **Precisa de extensão para eliminar a penalidade:**
  - Injetar demográficos (idade e sexo) via **late fusion** no classifier head (tarefa acima)
  - Documentar na tese o tradeoff escolhido: SimplifiedBEHRT vs. BEHRT completo

  ### Late fusion vs. early fusion (concatenação no CLS) — decisão tomada

  **Early fusion** (concatenar age no embedding CLS antes do encoder): distorce o espaço residual do Transformer. O CLS token tem semântica de agregação global; injetar um escalar dimensional incompatível antes das camadas de atenção força o modelo a "neutralizar" o sinal demográfico no residual stream ou a aprender atenção enviesada. Requer projeção e aumenta complexidade sem clareza de ganho.

  **Late fusion** (concatenar age + sex ao CLS poolado antes do classifier head): **escolha correta para este cenário**. O Transformer aprende a representação da sequência de exames sem interferência demográfica. O classifier head recebe `[CLS_output ∈ R^64 || age_norm ∈ R^1 || sex_bin ∈ R^1]` e aprende a ponderar ambas as fontes de sinal independentemente. Não altera os pesos do Transformer. Compatível com FL (demographics ficam locais em cada hospital, nunca são transmitidos).

  **Implementação concreta da late fusion:**
  ```python
  # model.py — SimplifiedBEHRT.forward()
  def forward(self, x, demographics=None):
      # ... (encoder atual, sem alteração)
      pooled = cls_output  # shape: (B, embed_dim)
      if demographics is not None:
          pooled = torch.cat([pooled, demographics], dim=-1)  # (B, embed_dim + D)
      return self.classifier(pooled)

  # ModelConfig: dois classifier heads ou um head com dim condicional
  # Opção mais limpa: único head com embed_dim + max_demo_dim, zerar demo quando ausente
  ```

- [ ] **Definir janela temporal da predição**

  O modelo é treinado com o histórico completo da internação. Na prática clínica, a predição ocorre com dados parciais. Falta definir formalmente qual janela usar (ex.: fim do 1º, 3º ou 5º dia de internação). Decisão clínica — orientadora ou literatura devem embasar.

- [ ] **Validar rótulos de desfecho clinicamente**

  4 classes derivadas do FAPESP: alta, internação prolongada, UTI, óbito. Verificar se têm suporte na literatura como categorias de prognóstico independentes — ou se devem ser substituídas por escores de gravidade (ex.: SOFA score).

- [ ] **Validar fórmula do risk score clinicamente**

  Score escalar `sum(prob × linspace(0,1,n))` — sem justificativa clínica, escolha técnica. Substituir por fórmula com embasamento na literatura ou justificar formalmente para a defesa.

- [ ] **Revisão clínica do mapeamento LOINC (22 analitos)**

  Mapeamento feito de forma técnica; deve ser verificado por profissional de saúde ou terminologista antes de uso clínico.

- [ ] Avaliação com AUC-ROC, sensibilidade e especificidade em estudo retrospectivo
- [ ] Fine-tuning em corpus clínico brasileiro (MIMIC-BR ou equivalente)
- [ ] Substituir DistilGPT-2 por LLM em português (Maritaca, Llama-PT) no módulo RAG

### Infraestrutura

- [x] ~~**Ambiente wire-production (Docker Compose)**~~
- [ ] Redis para rate limiting (ver seção Dependências de Produção)
- [ ] Monitoramento com Prometheus + Grafana para métricas de treino federado
- [ ] Message broker (RabbitMQ ou Redis) para orquestração de rounds
- [ ] Chamadas gRPC diretas do scheduler para o servidor Flower

### Custo computacional e energético

- [ ] **Medir e documentar custo computacional de um round de treinamento FL**

  Tempo de treinamento, uso de CPU/GPU e consumo de memória por round — em simulação e (futuramente) com clientes reais. Relevante para cenários de restrição energética (hospitais com infraestrutura elétrica limitada ou instável).

- [ ] **Avaliar viabilidade em hardware de baixo consumo**

  Verificar se o modelo pode ser treinado/inferido em hardware mais modesto (ex.: sem GPU dedicada) e qual é o impacto na latência e precisão. Considerar quantização e pruning se necessário.

### Regulatório

- [ ] Submissão ANVISA como Software como Dispositivo Médico (SaMD) — classificação provável Classe III (RDC 657/2022)
- [ ] Validação clínica prospectiva com parecer de comitê de ética (CEP/CONEP)
- [ ] Documentação técnica conforme Resolução CFM 2.227/2018 (sistemas de suporte à decisão clínica)
