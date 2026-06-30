# TODO — MOSAIC-FL

Última atualização: 2026-06-29 (pós-Exp 16 / pré-Exp 17)

Dividido em quatro partes:
- **Experimentos e validação** — o que precisa ser executado e medido antes da defesa
- **Qualidade profissional** — padrão de engenharia, independente de deploy real
- **Dependências de produção** — o que impede deploy em ambiente hospitalar real
- **Roadmap de produção** — funcionalidades para uso clínico completo

---

## Experimentos e Validação

> Esta seção documenta a fila de execuções e validações necessárias para completar os resultados do TCC.

### Série DP-FedAvg (Fase 2 — Privacidade Diferencial)

DP-FedAvg implementado (McMahan et al. 2018) em `client.py` + `fl_core.py`. Resultados ainda não coletados.

- [ ] **Exp 17 — DP-FedAvg σ=1,0, S=1,0**

  ```bash
  FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full
  ```
  Preencher template em `docs/Sumario_Treinamento_Parte2.md`. Métricas a coletar: Acc, Macro F1, AUC, ECE, P@3, dp_update_norm médio, fração de updates clipados.

- [ ] **Exp 18 — DP-FedAvg σ=0,5, S=1,0** (menos ruído)

  ```bash
  FL_DP_NOISE=0.5 FL_DP_CLIP=1.0 make training-full
  ```

- [ ] **Exp 19 — DP-FedAvg σ=2,0, S=1,0** (mais ruído)

  ```bash
  FL_DP_NOISE=2.0 FL_DP_CLIP=1.0 make training-full
  ```

  Após os três experimentos: construir tabela Acc × ε para o TCC (custo de privacidade com DP formal).

### Validação RAG — gemma3:4b

- [ ] **Validar qualidade das justificativas do gemma3:4b**

  Modelo baixado via `make ollama-setup` (~3,3 GB). Executar `make training-full` com Ollama ativo, coletar amostras de justificativas por classe e avaliar coerência clínica (comparar com distilgpt2). Registrar exemplos em `docs/Sumario_Treinamento_Parte2.md`.

### Análises pendentes para a defesa

- [ ] **Tabela unificada de experimentos** (Exp 1–19+)

  Tabela comparativa com todas as métricas relevantes em ordem cronológica, para compor a seção de resultados do TCC.

- [ ] **Diagrama de execução do pipeline** (`make training-full`)

  Fluxograma das 4 fases com durações reais e dependências. Pendente desde reunião com orientadora (2026-06-10).

- [ ] **Definir janela temporal da predição**

  O modelo usa o histórico completo da internação. Na prática clínica, a predição ocorre com dados parciais. Falta definir formalmente qual janela usar (ex.: fim do 1º, 3º ou 5º dia de internação). Decisão clínica — orientadora ou literatura devem embasar.

- [ ] **Validar rótulos de desfecho clinicamente**

  5 classes: `curado_pronto`, `curado_internado`, `melhora_pronto`, `melhora_internado_breve`, `melhora_internado_grave`. Verificar se têm suporte na literatura como categorias de prognóstico independentes em COVID-19 — ou se devem ser substituídas por escores de gravidade (ex.: SOFA score).

- [ ] **Confirmar colunas de magnitude e demográficos no dataset FAPESP**

  Verificar cobertura de `value`, `age`, `sex` antes de implementar late fusion com demográficos no modelo federado:

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

  Resultado do Pooled B (Exp 16) com late fusion demográfica: +0,39 p.p. — ganho confirmado no centralizado. Late fusion no federado ainda não testada separadamente.

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

  `FedProxClient` computa pesos inversamente proporcionais à frequência de cada classe no loader local e passa para `CrossEntropyLoss(weight=...)`. Avaliação (`evaluate`) usa critério sem peso para comparabilidade entre rounds.

- [x] ~~**Class weight clipping (`weights.clamp(max=15.0)`)**~~

  Peso de `melhora_pronto` no BPSP era 47,104 (85 amostras em 20.019). Teto de 15,0 mantém correção do desbalanceamento sem explosão de gradiente.

- [x] ~~**Gradient clipping (`clip_grad_norm(max_norm=1.0)`)**~~

  Implementado em `client.py` antes de `optimizer.step()`.

- [x] ~~**Seeding determinístico por rodada × cliente**~~

  `torch.manual_seed(seed + round * n_clients + client_id)` no início de cada `fit()`.

- [x] ~~**FedNova — normalização por passos efetivos τ_i (Wang et al. 2020)**~~

  Substitui FedAvg simples. BPSP: ~1.251 batches/rodada; HSL: ~226 batches/rodada (razão 5,5×). Sem FedNova, BPSP dominava a agregação. Ganho Exp 12 vs Exp 7: +8,08 p.p. (67,44% vs 59,36%). SCAFFOLD descartado — ver seção de conversa bruta abaixo.

- [x] ~~**Checkpoint guloso (salvar melhor rodada, não a última)**~~

  UPSERT no PostgreSQL a cada rodada com melhora. Incidente motivador: Exp 8 R91=66,61% vs R120=58,27% (gap 8,34 p.p. que seria perdido).

- [x] ~~**Checkpoint scoping por `training_id` (Migration 011)**~~

  Sem scoping, `load_best()` retornava o melhor checkpoint da história do banco. Incidente: Exp 9 avaliou modelo do Exp 8 sem perceber.

- [x] ~~**Implementar `_save_checkpoint` de verdade em `experiment_server.py`**~~

  `CheckpointStore` (ABC) com dois backends: `SQLiteCheckpointStore` (experimentos) e `PostgreSQLCheckpointStore` (`metrics.fl_checkpoints`, produção). Seleção automática via `get_checkpoint_store(FL_DB_URL)`. Integridade via SHA-256.

- [x] ~~**Integrar RAG com tensores reais (modo banco)**~~

  `ClinicalRAG` usa `_InMemoryStore` (numpy cosine similarity) quando `FL_DB_URL` está vazio — sem dependência de PostgreSQL em experimentos.

- [x] ~~**Ollama com fallback automático para HuggingFace**~~

  `_check_ollama_available()` detecta indisponibilidade no `__init__`. Se Ollama offline: loga WARNING, usa `RUNTIME_CFG.llm_hf_model` (distilgpt2) automaticamente. Modelo padrão: `gemma3:4b`.

- [x] ~~**Corrigir bugs de construção da knowledge base RAG**~~

  Bug 1: `[PAD]`/`[CLS]`/`[SEP]` apareciam como top attention tokens (`_is_clinical_token()` filter). Bug 2: `replace("","adulto")` inseria "adulto" entre cada caractere — corrigido com guard `if idade_exacta:`.

- [x] ~~**DP-FedAvg implementado (McMahan et al. 2018)**~~

  Cliente: clipa update Δ = w_final − w_global à norma S antes de enviar. Servidor: adiciona N(0,(σ·S/n)²) após agregação. Ativado por `FL_DP_NOISE=σ`. Experimentos (Exp 17/18/19) ainda não executados.

- [x] ~~**`hospital_id` explícito no DataSourceFactory e runner**~~
- [x] ~~**DataLoader cache no cliente (evita re-query ao banco a cada round)**~~
- [x] ~~**Exponential backoff com jitter no reconect do cliente FL**~~
- [x] ~~**RotatingFileHandler (20MB / 5 backups)**~~
- [x] ~~**`temperature` no retorno de `predict_proba()`**~~

- [ ] **Docstrings completas nos módulos públicos**

### Qualidade de código estático

- [x] ~~**Adicionar type hints completos**~~
- [x] ~~**Adicionar linting com ruff ao `make` e ao CI**~~
- [x] ~~**Configurar pre-commit hooks**~~

### Testes

- [x] ~~**Adicionar testes de contrato para `fit()` e `evaluate()`**~~
- [x] ~~**Testes de integração da API (503/503 passando)**~~

  `TestTokenizer` reescrito como `TestRecordsToTokens`. Mock de `predict_proba` corrigido. Namespace collision entre `tests/integration/` e `integration/` resolvida removendo `__init__.py`. `BigInteger` → `with_variant(Integer, "sqlite")` para autoincrement em SQLite.

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

  Vocabulário canônico construído de `knowledge.term_dictionary` + `knowledge.analyte_references` sem dados de pacientes, distribuído a todos os clientes FL. Garante que token IDs sejam idênticos entre hospitais.

- [x] ~~**Probabilidades por classe com incerteza via MC Dropout**~~

  `InferenceEngine.predict_proba()` executa 50 passes com `model.train()` (dropout ativo). Retorna por classe: `value` (média) e `uncertainty` (desvio padrão). Thread-safe via `threading.Lock()`.

- [x] ~~**`trained: bool` na resposta da API**~~
- [x] ~~**Exponential backoff com jitter no reconect do cliente FL**~~
- [x] ~~**DataLoader cache no cliente (evita re-query ao banco a cada round)**~~
- [x] ~~**`hospital_id` explícito no DataSourceFactory e runner**~~

- [x] ~~**Late fusion demográfica (idade + sexo)**~~

  Ramo separado concatenado ao [CLS] antes do classifier head. Ganho no BEHRT Pooled B (120 épocas): +0,39 p.p. Não testado separadamente no federado com FedNova.

- [x] ~~**DiaRelativoEmbedding (embedding de dias desde a admissão)**~~

  Ganho no ablation: +1,80 p.p. (Exp 4→Exp 6).

- [x] ~~**Calibração isotônica OvR (substituiu temperature scaling)**~~

  Temperature scaling falhou em 100% dos experimentos (8/8) — subconfiança sistemática não-uniforme: LBFGS minimiza NLL (converge para T>1 quando modelo está subconfiante), o que piora o ECE. Isotônica OvR: ECE 0,0575 → 0,0149 no Exp 15 (melhora de 74%).

- [x] ~~**`evaluate()` integrado no script de simulação com relatório pré/pós-calibração**~~

  Métricas incluem ECE, AUC-ROC, F1 e matriz de confusão. Resultados gravados em JSON em `experiments/logs/`.

### Configuração e documentação

- [x] ~~**Arquivo `.env.example`**~~
- [x] ~~**CHANGELOG.md**~~
- [x] ~~**CONTRIBUTING.md**~~
- [x] ~~**Glossário de conceitos — Seção 11 no Metodologia MOSAIC-FL**~~

  Todos os termos técnicos definidos com "Por que importa para este projeto" abaixo de cada definição.

- [ ] **Docstrings completas nos módulos públicos**
- [ ] **Excluir `docs/avaliacao_metodologia_mosaicfl.md`** (desatualizado)

---

## Dependências de Produção

> **Estas issues bloqueiam deploy em ambiente hospitalar real.**
> Nenhuma delas impede o TCC ou a simulação local.

### BLOQUEADOR: Rate limiter não funciona com múltiplos workers

- [ ] **Substituir `_SlidingWindowLimiter` in-process por Redis + `fastapi-limiter`**

  O limitador atual é por processo Python. Com Gunicorn e N workers, cada worker tem seu próprio contador — um cliente pode fazer `120 × N` req/min sem ser bloqueado. Em produção com 4 workers, o limite efetivo é 480 req/min.

  **O que fazer:** instalar `fastapi-limiter` + `redis-py`, configurar `FL_REDIS_URL`, e substituir `_rate_check()` por `RateLimiter` do fastapi-limiter.

### BLOQUEADOR: Timeout e circuit breaker no MC Dropout

- [ ] **Adicionar timeout interno e circuit breaker no MC Dropout**

  Passes sequenciais com lock por request — sob carga alta, requests enfileiram antes do timeout HTTP.

  **O que fazer:** adicionar timeout interno e considerar batches paralelos com `torch.vmap`.

### Importante: Rotação de chave HMAC

- [ ] **Estratégia de rotação para `FL_PATIENT_ID_SECRET`**

  Se o secret for comprometido ou precisar ser rotacionado, todos os hashes existentes ficam desvinculados dos registros originais.

  **O que fazer:** armazenar `key_version` junto com cada hash no banco. Na rotação, re-hash os registros existentes com a nova chave mantendo o `key_version` antigo como fallback temporário.

### Resolvidos

- [x] ~~**Temperature scaling pós-treinamento**~~ — substituído por isotônica OvR (ECE 0,0149 no Exp 15)
- [x] ~~**Conjunto de teste global no servidor (`_load_test_data()`)**~~
- [x] ~~**`FL_MC_SAMPLES` via env var**~~
- [x] ~~**API carrega checkpoint automaticamente ao reiniciar**~~

  `_lifespan(app)` substitui `@app.on_event("startup")` (deprecated). Chama `_get_engine()` antes de aceitar tráfego.

- [x] ~~**`_run_ingest` não é atômica**~~

  `_run_ingest` envolve todos os passos em `with _db.begin() as conn:`. Se qualquer passo falhar, tudo reverte.

- [x] ~~**Versão do modelo nas respostas da API**~~

  `checkpoint_round`, `checkpoint_at`, `model_version` (SHA-256 12 hex chars) gravados e propagados.

- [x] ~~**`on_event("startup")` deprecated → `lifespan` context manager**~~

- [x] ~~**Medir ECE com dados reais do FAPESP**~~

  Medido no Exp 15: ECE=0,0149 (calibração isotônica OvR) — abaixo do limiar de 0,05 definido para a defesa.

---

## Roadmap de Produção

### Fase 3 — Distribuído

- [ ] **Desktop como servidor Flower + notebook como cliente**

  Comunicação real entre nós (não simulada em memória). Requer TLS ou rede local configurada.

- [ ] **TLS mútuo (mTLS) entre servidor e clientes Flower**
- [ ] **Message broker (RabbitMQ ou Redis) para orquestração de rounds**
- [ ] **Chamadas gRPC diretas do scheduler para o servidor Flower**

### Fase 4 — API de Inferência

- [ ] **REST endpoint para prognóstico de novo paciente com modelo federado**
- [ ] **Auditoria de acesso e rastreabilidade conforme LGPD Art. 37**
- [ ] **Consentimento informado e designação de DPO**

### Dados e Integração

- [x] ~~**Exportador ClinicalPath**~~
- [x] ~~**Módulo FHIR R4 — `integration/fhir/` implementado e isolado**~~

  `InferenceOutput`, `FHIRExporter.to_risk_assessment()` (FHIR R4 válido), `loinc_map.py` (22 analitos mapeados). 33 testes unitários.

- [ ] **ClinicalPath: adicionar `FL_PROB_*` ao `list_exams.txt`**

  Pendente autorização do Prof. Claudio Linhares (email enviado em `docs/email_claudio.md`).

- [ ] **ClinicalPath: implementar geração do `network.txt` no exportador**
- [ ] **ClinicalPath: esquema de IDs do `time-metadata.txt` (JGraphX)**

  Bloqueado até resposta do Prof. Claudio.

- [ ] Integração FHIR com EPR dos hospitais (Tasy, MV, Soul MV)
- [ ] Conector genérico para prontuários eletrônicos brasileiros
- [ ] Detecção de out-of-distribution além do `value >= 0` atual

### Segurança e Privacidade

- [ ] TLS mútuo (mTLS) entre servidor e clientes Flower
- [x] ~~**Differential Privacy nos pesos — DP-FedAvg implementado**~~ (experimentos Exp 17/18/19 pendentes)
- [ ] Auditoria de acesso e rastreabilidade conforme LGPD Art. 37
- [ ] Consentimento informado e designação de DPO

### Modelo

- [x] ~~**Temperature scaling pós-treinamento**~~ — substituído por isotônica OvR
- [x] ~~**Medir ECE com dados reais do FAPESP**~~ — ECE=0,0149 (Exp 15)

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
  - Injetar demográficos (idade e sexo) via **late fusion** no classifier head (implementado no Pooled B — testar no federado)
  - Documentar na tese o tradeoff escolhido: SimplifiedBEHRT vs. BEHRT completo

  ### Late fusion vs. early fusion — decisão tomada

  **Late fusion** (concatenar age + sex ao CLS poolado antes do classifier head): **escolha correta para este cenário**. O Transformer aprende a representação da sequência de exames sem interferência demográfica. O classifier head recebe `[CLS_output ∈ R^64 || age_norm ∈ R^1 || sex_bin ∈ R^1]`. Compatível com FL (demographics ficam locais em cada hospital).

  ```python
  # model.py — SimplifiedBEHRT.forward()
  def forward(self, x, demographics=None):
      # ... (encoder atual, sem alteração)
      pooled = cls_output  # shape: (B, embed_dim)
      if demographics is not None:
          pooled = torch.cat([pooled, demographics], dim=-1)  # (B, embed_dim + D)
      return self.classifier(pooled)
  ```

- [ ] **Validar fórmula do risk score clinicamente**

  Score escalar `sum(prob × linspace(0,1,n))` — sem justificativa clínica, escolha técnica. Substituir por fórmula com embasamento na literatura ou justificar formalmente para a defesa.

- [ ] **Validar rótulos de desfecho clinicamente**

  5 classes derivadas do FAPESP. Verificar se têm suporte na literatura como categorias de prognóstico independentes — ou se devem ser substituídas por escores de gravidade (ex.: SOFA score).

- [ ] **Revisão clínica do mapeamento LOINC (22 analitos)**

  Mapeamento feito de forma técnica; deve ser verificado por profissional de saúde ou terminologista antes de uso clínico.

- [ ] Avaliação com AUC-ROC, sensibilidade e especificidade em estudo retrospectivo
- [ ] Fine-tuning em corpus clínico brasileiro (MIMIC-BR ou equivalente)
- [x] ~~**Substituir DistilGPT-2 por LLM em português no módulo RAG**~~ — gemma3:4b implementado via Ollama (validação qualitativa pendente)

### Infraestrutura

- [x] ~~**Ambiente wire-production (Docker Compose)**~~
- [ ] Redis para rate limiting (ver seção Dependências de Produção)
- [ ] Monitoramento com Prometheus + Grafana para métricas de treino federado
- [ ] Message broker (RabbitMQ ou Redis) para orquestração de rounds

- [ ] **Medir e documentar custo computacional de um round de treinamento FL**

  Tempo de treinamento, uso de CPU e consumo de memória por round — em simulação e (futuramente) com clientes reais. Duração atual do `make training-full` completo: ~9h43min (583 min) em CPU.

- [ ] **Refactoring MVP (executar SOMENTE após todas as fases implementadas)**

  Modularização + configurações em banco (migration + seed). Não executar enquanto há experimentos ativos.

### Regulatório

- [ ] Submissão ANVISA como Software como Dispositivo Médico (SaMD) — classificação provável Classe III (RDC 657/2022)
- [ ] Validação clínica prospectiva com parecer de comitê de ética (CEP/CONEP)
- [ ] Documentação técnica conforme Resolução CFM 2.227/2018 (sistemas de suporte à decisão clínica)

---

avaliar comparação com SCAFFOLD ou FedNova — que são alternativas diretas ao FedProx para non-IID severo.

SCAFFOLD e FedNova

  O problema central do seu projeto é o client drift — o HSL aprende numa direção (melhora_pronto domina) e o BPSP aprende em outra (quase sem melhora_pronto), e o FedAvg faz uma média
  que dilui os dois. O FedProx tenta controlar isso com o termo µ, mas de forma grosseira.

  SCAFFOLD (Karimireddy et al., 2020) resolve o drift de forma mais elegante: cada cliente mantém uma variável de controle que estima o quanto ele está desviando do gradiente global. A
  cada rodada, o cliente corrige sua atualização local subtraindo o próprio drift e adicionando o drift médio global. É como se cada hospital dissesse: "sei que minha população é 
  enviesada nessa direção — vou compensar antes de enviar os pesos". Resultado: convergência mais estável em non-IID severo, sem precisar de µ alto que penaliza aprendizado local.

  FedNova (Wang et al., 2020) ataca um problema diferente mas relacionado: quando os clientes fazem números diferentes de passos locais (épocas), a agregação FedAvg fica distorcida —
  quem treinou mais domina proporcionalmente mais do que deveria. FedNova normaliza a contribuição de cada cliente pelo número de passos que ele deu, tornando a agregação justa
  independentemente de quantas épocas cada um rodou.

  ---
  Por que importa para o seu TCC:

  No seu cenário, BPSP tem 20.019 amostras e HSL tem 3.621 — e as distribuições são opostas. O FedProx com µ=0,1 reduziu o drift mas não convergiu em 120 rodadas. 
  
  SCAFFOLD
  provavelmente convergiria mais rápido porque corrigiria o drift ativamente em vez de apenas penalizá-lo.

  Implementar SCAFFOLD ou FedNova não é requisito para o TCC — mas mencionar que eles existem e que seriam a evolução natural é exatamente o tipo de "trabalhos futuros" que banca
  examinadora e revisores de artigo valorizam. Você apresenta os resultados do FedProx, explica o limite que encontrou, e aponta o caminho.

porque motivo voce acha que uma das implementacoes nao eh o foco do meu trabalho? eu tenho medo que o scaffold enviese os resultados, acredito que o fednova teria menos eveitos      
  colaterais                                                                                                                                                                            

● Boa pergunta — assumi errado. Fui conservador sem motivo: o projeto está em evolução ativa com experimentos ainda em curso, e vocação para publicação. Implementar FedNova é
  completamente dentro do escopo.

  Seu instinto sobre SCAFFOLD está correto. O problema específico do SCAFFOLD no seu cenário:

  As variáveis de controle do SCAFFOLD são estimativas do gradiente global que cada cliente mantém localmente. Com apenas 2 clientes e distribuições tão opostas (melhora_pronto 61,5%
  vs 0,4%), a variável de controle do BPSP seria calculada quase inteiramente sobre curado_pronto — e ela passaria a corrigir o HSL na direção errada. O remédio vira veneno quando a
  heterogeneidade é extrema e o número de clientes é pequeno.

  FedNova é mais adequado para o seu caso específico porque:

  O problema que o FedNova resolve é diferente do SCAFFOLD — ele ataca a inconsistência objetiva causada por números diferentes de passos locais. No seu cenário, BPSP treina sobre
  20.019 amostras e HSL sobre 3.621 — com o mesmo batch_size=16 e local_epochs=2, o BPSP dá ~2.502 passos e o HSL ~453 passos por rodada. O FedAvg trata isso como equivalente. FedNova
  normaliza cada cliente pela quantidade de passos que ele realmente deu antes de agregar — a contribuição do HSL deixa de ser diluída pelo simples fato de ter menos dados.

  Não introduz hiperparâmetro novo, não mantém estado adicional, e a matemática não cria novo vetor de viés.

  Impacto na implementação: a mudança é contida — aggregate_fedavg vira aggregate_fednova, o fit() do cliente passa a retornar o número de passos efetivos (τ), e o servidor normaliza
  antes de agregar. O resto do pipeline não muda.

claude, quando for avaliar o sistema, nao foque em um trabalho simples para TCC, foque em um sistema que tem que estar        
  prestes a ser um MVP solido em producao em relacao ao treinamento do modelo. As outras partes realmente sao importantes, mas  
  talvez nao fique tao boas assim, mas o modelo precisa estar em nivel profissional.

  deletar -> docs/avaliacao_metodologia_mosaicfl.md desatualizado

  make db-up && alembic upgrade head && make training-full -> subir banco, rodar migrations, iniciar treinamento

treinamento ocorrido no inicio do dia 30 de junho 2026

O que aconteceu conforme esperado:

  ┌──────────────────────┬───────────┬────────────────────────────────────┐
  │     Expectativa      │ Previsto  │               Obtido               │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ Accuracy global      │ 65–72%    │ 65,90% ✓                           │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ F1 macro             │ 0,45–0,55 │ 0,4905 ✓                           │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ AUC macro            │ 0,79–0,83 │ 0,8105 ✓                           │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ Melhor rodada        │ R60–R110  │ R77 ✓                              │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ curado_internado F1  │ 0,000     │ 0,000 ✓                            │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ melhora_int_grave F1 │ 0,25–0,40 │ 0,3215 ✓                           │
  ├──────────────────────┼───────────┼────────────────────────────────────┤
  │ HSL instável         │ sim       │ sim (acc=33,19%, loss crescente) ✓ │
  └──────────────────────┴───────────┴────────────────────────────────────┘

  ---
  O que não aconteceu conforme esperado:

  ┌───────────────┬──────────┬───────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │  Expectativa  │ Previsto │            Obtido             │                                 Problema                                 │
  ├───────────────┼──────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ ECE isotônica │ < 0,02   │ 0,0311 ✗                      │ Modelo subconfiante (T=1,237)                                            │
  ├───────────────┼──────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ HSL accuracy  │ 35–42%   │ 33,19% ✗                      │ Ficou abaixo do intervalo                                                │
  ├───────────────┼──────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ τ_eff médio   │ 40–80    │ 1.095 ✗                       │ Estimativa estava errada — valor é correto matematicamente, previsão não │
  ├───────────────┼──────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Convergência  │ esperada │ não convergiu (120 rodadas) ✗ │ Único run do projeto sem convergência                                    │
  └───────────────┴──────────┴───────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  ---                   
    ---
  O que aconteceu e não estava previsto:

  - Todas as classes melhoraram em relação ao Bloco 1 (exceto curado_internado que era 0 dos dois lados) — efeito do critério F1 macro.
  - Inversão FL vs Pooled: no Bloco 1 T15 o FL superava o Pooled; agora o Pooled supera em 3,61 p.p. Split corrigido é mais honesto.
  - Critério F1 macro fez diferença real: R77 (F1=0,4905) vs o que seria R58 por accuracy (Acc=68,15%, F1=0,4819) — escolhemos um modelo 2,25 p.p. menos acurado mas 0,0086 mais
  equilibrado entre classes.

instalacao do driver placa de video - gpu
! sudo mokutil --list-new; echo ---; sudo efibootmgr -v; echo ---; ls /boot/efi/EFI/ /boot/efi/EFI/*/; echo ---; dpkg -l | grep shim

 sudo mokutil --import /var/lib/shim-signed/mok/MOK.der

 sudo reboot
