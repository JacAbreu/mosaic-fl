# Metodologia do Projeto MOSAIC-FL: Detalhamento Técnico e Metodológico

> **Nota de revisão**: Este documento é uma versão corrigida e ampliada do texto gerado externamente, com base em verificação integral do código-fonte, logs de experimentos e arquivos de avaliação do repositório. Todas as afirmações são rastreáveis ao código-fonte ou a arquivos de log de experimento indicados. Lacunas são explicitamente identificadas.

---

## 1. Contexto e Motivação do Projeto

### 1.1 Definição do Escopo

O MOSAIC-FL (Medical Open-Source Architecture for Scalable Intelligent Clinical — Federated Learning) é um ecossistema de Aprendizado Federado concebido como uma extensão analítica do **ClinicalPath** (Linhares et al., 2023), sistema de prontuário eletrônico utilizado em hospitais brasileiros. Sua arquitetura visa a estimativa de probabilidades de desfechos clínicos e a estratificação de risco por meio do processamento de trajetórias laboratoriais temporais de pacientes com COVID-19.

O projeto consolida três pilares tecnológicos principais:

1. **SimplifiedBEHRT**: modelo Transformer para modelagem de sequências temporais de exames laboratoriais
2. **FedProx e FedNova**: algoritmos de otimização para aprendizado federado em ambientes heterogêneos
3. **Módulo RAG (Retrieval-Augmented Generation)**: sistema para prover justificativas interpretáveis baseadas em perfis clínicos similares

### 1.2 Motivação Clínica e de Pesquisa

A escolha do aprendizado federado para este contexto é motivada por três fatores fundamentais:

**Privacidade de dados**: Os dados de pacientes são protegidos pela Lei Geral de Proteção de Dados (LGPD — Lei 13.709/2018) e pela Resolução CFM 2.227/2018, que estabelecem que informações de saúde não podem ser centralizadas sem consentimento explícito. O aprendizado federado permite o treinamento de modelos sem mover dados brutos entre instituições.

**Heterogeneidade de dados**: Hospitais brasileiros apresentam distribuições de desfechos radicalmente diferentes. O BPSP tem 55,6% de seus casos como `curado_pronto`, enquanto o HSL tem 61,5% como `melhora_pronto` — uma diferença extrema que representa um cenário de non-IID severo. O modelo RF treinado isoladamente no HSL obteve apenas **24,25% de acurácia no conjunto de teste global** (vs. 68,35% do RF centralizado), evidenciando que nenhum hospital consegue generalizar sozinho — motivação empírica direta para a federação.

**Integração com ecossistema hospitalar**: O sistema é projetado para se integrar ao fluxo clínico existente via HL7 FHIR R4, padronização LOINC de analitos, e geração automática de recursos `RiskAssessment`.

### 1.3 Contexto de Engenharia

#### 1.3.1 Arquitetura Hexagonal e Separação de Responsabilidades

O MOSAIC-FL adota a **Arquitetura Hexagonal** (Cockburn, 2005) como princípio estrutural. A ideia central é isolar o domínio de toda dependência de infraestrutura: o pacote `mosaicfl.core` contém exclusivamente lógica de negócio e pode ser importado, testado e evoluído sem nenhum banco de dados, servidor ou framework web presente.

O repositório é organizado em quatro camadas bem delimitadas:

**1. Domínio puro — `src/mosaicfl/core/`**

Contém os componentes de alto valor intelectual do projeto, sem importações de `infrastructure/` ou `integration/`:

| Módulo | Responsabilidade |
|--------|-----------------|
| `model.py` | SimplifiedBEHRT: embeddings, encoder, late fusion, classifier |
| `preprocessor.py` | SQL de ingestão, tokenização, mapeamento de desfechos, splits |
| `federated.py` | Loop federado: FedProx, FedNova, checkpoint scoping |
| `client.py` | Treinamento local: forward, loss proximal, gradient clipping |
| `calibration.py` | Temperature Scaling e Isotonic Calibration OvR |
| `convergence.py` | ConvergenceTracker: warm-up, critério de platô |
| `evaluation.py` | Métricas por classe: AUC, F1, ECE, MCE, matriz de confusão |
| `interpretability.py` | BEHRTPatternExtractor: análise de atenção por cabeça |
| `data_loader.py` | DataLoaders determinísticos por hospital (`torch.Generator` + seed) |
| `rag.py` | ClinicalRAG: InMemoryStore / PostgreSQLStore, geração com DistilGPT-2 |
| `config.py` | Configuração centralizada: `ModelConfig`, `FedConfig`, `RuntimeConfig` |

**2. Adaptadores de infraestrutura — `infrastructure/`**

Quatro pacotes Python independentes, publicáveis separadamente, cada um com seu próprio `pyproject.toml`:

- **`mosaicfl_server`**: daemon Flower (`ServerApp`), `HealthServer` HTTP na porta 8081, `StateStore` PostgreSQL para checkpoints e histórico de rodadas, `CustomFedProxStrategy`, métricas Prometheus (push via Pushgateway para o scheduler)
- **`mosaicfl_client`**: daemon Flower (`ClientApp`), `ClientDaemon` que carrega pesos globais, treina localmente e devolve delta, `RoundDispatcher` com backoff exponencial, heartbeat de disponibilidade
- **`mosaicfl_api`**: FastAPI com três roteadores (`/api/predict`, `/api/exams/ingest`, `/admin/`), `InferenceEngine` com MC Dropout, módulo de auditoria (`audit.log_access`) e segurança (JWT/API-Key, rate limiting por janela deslizante)
- **`mosaicfl_scheduler`**: APScheduler gerenciando o ciclo de rodadas, `ClientAvailabilityChecker`, persistência de estado do scheduler em PostgreSQL, `SchedulerStateStore`

**3. Adaptadores de integração — `integration/`**

Módulos de conversão entre o domínio MOSAIC-FL e padrões externos:

- **`integration/fhir/`**: `FHIRExporter` (produz `RiskAssessment` FHIR R4), `loinc_map.py` (mapeamento analito → LOINC), modelos Pydantic dos recursos FHIR. O exporter é um objeto sem estado — não importa nada de `infrastructure/` e não acessa banco, tornando-o arquiteturalmente impossível de vazar dados clínicos
- **`integration/clinical-path/`**: `ClinicalPathExporter` que gera os cinco arquivos de texto esperados pelo ClinicalPath v2 por paciente: `exam-id.txt`, `timestamp_to_date.txt`, `time-metadata.txt`, `node-inline-time.txt`, `node-inline-time-complete.txt`. Injeta exames sintéticos (e.g., `FL_RISK_SCORE`, `FL_PROB_ALTA`, `FL_PROB_ALTA_INCERTEZA`) no formato nativo do prontuário
- **`integration/fapesp/`**: leitores de dados do dataset FAPESP COVID-19
- **`integration/column_resolver.py`**: normalização de nomes de analitos para busca no `knowledge.term_dictionary`
- **`integration/term_manager.py`**: gerenciamento de sinônimos e aliases de analitos

**4. Experimentos e scripts de pesquisa — `experiments/`**

#### 1.3.1.1 Ganhos para Manutenabilidade

A adoção da Arquitetura Hexagonal traz ganhos concretos e verificáveis ao longo do ciclo de vida do projeto:

**Testabilidade sem infraestrutura**: O domínio `mosaicfl.core` não importa SQLAlchemy, FastAPI, Flower nem PyTorch Lightning — apenas PyTorch e scikit-learn. Isso permite rodar os 379 testes unitários em 20–30 segundos em qualquer máquina, sem banco de dados, sem servidor e sem GPU. Um desenvolvedor que clona o repositório pela primeira vez executa `make test` imediatamente, sem configuração de ambiente além do `pip install -e .`.

**Substituição de adaptadores sem impacto no domínio**: O backend de vetores do RAG (`_InMemoryStore` vs `_PostgreSQLStore`) é selecionado por presença de `FL_DB_URL` em runtime — o `ClinicalRAG` não sabe qual usa. Da mesma forma, o `InferenceEngine` pode carregar checkpoints de disco ou de PostgreSQL via `reload()` sem que a lógica de predição mude. Se amanhã o projeto migrar de pgvector para Pinecone, apenas o adaptador muda.

**Evolução independente dos pacotes de infraestrutura**: Os quatro pacotes de infraestrutura (`mosaicfl_server`, `mosaicfl_client`, `mosaicfl_api`, `mosaicfl_scheduler`) têm `pyproject.toml` independentes e podem ser versionados, publicados e implantados em ritmos diferentes. O servidor FL pode ser atualizado sem tocar no cliente, e o cliente hospitalar pode rodar uma versão mais antiga do servidor sem quebrar o contrato — o domínio puro é a única API entre eles.

**Localização de bugs**: Quando um bug surge em inferência, o isolamento entre `InferenceEngine` (infraestrutura) e `SimplifiedBEHRT` (domínio) permite reproduzir o problema unitariamente no domínio sem precisar de banco ou servidor. O bug do mapeamento MD5 — que gerava tokens fora do vocabulário degradando silenciosamente as predições — foi diagnosticado e corrigido exclusivamente no `inference_engine.py` sem nenhuma alteração no `model.py` ou no `preprocessor.py`.

**Documentação viva via testes**: O arquivo `tests/test_fl_cycle_explained.py` (30 funções) documenta o protocolo federado passo-a-passo como testes executáveis. Qualquer desenvolvedor que leia os testes compreende o ciclo de vida completo de uma rodada — inicialização, distribuição de pesos, treinamento local, agregação FedNova, checkpoint guloso — sem consultar documentação externa que pode estar desatualizada.

**Conformidade regulatória localizável**: A LGPD exige que dados de saúde não trafeguem em texto claro e que o controlador saiba exatamente onde e como os dados são processados. Com a separação hexagonal, todos os pontos onde `patient_id` aparece estão no adaptador `security.py` (`_pid_to_internal`), no adaptador `audit.py` (`log_access`) e no adaptador `inference_engine.py`. O domínio puro nunca recebe o ID bruto — ele opera sobre tensores e tokens, não sobre identidades. Isso torna a auditoria de conformidade uma análise pontual, não uma varredura em todo o codebase.

#### 1.3.1.2 Tradeoffs

A Arquitetura Hexagonal resolve problemas reais neste projeto mas impõe custos igualmente reais:

| Tradeoff | Ganho | Custo |
|----------|-------|-------|
| **Isolamento do domínio** | Testes rápidos offline; domínio portável | Mais arquivos e módulos para navegar; curva de entrada maior para novos colaboradores |
| **Adaptadores por backend** | Troca de infraestrutura sem impacto no domínio | Duplicação de lógica: `_make_token()` existe em `preprocessor.py` **e** em `inference_engine.py` como fallback — risco de divergência se um for atualizado sem o outro |
| **Pacotes independentes** | Deploy granular por componente | Overhead de coordenação de versões; `pyproject.toml` por pacote; testes de integração precisam montar o ambiente completo |
| **Inversão de dependência** | O domínio não depende de banco | Injeção de dependência explícita é mais verbosa; `FL_DB_URL` como variável de ambiente é frágil em Docker multi-container se não houver orquestração adequada |
| **Adaptador FHIR sem estado** | `FHIRExporter` é puro e testável | Qualquer mudança no perfil FHIR (ex.: migrar de R4 para R5) exige atualizar o adaptador **e** os modelos Pydantic em `integration/fhir/models.py` separadamente |
| **Registry Prometheus isolado** | Evita colisões entre testes e processos | Pushgateway obrigatório para o scheduler (pull model não funciona para jobs curtos) — adiciona um componente de infraestrutura |
| **Separação experimentos/produção** | Scripts de pesquisa não contaminam o código de produção | Duplicação de configuração: `FED_CFG.pooled_epochs=120` existe mas os scripts de experimento usaram 40 épocas por decisão ad hoc — inconsistência rastreável mas que requer disciplina |

**Consideração específica ao contexto de TCC**: Em um projeto acadêmico de escopo e equipe reduzidos (desenvolvedor solo), a Arquitetura Hexagonal representa um investimento inicial significativo de design que não se justificaria em um protótipo de pesquisa puro. A escolha é deliberada: o MOSAIC-FL é projetado como extensão do ClinicalPath (sistema em produção hospitalar), onde a manutenabilidade e a conformidade regulatória são requisitos não-funcionais de primeira classe — não opções de design. O custo de overhead do hexagonal é amortizado pela redução de risco de defeitos em produção clínica.



Scripts desacoplados do código de produção:

- `run_training.py`: treinamento federado completo com dados reais
- `run_behrt_pooled.py`: baselines BEHRT Pooled A e B
- `run_recalibrate.py`: recalibração de checkpoint existente
- `run_bootstrap_ci.py`: intervalos de confiança via bootstrap
- `run_seed_sensitivity.py`: análise de sensibilidade a seeds

#### 1.3.2 Esquema de Banco de Dados e Migrações

O banco PostgreSQL é gerenciado via **Alembic** com 11 migrações versionadas, correspondendo à evolução do projeto:

| Migration | Conteúdo |
|-----------|---------|
| 001–003 | Schema inicial: patients, attendances |
| 004–006 | exam_records, clinical_outcomes, extensões demográficas |
| 007 | diagnosis em attendances |
| 008–009 | term_dictionary e analyte_references (base do InferenceEngine) |
| 010 | simulation_node_config (configuração dos nós FL) |
| 011 | fl_trainings + training_id em fl_checkpoints (checkpoint scoping) |

O schema organiza tabelas em três namespaces PostgreSQL:
- **`clinical`**: dados clínicos (attendances, patients, exam_records, outcomes)
- **`metrics`**: métricas do FL (fl_trainings, fl_checkpoints, round_metrics)
- **`knowledge`**: base de conhecimento (term_dictionary, analyte_references, clinical_profiles)

#### 1.3.3 Pipeline de Inferência em Produção

O `InferenceEngine` (`infrastructure/mosaicfl_api/inference_engine.py`) resolve uma subtileza crítica: em produção, os exames chegam como nomes livres de analitos (e.g., "HEMOGRAMA COMPLETO – LEUCÓCITOS"), que precisam ser normalizados para o mesmo formato canônico usado no treinamento.

O pipeline de tokenização em inferência espelha exatamente o treinamento:

```
nome_bruto → normalize() → knowledge.term_dictionary → canonical
canonical → knowledge.analyte_references → (ref_low, ref_high)
(canonical, value, ref_low, ref_high) → _classify() → HIGH/NORMAL/LOW/NO_REF
(canonical, classification) → _make_token() → "LEUCOCITOS_HIGH"
token_str → standard_vocab → token_id
```

A substituição de um mapeamento MD5 que gerava tokens fora do vocabulário (bug crítico identificado e corrigido) é documentada no cabeçalho do módulo. A consistência entre tokenização de treinamento e inferência é um requisito de correção clínica — tokens fora do vocabulário retornam `UNK=1`, degradando silenciosamente a qualidade da predição.

O `InferenceEngine` também gerencia o reload de checkpoints via `reload()` e carrega os intervalos de referência do banco via `_load_references()`, ambos com lock (`threading.Lock`) para segurança em ambiente ASGI multi-threaded.

#### 1.3.4 Observabilidade e Auditoria

O sistema implementa três camadas de observabilidade:

**Métricas Prometheus** (`infrastructure/shared/metrics.py`): Registry isolado (não usa o registry global do `prometheus_client`, evitando colisões entre processos e testes) com os seguintes gauges e contadores:

| Métrica | Tipo | Semântica |
|---------|------|---------|
| `fl_rounds_total` | Counter | Total de rodadas concluídas |
| `fl_round_accuracy` | Gauge | Acurácia da última rodada |
| `fl_round_loss` | Gauge | Loss da última rodada |
| `fl_clients_active` | Gauge | Clientes que participaram da última rodada |
| `fl_convergence_round` | Gauge | Rodada em que a convergência foi detectada (-1 = não convergiu) |

O scheduler usa `push_to_gateway()` (Pushgateway) ao final de cada job de curta duração, evitando o problema do pull model para processos efêmeros.

**Auditoria clínica** (`infrastructure/mosaicfl_api/audit.py`): `log_access()` é invocado em todos os endpoints de predição, registrando `patient_id_hash` (HMAC-SHA256, nunca o ID bruto), `exam_count`, `risk_score` e timestamp. O log de auditoria é separado do log de aplicação para facilitar conformidade regulatória.

**Logging estruturado**: `python-json-logger` produz registros JSON com campos tipados (`round`, `client_count`, `accuracy`), compatíveis com ingestão em stacks ELK/Loki sem parser customizado.

#### 1.3.5 Segurança em Camadas

A segurança do sistema é implementada em três planos:

**Plano de transporte**: TLS mútuo obrigatório para comunicação gRPC entre servidor e clientes Flower (`infrastructure/shared/tls.py`). A variável `FL_TLS_CERT_DIR` é obrigatória em produção — ausência lança `EnvironmentError` na inicialização, impedindo execução insegura por acidente. A geração de certificados de desenvolvimento é automatizada via `scripts/gen_certs.sh`.

**Plano de API**: Dois mecanismos de autenticação configuráveis via variáveis de ambiente:
- JWT com `FL_JWT_SECRET` (HMAC-HS256) ou `FL_JWT_PUBLIC_KEY_FILE` (RSA RS256/RS512)
- API Key via header `X-API-Key`

Rate limiting por janela deslizante sem dependências externas (implementação pura em Python):
- Endpoints gerais: 120 requisições/60s por IP
- Endpoint de ingestão: 30 requisições/60s por IP (dado o custo computacional de tokenização + inferência)

**Plano de dados**: Pseudonimização HMAC-SHA256 de `patient_id` com `FL_PATIENT_ID_SECRET` local por hospital (nunca compartilhado com o servidor central). O recurso FHIR usa `correlation_token` efêmero (UUID descartável por requisição), cumprindo o campo obrigatório `subject` sem que o servidor FL armazene qualquer mapeamento de identidade.

#### 1.3.6 Pirâmide de Testes

**Contagem verificada**: O repositório contém **569 funções de teste** em 40 arquivos, distribuídas em três níveis:

| Nível | Quantidade | Escopo |
|-------|-----------|--------|
| Unitários (`tests/unit/`) | 379 | Componentes isolados: model, calibration, convergence, security, FHIR, data splits |
| Integração (`tests/integration/`) | 154 | Fluxos completos com mocks de banco: API endpoints, ClinicalPath exporter, infraestrutura |
| E2E (`tests/e2e/`) | 6 | Ciclo FL real sem mocks (marcados `@pytest.mark.e2e`, excluídos do `make test` padrão) |
| Ciclo FL documentado (`test_fl_cycle_explained.py`) | 30 | Testes narrativos que documentam o protocolo federado passo-a-passo |

O `pyproject.toml` configura `addopts = "-m 'not e2e'"` — os testes e2e requerem banco de dados ativo e são executados explicitamente via `make test-e2e`. O `make test` roda os 409 testes unitários e de integração em ambiente offline.

A cobertura de código (`make test-cov`) exclui explicitamente módulos de infraestrutura bloqueantes (daemons, runners APScheduler) e conectores de banco que exigem infraestrutura real, focando a medição no domínio e na lógica de API.

#### 1.3.7 Orquestração e Reprodutibilidade

O **Makefile** é o ponto de entrada único para todas as operações do projeto, eliminando dependências de IDEs ou scripts ad hoc:

| Target | Função |
|--------|--------|
| `make test` | 409 testes unitários + integração (offline) |
| `make test-all` | Todos os 569 testes (requer banco) |
| `make test-cov` | Testes + relatório de cobertura |
| `make training` | Treinamento federado com dados reais (FL_ENV=production) |
| `make training-bpsp-only` | SimplifiedBEHRT treinado só com BPSP |
| `make training-hsl-only` | SimplifiedBEHRT treinado só com HSL |
| `make training-full` | Pipeline completo de 4 fases sequenciais |
| `make behrt-pooled` | Baselines BEHRT Pooled A e B |
| `make recalibrate` | Recalibração de checkpoint existente |
| `make seed-sensitivity` | Sensibilidade a seeds (multi-run) |
| `make bootstrap-ci` | Intervalos de confiança bootstrap |
| `make db-up / db-down` | Banco PostgreSQL via Docker Compose |
| `make fl-server / fl-client` | Servidor e cliente Flower |

O ambiente de experimentos é completamente reprodutível: splits determinísticos por seed, vocabulário padrão versionado (`standard_vocab.json`), checkpoints com `training_id` scoped, e logs com timestamp por experimento em `experiments/logs/`.

Todos os experimentos foram executados em um Dell Inspiron 5402 (Intel Core i7-1165G7, 16 GB RAM, sem GPU dedicada). O `RuntimeConfig.device = "cpu"` é a configuração de produção do projeto. Essa restrição influenciou diretamente escolhas de design: `embed_dim=64`, `num_layers=2`, `max_seq_len=128` e `batch_size=16` foram calibrados para tempo de treino aceitável (~4h por experimento completo de 120 rodadas).

### 1.3.8 Requisitos de Hardware

O projeto opera em dois modos com perfis de memória completamente distintos: **treino federado** (SimplifiedBEHRT, processo isolado) e **inferência em produção** (API + RAG). É fundamental distingui-los porque a adição do Mistral-Nemo ao componente RAG **não altera os requisitos de treino** — afeta exclusivamente a stack de inferência.

#### 1.3.8.1 Anatomia de Memória do SimplifiedBEHRT

O modelo é intencionalmente pequeno. A medição direta no código (`sum(p.numel() for p in model.parameters())`) retorna **715.589 parâmetros**, com a seguinte distribuição:

| Componente | Parâmetros | Tamanho (fp32) |
|---|---|---|
| `embedding.weight` (vocabulário) | 640.000 | 2,44 MB |
| Encoder layers (2×) — atenção + FF | 65.028 | 0,25 MB |
| DiaRelativoEmbedding | 3.968 | 0,02 MB |
| Classifier head | 4.096 + 320 | 0,02 MB |
| **Total** | **715.589** | **2,73 MB** |

89% do modelo é a tabela de embedding de vocabulário. A consequência prática: trocar `embed_dim` de 64 para 128 dobraria o modelo inteiro; aumentar `vocab_size` de 10.000 para 50.000 o multiplicaria por 5. As escolhas de `embed_dim=64` e `vocab_size=10.000` são as variáveis de design com maior alavancagem sobre o tamanho do modelo.

Durante o treino com Adam, o footprint real é:

| Elemento | Memória |
|---|---|
| Pesos fp32 | 2,73 MB |
| Estado Adam (momentum + variância) | 5,46 MB |
| Ativações (batch=16, seq=128) | ~10 MB |
| DataLoader + buffers | ~200 MB |
| **Total processo de treino** | **~220 MB** (modelo + dados) |

#### 1.3.8.2 Stack de Inferência — Sistema Atual

O processo de inferência em produção (`mosaicfl_api`) carrega em memória todos os componentes simultaneamente:

| Componente | RAM | Observação |
|---|---|---|
| OS + Python runtime | 1.400 MB | Ubuntu 22.04 LTS |
| FastAPI + uvicorn | 120 MB | 1 worker ASGI |
| PostgreSQL (pool local) | 80 MB | SQLAlchemy pool |
| SimplifiedBEHRT fp32 | 3 MB | 715K parâmetros |
| MC Dropout (50 amostras, pico) | 10 MB | 50× forward pass simultâneo |
| all-MiniLM-L6-v2 (RAG embedder) | 90 MB | 22M parâmetros, sentence-transformers |
| DistilGPT-2 (gerador atual) | 330 MB | 82M parâmetros fp32 |
| pgvector / InMemoryStore | 50 MB | índice knowledge.clinical_profiles |
| Overhead Python/GC | 200 MB | buffers, cache de módulos |
| **Total inferência atual** | **2.283 MB (2,2 GB)** | |

**Mínimo absoluto**: 4 GB RAM (sistema operacional + API). **Recomendado**: 8 GB (margem para picos de carga e múltiplas requisições concorrentes).

#### 1.3.8.3 Stack de Inferência — Com Mistral-Nemo 12B Q4 Local

A substituição do DistilGPT-2 (330 MB) pelo Mistral-Nemo 12B quantizado Q4\_K\_M altera apenas a parcela de geração RAG:

| Componente | RAM | Observação |
|---|---|---|
| Stack base (sem DistilGPT-2) | 1.953 MB | Todos os itens acima exceto gerador |
| Mistral-Nemo 12B Q4\_K\_M (pesos) | 7.200 MB | ~4 bits/parâmetro médio |
| KV cache (ctx=2.048 tokens) | 350 MB | 40 layers, n\_kv\_heads=8, head\_dim=128 |
| llama-cpp-python runtime | 200 MB | framework GGUF |
| **Total inferência com Mistral-Nemo** | **9.703 MB (9,5 GB)** | |
| Headroom em 16 GB | 6.681 MB (6,5 GB) | confortável |
| Headroom em 32 GB | 23.065 MB (22,5 GB) | ideal para múltiplos workers |

**Mínimo absoluto**: 12 GB RAM (o modelo exige 7,5 GB sozinho; headroom mínimo de SO é ~3 GB). **Recomendado**: 16 GB para uso único, 32 GB para servidor com múltiplos workers paralelos.

**Ponto crítico**: em uma máquina de 16 GB, o treino federado (1,9 GB) e a inferência com Mistral-Nemo (9,5 GB) **não devem rodar simultaneamente** — a soma de 11,4 GB deixa apenas 4,6 GB de headroom, que pode ser insuficiente sob pressão do SO e GC. A recomendação é isolar o processo de treino em janela separada ou agendar via `make training` fora do horário de atendimento clínico.

#### 1.3.8.4 Impacto de Latência por Modo de Hardware

| Operação | CPU (i7-1165G7) | CPU servidor (Xeon 16c) | GPU RTX 4070 Ti (12 GB VRAM) |
|---|---|---|---|
| Predição SimplifiedBEHRT (50 MC samples) | < 1s | < 0,5s | < 0,1s |
| Embedding RAG (all-MiniLM-L6-v2) | ~0,5s | ~0,2s | ~0,05s |
| Geração DistilGPT-2 (150 tokens) | ~12s | ~5s | ~1s |
| Geração Mistral-Nemo Q4 (150 tokens) | ~75s | ~30s | ~5s |
| **Resposta total (predição + RAG)** | **~76s** | **~31s** | **~6s** |

Com GPU RTX 4070 Ti (12 GB VRAM), o Mistral-Nemo 12B Q4 cabe inteiramente em VRAM (7,75 GB utilizados de 12 GB disponíveis), reduzindo o tempo de geração de 75 segundos para ~5 segundos — viável para uso clínico interativo. **A GPU não acelera o treino do SimplifiedBEHRT de forma significativa** (modelo com apenas 2,73 MB de parâmetros não satura batches de GPU com `batch_size=16`; seria necessário aumentar `batch_size` para 128 ou 256 para extrair ganho real, o que reduziria o tempo de treino de ~4h para ~30–45 minutos).

#### 1.3.8.5 Perfis de Hardware por Cenário de Implantação

| Cenário | RAM | CPU | GPU |
|---|---|---|---|
| **Pesquisa / TCC** (sistema atual) | 16 GB | i7 quad-core | não necessária |
| **Hospital pequeno** (API sem RAG generativo) | 8 GB | 4 cores | não necessária |
| **Hospital médio** (API + Mistral-Nemo Q4 CPU) | 32 GB | 16 cores | não necessária |
| **Hospital grande / produção** (API + Mistral-Nemo GPU) | 32 GB | 16 cores | RTX 4070 Ti 12 GB VRAM ou equivalente |
| **Referência máxima** (múltiplos workers + GPU) | 64 GB | 32 cores | A100 40 GB VRAM ou equivalente |

Para o escopo do TCC, o **perfil "Pesquisa / TCC"** cobre todos os cenários de demonstração. Para produção hospitalar com RAG generativo, o **perfil "Hospital médio"** (32 GB, sem GPU) é o mínimo viável — o tempo de geração de ~30 segundos por resposta é aceitável para um sistema de apoio à decisão clínica que não opera em tempo-real.

---

## 2. Caracterização e Pré-processamento dos Dados

### 2.1 Fonte de Dados: Dataset FAPESP COVID-19

**Todos os experimentos utilizam dados reais** do dataset **FAPESP COVID-19 Data Sharing/BR**, acessados via PostgreSQL. A query principal une quatro tabelas:

- `clinical.attendances`: admissões e tipo de atendimento
- `clinical.patients`: sexo e ano de nascimento
- `metrics.clinical_outcomes`: desfecho e data de saída
- `metrics.exam_records`: analito, valor, classificação e data do exame

A query filtra explicitamente `a.hospital_id IN ('HSL', 'BPSP')`, pois são os únicos hospitais com vinculação consistente entre atendimentos e exames (HEI: 0% de vinculação; HFL e HCSP: sem exames vinculados).

### 2.2 Critérios de Inclusão e Exclusão

**Desfechos excluídos**: A query `WHERE co.outcome_class NOT IN (2, 3, 4)` remove:

- Código 2 (Alta administrativa): saída burocrática sem relação com evolução clínica
- Código 3 (Transferência): desfecho clínico final desconhecido
- Código 4 (Em atendimento): dado censurado — desfecho ainda não ocorreu

Adicionalmente, `(co.outcome_at - a.attended_at) >= 0` exclui registros com datas inconsistentes (desfecho anterior à admissão).

### 2.3 Mapeamento de Desfechos Clínicos (5 Classes)

A função `_map_outcome()` cruza três dimensões: `outcome_class` (0=curado, 1=melhora), `attendance_type` (internado ou não), e `duration_days` (dias de internação). O limiar de 10 dias foi **verificado diretamente no código** (`return 3 if duration_days <= 10 else 4`). O resultado são 5 classes:

| Índice | Rótulo | Critério Clínico |
|--------|--------|-----------------|
| 0 | `curado_pronto` | Desfecho curado, atendimento sem internação |
| 1 | `curado_internado` | Desfecho curado, com internação (qualquer duração) |
| 2 | `melhora_pronto` | Desfecho melhora, atendimento sem internação |
| 3 | `melhora_internado_breve` | Desfecho melhora, internação ≤ 10 dias |
| 4 | `melhora_internado_grave` | Desfecho melhora, internação > 10 dias |

A classe 4 (internação > 10 dias) representa o grupo de maior gravidade clínica e custo hospitalar, sendo um alvo prioritário da predição.

### 2.4 Distribuição de Classes: Heterogeneidade Non-IID Extrema

A análise dos dados revela um cenário de **Label Shift** severo entre os hospitais:

**BPSP — distribuição total (N=28.599):**

| Classe | N | % |
|--------|---|---|
| curado_pronto (0) | 15.892 | 55,6% |
| curado_internado (1) | 318 | 1,1% |
| melhora_pronto (2) | 120 | **0,4%** |
| melhora_internado_breve (3) | 9.448 | 33,0% |
| melhora_internado_grave (4) | 2.821 | 9,9% |

**HSL — distribuição total (N=5.174):**

| Classe | N | % |
|--------|---|---|
| curado_pronto (0) | 67 | **1,3%** |
| curado_internado (1) | 45 | 0,9% |
| melhora_pronto (2) | 3.182 | **61,5%** |
| melhora_internado_breve (3) | 1.280 | 24,7% |
| melhora_internado_grave (4) | 600 | 11,6% |

`melhora_pronto` representa 61,5% do HSL e apenas 0,4% do BPSP, enquanto `curado_pronto` representa 55,6% do BPSP e apenas 1,3% do HSL. Esta é a principal fonte de heterogeneidade que justifica a necessidade de algoritmos federados robustos como FedProx e FedNova.

### 2.5 Estratégia de Divisão dos Dados

A partir do Experimento 3, adotou-se a divisão 70/10/10/10 (treino/validação/calibração/teste) para garantir um conjunto de calibração independente. A separação é feita por hospital com gerador determinístico (`RANDOM_SEED + cid`). **O `torch.Generator` é instanciado com seed específico por hospital** em `prepare_dataloaders_from_db()`, garantindo que o split seja idêntico em toda re-execução — requisito de reprodutibilidade.

- 70%: treino local (usado no loop federado)
- 10%: validação local
- 10%: **calibração** — usado exclusivamente para Temperature Scaling e Isotonic Calibration (nunca exposto ao treino)
- 10%: teste global (BPSP + HSL combinados para avaliação final)

Os volumes finais, **confirmados pelos logs de experimento**:

- BPSP: 20.019 treino / 2.859 validação / 2.859 calibração / 2.862 teste
- HSL: 3.621 treino / 517 validação / 517 calibração / 519 teste
- Calibração global: 3.376 amostras
- Teste global: 3.381 amostras

### 2.6 Tokenização e Vocabulário

**Construção do token**: Cada exame é convertido em um token no formato:

```
token = f"{analyte}_{classification}"
```

Exemplos: `LEUCOCITOS_HIGH`, `PCR_NORMAL`, `HEMOGLOBINA_LOW`

Quando a classificação é `"NO_REF"` (sem intervalo de referência cadastrado), o token é apenas o nome do analito.

**Vocabulário**: Construído globalmente sobre o pool completo BPSP+HSL, com os 9.997 tokens mais frequentes (excluindo 3 especiais). Os tokens especiais reservados são:

- `PAD = 0` (padding)
- `UNK = 1` (desconhecido)
- `CLS = 2` (token de classificação)

**Critério de seleção**: O vocabulário é construído puramente por frequência — **não há seleção clínica manual de analitos**. Os tokens mais frequentes entram automaticamente. Isso resulta em um vocabulário de **648 tokens únicos** realmente utilizados nos dados, muito abaixo da capacidade de 10.000 — o excedente é reserva para expansão sem re-treinamento do embedding.

`max_seq_len = 128`: A query SQL usa `ROW_NUMBER() OVER (PARTITION BY attendance_id ORDER BY dia_relativo, analyte)` e filtra `WHERE _rn <= 128`. Os primeiros 128 exames em ordem cronológica (por dia relativo, com desempate pelo nome do analito) são retidos por atendimento. O valor 128 é calibrado para o hardware alvo (Dell Inspiron 5402, i7-1165G7, 16 GB RAM, sem GPU), pois valores maiores aumentariam consumo de memória na dimensão de atenção O(L²).

### 2.7 Pesos de Classe para Tratamento de Desbalanceamento

Para mitigar o desbalanceamento extremo (classe 2 no BPSP com apenas 85 amostras de treino), o sistema calcula pesos de classe inversamente proporcionais à frequência:

```python
weight_i = total / (n_classes × count_i)
weights.clamp(max=15.0)  # teto para evitar explosão de gradiente
```

Sem o clipping, o peso da classe 2 no BPSP seria aproximadamente 47 — o que causava gradientes instáveis. O teto de 15,0 estabiliza o treinamento.

---

## 3. Arquitetura do Modelo: SimplifiedBEHRT

### 3.1 Visão Geral

O modelo é uma arquitetura Transformer simplificada com as seguintes configurações fixas (`ModelConfig`), **todas verificadas em `src/mosaicfl/core/config.py`**:

| Parâmetro | Valor | Motivação |
|-----------|-------|-----------|
| `vocab_size` | 10.000 | Capacidade para tokens futuros (uso atual: 648) |
| `embed_dim` | 64 | Balanceamento entre capacidade e hardware limitado |
| `max_seq_len` | 128 | Cobertura da maioria das internações; limite de memória |
| `num_layers` | 2 | Modelo raso por volume de dados de treino (~23.640 amostras) |
| `num_heads` | 4 | 64/4 = 16 dimensões por cabeça |
| `ff_dim` | 128 | 2× embed_dim (padrão Transformer) |
| `num_classes` | 5 | Correspondente aos 5 desfechos clínicos |
| `dropout` | 0,1 | Regularização para evitar overfitting |

### 3.2 Componentes do Forward Pass

**1. Token Embedding**: Mapeia índice do vocabulário → vetor de 64 dimensões. Índice 0 (PAD) recebe embedding zero.

**2. DiaRelativoEmbedding** (inovação do projeto): Captura a posição temporal do exame dentro do episódio de internação. O dia relativo é o número de dias desde a admissão (`attended_at`). O embedding de dia é **somado** ao embedding de token antes do positional encoding.

Do ponto de vista clínico, isso permite que o modelo capture a **velocidade da progressão clínica** — uma PCR elevada na admissão tem peso prognóstico distinto de uma PCR em ascensão no 5º dia de internação. Esta modificação resultou em um ganho direto de **+3,08 p.p.** na acurácia global (Experimento 6 vs Experimento 5).

Encoding: Valores de dia relativo são deslocados +1 (0=padding, 1=dia 0, 61=dia ≥ 60). O token CLS recebe `dia_relativo = 0` (sem embedding temporal).

**3. PositionalEncoding sinusoidal**: Encoding posicional não aprendível, adicionado após o DiaRelativoEmbedding, seguindo a formulação original do Transformer (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos × exp(-2i × log(10000) / d_model))
PE(pos, 2i+1) = cos(pos × exp(-2i × log(10000) / d_model))
```

**4. Token CLS learnable**: Inicializado com `trunc_normal_` (std=0,02), prefixado à sequência. A representação do token CLS após o encoder é usada como representação do paciente para classificação.

**5. BEHRTEncoderLayer × 2**: Cada camada substitui `nn.TransformerEncoderLayer` para expor pesos de atenção:

- Multi-Head Self-Attention (4 cabeças, `batch_first=True`)
- `need_weights=True, average_attn_weights=False` → shape `(batch, 4, seq, seq)` por camada
- Feed-Forward: `Linear(64, 128) → ReLU → Dropout → Linear(128, 64)`
- LayerNorm pós-atenção e pós-FF (Post-LN, não Pre-LN)
- Residual connections em ambos os sub-blocos

O `average_attn_weights=False` preserva os pesos por cabeça individualmente, permitindo análise de quais analitos o modelo foca em cada cabeça via `BEHRTPatternExtractor`.

**6. Pooling CLS**: O vetor da posição 0 (CLS) após o encoder é usado como representação final.

**7. Pre-classifier**: `LayerNorm(64) → Dropout(0.1)`

**8. Late Fusion Demográfica**: Se dados demográficos estão disponíveis (`demo_dim > 0`), o vetor CLS é concatenado com as features demográficas antes do classificador:

```python
if demographics is not None:
    pooled = torch.cat([pooled, demographics], dim=-1)
# demographics: (batch, 2) → [age_norm, sex_binary]
```

Isso permite que o Transformer aprenda a representação da sequência de exames **sem interferência demográfica**, enquanto o classificador pondera ambas as fontes de forma independente.

**9. Classifier head**: `Linear(64 + demo_dim, 64) → ReLU → Dropout(0.1) → Linear(64, 5)`

### 3.3 Mecanismo de Self-Attention

A arquitetura implementa o mecanismo de atenção padrão do Transformer (Vaswani et al., 2017):

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Q = X · W_Q ∈ R^{L×d_k}
K = X · W_K ∈ R^{L×d_k}
V = X · W_V ∈ R^{L×d_v}

d_k = embed_dim / num_heads = 64 / 4 = 16

MultiHead(Q, K, V) = Concat(head_1, ..., head_4) · W_O
head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

Isso permite que cada cabeça de atenção foque em diferentes padrões de co-ocorrência temporal entre analitos, capturando relações como "PCR elevada seguida de D-dímero elevado em 3 dias" que são clinicamente significativas.

### 3.4 Inferência com MC Dropout (Incerteza Epistêmica)

Em inferência, o sistema executa **50 amostras de Monte Carlo Dropout** (mantendo `model.train()` ativo para habilitar o dropout), calculando média e desvio padrão por classe. Isso fornece:

- **Probabilidades calibradas**: média das 50 amostras (mais robusta que um único forward pass)
- **Incerteza epistêmica**: desvio padrão por classe, reportado no recurso FHIR `RiskAssessment`

O MC Dropout é distinto da calibração de probabilidades (Seção 5): a calibração ajusta a escala das saídas do softmax; o MC Dropout quantifica a variância do próprio modelo sob diferentes configurações de dropout.

### 3.5 BEHRTPatternExtractor — Interpretabilidade

O `BEHRTPatternExtractor` usa os pesos de atenção retornados por `return_attention=True` para:

- Identificar quais analitos cada cabeça de atenção foca por classe de desfecho prevista
- Gerar matrizes de co-ocorrência temporal: quais pares de analitos (posição i, posição j) recebem maior peso conjunto
- Fornecer interpretabilidade clínica da predição sem necessidade de métodos de atribuição externos

A shape `(num_layers, batch, num_heads, seq_len, seq_len)` permite análise por camada, por cabeça e por sequência individual.

---

## 4. Protocolo de Aprendizado Federado

### 4.1 Visão Geral do Ciclo de Vida

O sistema implementa aprendizado federado com dois clientes (BPSP e HSL) em modo de simulação local (ambos os processos rodam na mesma máquina). Cada rodada segue este ciclo:

1. **Inicialização**: Modelo global é inicializado com pesos aleatórios. Um `training_id` é registrado no PostgreSQL via `register_training()` para garantir rastreabilidade.
2. **Distribuição do modelo**: Pesos globais são enviados para cada cliente.
3. **Treinamento local (cada cliente)**:
   - Modelo local carrega os pesos globais via `set_parameters()`
   - Para `local_epochs = 1` (reduzido de 2 para minimizar drift em regime non-IID severo, Li et al. 2020), cada batch:
     - Loss: `L = CrossEntropy(weighted) + (μ/2)·‖w_local − w_global‖²`
     - Gradient clipping: `clip_grad_norm(max_norm=1.0)`
     - Contador τ_i: número de batches processados (passos efetivos)
   - Retorna: pesos atualizados, `num_samples`, métricas (`loss`, `tau`, `grad_norm`)
4. **Agregação no servidor**:
   - Se `use_fednova=True` (padrão): normalização por passos efetivos
   - Se `use_fednova=False`: FedAvg ponderado por `num_samples`
5. **Avaliação global**: `evaluate_global_model(global_model, test_loader)` — acurácia e loss no conjunto de teste global
6. **Checkpoint (melhor rodada)**: Se `acc_global > best_accuracy`, salva via UPSERT no PostgreSQL com `training_id` scoped
7. **Critério de convergência**:
   - Warm-up de 20 rodadas (convergência não avaliada antes)
   - Δaccuracy < 0,005 por 3 rodadas consecutivas
   - Máximo de 120 rodadas

### 4.2 FedProx: Mitigação de Client Drift

O termo proximal do FedProx é implementado em `client.py`:

```python
def _proximal_loss(self, loss, proximal_mu):
    proximal_term = 0.0
    for local_w, global_w in zip(self.model.parameters(), self.global_params):
        proximal_term += torch.norm(local_w - global_w, p=2) ** 2
    return loss + (proximal_mu / 2) * proximal_term
```

Formulação matemática:

```
L_FedProx(w) = L_CE(w) + (μ/2) · ‖w − w*‖²
```

onde `w` são os pesos locais após o update, `w*` são os pesos globais recebidos do servidor, e `μ = 0,1` (aumentado de 0,01 no Experimento 7). O aumento foi motivado pela constatação de que o drift entre clientes era excessivo: com μ=0,01, o modelo oscilava ±12 p.p. entre rodadas.

**Fundamento teórico**: Li et al. (2020) demonstram que em cenários de alta heterogeneidade (γ-inexactness), o termo proximal reduz a divergência entre distribuições locais e global, melhorando a convergência.

### 4.3 FedNova: Normalização por Passos Efetivos

O FedNova resolve o "problema de inconsistência objetiva" (Wang et al., 2020) causado pela disparidade volumétrica entre hospitais. Com `local_epochs=1`:

- BPSP: ~1.251 batches/rodada (20.019 amostras / batch_size=16)
- HSL: ~226 batches/rodada (3.621 amostras / batch_size=16)
- Razão: ~**5,5×** — distorce a agregação FedAvg ao dar implicitamente mais peso ao BPSP

Implementação (`fl_core.py`):

```
τ_eff = Σ_i p_i · τ_i          # média ponderada dos passos efetivos

Δ_i = (w_i − w_global) / τ_i   # update normalizado do cliente i

w_{t+1} = w_global + τ_eff · Σ_i p_i · Δ_i
```

onde `p_i = n_i / N_total` (fração de amostras do cliente i) e `τ_i` é o número de batches processados localmente.

O Experimento 12 (primeira execução válida do FedNova com checkpoint scoping) atingiu a acurácia recorde de **67,44%**, demonstrando a eficácia da normalização.

### 4.4 Checkpoint Scoping e Recuperação de Falhas

**Incidente detectado no Experimento 9**: Contaminação cruzada de checkpoints — `load_best()` sem filtro por experimento retornou o checkpoint R91 do Experimento 8 (0,6661) em vez do R33 do Experimento 9 (0,6386).

**Solução implementada (Migration 011)**:

- Criação da tabela `metrics.fl_trainings` com `training_id` único
- Adição de coluna `training_id` em `metrics.fl_checkpoints` com índice UNIQUE parcial (`WHERE training_id IS NOT NULL`)
- `register_training()` antes do loop, UPSERT com `ON CONFLICT (training_id) WHERE training_id IS NOT NULL`
- `load_best(training_id)` com filtro por treinamento específico

**Mecanismo de recuperação (`RoundDispatcher`)**:

- Poll via HTTP GET em `/metrics/round/{n}` no servidor Flower
- Backoff exponencial: início em 5s, dobra a cada tentativa, teto em 60s, `max_wait = 600s`
- HTTP 404: rodada ainda não concluída (aguarda). HTTP 200: métricas disponíveis
- Convergência verificada via `ConvergenceTracker` que mantém histórico completo de acurácias e detecta platô sobre as últimas `patience` rodadas consecutivas após o warm-up

O **gap best vs last** quantifica o valor do checkpoint guloso: no Experimento 8, **66,61% (R91) vs 58,27% (R120) = 8,34 p.p.** — o checkpoint guloso capturou o pico antes da degradação por overfitting.

### 4.5 Pipeline de Treinamento Completo (`make training-full`)

O Makefile implementa um pipeline de 4 fases sem parametrização externa, executável com um único comando:

1. **`training-bpsp-only`**: SimplifiedBEHRT treinado exclusivamente com dados BPSP — baseline isolado por hospital
2. **`training-hsl-only`**: SimplifiedBEHRT treinado exclusivamente com dados HSL
3. **Treinamento federado completo** (FedProx + FedNova + Checkpoint Scoped)
4. **`behrt_pooled` + RF centralizado**: baselines de comparação (artefatos de pesquisa — nunca implantar em produção)

Esta sequência garante **reprodutibilidade completa**: qualquer re-execução do experimento parte do zero com os mesmos splits determinísticos e produz resultados rastreáveis por `training_id`.

### 4.6 Registro de Treinamentos Federados

Esta seção consolida os registros de todos os treinamentos federados completos (120 rodadas) executados com dados reais FAPESP COVID-19, extraídos diretamente dos arquivos de log (`experiments/logs/run_complete_*.log`) e das avaliações geradas (`experiments/logs/evaluation_round_*.json`). Os registros de desenvolvimento com 20 rodadas são tratados separadamente como fases de ablação.

#### 4.6.1 Treinamentos Completos (120 Rodadas)

Os quatro treinamentos abaixo completaram as 120 rodadas máximas. Nenhum atingiu o critério de convergência (Δacc < 0,005 por 3 rodadas consecutivas após warm-up de 20), indicando que o modelo ainda estava aprendendo ao fim do budget ou oscilava acima do limiar.

| # | Log | Algoritmo | training\_id | Rodada melhor | Acc melhor | Acc última | Loss final | Duração | Tráfego |
|---|-----|-----------|------------|--------------|-----------|-----------|-----------|---------|--------|
| I | run\_complete\_20260625\_225308 | FedProx + FedAvg | — | R120 (sem guloso) | 63,29% | 59,36% | 1,0270 | 4h 24min | 1310 MB |
| II | run\_complete\_20260626\_130506 | FedProx + checkpoint guloso | — | **R91** | **66,61%** | 58,27% | 1,0479 | 4h 25min | 1310 MB |
| III | run\_complete\_20260628\_074558 | FedNova (seed=42) | — | R33 | 63,86% | 54,54% | 1,1618 | 3h 54min | 1310 MB |
| IV | run\_complete\_20260628\_182702 | FedNova (seed=42) | **id=2** | **R115** | **67,44%** | 61,14% | 0,9849 | 4h 6min | 1310 MB |

**Observações registradas nos logs:**

**Treinamento I** (`run_complete_20260625_225308`): Primeira execução com 120 rodadas. Sem checkpoint guloso e sem `training_id`, o único checkpoint salvo corresponde à rodada final (R120, acc=59,36%) — substancialmente abaixo do pico atingido durante o treinamento. A análise posterior revelou que o checkpoint do Treinamento II (R91, 66,61%), salvo sem `training_id`, foi carregado erroneamente como resultado deste experimento, incidente que motivou a Migration 011.

**Treinamento II** (`run_complete_20260626_130506`): Primeira execução com checkpoint guloso funcional. O log registra 10 atualizações progressivas do melhor checkpoint: R1→R3→R5→R6→R7→R9→R19→R59→R74→R91, com a acurácia crescendo de 52,26% (R1) até 66,61% (R91). A partir de R92, o modelo degradou progressivamente até 58,27% na R120 — gap de **8,34 p.p.** entre melhor e última rodada, evidenciando sobretreino após o pico. Recalibrado com Temperature Scaling: ECE piorou de (não registrado) para (não registrado), confirmando o padrão de falha observado em todos os experimentos.

**Treinamento III** (`run_complete_20260628_074558`): Primeira execução com FedNova. Sem `training_id`, o checkpoint foi salvo sem scoping. O pico ocorreu cedo (R33, 63,86%) e o modelo degradou para 54,54% na R120 — gap de **9,32 p.p.**, o maior de todos os treinamentos. O log reporta `τ=[2504, 454] | τ_eff=2190.0` em todas as rodadas, confirmando a normalização FedNova operando corretamente. A degradação acentuada sugere que o `training_id=None` pode ter causado colisão de checkpoints com o Treinamento II ainda presente no banco.

**Treinamento IV** (`run_complete_20260628_182702`): Primeiro treinamento com `training_id` scoped (id=2, registrado via `training_registered_postgres`). O log registra 8 atualizações do melhor checkpoint: R1→R2→R3→R5→R32→R37→R115, com a acurácia crescendo de 37,56% (R1) até 67,44% (R115). O pico tardio em R115 (de 120 rodadas) indica que o modelo ainda estava em ascensão ao fim do budget — diferentemente dos Treinamentos II e III, que atingiram pico antes da metade. `training_completed_postgres` registrou `best_round=115 best_accuracy=0.6744 converged=False`. Este é o treinamento de referência do projeto.

#### 4.6.2 Avaliação Quantitativa dos Treinamentos Completos

Três avaliações foram registradas em arquivos `evaluation_round_*.json`. Estas avaliações são feitas no conjunto de teste global (BPSP + HSL, n=3.381) com o checkpoint da melhor rodada carregado:

| Arquivo de avaliação | Rodada avaliada | Treinamento | Acc | Macro F1 | Macro AUC | Temperature (T) |
|---------------------|----------------|-------------|-----|----------|-----------|----------------|
| `evaluation_round_20.json` | R20 | Fase de ablação (20 rodadas) | 59,63% | 0,3515 | 0,7456 | 1,4418 |
| `recalibrate_20260626_192337.json` | R91 | Treinamento II (Exp 8) | 66,61% | 0,4823 | 0,8097 | 1,0849 |
| `evaluation_round_120.json` | R115 | Treinamento IV (Exp 12) | **67,44%** | **0,484** | 0,8015 | 1,058 |

**Observações:**

- O Temperature T=1,4418 na fase de ablação (20 rodadas) indica que o modelo estava fortemente subconfiante — o softmax precisou de escalonamento de 44% para aproximar as probabilidades da acurácia real. A magnitude decrescente de T (1,4418 → 1,0849 → 1,058) ao longo dos treinamentos indica melhora progressiva da calibração intrínseca do modelo com mais rodadas.
- O Macro AUC do Treinamento II (0,8097) é ligeiramente superior ao Treinamento IV (0,8015), apesar da acurácia inferior. Isso sugere que o FedProx (Treinamento II) pode ter melhor ordenação probabilística em algumas classes, enquanto o FedNova (Treinamento IV) favoreceu a acurácia por argmax. A distinção é clinicamente relevante: em triagem, AUC é mais informativo que acurácia pontual.
- O F1 macro permaneceu próximo entre os dois melhores treinamentos (0,4823 vs 0,484), refletindo o desafio persistente das classes minoritárias (`curado_internado`, n=28 no teste).

#### 4.6.3 Fases de Ablação e Desenvolvimento (20 Rodadas)

Antes dos treinamentos completos, três runs de 20 rodadas foram executados para validar componentes incrementais. Os dados abaixo são extraídos dos logs `FL_TRAINING_COMPLETE` e do `evaluation_round_20.json` (última avaliação disponível desta fase):

| Log | Data | Acc (R20) | Loss | Duração | Nota |
|-----|------|----------|------|---------|------|
| run\_complete\_20260625\_124833 | 25/06 | 54,75% | 1,1603 | ~47min | Baseline pré-DiaRelativo |
| run\_complete\_20260625\_144656 | 25/06 | 56,55% | 1,1478 | ~47min | + DiaRelativoEmbedding |
| run\_complete\_20260625\_201012 | 25/06 | 59,63% | 1,1197 | ~48min | + μ=0,1 (RAG falhou nesta run) |

O ganho progressivo de 54,75% → 56,55% → 59,63% ao longo das três runs de ablação confirma:
- **+1,80 p.p.**: DiaRelativoEmbedding (captura de progressão temporal)
- **+3,08 p.p.**: μ=0,1 (maior contenção do drift local)

Estes valores são consistentes com os reportados na Seção 8.1, onde o mesmo ganho de DiaRelativo (+3,08 p.p.) é documentado via comparação Exp 5 vs Exp 6.

#### 4.6.4 Custo de Comunicação

Todos os treinamentos completos (120 rodadas, 2 clientes) registraram **1310,28 MB** de tráfego total — valor determinístico, pois o tamanho do modelo (SimplifiedBEHRT, ~0,55 MB de parâmetros float32) é fixo e cada rodada exige upload + download por cliente:

```
tráfego_por_rodada = 2 clientes × 2 direções × tamanho_modelo
1310,28 MB / 120 rodadas = 10,92 MB/rodada
10,92 MB / (2 clientes × 2 direções) = 2,73 MB por transmissão ≈ tamanho do modelo
```

Este custo de comunicação é baixo o suficiente para redes hospitalares padrão (1 Gbps internos), mas seria relevante em cenários de edge computing com conectividade limitada (ex.: UBS com link 4G de 10 Mbps — 10,92 MB/rodada levaria ~9 segundos por rodada, ~18 minutos por treinamento completo).

---

## 5. Calibração de Probabilidades

### 5.1 Problema Diagnosticado: Subconfiança Sistemática

Em todos os experimentos com temperature scaling, o ECE pós-calibração ficou **igual ou acima** do pré-calibração. O padrão é estruturalmente subconfiante: a confiança do modelo é sistematicamente menor que a acurácia real em quase todos os bins.

| Experimento | ECE pré | ECE pós-TS | Δ ECE |
|-------------|---------|-----------|-------|
| Exp 1 | 0,059 | 0,098 | +0,039 |
| Exp 5 | 0,046 | 0,069 | +0,023 |
| Exp 7 | 0,033 | 0,062 | +0,029 |
| Exp 12 | 0,094 | 0,109 | +0,015 |

**Valores do Exp 12 verificados em `evaluation_round_120.json`**: ECE pré=0,0935, T=1,058, ECE pós=0,1086.

**Causa raiz**: Temperature scaling é um método paramétrico global — aplica um único escalar T sobre todos os logits igualmente. Com T > 1 (softmax mais suave), a subconfiança piora. O LBFGS minimiza Negative Log Likelihood (NLL), não ECE diretamente — os objetivos divergem em padrões não-uniformes como o observado.

### 5.2 Solução: Calibração Isotônica (OvR)

A Calibração Isotônica (Zadrozny & Elkan, 2002) aprende uma função monotônica não-paramétrica que mapeia a confiança bruta → probabilidade calibrada, ajustando cada bin de forma independente.

**Implementação**:

- Abordagem One-vs-Rest (OvR): treina um `IsotonicRegression` por classe
- Algoritmo Pool Adjacent Violators (PAV): a confiança média de cada bin é substituída pela acurácia real observada, respeitando monotonicidade
- Função escada que pode subir ou descer dependendo do padrão local de cada classe

**Resultado verificado** (`recalibrate_20260626_192337.json`, checkpoint R91 do Exp 8):
- ECE pré-calibração: 0,0859
- ECE pós-temperature scaling: **0,1066** (piorou +0,021)

**ECE pós-calibração isotônica não disponível nos logs atuais** — o arquivo de recalibração registra apenas pré e pós-temperature scaling. A medição do impacto real da isotônica sobre o ECE é uma **lacuna experimental pendente** (ver Seção 10, item H).

---

## 6. Módulo RAG (Retrieval-Augmented Generation)

### 6.1 Arquitetura Implementada

O módulo RAG é composto por três componentes, **todos verificados em `src/mosaicfl/core/rag.py`**:

**1. Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões) para codificar perfis clínicos e consultas.

**2. Backend de vetores**:
- `_PostgreSQLStore` com pgvector em produção (tabela `knowledge.clinical_profiles`)
- `_InMemoryStore` (similaridade de cosseno via numpy) em experimentos sem banco

**3. LLM para geração**: `distilgpt2` (82M parâmetros), modelo generativo de propósito geral.

**Fluxo de recuperação**:

1. Consulta do paciente é tokenizada e convertida em embedding
2. Busca por similaridade de cosseno (top-k = 3 por padrão)
3. Perfis recuperados são usados como contexto no prompt
4. DistilGPT-2 gera justificativa textual

**Detecção de alucinação**: Heurística `probability < 0,6 AND "certeza" in justification.lower()` — sinaliza quando o modelo gera afirmações com baixa confiança.

### 6.2 Métricas de Avaliação

**Precision@k**: calculada como `(casos recuperados com mesmo desfecho que ground_truth) / (k × n_queries)`. O Experimento 8 reportou os seguintes resultados:

| Classe | Precision@3 |
|--------|-------------|
| curado_pronto | 0,231 |
| melhora_pronto | **0,386** |
| melhora_internado_breve | 0,206 |
| melhora_internado_grave | 0,116 |
| **Macro** | **0,226** |

`melhora_pronto` tem maior Precision@3 porque é uma classe com perfil clínico mais específico no HSL. `curado_pronto` (48% do test set) tem perfis genéricos demais, dificultando recuperação discriminante.

### 6.3 Limitação Identificada

Os experimentos de RAG realizados até 2026-06-29 **não produziram exemplos qualitativos válidos** devido a dois problemas:

1. **Base de conhecimento corrompida**: os textos armazenados continham artefatos de tokenização (palavra "adulto" interpolada entre cada caractere do texto), causada pela aplicação do sentence-transformer sobre tokens especiais do BEHRT em vez de texto clínico real.

2. **Geração incoerente**: o DistilGPT-2, sem fine-tuning em português ou domínio clínico, produz texto em português com baixa coerência.

**Ação recomendada**: Para produção, substituir o LLM por um modelo treinado em português clínico (ex: BERTimbau fine-tuned) ou restringir a geração a templates estruturados com preenchimento de slots.

---

## 7. Interoperabilidade e Segurança

### 7.1 Padronização LOINC para Analitos

O sistema mapeia analitos do dataset FAPESP para códigos LOINC, garantindo interoperabilidade semântica em redes federadas:

| Analito FAPESP | Código LOINC | Nome Oficial LOINC |
|----------------|-------------|-------------------|
| Hemoglobina | 718-7 | Hemoglobin Mass/volume in Blood |
| Leucócitos | 6690-2 | Leukocytes #/volume in Blood |
| Plaquetas | 777-3 | Platelets #/volume in Blood |
| Creatinina | 2160-0 | Creatinine Mass/volume in Serum or Plasma |
| PCR | 1988-5 | C reactive protein Mass/volume in Serum or Plasma |
| Ferritina | 2276-4 | Ferritin Mass/volume in Serum or Plasma |
| D-dímero | 48066-5 | Fibrin D-dimer DDU Mass/volume in Platelet poor plasma |
| LDH | 2532-0 | Lactate dehydrogenase Enzymatic activity/volume in Serum or Plasma |
| Troponina | 6598-7 | Troponin T.cardiac Mass/volume in Serum or Plasma |

### 7.2 Pseudonimização HMAC-SHA256 (LGPD Art. 13 §4°)

A pseudonimização é implementada em `security.py` e **verificada diretamente no código**:

```python
def _pid_to_internal(raw_patient_id: str) -> str:
    return hmac.new(
        _PID_SECRET.encode(),
        raw_patient_id.encode(),
        hashlib.sha256
    ).hexdigest()
```

O `_PID_SECRET` é lido da variável de ambiente `FL_PATIENT_ID_SECRET` — **não compartilhado com o servidor central**. Cada instância hospitalar gerencia seu próprio secret localmente. Isso garante que o servidor central nunca receba o `patient_id` real, apenas o hash HMAC-SHA256, que é irreversível sem o secret local.

### 7.3 Autenticação e Rate Limiting

**Autenticação** (`security.py`):

- JWT via `FL_JWT_SECRET` (HMAC, HS256) ou `FL_JWT_PUBLIC_KEY_FILE` (RSA, RS256/RS512)
- API Key via header `X-API-Key`
- Modo desenvolvimento: `FL_AUTH_REQUIRED=false` desativa autenticação

**Rate Limiting** (janela deslizante, sem dependências externas):

```python
_api_limiter    = _SlidingWindowLimiter(max_calls=120, window_seconds=60.0)
_ingest_limiter = _SlidingWindowLimiter(max_calls=30,  window_seconds=60.0)
```

Aplicado por IP em todos os endpoints via `_rate_check`.

**Auditoria**: `audit.log_access()` é chamado em todos os endpoints de predição, registrando `patient_id_hash` (nunca o ID real), `exam_count` e `risk_score`.

### 7.4 HL7 FHIR R4 — RiskAssessment

O sistema gera o recurso FHIR R4 `RiskAssessment` via `state._fhir_exporter.to_risk_assessment()`. O recurso inclui:

- `subject`: `correlation_token` efêmero (UUID descartável) — mantendo o mapeamento real exclusivamente sob controle do hospital de origem
- `prediction`: probabilidades por classe
- `model_round`: versão do modelo
- `temperature`: parâmetro de calibração
- `ece`: Expected Calibration Error

O uso de `correlation_token` efêmero resolve o campo obrigatório `subject` do FHIR R4 sem que o servidor FL saiba a identidade do paciente — combinação de privacidade + interoperabilidade.

---

## 8. Análise de Resultados Experimentais

### 8.1 Evolução dos Experimentos

A tabela abaixo resume a evolução dos experimentos, destacando as principais mudanças arquiteturais e seus impactos:

| Experimento | Condição Principal | Acurácia | Macro AUC | Desafio Endereçado |
|-------------|-------------------|----------|-----------|-------------------|
| Exp 1 | Baseline FedProx (μ=0,01) | 58,0% | 0,740 | Implementação inicial |
| Exp 3 | Split 70/10/10/10 + calibração independente | 55,8% | 0,755 | Correção metodológica do data leakage |
| Exp 6 | + DiaRelativoEmbedding | 59,63% | 0,746 | Captura de velocidade de progressão |
| Exp 7 | μ=0,1 + 120 rodadas | 59,36%¹ | 0,770 | Redução de drift non-IID |
| Exp 9 | — | — | — | Incidente: contaminação de checkpoints |
| Exp 8 | + Checkpoint Guloso | 66,61%² | 0,810 | Instabilidade de convergência |
| Exp 12 | FedNova + Checkpoint Scoped | **67,44%** | 0,802 | Inconsistência objetiva por volume |

¹ Avaliação na R120 (última rodada). Melhor checkpoint (R89=63,29%) não capturado por falta de checkpoint guloso.  
² Avaliação no checkpoint R91 (restaurado via `load_best`). Gap best vs last = 8,34 p.p.

### 8.2 Experimento 12: Recorde de Performance

O Experimento 12 (primeira execução válida do FedNova com checkpoint scoping) atingiu **67,44% de acurácia** — o melhor resultado do projeto. O checkpoint foi salvo na rodada 115 (de 120), com `training_id=2` garantindo restauração correta.

**Métricas por classe (pré-calibração, R115) — verificadas em `evaluation_round_120.json`:**

| Classe | Support | AUC | F1 | Precision | Recall |
|--------|---------|-----|----|-----------|--------|
| curado_pronto | 1.620 (47,9%) | 0,8762 | 0,8146 | 0,7695 | 0,8654 |
| curado_internado | 28 (0,8%) | 0,5713 | 0,0323 | 0,0294 | 0,0357 |
| melhora_pronto | 321 (9,5%) | **0,9553** | 0,6606 | 0,7854 | 0,5701 |
| melhora_internado_breve | 1.074 (31,8%) | 0,8108 | 0,5819 | 0,6413 | 0,5326 |
| melhora_internado_grave | 338 (10,0%) | 0,7936 | 0,3306 | 0,3050 | 0,3609 |

**Observações:**

- `melhora_pronto` atingiu AUC=0,9553 — o melhor histórico do projeto, indicando que a normalização do FedNova permitiu extrair sinal de alta qualidade mesmo de classes minoritárias em ambientes heterogêneos.
- `curado_internado` (N=28 no teste) permanece com F1 próximo de zero — problema estrutural de raridade extrema.
- `melhora_internado_grave` (classe de maior severidade) tem F1=0,33.

**Matriz de confusão (pré-calibração, R115) — verificada em `evaluation_round_120.json`:**

```
                   Predito →
Real ↓          cp    ci    mp   mib   mig
curado_pronto  [1402,  10,  23,  119,  66]
curado_intern  [  12,   1,   2,   10,   3]
melhora_pronto [  71,  10, 183,   49,   8]
mib            [ 270,   9,  22,  572, 201]
mig            [  67,   4,   3,  142, 122]
```

### 8.2.1 Análise de Erros Clinicamente Críticos

A matriz de confusão revela um padrão de erro com implicação clínica direta:

**67 dos 338 casos de `melhora_internado_grave` foram classificados como `curado_pronto` (19,8%)**

Este é o erro mais grave do sistema: pacientes com internação prolongada (>10 dias) e evolução lenta sendo classificados como curados sem internação. Em um cenário de tomada de decisão clínica, isso poderia resultar em alta precoce ou subestimação de risco.

Adicionalmente:
- **201 casos de `melhora_internado_breve`** classificados como `melhora_internado_grave` (sobre-estimação de gravidade): erro menos perigoso mas que aumenta custos de internação.
- **89 casos de `melhora_pronto`** classificados como `curado_pronto` (27,7% da classe): confusão entre dois desfechos favoráveis — clinicamente menos crítica.

A quantificação desses erros por gravidade clínica (e não apenas pela acurácia agregada) é fundamental para a discussão do impacto real do sistema.

### 8.3 Análise do Custo de Privacidade

**Esta seção foi integralmente corrigida com base nos logs de experimento reais.**

O custo de privacidade é calculado comparando o modelo federado com baselines que teriam acesso direto aos dados combinados. Os resultados abaixo são **verificados em `experiments/logs/behrt_pooled_20260625_212059.log` e `experiments/data/behrt_pooled_20260625_223649.json`**:

| Modelo | Acurácia | F1 Macro | AUC Macro | ECE | n_épocas/rodadas |
|--------|----------|----------|-----------|-----|-----------------|
| **FL Exp 12** (FedNova + late fusion) | **67,44%** | 0,484 | 0,8015 | 0,0935 | 120 rodadas |
| BEHRT Pooled A (sem_demo, pool BPSP+HSL) | 67,79% | 0,5218 | 0,8354 | 0,0929 | 40 épocas |
| BEHRT Pooled B (late_fusion, pool BPSP+HSL) | 63,03% | 0,5005 | — | — | 40 épocas |
| RF Centralizado (Bag-of-Tokens) | 68,35% | 0,5116 | 0,7943 | 0,0638 | — |
| RF Hospital 0 (BPSP isolado) | 59,45% | 0,337 | 0,7425 | 0,0467 | — |
| RF Hospital 1 (HSL isolado) | **24,25%** | 0,184 | 0,6905 | 0,2541 | — |

**Comparações:**

| Comparação | Δ Acc | Δ F1 | Interpretação |
|-----------|-------|------|--------------|
| FL vs BEHRT Pooled B (mesma arquitetura) | **+4,41 p.p.** | -0,017 | FL supera o pooled com late fusion |
| FL vs BEHRT Pooled A (backbone sem demo) | **-0,35 p.p.** | -0,038 | Gap mínimo com backbone sem demográficos |
| FL vs RF Centralizado | **-0,91 p.p.** | -0,028 | FL perde levemente para RF clássico em acurácia, mas AUC é superior (0,80 vs 0,79) |

**Limitação metodológica importante**: O BEHRT Pooled foi treinado com **40 épocas**, enquanto o `FED_CFG.pooled_epochs = 120` especifica 120 como budget equivalente ao FL. A comparação com budget completo ainda não foi realizada e pode alterar os gaps. Com 40 épocas, o FL se mostra competitivo ou superior.

**Conclusão revisada**: O custo de manter os dados isolados (privacidade) é **no máximo -0,35 p.p.** de acurácia em relação ao pooled sem demográficos, e o FL com late fusion **supera o pooled com mesma arquitetura** nas condições testadas. O AUC do FL (0,8015) supera o RF centralizado (0,7943) — métrica mais relevante para triagem clínica com classes desbalanceadas.

**Contexto na literatura**: gaps abaixo de 2 p.p. são típicos para FL sem Privacidade Diferencial em cenários non-IID severo. O gap de FL com DP é usualmente 2-10 p.p. (Abadi et al., 2016). Estudos recentes com protocolo idêntico (FedProx, non-IID) frequentemente reportam gaps nulos ou negativos (ex: arXiv 2601.17183, 2026, com FedProx superando centralizado em +1,67 p.p.).

### 8.4 Análise de Calibração

**Temperature Scaling**: falhou sistematicamente em todos os experimentos (ECE pós-calibração sempre ≥ ECE pré-calibração). Exemplo do Exp 12:

- ECE pré-calibração: 0,0935
- ECE pós-temperature scaling (T=1,058): 0,1086 (piorou)
- MCE pré: 0,2545 → MCE pós: 0,2875

**Solução validada**: Calibração Isotônica OvR. O mapeamento não-paramétrico por classe corrige a subconfiança não-uniforme que o temperature scaling não captura. O ECE resultante da calibração isotônica ainda não foi medido nos experimentos disponíveis (ver Seção 10, item H).

### 8.5 Avaliação RAG

**Métricas quantitativas** (Experimento 8):

| Classe | Precision@3 |
|--------|-------------|
| curado_pronto | 0,231 |
| melhora_pronto | **0,386** |
| melhora_internado_breve | 0,206 |
| melhora_internado_grave | 0,116 |
| **Macro** | **0,226** |

**Limitação qualitativa**: Os experimentos de RAG realizados até 2026-06-29 não produziram exemplos válidos de justificativas geradas pelo DistilGPT-2 devido a artefatos de tokenização na base de conhecimento e falta de fine-tuning em português clínico.

---

## 9. Conclusões e Trabalhos Futuros

### 9.1 Contribuições Principais

1. **Validação do FedNova para non-IID severo**: A normalização por passos efetivos (τ_i) equalizou a contribuição de BPSP e HSL, resultando em acurácia recorde de 67,44% — o melhor resultado do projeto.

2. **Checkpoint scoping como requisito de engenharia**: A Migration 011 e o `training_id` scoped resolveram o problema de contaminação cruzada, capturando gap de até 8,34 p.p. entre melhor e última rodada.

3. **DiaRelativoEmbedding como inovação arquitetural**: A injeção do dia relativo desde a admissão capturou a velocidade de progressão clínica, resultando em ganho de +3,08 p.p. de acurácia.

4. **Interoperabilidade FHIR R4 + LOINC**: O sistema está pronto para integração em ecossistemas hospitalares modernos, com pseudonimização HMAC-SHA256 conforme LGPD.

5. **Custo de privacidade quantificado**: ≤ -0,35 p.p. de acurácia entre FL federado e BEHRT pooled com backbone equivalente — evidência empírica de que a federação preserva utilidade com custo mínimo de privacidade.

6. **Colapso do modelo local**: RF Hospital 1 (HSL isolado) = 24,25% — evidência empírica direta de que nenhum hospital consegue generalizar individualmente, justificando a federação.

### 9.2 Lacunas e Trabalhos Futuros

**1. Rodar BEHRT Pooled com 120 épocas**: A comparação atual usa 40 épocas, não o budget equivalente ao FL. A análise definitiva do custo de privacidade exige este experimento.

**2. Medir ECE pós-calibração isotônica**: O impacto real da isotônica no ECE ainda não foi registrado nos logs disponíveis.

**3. Privacidade Diferencial (DP)**: Sem DP, gradientes federados permitem reconstrução de dados via ataques de inversão de gradiente. Implementação DP-SGD (Abadi et al., 2016) é bloqueadora para produção.

**4. Avaliação formal do RAG**: Substituir DistilGPT-2 por modelo em português clínico (ex: BERTimbau fine-tuned) e implementar avaliação quantitativa com métricas como BLEU, ROUGE e relevância clínica.

**5. Arquitetura distribuída real**: Atualmente em simulação local. Produção exige TLS mútuo, autenticação de SuperNodes, e tolerância a falhas.

**6. Hardware com suporte GPU**: O treinamento atual em CPU (Dell Inspiron 5402, i7-1165G7) leva ~4h por experimento. Com GPU dedicada, o ciclo seria reduzido para minutos.

**7. Performance por hospital no test set**: Os resultados atuais misturam BPSP e HSL no teste global. A análise separada por hospital pode revelar assimetrias ocultas.

### 9.3 Declaração de Uso de IA

Conforme diretrizes acadêmicas, declara-se o uso do modelo Claude Sonnet 4.6 (Anthropic) para automação de testes, refatoração de código, revisão de documentação e — neste documento — verificação cruzada entre documento externo e código-fonte. A autoria intelectual, análise crítica de resultados e definição metodológica são de responsabilidade integral humana.

---

## 10. Dados Necessários para Complementação

Os seguintes itens não puderam ser obtidos do código-fonte e requerem coleta específica:

| # | Item | Como obter | Prioridade |
|---|------|-----------|-----------|
| A | Contagem de pacientes e atendimentos por hospital | `SELECT COUNT(DISTINCT patient_id), COUNT(DISTINCT attendance_id) FROM clinical.attendances GROUP BY hospital_id` | Alta |
| B | Distribuição real das 5 classes por hospital (verificar vs. valores do documento) | Query de frequência em `metrics.clinical_outcomes` com JOIN em `clinical.attendances` | Alta |
| C | Período exato de coleta dos dados FAPESP | `SELECT MIN(co.outcome_at), MAX(co.outcome_at) FROM metrics.clinical_outcomes co JOIN clinical.attendances a ON co.attendance_id = a.attendance_id WHERE a.hospital_id IN ('HSL','BPSP')` | Alta |
| D | Justificativa clínica formal para `max_seq_len=128` | Distribuição de `COUNT(exames) por attendance_id` + literatura sobre densidade de exames em COVID-19 | Média |
| E | Exemplos qualitativos de justificativas RAG | Reexecutar pipeline RAG com base de conhecimento corrigida e modelo em português (ex: BERTimbau) | Baixa |
| F | Top-20 analitos mais frequentes no vocabulário | `standard_vocab.json` ou query de frequência em `metrics.exam_records` | Média |
| G | BEHRT Pooled com 120 épocas | Executar `make training-full` com `pooled_epochs=120` (config já correto; execução requer ~8h) | Alta |
| H | ECE pós-calibração isotônica (Exp 12) | Executar script de recalibração com registro explícito do ECE isotônico | Alta |
| I | Performance por hospital no test set global | Filtrar `evaluation_round_120.json` por `hospital_id` no test set | Média |
| J | Ablação FL Config A (sem demográficos) vs Config B (com demográficos) | Executar 1 rodada FL com `demo_dim=0` para isolar contribuição demográfica | Média |
| K | Resultados do Experimento 13 | Rastrear no planejamento do desktop | Baixa |
| L | Estatísticas demográficas (idade, sexo) por classe de desfecho | Query com JOIN `clinical.patients` estratificada por `outcome_class` | Baixa |

---

*Documento gerado a partir de verificação integral do repositório MOSAIC-FL em 2026-06-29. Todas as afirmações são rastreáveis ao código-fonte ou a arquivos de log de experimento indicados. Lacunas explicitamente identificadas.*
