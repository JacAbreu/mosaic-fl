# MOSAIC-FL — Diagramas de Arquitetura (C4 + UML)

> Este documento complementa `docs/FLUXO_APRENDIZADO_FEDERADO.md` (que descreve o *fluxo de dados* ponta a ponta) com uma visão *arquitetural* do sistema, em dois vocabulários de diagrama diferentes, para leitores com backgrounds distintos:
>
> - **Modelo C4** (Simon Brown) — do nível mais amplo (Contexto) ao mais específico (Código), em 4 níveis. É o vocabulário preferido pela autora.
> - **UML** (sequência e caso de uso) — para leitores da banca ou da literatura que não conhecem C4 e preferem a notação UML clássica.
>
> **Convenção de renderização:** todos os diagramas C4 e de sequência estão em **Mermaid** (mesma ferramenta usada em `FLUXO_APRENDIZADO_FEDERADO.md` — renderiza nativamente no GitHub e na maioria dos visualizadores Markdown). O diagrama de caso de uso é a única exceção: Mermaid **não tem** tipo de diagrama de caso de uso nativo (só flowchart, sequência, classe, estado, ER, C4, mindmap, timeline, gantt, sankey). Para ter a notação UML genuína (atores-boneco + elipses de caso de uso, `<<include>>`/`<<extend>>`), esse diagrama está em **PlantUML**, que exige um renderizador à parte (ver nota na seção 6).

---

## Índice

1. [Os três cenários representados](#1-os-três-cenários-representados)
2. [Nível 1 — Contexto do Sistema (C4)](#2-nível-1--contexto-do-sistema-c4)
3. [Nível 2 — Contêineres (C4)](#3-nível-2--contêineres-c4)
4. [Nível 3 — Componentes (C4), um por cenário](#4-nível-3--componentes-c4-um-por-cenário)
5. [Nível 4 — Código (C4), classes-chave por cenário](#5-nível-4--código-c4-classes-chave-por-cenário)
6. [Diagramas de Sequência (UML)](#6-diagramas-de-sequência-uml)
7. [Diagrama de Caso de Uso (UML)](#7-diagrama-de-caso-de-uso-uml)

---

## 1. Os três cenários representados

O projeto tem três fluxos operacionalmente distintos, que usam containers e componentes parcialmente diferentes. Os níveis 1 e 2 do C4 são únicos (mostram o sistema inteiro); a partir do nível 3, cada cenário ganha seu próprio diagrama:

| Cenário | O que representa | Containers principais envolvidos |
|---|---|---|
| **A — Comunicação Federada** | Troca de pesos entre SuperLink, ServerApp e os SuperNodes dos hospitais, round a round | SuperLink, ServerApp, SuperNode BPSP, SuperNode HSL |
| **B — Treinamento** | Orquestração das 4 fases do pipeline de pesquisa (`make training-full`), agregação FedNova, calibração, checkpoint | Ambiente de Pesquisa (`experiments/`), PostgreSQL |
| **C — Disponibilização via API + RAG** | Requisição de um hospital à API de inferência, predição com MC Dropout, geração da justificativa clínica via RAG | API de Inferência, PostgreSQL, Ollama |

---

## 2. Nível 1 — Contexto do Sistema (C4)

Visão mais ampla: o MOSAIC-FL como caixa única, e quem/o que interage com ele. Um único diagrama cobre os três cenários — os atores aparecem todos juntos porque, do ponto de vista de contexto, todos "conversam com o sistema", independente do cenário interno.

```mermaid
flowchart TB
    PESQ["👤 Pesquisadora / Administradora\n[Pessoa]\nOpera o treinamento, monitora\nconvergência, ajusta configuração"]

    HOSP_BPSP["🏥 Hospital BPSP\n[Organização externa]\nFornece dados clínicos locais\ne executa nó federado (SuperNode)"]
    HOSP_HSL["🏥 Hospital HSL\n[Organização externa]\nFornece dados clínicos locais\ne executa nó federado (SuperNode)"]

    SIST_EXT["🖥️ Sistema Hospitalar Cliente\n[Sistema externo]\nConsulta predições de risco\nvia API REST"]

    CP["📊 ClinicalPath\n[Sistema externo]\nVisualização temporal do\nquadro clínico (Linhares et al. 2023)"]
    RNDS["🌐 Sistema receptor FHIR / RNDS\n[Sistema externo]\nConsome RiskAssessment R4"]

    MOSAIC["🔷 MOSAIC-FL\n[Sistema de Software]\nAprendizado federado clínico:\nFedNova + BEHRT + RAG\npara predição de desfecho"]

    OLLAMA["🤖 Ollama (gemma3:4b)\n[Sistema externo]\nServiço local de LLM para\ngeração de justificativa RAG"]

    PESQ -->|"opera, monitora,\nconfigura em runtime"| MOSAIC
    HOSP_BPSP <-->|"pesos do modelo\n(nunca dados brutos)"| MOSAIC
    HOSP_HSL <-->|"pesos do modelo\n(nunca dados brutos)"| MOSAIC
    SIST_EXT -->|"POST /api/predict\nPOST /api/exams/ingest"| MOSAIC
    MOSAIC -->|"exporta arquivos\nPatientExport"| CP
    MOSAIC -->|"exporta\nRiskAssessment"| RNDS
    MOSAIC <-->|"HTTP localhost:11434\ngeração de texto"| OLLAMA

    style MOSAIC fill:#1168bd,stroke:#0b4884,color:#fff
    style PESQ fill:#08427b,stroke:#052e56,color:#fff
    style HOSP_BPSP fill:#08427b,stroke:#052e56,color:#fff
    style HOSP_HSL fill:#08427b,stroke:#052e56,color:#fff
    style SIST_EXT fill:#08427b,stroke:#052e56,color:#fff
    style CP fill:#999,stroke:#666,color:#fff
    style RNDS fill:#999,stroke:#666,color:#fff
    style OLLAMA fill:#999,stroke:#666,color:#fff
```

**Nota de privacidade estrutural:** a única seta bidirecional entre hospitais e o sistema carrega **pesos do modelo**, nunca dados clínicos brutos — essa é a garantia central que o resto dos diagramas (a partir do nível 2) precisa preservar visualmente.

---

## 3. Nível 2 — Contêineres (C4)

Cada retângulo abaixo é um processo/serviço deployável de forma independente. Um único diagrama cobre os três cenários; as legendas de aresta indicam a que cenário (A/B/C) cada interação pertence.

```mermaid
flowchart TB
    subgraph EXTERNO["Fora do MOSAIC-FL"]
        PESQ["👤 Pesquisadora"]
        SIST_EXT["🖥️ Sistema Hospitalar Cliente"]
        OLLAMA["🤖 Ollama\n[Serviço externo]\ngemma3:4b"]
    end

    subgraph MOSAIC["MOSAIC-FL"]
        SUPERLINK["SuperLink\n[flower-superlink]\nRoteamento gRPC + persistência\nde estado da federação"]

        SERVERAPP["ServerApp\n[infrastructure/mosaicfl_server]\nProductionFedProxStrategy:\nagregação, checkpoint, watchdog"]

        SUPERNODE_A["SuperNode BPSP\n[infrastructure/mosaicfl_client]\nFedProxClient + DataSourceFactory\n(roda fisicamente no BPSP)"]
        SUPERNODE_B["SuperNode HSL\n[infrastructure/mosaicfl_client]\nFedProxClient + DataSourceFactory\n(roda fisicamente no HSL)"]

        SCHED["Scheduler\n[infrastructure/mosaicfl_scheduler]\nDispara rounds periodicamente,\nverifica quórum de clientes"]

        API["API de Inferência\n[infrastructure/mosaicfl_api]\nFastAPI — /predict, /ingest,\n/patients, /fl/status"]

        PESQUISA["Ambiente de Pesquisa\n[experiments/]\nPipeline de 4 fases:\nBPSP-only → HSL-only →\nFederado → Pooled baseline"]

        DB[("PostgreSQL\nschemas: clinical, metrics,\nknowledge (+ pgvector)")]
    end

    PESQ -->|"A: inicia federação"| SUPERLINK
    PESQ -->|"B: make training-full[-cuda]"| PESQUISA
    PESQ -->|"C: consulta /fl/status"| API

    SUPERLINK <-->|"A: Fleet API :9091"| SUPERNODE_A
    SUPERLINK <-->|"A: Fleet API :9091"| SUPERNODE_B
    SUPERLINK <-->|"A: AppIo API :9092"| SERVERAPP
    SCHED -->|"A: dispara round\n(verifica quórum)"| SERVERAPP

    SERVERAPP -->|"A: save/load checkpoint"| DB
    SUPERNODE_A -->|"A: lê dados locais\n(FL_DATA_SOURCE=sgbd)"| DB
    SUPERNODE_B -->|"A: lê dados locais\n(FL_DATA_SOURCE=sgbd)"| DB

    PESQUISA -->|"B: save checkpoint,\nround_history, métricas"| DB
    PESQUISA -->|"B: simula ambos os\nclientes localmente (Ray)"| PESQUISA

    SIST_EXT -->|"C: POST /api/predict"| API
    API -->|"C: load_best(training_id)"| DB
    API -->|"C: gera justificativa"| OLLAMA

    style MOSAIC fill:none,stroke:#1168bd,stroke-width:2px
    style SUPERLINK fill:#438dd5,stroke:#2e6295,color:#fff
    style SERVERAPP fill:#438dd5,stroke:#2e6295,color:#fff
    style SUPERNODE_A fill:#438dd5,stroke:#2e6295,color:#fff
    style SUPERNODE_B fill:#438dd5,stroke:#2e6295,color:#fff
    style SCHED fill:#438dd5,stroke:#2e6295,color:#fff
    style API fill:#438dd5,stroke:#2e6295,color:#fff
    style PESQUISA fill:#438dd5,stroke:#2e6295,color:#fff
    style DB fill:#438dd5,stroke:#2e6295,color:#fff
```

**Nota sobre o Ambiente de Pesquisa:** ao contrário dos demais containers (que são serviços de produção, deployáveis independentemente — inclusive via Docker/Helm, ver README), `experiments/` roda localmente e **simula** os dois clientes num único processo (via Ray), em vez de comunicação real entre máquinas. É o modo usado para gerar os resultados reportados no TCC. O modo "Rede Federada Real" (desktop + notebook, documentado no README) usa os containers de produção (SuperLink/ServerApp/SuperNode) de fato distribuídos.

---

## 4. Nível 3 — Componentes (C4), um por cenário

### 4.A — Comunicação Federada

Zoom nos containers `ServerApp` e `SuperNode`, mostrando os módulos internos responsáveis pela troca de pesos round a round.

```mermaid
flowchart TB
    subgraph SERVERAPP["Container: ServerApp"]
        SUPERLINK_ENTRY["runner/superlink.py\napp = ServerApp(...)\nentrypoint `flwr run`"]
        STRAT_CORE["strategy/core.py\nProductionFedProxStrategy\n__init__, aggregate_fit,\naggregate_evaluate"]
        STRAT_FIT["strategy/fit_config_mixin.py\nconfigure_fit,\n_load_global_weights"]
        STRAT_WATCH["strategy/watchdog_mixin.py\n_restore_from_state,\nwatchdog de timeout por round"]
        STRAT_CAL["strategy/calibration_mixin.py\n_run_calibration\n(isotônica pós-convergência)"]
        STATE_STORE["state_store.py\nTrainingState +\nTrainingStateStore"]
        CFG_LOADER["config_loader.py\nChromaDBConfigLoader /\nFileConfigLoader\n(config em runtime)"]
    end

    subgraph SUPERNODE["Container: SuperNode (por hospital)"]
        SUPERNODE_ENTRY["runner/supernode.py\napp = ClientApp(...)\n_client_fn"]
        LEGACY_CLIENT["runner/legacy_client.py\nProductionClient\n(reconexão com backoff)"]
        FEDPROX_CLIENT["mosaicfl.core.client\nFedProxClient\nfit(), evaluate(),\ntermo proximal, DP clip"]
        DS_FACTORY["datasource/factory.py\nDataSourceFactory\nsimulated | sgbd | csv"]
        HEARTBEAT["heartbeat.py\nRegistry JSON de status"]
    end

    SUPERLINK_ENTRY --> STRAT_CORE
    STRAT_CORE --> STRAT_FIT
    STRAT_CORE --> STRAT_WATCH
    STRAT_CORE --> STRAT_CAL
    STRAT_WATCH --> STATE_STORE
    STRAT_FIT -->|"lê config do round"| CFG_LOADER

    SUPERNODE_ENTRY --> LEGACY_CLIENT
    LEGACY_CLIENT --> FEDPROX_CLIENT
    FEDPROX_CLIENT --> DS_FACTORY
    LEGACY_CLIENT --> HEARTBEAT

    STRAT_FIT -.->|"envia pesos globais\n+ config (μ, round)"| FEDPROX_CLIENT
    FEDPROX_CLIENT -.->|"retorna pesos locais\n+ τ (passos efetivos)"| STRAT_CORE
```

### 4.B — Treinamento (pipeline de pesquisa)

Zoom no container `Ambiente de Pesquisa`, mostrando a orquestração das 4 fases e a mecânica de agregação.

```mermaid
flowchart TB
    subgraph PESQUISA["Container: Ambiente de Pesquisa (experiments/)"]
        RUNNER["training_runner/run_training.py\nEntrypoint por fase\n(chamado pelo Makefile)"]
        ORCH["training/core/orchestrator.py\nFederatedTraining\ncarrega dados, dispara FL,\nRAG, baseline, ablation"]

        subgraph FLCORE["training/core/fl_core/"]
            ROUTER["router.py\nrun_federated_learning\n(dispatch manual vs Ray)"]
            MANUAL["manual_loop.py\nrun_federated_learning_manual\n(loop federado principal)"]
            RAYLOOP["ray_loop.py\nrun_federated_learning_ray\n(simulação paralela)"]
            AGG["aggregation.py\naggregate_fedavg,\naggregate_fednova,\napply_dp_noise"]
            EVAL["evaluation.py\nevaluate_global_model\n(accuracy, f1_macro,\nper_class_f1)"]
        end

        DATALOADERS["training/core/dataloaders.py\nprepare_dataloaders_from_db\nsplit 70/10/10/10,\nseed independente por hospital"]
        ABLATION["training/core/ablation.py\nrun_ablation_demographics\n(late fusion, multi-seed)"]
        BASELINES["training/core/baselines.py\nRF Bag-of-Tokens,\nBEHRT Pooled"]

        CKPT["infrastructure/shared/\ncheckpoint_store/\nSQLite | PostgreSQL"]
        METRICS["infrastructure/shared/\nmetrics_store/\nSQLite | PostgreSQL"]
    end

    RUNNER --> ORCH
    ORCH --> DATALOADERS
    ORCH --> ROUTER
    ROUTER --> MANUAL
    ROUTER --> RAYLOOP
    MANUAL --> AGG
    MANUAL --> EVAL
    MANUAL -->|"checkpoint guloso\nf1_macro > best_f1"| CKPT
    MANUAL -->|"round_duration_s,\ntau_eff, f1_macro"| METRICS
    ORCH --> ABLATION
    ORCH --> BASELINES
    ORCH -->|"RAG (fase federada,\npós-convergência)"| ORCH
```

### 4.C — Disponibilização via API + RAG

Zoom no container `API de Inferência`, mostrando o caminho de uma predição até a justificativa clínica.

```mermaid
flowchart TB
    subgraph API["Container: API de Inferência"]
        ROUTER_PRED["routers/prediction.py\nPOST /api/predict\nPOST /api/exams/ingest"]
        ROUTER_PAT["routers/patients.py\nGET /api/patients/{id}\nPOST outcome"]
        ROUTER_ADM["routers/admin.py\nGET /fl/status\nPOST /fl/reload"]

        STATE["state.py\nSingleton engine +\nfallback ao CheckpointStore"]

        subgraph IE["inference_engine/"]
            ENGINE["engine.py\nInferenceEngine\nMC Dropout (N amostras)"]
            TOKEN["tokenization.py\nrecords_to_tokens\n(idêntico ao treino)"]
        end

        subgraph RAGPKG["mosaicfl.core.rag"]
            RAG_CORE["__init__.py\nClinicalRAG\nbuild_knowledge_base,\ngenerate_justification"]
            RAG_STORES["stores.py\n_InMemoryStore,\n_PostgreSQLStore"]
            RAG_LLM["llm_backends.py\nOllama HTTP |\nHuggingFace fallback"]
        end

        INTERP["mosaicfl.core.interpretability\nBEHRTPatternExtractor\nextract_top_patterns\n(atenção → padrões)"]

        subgraph DBPKG["db/"]
            DB_CLINICAL["clinical_mixin.py\nrisco, exames,\ndesfechos"]
            DB_PATIENTS["patients_mixin.py\npacientes, atendimentos"]
        end

        SEC["security.py\nJWT, API Key,\nrate limiting"]
        AUDIT["audit.py\nAudit log LGPD"]
    end

    ROUTER_PRED --> SEC
    ROUTER_PRED --> STATE
    STATE --> ENGINE
    ENGINE --> TOKEN
    ENGINE -->|"padrões de atenção\ndo checkpoint carregado"| INTERP
    INTERP --> RAG_CORE
    RAG_CORE --> RAG_STORES
    RAG_CORE --> RAG_LLM
    ROUTER_PRED --> DB_CLINICAL
    ROUTER_PAT --> DB_PATIENTS
    ROUTER_PRED --> AUDIT
```

---

## 5. Nível 4 — Código (C4), classes-chave por cenário

O C4 trata o nível 4 como opcional — em geral só vale a pena diagramar as classes com maior densidade de decisão de design. Um diagrama de classes por cenário, com os métodos e atributos que efetivamente aparecem nas discussões deste projeto (não é um diagrama de classes exaustivo).

### 5.A — Comunicação Federada: `ProductionFedProxStrategy` e `FedProxClient`

```mermaid
classDiagram
    class ProductionFedProxStrategy {
        <<Servidor — herda fl.server.strategy.FedProx>>
        +CHECKPOINT_DIR
        +LOG_DIR
        +__init__(proximal_mu, min_clients, ...)
        +configure_fit(round, params, client_manager)
        +aggregate_fit(round, results, failures)
        +aggregate_evaluate(round, results, failures)
        -_load_global_weights()
        -_restore_from_state()
        -_start_round_watchdog()
        -_save_state()
        -_run_calibration()
    }
    class _FitConfigMixin {
        +configure_fit(...)
        -_load_global_weights()
    }
    class _WatchdogMixin {
        -_restore_from_state()
        -_start_round_watchdog()
        -_cancel_round_watchdog()
    }
    class _CalibrationMixin {
        -_run_calibration()
    }
    ProductionFedProxStrategy --|> _FitConfigMixin
    ProductionFedProxStrategy --|> _WatchdogMixin
    ProductionFedProxStrategy --|> _CalibrationMixin

    class FedProxClient {
        <<Cliente — herda fl.client.NumPyClient>>
        -client_id
        -model : SimplifiedBEHRT
        -global_params
        +set_parameters(parameters)
        +get_parameters(config)
        +fit(parameters, config) tuple
        +evaluate(parameters, config) tuple
        -_proximal_loss(loss, mu)
        -_compute_class_weights(loader)
    }
    ProductionFedProxStrategy ..> FedProxClient : envia pesos globais\n+ config (μ, round)
    FedProxClient ..> ProductionFedProxStrategy : retorna pesos locais\n+ τ (passos efetivos)
```

### 5.B — Treinamento: `FederatedTraining` e a agregação

```mermaid
classDiagram
    class FederatedTraining {
        <<orchestrator.py>>
        +train() dict
        -_load_data()
        -_run_federated()
        -_run_baselines()
        -_run_ablation()
        -_run_rag()
    }
    class AggregationFunctions {
        <<aggregation.py — funções livres>>
        +aggregate_fedavg(results) params
        +aggregate_fednova(results, tau_list) params
        +apply_dp_noise(params, sigma, clip) params
    }
    class ManualLoop {
        <<manual_loop.py>>
        +run_federated_learning_manual(config) history
    }
    class CheckpointStore {
        <<ABC>>
        +save(round, params, metrics)
        +load_best(training_id) dict
        +register_training(config) training_id
        +save_round_history(round, tau_eff, f1_macro)
    }
    FederatedTraining --> ManualLoop : dispara via router.py
    ManualLoop --> AggregationFunctions : agrega a cada round
    ManualLoop --> CheckpointStore : checkpoint guloso\n(f1_macro > best_f1)
    CheckpointStore <|.. SQLiteCheckpointStore
    CheckpointStore <|.. PostgreSQLCheckpointStore
```

### 5.C — API + RAG: `InferenceEngine` e `ClinicalRAG`

```mermaid
classDiagram
    class InferenceEngine {
        <<engine.py>>
        -model : SimplifiedBEHRT
        -vocab
        -temperature
        +load_from_store(checkpoint dict)
        +predict(records) dict
        -_mc_dropout_sample(x, n_samples)
    }
    class ClinicalRAG {
        <<rag/__init__.py>>
        -_store : _InMemoryStore | _PostgreSQLStore
        -_llm_backend : str
        +build_knowledge_base(patterns)
        +generate_justification(query, top_k) str
        -_check_ollama_available()
    }
    class BEHRTPatternExtractor {
        <<interpretability.py>>
        +extract_top_patterns(model, loader) list
    }
    class PatientDB {
        <<db/__init__.py — composição de mixins>>
        +add_exams_bulk(records)
        +upsert_patient(patient)
        +add_risk(patient_id, prediction)
        +record_outcome(patient_id, outcome)
    }
    InferenceEngine --> BEHRTPatternExtractor : padrões de atenção\ndo checkpoint
    BEHRTPatternExtractor --> ClinicalRAG : perfis clínicos
    InferenceEngine ..> PatientDB : histórico de risco
```

---

## 6. Diagramas de Sequência (UML)

Notação UML de sequência padrão, em Mermaid `sequenceDiagram` — renderiza como diagrama de sequência UML genuíno (atores/objetos com linha de vida, mensagens síncronas/assíncronas, ativação).

### 6.A — Rodada Federada Completa

```mermaid
sequenceDiagram
    participant SL as SuperLink
    participant SA as ServerApp\n(ProductionFedProxStrategy)
    participant CB as SuperNode BPSP\n(FedProxClient)
    participant CH as SuperNode HSL\n(FedProxClient)
    participant DB as PostgreSQL

    SA->>DB: load_best(training_id) — restaura\nestado se houver interrupção anterior
    SA->>SL: configure_fit(round, pesos_globais, config)
    SL->>CB: FitIns(pesos_globais, config{mu, round})
    SL->>CH: FitIns(pesos_globais, config{mu, round})

    par Treino local paralelo
        CB->>CB: fit() — épocas locais,\nloss + termo proximal FedProx,\nclip DP se habilitado
    and
        CH->>CH: fit() — idem
    end

    CB->>SL: FitRes(pesos_locais, n_amostras, τ_BPSP)
    CH->>SL: FitRes(pesos_locais, n_amostras, τ_HSL)
    SL->>SA: aggregate_fit(resultados)
    SA->>SA: aggregate_fednova(resultados, τ)\nnormaliza por passos efetivos
    SA->>DB: save(round, pesos_agregados) — se\nf1_macro > best_f1 (checkpoint guloso)

    SA->>SL: configure_evaluate(round, pesos_globais)
    SL->>CB: EvaluateIns
    SL->>CH: EvaluateIns
    CB->>SL: EvaluateRes(loss, accuracy)
    CH->>SL: EvaluateRes(loss, accuracy)
    SL->>SA: aggregate_evaluate(resultados)
    SA->>SA: ConvergenceTracker.update()\nΔ f1_macro < threshold?

    alt convergiu ou max_rounds atingido
        SA->>SA: _run_calibration()\n(isotônica OvR)
        SA->>DB: save_round_history(...)\ncomplete_training()
    else não convergiu
        SA->>SL: próximo round (repete)
    end
```

### 6.B — Pipeline de Treinamento (`make training-full`)

```mermaid
sequenceDiagram
    participant M as Makefile
    participant R as run_training.py
    participant O as FederatedTraining\n(orchestrator)
    participant FL as fl_core (manual_loop)
    participant CK as CheckpointStore
    participant RAG as ClinicalRAG

    M->>R: fase 1/4 — FL_INCLUDE_HOSPITALS=BPSP
    R->>O: train()
    O->>O: prepare_dataloaders_from_db()\nseed 1042 (BPSP)
    O->>FL: run_federated_learning(config)
    FL->>CK: checkpoint guloso por round
    FL-->>O: history (accuracy, f1_macro, ...)
    O-->>R: resultado fase 1
    R-->>M: evaluation_best_r{N}_of_{120}.json

    M->>R: fase 2/4 — FL_INCLUDE_HOSPITALS=HSL
    R->>O: train() (idem, seed 1043)
    O-->>R: resultado fase 2

    M->>R: fase 3/4 — federado (BPSP+HSL)
    R->>O: train()
    O->>FL: run_federated_learning(config)
    FL->>CK: checkpoint guloso
    FL->>RAG: extract_top_patterns() + build_knowledge_base()\n(pós-convergência)
    RAG-->>FL: Precision@3
    O-->>R: resultado fase 3\n+ last_federated_training_id.txt

    M->>R: fase 4/4 — BEHRT Pooled + RF baseline
    R->>O: train() (sem privacidade, budget equivalente)
    O-->>R: resultado fase 4

    R-->>M: pipeline completo — logs em experiments/logs/
```

### 6.C — Requisição de Inferência com Justificativa RAG

```mermaid
sequenceDiagram
    actor H as Sistema Hospitalar
    participant API as routers/prediction.py
    participant SEC as security.py
    participant ST as state.py
    participant IE as InferenceEngine
    participant INT as BEHRTPatternExtractor
    participant RAG as ClinicalRAG
    participant OL as Ollama (gemma3:4b)
    participant DB as PatientDB

    H->>API: POST /api/predict {patient_id, records}
    API->>SEC: valida JWT/API Key, rate limit
    SEC-->>API: autorizado
    API->>ST: get_engine()
    ST->>ST: fallback: .pt em disco →\nCheckpointStore.load_best() → None
    ST-->>API: InferenceEngine carregado

    API->>IE: predict(records)
    IE->>IE: records_to_tokens()\n(tokenização idêntica ao treino)
    IE->>IE: MC Dropout — N forward passes
    IE-->>API: probabilidades por classe + incerteza

    API->>INT: extract_top_patterns(modelo, paciente)
    INT-->>API: padrões de atenção clínicos

    API->>RAG: generate_justification(padrões)
    RAG->>RAG: build_knowledge_base() — consulta\nperfis similares (ChromaDB/pgvector)
    RAG->>OL: POST /api/generate (prompt + contexto)
    OL-->>RAG: texto da justificativa
    RAG-->>API: justificativa clínica

    API->>DB: add_risk(patient_id, prediction)
    API-->>H: PredictResponse{probabilidades,\nrisk_score, justificativa, incerteza}
```

---

## 7. Diagrama de Caso de Uso (UML)

**Nota de renderização:** este é o único diagrama do documento em **PlantUML**, não Mermaid — Mermaid não tem tipo de diagrama de caso de uso. Para visualizar: cole o bloco abaixo em [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/) (servidor público), use a extensão "PlantUML" do VS Code, ou rode localmente com `plantuml diagrama.puml` (requer Java + Graphviz).

```plantuml
@startuml MOSAIC-FL_Casos_de_Uso
left to right direction
skinparam packageStyle rectangle

actor "Pesquisadora /\nAdministradora" as Pesq
actor "Hospital BPSP\n(nó federado)" as HospA
actor "Hospital HSL\n(nó federado)" as HospB
actor "Sistema Hospitalar\nCliente" as SistExt
actor "Ollama" as Ollama <<sistema externo>>

rectangle "MOSAIC-FL" {

  package "Cenário A — Comunicação Federada" {
    usecase "Iniciar rodada\nfederada" as UC1
    usecase "Treinar modelo\nlocalmente (FedProx)" as UC2
    usecase "Agregar pesos\n(FedNova)" as UC3
    usecase "Verificar\nconvergência" as UC4
    usecase "Recuperar sessão\ninterrompida" as UC5
  }

  package "Cenário B — Treinamento (pesquisa)" {
    usecase "Executar pipeline\nde 4 fases" as UC6
    usecase "Calibrar modelo\n(isotônica OvR)" as UC7
    usecase "Rodar ablação\ndemográfica" as UC8
    usecase "Comparar com\nbaselines (RF, Pooled)" as UC9
    usecase "Aplicar Differential\nPrivacy (DP-FedAvg)" as UC10
  }

  package "Cenário C — API + RAG" {
    usecase "Consultar predição\nde risco" as UC11
    usecase "Gerar justificativa\nclínica (RAG)" as UC12
    usecase "Ingerir novos\nexames" as UC13
    usecase "Exportar para\nClinicalPath" as UC14
    usecase "Exportar\nRiskAssessment (FHIR)" as UC15
    usecase "Registrar desfecho\ntardio (ground truth)" as UC16
  }

  package "Transversal" {
    usecase "Gerenciar vocabulário\nclínico (term_manager)" as UC17
    usecase "Monitorar saúde do\nsistema (health/metrics)" as UC18
    usecase "Configurar hiperparâmetros\nem runtime" as UC19
  }
}

Pesq --> UC1
Pesq --> UC6
Pesq --> UC19
Pesq --> UC18
Pesq --> UC17

HospA --> UC2
HospB --> UC2
UC1 ..> UC2 : <<include>>
UC2 ..> UC3 : <<include>>
UC3 ..> UC4 : <<include>>
UC4 ..> UC5 : <<extend>>\n(se interrompido)

UC6 ..> UC1 : <<include>>\n(simula localmente)
UC6 ..> UC7 : <<include>>
UC6 ..> UC9 : <<include>>
UC6 ..> UC8 : <<extend>>
UC6 ..> UC10 : <<extend>>\n(se FL_DP_NOISE>0)

SistExt --> UC11
SistExt --> UC13
SistExt --> UC16
UC11 ..> UC12 : <<include>>
UC12 --> Ollama
UC11 ..> UC14 : <<extend>>
UC11 ..> UC15 : <<extend>>
UC13 ..> UC17 : <<include>>\n(scan_analytes)

@enduml
```

---

## Referência cruzada de nomes — diagrama ↔ código

Para evitar ambiguidade entre o nome usado nos diagramas e o caminho real no repositório (pós-modularização de 2026-07-01):

| Nome no diagrama | Caminho no repositório |
|---|---|
| ServerApp | `infrastructure/mosaicfl_server/` |
| SuperNode | `infrastructure/mosaicfl_client/` |
| Scheduler | `infrastructure/mosaicfl_scheduler/` |
| API de Inferência | `infrastructure/mosaicfl_api/` |
| Ambiente de Pesquisa | `experiments/` (`training_runner/` + `training/core/`) |
| `ProductionFedProxStrategy` | `infrastructure/mosaicfl_server/strategy/` (pacote, mixins) |
| `FedProxClient` | `src/mosaicfl/core/client.py` |
| `FederatedTraining` | `experiments/training/core/orchestrator.py` |
| `InferenceEngine` | `infrastructure/mosaicfl_api/inference_engine/` (pacote) |
| `ClinicalRAG` | `src/mosaicfl/core/rag/` (pacote) |
| `PatientDB` | `infrastructure/mosaicfl_api/db/` (pacote, mixins) |
| `CheckpointStore` | `infrastructure/shared/checkpoint_store/` (pacote) |
| `MetricsStore` | `infrastructure/shared/metrics_store/` (pacote) |

Este documento reflete a estrutura pós-modularização (ver `docs/Linha_do_Tempo_MOSAIC-FL.md`, Parte 10). Se novos módulos forem criados ou renomeados, atualizar esta tabela primeiro — os diagramas C4 nível 3/4 dependem diretamente dela.
