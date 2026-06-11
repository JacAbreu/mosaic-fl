# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

## [Unreleased]

### Adicionado
- `SequencePipeline` em `preprocessor.py` — pipeline temporal para dados reais do PostgreSQL:
  - `build()` — query + tokenização + tensores para um hospital (ou todos se `hospital_id=None`)
  - `build_per_hospital()` — query única, divide tensores por hospital (simulação FL)
  - `_load_dataframe()` — conexão/query extraída para reutilização sem duplicação
  - Parâmetro `hospital_id: Optional[str]` — modo produção (cliente FL real) vs simulação
- `SequencePipelineInicial` preservada como referência histórica da abordagem binária
- `prepare_dataloaders_from_db()` em `run_experiments_v2.py` — cria DataLoaders FL diretamente do banco via `build_per_hospital()`
- `FL_DB_URL` e `FL_ENV` como variáveis de ambiente e aliases em `config.py` (via `RuntimeConfig`)
- Guarda de produção em `load_with_fallback()`: `FL_ENV=production` bloqueia sintético e exige `FL_DB_URL`
- Migration `006_extend_patients_attendances` — formaliza `municipality`, `cep_prefix` (patients), `clinic_id` (attendances)
- Migration `007_add_diagnosis_to_attendances` — adiciona `suspected_diagnosis TEXT` e `confirmed_diagnosis TEXT` a `clinical.attendances`, com índice parcial
- Mapeamentos semânticos para `suspected_diagnosis` e `confirmed_diagnosis` em `integration/column_resolver.py`
- Extração de `suspected_diagnosis` e `confirmed_diagnosis` em `integration/fapesp/outcomes_extract.py`
- `docs/FLUXO_APRENDIZADO_FEDERADO.md` — documento técnico com diagramas Mermaid do pipeline completo, série temporal BEHRT, rodadas federadas e RAG
- `test_fl_cycle_explained.py` — documentação executável do ciclo FL completo (29 testes)
- `test_infrastructure.py` — cobertura dos daemons de produção com mocks (61 testes)
- `.env.example` — referência de variáveis de ambiente necessárias
- `.pre-commit-config.yaml` — hooks de lint e formatação automáticos

### Corrigido
- `MODEL_CFG.num_classes`: 2 → 5 (faixas de duração de internação como label, não binário)
- `set_parameters` em `client.py`: docstring corrigida — usa `state_dict()` (treináveis + buffers), não `model.parameters()`
- `preprocess_v2.py`: `select_dtypes(include=['object'])` → `['object', 'str']` (Pandas4Warning)
- `preprocess_v2.py`: `normalize_units` cast explícito para float antes de multiplicação
- `data_loader.py`: `_convert_desfecho` usa `pd.api.types.is_numeric_dtype()` em vez de comparação com `object`
- `benchmark.py`: reescrito completamente para usar imports v2; corrigidos SyntaxError em strings literais e `monitor.stop()` duplo
- Nomes de diretório na documentação: `infrastructure/server/` → `infrastructure/mosaicfl_server/`

### Alterado
- `run_experiments_v2.py`: usa `FL_DB_URL` para carregar tensores reais quando configurado; fallback para CSV/sintético apenas em `FL_ENV=development`
- `_build_tensors()` em `SequencePipeline`: retorna `(sequences, labels, hospital_ids)` — rastreia hospital por sequência
- Licença corrigida de MIT para Apache 2.0 em `pyproject.toml` e README
- Roadmap de produção movido do README para `TODO.md`

## [0.2.0] — 2026-06-03

### Adicionado
- `SimplifiedBEHRT` v2 com CLS token pooling e `BEHRTEncoderLayer` customizado (expõe pesos de atenção)
- `FedProxClient` v2 com termo proximal correto e tratamento de exceção por batch
- `ConvergenceTracker` com stable_count incremental
- `EHRPreprocessor` v2 com normalização de unidades médicas (kg/lb, anos/meses)
- `data_loader.py` com Strategy Pattern: SGBD → CSV → sintético
- `RAGSystem` v2 com ChromaDB + DistilGPT-2 e truncagem de prompt
- Daemons de produção: `server_daemon.py`, `client_daemon.py`, `scheduler_daemon.py`
- `ProductionFedProxStrategy` com checkpoint e exportação de métricas JSON
- `scheduler_daemon.py` com APScheduler, heartbeat e estado persistente
- Suite de testes `test_mosaicfl.py` e `test_v2_core.py`

## [0.1.0] — 2026-05-01

### Adicionado
- Implementação v1 com dados sintéticos (BEHRT mean pooling, FedProx básico)
- 5 experimentos do TCC: preprocessamento, efeito equalizador FL, heterogeneidade não-IID, RAG, eficiência
- RAG v1 com ChromaDB
- `run_experiments.py` e `run_experiments_v2.py`
