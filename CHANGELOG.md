# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

## [Unreleased]

### Adicionado
- `test_fl_cycle_explained.py` — documentação executável do ciclo FL completo (29 testes)
- `test_infrastructure.py` — cobertura dos daemons de produção com mocks (61 testes)
- `.env.example` — referência de variáveis de ambiente necessárias
- `.pre-commit-config.yaml` — hooks de lint e formatação automáticos
- `CHANGELOG.md` — este arquivo
- `TODO.md` — roadmap de qualidade profissional e produção
- Seções de Testes, Rodando Localmente e Benchmark no README
- `make lint` e `make fmt` no Makefile

### Corrigido
- `preprocess_v2.py`: `select_dtypes(include=['object'])` → `['object', 'str']` (Pandas4Warning)
- `preprocess_v2.py`: `normalize_units` cast explícito para float antes de multiplicação
- `data_loader.py`: `_convert_desfecho` usa `pd.api.types.is_numeric_dtype()` em vez de comparação com `object`
- `benchmark.py`: reescrito completamente para usar imports v2; corrigidos SyntaxError em strings literais e `monitor.stop()` duplo
- `test_v2_core.py`: buffers Long clampados ao range válido do vocab em `test_fedavg_aggregation_preserves_shape` (eliminava falha dependente de ordem de execução dos testes)
- `infrastructure/mosaicfl_client/heartbeat.py`: recuperação de JSON corrompido usa `json.dumps` correto
- Nomes de diretório na documentação: `infrastructure/server/` → `infrastructure/mosaicfl_server/`

### Alterado
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
