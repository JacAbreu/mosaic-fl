# TODO — MOSAIC-FL

Dividido em duas partes:
- **Qualidade profissional** — o que falta para o código atingir padrão de engenharia de software profissional, independente de deploy real
- **Roadmap de produção** — funcionalidades necessárias para uso clínico real

---

## Qualidade Profissional

### Consistência de código

- [x] ~~**Corrigir docstring de `set_parameters` em `client.py`**~~

  ~~A docstring dizia "Usa model.parameters()" mas o código usa `state_dict().keys()` (inclui buffers). A contradição criava expectativa errada de que buffers não eram sincronizados.~~
  Corrigido: docstring e comportamento alinhados — state_dict é usado em set e get (treináveis + buffers).

- [x] ~~**Eliminar `from .config import *` em todos os módulos v2**~~

  ~~Wildcard imports poluem o namespace de cada módulo com ~20 constantes, dificultam entender de onde vem cada símbolo, e causam conflitos silenciosos se dois módulos definirem o mesmo nome. Substituir por imports explícitos: `from .config import VOCAB_SIZE, EMBED_DIM, ...`.~~

- [x] ~~**Unificar os dois `ConvergenceTracker`**~~

  ~~Existia um em `server_v2.py` (stable_count incremental) e outro em `infrastructure/mosaicfl_server/strategy.py` (janela deslizante).~~
  Unificado em `src/mosaicfl/core/convergence.py` (janela deslizante) — fonte única importada por todos os adapters.

- [ ] **Implementar `_save_checkpoint` de verdade em `experiment_server.py`**

  Hoje o método apenas registra um caminho no histórico sem escrever nada em disco. Implementar com `torch.save(model.state_dict(), path)` e carregar com `torch.load`.

- [ ] **Integrar RAG com tensores reais (modo banco)**

  `run_rag_pipeline()` em `run_experiments_v2.py` depende de `df_raw` (DataFrame com coluna `desfecho`), que não existe no modo banco de dados (`loaded_from_db=True`). Adaptar para construir `patient_data` a partir dos tensores/vocab do SequencePipeline.

- [x] ~~**Corrigir `fit_metrics_aggregation_fn=weighted_average`**~~

  ~~`weighted_average` espera o dicionário `{"accuracy": float}`. Quando usado como `fit_metrics_aggregation_fn`, recebe `{"loss": float}` — a chave `"accuracy"` não existe e retorna `{}`.~~

  Implementada opção C: `_weighted_average(metrics, key)` como implementação privada + `weighted_average_accuracy` e `weighted_average_loss` como funções públicas nomeadas. O alias `weighted_average = weighted_average_accuracy` preserva compatibilidade. Call sites em `server_v2.py`, `runner.py` e `strategy.py` atualizados. 8 novos testes em `TestWeightedAverageLoss` incluindo reprodução do bug original.

- [x] ~~**Corrigir `communication_mb` no histórico**~~

  ~~Em `server_v2.py` a estimativa é `round * 2.0` MB — um número arbitrário que cresce linearmente com o número de rounds sem relação com o tamanho real do modelo.~~

  Implementado: `_MODEL_SIZE_MB = 2.745 MB` calculado uma vez do `state_dict` real. `communication_mb` agora é `len(results) * _MODEL_SIZE_MB * 2` (upload + download por cliente que participou da rodada).

### Qualidade de código estático

- [x] ~~**Adicionar type hints completos**~~

  ~~Os módulos v2 têm type hints parciais. Completar com anotações em todos os métodos públicos e habilitar `mypy` no CI. Type hints servem como documentação executável e pegam classes inteiras de bugs antes de rodar.~~

- [x] ~~**Adicionar linting com ruff ao `make` e ao CI**~~

  ~~O `ruff` já está no `setup` do Makefile mas não há target `make lint` funcional nem verificação no CI. Adicionar `make lint` e `make fmt`, e bloquear merge se o lint falhar no GitHub Actions.~~

- [x] ~~**Configurar pre-commit hooks**~~

  ~~Sem hooks, é fácil commitar código com imports não usados, formatação inconsistente ou strings de debug. Adicionar `.pre-commit-config.yaml` com `ruff`, `ruff-format` e detecção de debug statements.~~

### Testes

- [x] ~~**Adicionar testes de contrato para `fit()` e `evaluate()`**~~

  ~~Os testes atuais verificam que os métodos rodam, mas não validam os tipos de retorno de forma rigorosa. Um teste de contrato verifica que `fit()` sempre retorna `(List[np.ndarray], int, {"loss": float})` — se alguém mudar a chave de `"loss"` para `"train_loss"`, o teste quebra imediatamente.~~

  Implementado em `TestFitContract` (15 testes) e `TestEvaluateContract` (15 testes) em `test_v2_core.py`. Verificam: tipos Python exatos (`type(x) is int/float`), shapes invariantes, `n_samples` exato contra o dataset, compatibilidade com `weighted_average_loss`/`weighted_average_accuracy`, e dois testes "negativos" que demonstram o que quebra quando a chave é renomeada. Descobriu incidentalmente que `cls_token_id` (int64) é incluído no state_dict — comportamento correto e agora documentado no teste.


- [ ] **Teste de integração end-to-end real (sem mocks)**

  Os testes de infraestrutura usam mocks para tudo. Existe `test_fl_cycle_explained.py` que testa o ciclo com dados reais, mas não testa os daemons de ponta a ponta. Adicionar um teste que sobe servidor + cliente em threads/processos separados, executa 1 round real e verifica que o modelo foi atualizado. Isso pega bugs de integração que mocks nunca encontram.

### Observabilidade

- [x] ~~**Structured logging em JSON**~~

  ~~Hoje os logs são strings livres (`logger.info("Rodada %d: ...", round)`). Em produção, logs precisam ser parseáveis por ferramentas como Loki, Datadog ou CloudWatch. Substituir por structured logging:~~
  ```python
  logger.info("round_completed", extra={"round": 3, "accuracy": 0.82, "loss": 0.31})
  ```
  Implementado em `infrastructure/logging_setup.py` com `python-json-logger`. Saída JSON por padrão (`FL_LOG_FORMAT=json`); fallback texto com `FL_LOG_FORMAT=text`. Eventos estruturados: `round_dispatched`, `round_completed`, `convergence_reached`, `checkpoint_saved`, `proximal_mu_updated`, `config_written`, `client_registered`, etc. Todos os daemons e strategy atualizados.

- [ ] **Health check endpoint nos daemons**

  `server_daemon.py` e `client_daemon.py` não expõem nenhum endpoint HTTP de health check. Kubernetes, Docker Swarm e load balancers precisam de `/healthz` para saber se o processo está vivo. Adicionar um servidor HTTP mínimo (FastAPI ou http.server) que responda 200 quando o daemon está operacional.

- [ ] **Métricas Prometheus**

  O TODO de infraestrutura menciona Prometheus + Grafana, mas o pré-requisito é expor as métricas. Adicionar `prometheus_client` nos daemons para publicar: `fl_round_total`, `fl_accuracy`, `fl_loss`, `fl_clients_active`. Sem isso, o Grafana não tem o que exibir.

### Configuração e segredos

- [ ] **Separar configuração de ambiente da configuração de modelo**

  `config.py` mistura hiperparâmetros de modelo (`VOCAB_SIZE`, `EMBED_DIM`) com configurações de ambiente (`DEVICE`, `OMP_NUM_THREADS`, `DATA_PATH`). Os primeiros são fixos para o experimento; os segundos variam por máquina. Separar em `model_config.py` (versionado) e carregar o resto de variáveis de ambiente ou de um `.env` via `python-dotenv`.

- [x] ~~**Arquivo `.env.example`**~~

  ~~Nenhum colaborador sabe quais variáveis de ambiente o projeto precisa sem ler o código. Criar `.env.example` com todas as variáveis documentadas. O `.env` real fica no `.gitignore`.~~

### Documentação

- [x] ~~**CHANGELOG.md**~~

  ~~Não existe registro de o que mudou entre versões. Qualquer colaborador que volte após semanas não sabe o que foi alterado. Manter um `CHANGELOG.md` no formato Keep a Changelog com entradas em `## [Unreleased]` a cada mudança relevante.~~

- [x] ~~**CONTRIBUTING.md**~~

  ~~Não existe guia de contribuição. Documentar: como configurar o ambiente, padrão de commits (Conventional Commits), como rodar os testes, como abrir um PR. Sem isso, qualquer novo contribuidor improvisa.~~

- [ ] **Docstrings completas nos módulos públicos**

  `model_v2.py` tem docstrings boas. `client_v2.py` tem a docstring do módulo desatualizada (menciona "VERSÃO CORRIGIDA" com lista de mudanças — isso é changelog, não docstring). Docstrings de módulo devem descrever o que o módulo faz, não o histórico de edições.

---

## Roadmap de Produção

### Dados e Integração

- [x] ~~**Exportador ClinicalPath**~~

  ~~Exporta predições do MOSAIC-FL como exame sintético `FL_RISK_SCORE` nos arquivos plain-text do ClinicalPath v2. Implementado em `integration/clinical-path/` com `ClinicalPathExporter`, modelos `ExamRecord`/`RiskPrediction`/`PatientExport`/`ClinicalPhase` e 38 testes de contrato. Referência `ref_high=0.3` faz ClinicalPath exibir dias com score acima de 0.3 na cor de risco.~~

- [ ] Integração HL7 FHIR com EPR dos hospitais
- [ ] Conector genérico para prontuários eletrônicos brasileiros (MV, Tasy, Soul MV)

### Segurança e Privacidade

- [ ] TLS mútuo (mTLS) entre servidor e clientes Flower
- [ ] Differential Privacy nos pesos (Gaussian mechanism, ε-δ DP)
- [ ] Auditoria de acesso e rastreabilidade conforme LGPD
- [ ] Consentimento informado e designação de DPO

### Modelo

- [ ] Fine-tuning em corpus clínico brasileiro (MIMIC-BR ou equivalente)
- [ ] Substituir DistilGPT-2 por LLM em português (Maritaca, Llama-PT)
- [ ] Avaliação com AUC-ROC, sensibilidade e especificidade em estudo retrospectivo

### Infraestrutura

- [x] ~~**Ambiente wire-production (Docker Compose)**~~

  ~~Ambiente de simulação completo em uma única máquina: `fl-server`, `fl-client-1/2/3`, `scheduler`, `api`. Implementado em `wire-production/` com `Dockerfile.wire` (imagem unificada), `docker-compose.yml` (6 serviços, 3 volumes nomeados, healthchecks), `.env.example` e `seed/generate_data.py` para geração de exames sintéticos.~~

- [ ] Chamadas gRPC diretas do scheduler para o servidor Flower (hoje o scheduler é apenas supervisor)
- [ ] Message broker (RabbitMQ ou Redis) para orquestração de rounds
- [x] ~~**Integração com `fl.server.Driver` do Flower SDK para controle programático**~~

  ~~Driver/Grid API para manter controle do loop FL no código Python.~~

  Implementado via `configure_fit` override na `ProductionFedProxStrategy`: lê config do ChromaDB (ou arquivo) antes de cada round, permitindo alterar `proximal_mu`, pausar ou parar sem reiniciar o servidor. Backend selecionável por `FL_CONFIG_BACKEND=chroma|file`. 55 testes em `test_config_loader.py`. Migração para PostgreSQL quando disponível.

- [ ] Monitoramento com Prometheus + Grafana para métricas de treino federado

### Regulatório

- [ ] Submissão ANVISA como Software como Dispositivo Médico (SaMD) Classe II/III
- [ ] Validação clínica prospectiva com parecer de comitê de ética (CEP/CONEP)
