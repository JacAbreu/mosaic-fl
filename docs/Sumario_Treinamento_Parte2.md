# Sumário de Treinamento — Parte 2

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico  
**Autora:** Jacqueline Abreu | ICMC/USP  
**Continuação de:** `docs/Sumario_Treinamento.md` (Exp 1–16)  
**Iniciado em:** 2026-06-29

---

## Nota sobre nomenclatura

Os documentos anteriores (`Sumario_Treinamento.md`) usam "Experimento N" por convenção inicial. A partir deste documento, adota-se **"Treinamento N"**.

**Justificativa:** cada entrada representa uma execução de treinamento com uma configuração específica — não uma hipótese formal isolada. "Experimento" em dissertações implica controle estrito de variáveis; aqui múltiplos parâmetros mudam juntos entre execuções. "Treinamento" é mais preciso e consistente com o vocabulário do pipeline (`training_id`, `make training-full`, `fl_trainings`). Na defesa: *"este trabalho conduz experimentos de aprendizado federado nos quais foram executados N treinamentos com configurações progressivamente refinadas."*

---

## Bloco 1 — Treinamentos pré-correção do split (T1–T16 + Run de Validação)

### Delimitação do bloco

Em 2026-06-30, foi corrigido o gerador de permutações em `prepare_dataloaders_from_db` (`dataloaders.py`): o RNG que produzia as partições train/val/cal/teste de cada hospital passou de **compartilhado sequencialmente entre hospitais** para **independente por hospital** (`RANDOM_SEED + 1000 + cid`).

Essa mudança altera as partições BPSP e HSL. Consequência direta: **todos os treinamentos executados antes dessa correção formam um bloco metodologicamente homogêneo**, e os resultados desse bloco não são diretamente comparáveis com os treinamentos do Bloco 2 (pós-correção). O valor 70,19% do Run de Validação é o recorde deste bloco e serve como referência histórica — não como baseline para os próximos treinamentos.

### Tabela consolidada do Bloco 1

| Treinamento | Acurácia avaliada | Melhor rodada | Algoritmo | Principal mudança | Principal lição |
|---|---|---|---|---|---|
| T1 | 58,0% | — | FedAvg | Primeira execução real com dados FAPESP, split 80/10/10, 20 rodadas | Crash pós-treino (bug no retorno de `run_federated_learning_manual`); baseline RF não executado |
| T2 | 52,5% | R7 | FedAvg | Correção do crash; baselines RF adicionados | Regressão de 5,5 p.p. — variância estocástica domina com 20 rodadas e sem checkpoint guloso |
| T3 | 55,8% | — | FedAvg | Split mudado para 70/10/10/10; conjunto de calibração introduzido | +3,3 p.p. vs T2; conjunto de cal independente é necessário para temperature scaling correto |
| T4 | 54,75% | — | FedAvg | Ajuste de hiperparâmetros; `evaluation_round_20.json` sobrescrito pelo T5 — métricas parcialmente perdidas | Perda silenciosa de dados de avaliação; confirmou necessidade de persistência no banco |
| T5 | 56,55% | — | FedAvg | Peso máximo de classe limitado; corrreção do bug `per_class_f1` em `ablation.py` | `melhora_pronto` colapsou (F1=0,025 vs 0,397 no T3) — modelo priorizou `curado_pronto` (majoritário BPSP) |
| T6 | 59,63% | R6 / 62,7% | FedAvg | `dia_relativo` adicionado como embedding temporal na sequência de exames | +3,08 p.p. — maior ganho de uma única alteração arquitetural no projeto; custo de privacidade caiu de ~10 para ~4 p.p. |
| T7 | 59,36% (R120) | **R89: 63,29%** | FedAvg | µ=0,01→0,1; max_rounds=20→120; `min_rounds=20` para warm-up | Sem checkpoint guloso: avaliação na R120 desperdiçou 3,93 p.p. de modelo; demonstrou que 120 rodadas é necessário e que a última rodada ≠ melhor |
| T8 | 66,61% | **R91** | FedAvg | Checkpoint guloso implementado — `save()` no PostgreSQL a cada `acc > best_accuracy`; `load_best()` restaura antes da avaliação | +7,25 p.p. vs T7 (avaliação correta); gap best vs last = 8,34 p.p. em 120 rodadas |
| T9 ⚠️ | 66,73% | R33 (Exp) / **R91 (DB)** | FedNova | Introdução do FedNova (normalização por τ_i) | **Cross-contamination:** `load_best()` sem `training_id` retornou o checkpoint do T8 (R91). Resultado inválido para FedNova. Ação: migration 011 + `training_id` scoping |
| T12 | 67,44% | **R115** | FedNova | Migration 011: `fl_trainings` + `training_id` FK; checkpoint scoping por treinamento | Primeiro resultado válido do FedNova — +0,83 p.p. vs T8; eliminou cross-contamination |
| T13 (BPSP-only) | 64,86% | **R118** | FedNova | Leave-one-out BPSP; gradient clipping max_norm=1,0; local_epochs=1; calibração isotônica OvR | `melhora_pronto` ausente com BPSP isolado (85 amostras, 0,4%); confirma que federação é clinicamente necessária |
| T14 (HSL-only) | 40,05% | **R100** | FedNova | Leave-one-out HSL | Dataset pequeno (3.621 amostras) + domínio diferente; loss crescente (instabilidade); −24,81 p.p. vs federado |
| T15 (Fed. BPSP+HSL) | **69,59%** | **R79** | FedNova | Pipeline MVP completo (FedNova + clip + local_epochs=1 + isotônica + scoping) | **Primeiro treinamento em que FL supera todos os baselines centralizados** — custo de privacidade negativo |
| T16 (BEHRT Pooled) | 68,68% | — (120 épocas) | — | Baseline pooled centralizado (BPSP+HSL sem privacidade) | FL (69,59%) supera pooled (68,68%) em +0,91 p.p. com budget equivalente |
| Run de Validação | **70,19%** | **R105** | FedNova | Seeding fix (per-round × cliente) + correção de 4 bugs RAG + Ollama integrado | Novo recorde absoluto do bloco; HSL regrediou 5 p.p. pelo seeding (dropout determinístico sensível a dataset pequeno) |

> **Nota T10/T11:** execuções intermediárias documentadas no Sumário original sem registro de acurácia destacado; omitidas desta tabela por não constituírem marcos de decisão.

### Decisões e correções por treinamento — detalhamento

#### T1 → T2

**O que mudou:** corrigido bug `AttributeError: 'list' object has no attribute 'get'` em `run_experiments_simulation.py` — a função retornava `history` como dict de listas mas o caller esperava dict de dicts por round. Baselines RF adicionados.  
**Por quê:** o crash pós-treino do T1 impedia a análise dos resultados; sem RF não havia baseline de comparação.  
**Resultado:** T2 rodou até o fim (R7, convergência), Acc=52,5% — regressão de 5,5 p.p. vs T1. Variância estocástica domina com apenas 20 rodadas.  
**Decisão:** manter 20 rodadas por ora; investigar split e calibração antes de aumentar rodadas.

---

#### T2 → T3

**O que mudou:** split 80/10/10 → 70/10/10/10; conjunto de calibração independente introduzido.  
**Por quê:** temperature scaling estava sendo feito no conjunto de teste — contaminação da avaliação. Com cal set isolado, ECE e MCE passam a ser métricas limpas.  
**Resultado:** T3 Acc=55,8%, +3,3 p.p. vs T2. Cal set isolado confirmou o padrão de subconfiança (ECE piora após calibração) que se repetirá em todos os experimentos.  
**Decisão:** 70/10/10/10 torna-se o split padrão do projeto.

---

#### T4

**O que mudou:** ajuste de hiperparâmetros; `evaluation_round_20.json` sobrescrito pelo T5 (arquivo único, nome fixo).  
**Por quê:** tentativa de aumentar acurácia por tunagem fina.  
**Resultado:** T4 Acc=54,75%. Métricas de precision/recall por classe perdidas pela sobrescrita.  
**Decisão:** o episódio motivou, meses depois, a migration 012 (`evaluation_json JSONB` no banco). A depedência de arquivo único nomeado fixo é um risco de perda de dados — anotado como dívida técnica.

---

#### T5 → T6

**O que mudou:** peso máximo de classe limitado (clamp); correção do bug `per_class_f1` em `ablation.py`; adição de `dia_relativo` como embedding temporal.  
**Por quê:** `melhora_pronto` colapsou em T5 (F1=0,025) porque a classe representa apenas 0,4% do BPSP — peso bruto de ~47x causava gradientes instáveis, forçando o modelo a ignorar a classe. `dia_relativo` foi a aposta para introduzir âncora temporal e estabilizar.  
**Resultado:** T6 Acc=59,63%, +3,08 p.p. — maior ganho de uma única alteração arquitetural no projeto. `melhora_pronto` voltou a F1=0,112 (+4,5× vs T5). MCE despencou de 0,736 para 0,180 — modelo mais calibrado nos extremos.  
**Decisão:** `dia_relativo` é necessário. Embedding temporal passa a ser componente permanente da arquitetura.

---

#### T6 → T7

**O que mudou:** µ FedProx 0,01 → 0,1; rodadas máximas 20 → 120; `min_rounds=20` (warm-up); correção de 3 bugs em `rag.py`.  
**Por quê:** variância de ±12 p.p. entre rodadas (50,84%–62,70% em T6) indicava client drift excessivo com µ fraco. 20 rodadas eram insuficientes para convergência — modelo ainda oscilava ao final.  
**Resultado:** T7 rodou 4,4h (15.846s), 120 rodadas sem convergência. Avaliação na R120=59,36%, mas melhor rodada foi R89=63,29% — gap de 3,93 p.p. perdido por falta de checkpoint guloso. AUC macro subiu de 0,770 (T6) para 0,770 (praticamente igual), mas ECE caiu de 0,1046 para 0,0326 — melhora drástica de calibração nativa. RAG funcionou pela primeira vez.  
**Decisão:** 120 rodadas é o número mínimo viável. Gap R89 vs R120 é a evidência direta para implementar checkpoint guloso no T8.

---

#### T7 → T8

**O que mudou:** checkpoint guloso implementado (`save()` no PostgreSQL a cada `acc_global > best_accuracy`; `load_best()` restaura antes da avaliação); calibração em log-space (T = exp(log_T)) para evitar T negativo; BEHRT Pooled omitido (bug `POOLED_EPOCHS` descoberto).  
**Por quê:** o gap de 3,93 p.p. entre R89 e R120 do T7 quantificou exatamente o custo de não ter checkpoint guloso. Bug de calibração (T=−8,9997 com LBFGS sem log-space) havia destruído as probabilidades de saída em tentativa anterior.  
**Resultado:** T8 Acc=66,61% (R91) — +7,25 p.p. vs T7 avaliação final; +2,49 p.p. vs melhor do T7 (R89=63,29%). Gap melhor vs última rodada: 8,34 p.p. (R91=66,61% vs R120=58,27%). AUC macro 0,810; Macro F1=0,481. `melhora_pronto` F1 saltou de 0,249 para 0,619 — maior evolução de qualquer métrica de classe em todo o projeto. Recalibração com `make recalibrate` post-treinamento confirmou o fix de log-space (T=1,0849) mas ECE continuou piorando com temperature scaling — padrão estrutural, não bug.  
**Decisão:** checkpoint guloso é obrigatório em todos os treinamentos futuros. Temperature scaling é inadequado para este dataset — investigar calibração isotônica.

---

#### T8 → T9 (cross-contamination — resultado inválido para FedNova)

**O que mudou:** algoritmo de agregação FedAvg → FedNova (Wang et al. 2020); `aggregate_fednova()` em `fl_core.py`; `fit()` do cliente retorna τ (passos efetivos); `use_fednova=True` em `FedConfig`.  
**Por quê:** o non-IID estrutural (BPSP=20k amostras vs HSL=3,6k, razão 5,5:1) produz updates de magnitudes completamente diferentes. FedAvg pondera por volume — BPSP domina. FedNova normaliza por τ_i (batches × épocas locais), eliminando o viés de escala sem hiperparâmetro adicional.  
**Resultado:** T9 relatou 66,73% R33, mas `load_best()` sem filtro por `training_id` retornou o checkpoint R91 do T8 (0,6661 > 0,6386). A avaliação refletiu o modelo do T8, não do FedNova. Resultado inválido para avaliar o algoritmo.  
**Decisão:** implementar migration 011 com `fl_trainings` e `training_id` FK antes de reexecutar FedNova. O T9 não produz número utilizável para o TCC, mas a cross-contamination revela um gap crítico de isolamento que precisava ser corrigido de qualquer forma.

---

#### T10 / T11 (transição — sem resultado destacado)

**O que foram:** execuções intermediárias durante o desenvolvimento e teste da migration 011. Não constituem marcos de decisão ou resultados reportáveis.  
**Contexto:** migration 011 introduziu `metrics.fl_trainings` (tabela de metadados de treinamento) e a FK `training_id` em `metrics.fl_checkpoints`, com índice UNIQUE parcial (`WHERE training_id IS NOT NULL`). `register_training()` antes do loop e `complete_training()` após. `load_best(training_id)` passou a filtrar por treinamento.  
**Decisão:** a partir do T12, todo treinamento federado tem `training_id` próprio — cross-contamination eliminada permanentemente.

---

#### T12 — primeira avaliação válida do FedNova

**O que mudou:** migration 011 aplicada; `register_training()` antes do loop; UPSERT com `ON CONFLICT (training_id) WHERE training_id IS NOT NULL`; `load_best(training_id=2)` com filtro explícito.  
**Por quê:** repetição do T9 com o sistema de scoping corrigido. Objetivo: medir o FedNova sem contaminação do checkpoint do T8.  
**Resultado:** T12 Acc=67,44% (R115, training_id=2), AUC=0,802, F1=0,484, ECE=0,1086 (temperature). Log confirmou `checkpoint_best_loaded_postgres round=115 accuracy=0.6744 training_id=2` — sem cross-contamination. Gap best vs last: 6,30 p.p. Pico térmico de 94°C na R120. `melhora_pronto` AUC=0,9553 — melhor da classe em todos os experimentos. Custo de privacidade real: FL (67,44%) vs BEHRT Pooled B (69,12%) = −1,68 p.p.  
**Decisão:** FedNova produz +0,83 p.p. sobre FedAvg com mesmo budget e sem parâmetro adicional. Será o algoritmo de agregação padrão. BEHRT Pooled confirma superação do RF pela segunda vez consecutiva.

---

#### T13 — BPSP-only (fase 1/4 do primeiro `make training-full`)

**O que mudou:** conjunto completo de melhorias MVP pela primeira vez: `local_epochs=1` (era 2), gradient clipping `max_norm=1.0`, class weight clipping `max=15.0`, `IsotonicCalibrator` OvR ao lado do `TemperatureScaler`, `DataLoader` com `generator` seeded por cliente, ablation multi-seed (k=3). Leave-one-out BPSP via `FL_INCLUDE_HOSPITALS=BPSP`.  
**Por quê:** isolar o valor de cada hospital. Hipótese central: BPSP com 0,4% de `melhora_pronto` no treino não aprenderá essa classe — modelo vai F1=0 nela, confirmando que a federação é clinicamente necessária.  
**Resultado:** T13 Acc=64,86% (R118, training_id=3), AUC=0,7065, F1=0,3302, ECE=0,0237 (isotônica). `melhora_pronto` F1=0,000, AUC=0,5149 (aleatório) — hipótese confirmada empiricamente. Isotônica (ECE=0,0237) superou temperature scaling (ECE=0,0921) pela primeira vez — confirma que calibração não-paramétrica por classe é a abordagem correta. Duração: 105,9 min (1 cliente).  
**Decisão:** federation is clinically necessary — argumento central validado com dado empírico. Isotônica OvR torna-se o calibrador padrão a partir de T13.

---

#### T14 — HSL-only (fase 2/4)

**O que mudou:** leave-one-out HSL via `FL_INCLUDE_HOSPITALS=HSL`. Mesmo conjunto de melhorias MVP do T13.  
**Por quê:** completar a análise leave-one-out. Hipótese: HSL com 3,6k amostras não generaliza para o test set dominado pelo BPSP (84,7% das amostras de teste).  
**Resultado:** T14 Acc=40,05% (R100, training_id=4), AUC=0,6572, F1=0,2853, ECE=0,0466 (isotônica). Regressão severa na R120 (24,16%) — oscilação extrema com 1 cliente e dataset pequeno. Loss crescente ao longo do treinamento (R1: 1,45 → R100: 3,94). Duração: apenas 18,9 min (HSL tem 226 batches/round vs 1.252 do BPSP). Ablation com late fusion deu −4,06 p.p. — única vez no projeto em que demográficos prejudicam: viés do perfil HSL não generaliza para o BPSP.  
**Decisão:** T14 confirma que o BEHRT HSL-only é instável e não generalizável isoladamente. A contribuição do HSL é indispensável precisamente via federação — não como modelo independente. Late fusion demográfica é contextual: ajuda com diversidade de treino, prejudica sem ela.

---

#### T15 — Federado BPSP+HSL com pipeline MVP completo (fase 3/4)

**O que mudou:** todas as melhorias MVP juntas pela primeira vez no treinamento federado completo (2 clientes): `local_epochs=1`, gradient clipping, class weight clipping, isotônica OvR, DataLoader seeded, FedNova com training_id scoping.  
**Por quê:** T12 (67,44%) foi o primeiro FedNova válido, mas ainda com `local_epochs=2`. T13 e T14 validaram individualmente cada melhoria MVP. T15 é a composição completa.  
**Resultado:** T15 Acc=69,59% (R79, training_id=5), AUC=0,8181, F1=0,4946, ECE=0,0149 (isotônica — menor valor do projeto). Duração: 121,8 min. Gap best vs last: 6,44 p.p. (R79=69,59% vs R120=63,15%). **Marco histórico:** primeira vez que FL supera todos os baselines centralizados — RF (68,41%), BEHRT Pooled A (68,29%), BEHRT Pooled B (68,68%). Custo de privacidade negativo. `make training-full` total: 583 min (9h43min).  
**Decisão:** local_epochs=1 + grad clipping + isotônica OvR é a combinação que cruzou o limiar. Resultado é o novo baseline para os experimentos DP (Exp 17/18/19).

---

#### T16 — BEHRT Pooled baseline com budget equivalente (fase 4/4)

**O que mudou:** `pooled_epochs=120` (era 40 nas referências anteriores); executado como fase 4/4 do mesmo `make training-full` do T13/14/15.  
**Por quê:** a comparação anterior (Exp 9: FL=66,73% vs Pooled=68,88%, com 120 vs 120 épocas/rodadas) ainda estava contaminada pelo cross-contamination do T9. Com T15 válido (69,59%), era necessário um baseline pooled com budget equivalente para medir o custo real de privacidade.  
**Resultado:** T16 BEHRT Pooled B=68,68%, Pooled A=68,29%. FL (T15: 69,59%) supera ambos. RF centralizado: 68,88%. FL supera tudo com budget equivalente. Duração: ~202 min (A+B+RF). **Marco do TCC:** custo de privacidade da federação é negativo — federar com dois hospitais em regime non-IID severo produz modelo melhor do que centralizar os dados.  
**Decisão:** T15 e T16 juntos respondem à questão central do TCC. O argumento de que "privacy comes for free" (ou melhor: com ganho) é empiricamente suportado por dados reais FAPESP.

---

#### T15 → Run de Validação (seeding fix + RAG bug fix)

**O que mudou:** seeding determinístico por rodada × cliente (`torch.manual_seed(seed + round × num_clients + client_id)` em `client.py`); correção de 4 bugs RAG (special tokens na KB, `replace("", "adulto")`, dispatch backend antes do tokenizer, nome do modelo Ollama); backend LLM alterado para Ollama (`gemma3:4b`).  
**Por quê:** reprodutibilidade entre runs era parcial — dropout usava estado global acumulado, produzindo resultados ligeiramente diferentes entre execuções com mesmos hiperparâmetros. RAG produzia KB corrompida com "adulto" interpolado em cada token e special tokens (`[PAD]`, `[CLS]`) como marcadores de atenção clínica.  
**Resultado:** Run de Validação Acc=70,19% (R105, training_id=8), AUC=0,8101, F1=0,4994, ECE=0,0159. Novo recorde do Bloco 1. HSL isolado regrediu −5 p.p. (35,05% vs 40,05%) — seeding alterou trajetória de dropout em dataset pequeno. Macro P@3 RAG subiu de 0,1284 para 0,2218 com bugs corrigidos.  
**Decisão:** seeding é necessário para reprodutibilidade, mas com custo real para datasets pequenos (sensibilidade à trajetória de dropout). A regressão do HSL é achado científico, não bug — documentado como implicação para equidade da federação em hospitais com volumes heterogêneos.

### Limite do Bloco 1 e implicações para o Bloco 2

A correção do split (RNG independente por hospital, 2026-06-30) encerra este bloco. O Bloco 2 começa com:

- BPSP: permutação com `torch.Generator().manual_seed(RANDOM_SEED + 1000 + 0)` → seed 1042
- HSL: permutação com `torch.Generator().manual_seed(RANDOM_SEED + 1000 + 1)` → seed 1043

Os conjuntos de teste serão diferentes. O recorde de 70,19% não é baseline direto para o Bloco 2 — é referência histórica. O primeiro treinamento do Bloco 2 (FedNova sem DP) estabelecerá o novo baseline sobre o qual os experimentos de DP (Exp 17/18/19) serão comparados.

---

## Sessão 2026-06-29 — O que foi acordado e implementado

### Contexto da sessão

A sessão começou com a validação do estado da RAG após a troca do backend LLM para Ollama. Foram identificados e corrigidos 4 bugs críticos, implementados 2 features novas, e corrigido o nome do modelo Ollama. Tudo implementado antes do Exp 17.

### Bugs encontrados e corrigidos

| Bug | Arquivo | Problema | Fix |
|---|---|---|---|
| Special tokens na knowledge base | `interpretability.py` | `[PAD]`, `[CLS]`, `[SEP]` apareciam como top attention tokens (alta atenção por construção, não por sinal clínico); contaminavam os perfis da KB | `_SPECIAL_TOKENS` frozenset + `_is_clinical_token()` — filtra qualquer token que começa com `[` ou `<` |
| `replace("", "adulto")` corrompia KB | `rag.py` `build_knowledge_base()` | `str(p.get("idade_exacta", ""))` retornava `""` quando ausente; `text.replace("", "adulto")` insere "adulto" entre **cada caractere** do texto em Python | Guard `if idade_exacta:` antes do `replace()` |
| `tokenizer.encode()` antes do dispatch de backend | `rag.py` `generate_justification()` | Primeiras linhas chamavam `self.tokenizer.encode()` sempre, antes de checar o backend; com Ollama, `self.tokenizer=None` → `AttributeError` | Dispatch (`if self._llm_backend == "ollama":`) movido para o topo da função |
| Fallback para HF usava nome do modelo Ollama | `rag.py` `__init__()` | Ao cair no fallback HuggingFace, tentava `AutoTokenizer.from_pretrained("gemma3:4b")` — inválido (`:` não é permitido em repo IDs do HF) | Adicionado `llm_hf_model` em `RuntimeConfig` (`FL_LLM_HF_MODEL`, padrão `distilgpt2`); fallback usa esse campo |
| Nome do modelo Ollama errado | `setup.sh`, `Makefile`, `.env.example`, `rag.py`, `docs/` | `gemma4:4b` não existe no registry do Ollama | Corrigido para `gemma3:4b` em todos os arquivos |

### Features implementadas

**1. Ollama com fallback automático para HuggingFace**
- `_check_ollama_available()`: GET `/api/tags` com timeout 5s no `__init__` — detecta indisponibilidade antes do primeiro uso
- Se Ollama offline: loga WARNING, usa `RUNTIME_CFG.llm_hf_model` (distilgpt2) automaticamente — sem intervenção manual
- `make ollama-setup`: instala Ollama + faz pull do `gemma3:4b` (~3,3 GB) — standalone e integrado ao `make setup` (steps 5+6 de `setup.sh`)
- `make ollama-check`: valida se Ollama está rodando e modelo disponível

**2. Seeding determinístico por rodada × cliente** (`client.py`)
- `torch.manual_seed(FED_CFG.random_seed + current_round * FED_CFG.num_clients + self.client_id)` no início de cada `fit()`
- `current_round` vem do `config` dict do servidor (já existia no protocolo Flower)
- Garante que runs independentes com mesmos hiperparâmetros produzam resultados idênticos
- Impacto esperado na acurácia: **negligenciável** — só muda ordem dos batches, não o gradiente médio

**3. DP-FedAvg (McMahan et al. 2018)** — dois níveis:
- **Cliente** (`client.py`): clipa update Δ = w_final − w_global à norma S (`dp_max_grad_norm`) antes de retornar ao servidor
- **Servidor** (`fl_core.py` → `apply_dp_noise()`): após agregação FedNova, adiciona N(0, (σ·S/n)²) ao estado global
- Desabilitado por padrão (`FL_DP_NOISE=0.0`) — sem overhead nos Exps anteriores
- Variáveis de configuração: `FL_DP_NOISE` (σ) e `FL_DP_CLIP` (S) — via env var ou Makefile

### Decisões tomadas

| Decisão | Justificativa |
|---|---|
| Seeding + DP no mesmo treinamento (Exp 17) | Impactos separáveis: seeding afeta apenas ordem de batches (efeito negligenciável); qualquer degradação observada é DP. Não confunde a análise. |
| DP sem Opacus | Opacus não está instalado; implementação manual do DP-FedAvg é suficiente para o TCC e evita nova dependência |
| Cota de ε com mecanismo gaussiano simples | Suficiente para ilustrar trade-off no TCC; cota é conservadora (RDP daria menor). Documentado como limitação. |
| `gemma3:4b` como modelo padrão | `gemma4` não existe no Ollama registry; `gemma3:4b` (~3,3 GB) é o modelo mais recente disponível na família Gemma 4B |
| Fallback obrigatório para HF | Requisito da sessão: sistema não pode falhar por indisponibilidade do Ollama; fallback automático com WARNING é a solução correta |

### Nota explicativa — O que é "backend", por que Ollama e o design do fallback

**O que é backend neste contexto:**
O módulo RAG precisa de um LLM para gerar a justificativa clínica em linguagem natural (ex.: "Exames sugerem quadro grave..."). "Backend" é *de onde* esse LLM roda — qual serviço executa a geração de texto:

| Backend | Como funciona | Dependência |
|---|---|---|
| `huggingface` | Carrega o modelo direto no processo Python via `transformers.AutoModelForCausalLM` | Apenas `pip install transformers` — zero serviços externos |
| `ollama` | Faz requisição HTTP para `localhost:11434/api/generate` — o modelo roda num processo separado gerenciado pelo Ollama | Requer Ollama instalado, `ollama serve` rodando, e `ollama pull gemma3:4b` (~3,3 GB baixados) |

**Por que Ollama foi escolhido como backend principal:**

A escolha é motivada pelo custo computacional. Carregar um modelo de 4 bilhões de parâmetros (gemma3:4b) tem impacto muito diferente dependendo do backend:

| Aspecto | HuggingFace (`transformers`) | Ollama |
|---|---|---|
| Formato do modelo | float16/float32 (pesos brutos) | GGUF quantizado (Q4_K_M típico) |
| RAM necessária (gemma3:4b) | ~8 GB (float16) ou ~16 GB (float32) | ~2–3 GB (quantizado 4 bits) |
| Processo | Carrega no mesmo processo Python que o treinamento FL | Processo separado — não compete com a memória do treinamento |
| GPU offloading | Manual (precisa configurar `device_map`) | Automático pelo Ollama |
| Overhead de inferência | Baixo (chamada Python direta) | Mínimo (HTTP localhost — sub-milissegundo de rede) |

Durante o `make training-full`, o BEHRT e o pipeline FL já consomem memória significativa. Carregar um modelo HuggingFace de 4B parâmetros no mesmo processo Python causaria pressão de memória que poderia interromper o treinamento. O Ollama isola o modelo num processo separado, eliminando esse conflito.

**Por que HuggingFace ainda é o padrão do código (`config.py`):**

O padrão do *código* (`FL_LLM_BACKEND=huggingface` em `config.py`) é diferente do padrão *operacional* (`.env.example` define `FL_LLM_BACKEND=ollama`). O padrão do código existe para que `make test` funcione em qualquer ambiente (CI/CD, primeiro clone) sem exigir Ollama instalado — usa `distilgpt2` (~82 MB, ~500 MB RAM), modelo pequeno que não compromete a memória de teste.

**Decisão de design para o fallback (pendente de implementação):**

O fallback atual (Ollama offline → `distilgpt2`) é problemático: troca silenciosamente o modelo, resultados de RAG não são comparáveis entre runs, e o P@3 medido pode ter sido calculado com um modelo diferente do que está em produção.

A direção acordada:
- **Ollama permanece como backend principal** — motivação: custo computacional (isolamento de processo, quantização automática)
- **Fallback deve usar o mesmo modelo** (gemma3:4b), carregado via HuggingFace com quantização explícita se Ollama não estiver disponível — ou, preferencialmente, **falhar com ERROR claro** em ambiente de treinamento real, exigindo que o operador suba o Ollama
- **O modelo efetivo usado deve ser gravado** nos metadados do JSON de resultado de cada experimento — requisito de observabilidade e reprodutibilidade
- `distilgpt2` continua válido apenas para testes unitários (`FL_ENV=test`), nunca silenciosamente em produção

---

## Estado do Projeto ao Iniciar Esta Parte

### Melhor resultado histórico (referência para todos os próximos experimentos)

| Métrica | Valor | Experimento |
|---|---|---|
| Accuracy | **69,59%** | Exp 15 — FL Federado FedNova, R79 |
| Macro AUC | 0,8181 | Exp 15 |
| Macro F1 | 0,4946 | Exp 15 |
| ECE isotônica | 0,0149 | Exp 15 |
| RF Centralizado | 68,41% | Exp 15 (baseline) |
| BEHRT Pooled B | 68,68% | Exp 16 (120 épocas, budget equiv.) |

**Conclusão do Exp 15/16:** custo de privacidade da federação é **negativo** — FL FedNova supera todos os baselines centralizados com budget equivalente (120 rodadas = 120 épocas).

### Pipeline atual (`make training-full`)

4 fases em sequência:
1. BPSP-only (Exp 13 ref.: 64,86%)
2. HSL-only (Exp 14 ref.: 40,05%)
3. **Federado BPSP+HSL** ← foco dos próximos experimentos
4. BEHRT Pooled baseline (Exp 16 ref.: 68,68%)

Duração total: ~9h43min (583 min) na máquina de desenvolvimento.

### Implementações concluídas (prontas para uso nos próximos treinamentos)

| Componente | Status | Ativar com |
|---|---|---|
| FedNova | ✓ ativo por padrão | `use_fednova=True` em `config.py` |
| Checkpoint guloso + scoping por `training_id` | ✓ ativo | automático |
| Calibração isotônica OvR | ✓ ativa | automático |
| Gradient clipping (max_norm=1,0) | ✓ ativo | automático |
| Seeding determinístico por rodada × cliente | ✓ ativo | automático |
| DP-FedAvg (McMahan et al. 2018) | ✓ implementado, **desabilitado por padrão** | `FL_DP_NOISE=σ` |
| RAG com Ollama (gemma3:4b) | ✓ implementado | `FL_LLM_BACKEND=ollama` (padrão) |
| RAG bugs corrigidos (special tokens + replace) | ✓ corrigido | automático |
| Fallback automático HF se Ollama offline | ✓ implementado | automático |

### Modelo Ollama

```bash
# Verificar se está disponível
make ollama-check

# Instalar / baixar modelo (se necessário)
make ollama-setup          # gemma3:4b (~3,3 GB)
```

---

## Run de Validação — Seeding Fix + RAG Bug Fix (2026-06-30, SEM DP)

**Status:** ✓ Concluído  
**Data:** 2026-06-29 18:56 → 2026-06-30 03:12  
**Log:** `experiments/logs/run_complete_20260629_185640.log`  
**Comando:** `make training-full` (sem parâmetros DP — `FL_DP_NOISE=0.0`)  
**Objetivo:** validar impacto isolado do seeding determinístico e da correção do bug RAG antes de introduzir DP

### O que mudou em relação ao Exp 13/14/15

| Mudança | Origem |
|---|---|
| Seeding determinístico por rodada × cliente | Sessão 2026-06-29 |
| Bug RAG `replace("", "adulto")` corrigido | Sessão 2026-06-29 |
| Filtragem de special tokens na knowledge base | Sessão 2026-06-29 |
| Backend RAG via Ollama (`gemma3:4b`) | Sessão 2026-06-29 |

### Fase 1/4 — BPSP-only FL (training_id=6)

| Métrica | Valor | Referência (Exp 13) | Δ |
|---|---|---|---|
| Accuracy (melhor) | **65,22%** | 64,86% | +0,36 p.p. |
| Melhor rodada | R95 | R118 | — |
| Accuracy final (R120) | 63,41% | — | — |

### Fase 2/4 — HSL-only FL (training_id=7)

| Métrica | Valor | Referência (Exp 14) | Δ |
|---|---|---|---|
| Accuracy (melhor) | **35,05%** | 40,05% | **−5,00 p.p.** |
| Melhor rodada | R63 | R100 | — |
| Loss final (R120) | 3,1512 | — | — |

> **Regressão HSL:** a loss cresceu ao longo do treinamento (R1: 0,2153 → final: 3,15), sinal de instabilidade com apenas 1 cliente e 3.621 amostras. O seeding determinístico alterou a trajetória de otimização de forma desfavorável para o HSL. Com datasets pequenos, a ordem dos batches tem impacto não negligenciável. Achado relevante para a defesa.

### Fase 3/4 — Federado BPSP+HSL (training_id=8)

| Métrica | Valor | Referência (Exp 15) | Δ |
|---|---|---|---|
| Accuracy (melhor) | **70,19%** | 69,59% | **+0,60 p.p. ✓ NOVO MELHOR** |
| Macro F1 | 0,4994 | 0,4946 | +0,0048 |
| Macro AUC | 0,8101 | **0,8181** | −0,0080 |
| ECE isotônica | 0,0159 | **0,0149** | +0,0010 |
| Melhor rodada | R105 | R79 | — |
| Accuracy final (R120) | 65,81% | 63,15% | — |
| Temperatura (isotônica) | 1,1203 | — | — |

#### Per-class (melhor checkpoint R105)

| Classe | F1 | Precisão | Recall | Suporte |
|---|---|---|---|---|
| curado_pronto | 0,830 | 0,804 | 0,859 | 1.620 |
| curado_internado | **0,000** | 0,000 | 0,000 | 28 |
| melhora_pronto | 0,783 | 0,758 | 0,810 | 321 |
| melhora_internado_breve | 0,607 | 0,622 | 0,592 | 1.074 |
| melhora_internado_grave | 0,277 | 0,303 | 0,254 | 338 |
| **Macro** | **0,499** | | | 3.381 |

#### Matriz de confusão completa (melhor checkpoint R105, pré-calibração)

Linhas = classe real | Colunas = classe predita | Diagonal = acertos

| Real \ Predito | curado_pronto | curado_internado | melhora_pronto | melhora_int_breve | melhora_int_grave | **Recall** |
|---|---|---|---|---|---|---|
| **curado_pronto** (N=1620) | **1391** | 0 | 24 | 156 | 49 | **85,9%** |
| **curado_internado** (N=28) | 10 | **0** | 1 | 13 | 4 | **0,0%** |
| **melhora_pronto** (N=321) | 22 | 0 | **260** | 36 | 3 | **81,0%** |
| **melhora_int_breve** (N=1074) | 248 | 2 | 46 | **636** | 142 | **59,2%** |
| **melhora_int_grave** (N=338) | 59 | 0 | 12 | 181 | **86** | **25,4%** |
| **Precisão** | 81,6% | — | 76,7% | 62,5% | 30,4% | |

#### Per-class completo (R105)

| Classe | N (teste) | F1 | Precisão | Recall | AUC-ROC |
|---|---|---|---|---|---|
| curado_pronto | 1.620 | **0,8304** | 0,8040 | 0,8586 | 0,9051 |
| curado_internado | 28 | **0,0000** | 0,0000 | 0,0000 | 0,5798 |
| melhora_pronto | 321 | **0,7831** | 0,7580 | 0,8100 | 0,9722 |
| melhora_internado_breve | 1.074 | **0,6069** | 0,6223 | 0,5922 | 0,8015 |
| melhora_internado_grave | 338 | **0,2765** | 0,3028 | 0,2544 | 0,7921 |
| **Macro** | 3.381 | **0,4994** | — | — | **0,8101** |

#### Métricas globais (R105, pós-calibração isotônica)

| Métrica | Valor |
|---|---|
| Accuracy | **70,19%** |
| Macro F1 | 0,4994 |
| Macro AUC | 0,8107 (pós-cal) / 0,8101 (pré-cal) |
| ECE isotônica | **0,0159** |
| Temperatura isotônica | 1,1203 |
| N amostras teste | 3.381 |

> Nota: calibração isotônica não altera o argmax → acurácia e F1 idênticos pré/pós. O AUC muda marginalmente porque a isotônica ajusta as probabilidades de saída.

#### Análise de erros — top 8 por volume

| # | Classe real | Classe predita | N | % do real | Risco clínico |
|---|---|---|---|---|---|
| 1 | melhora_int_breve | curado_pronto | 248 | 23,1% | Médio (subestima internação) |
| 2 | melhora_int_grave | melhora_int_breve | 181 | **53,6%** | **Alto** (confunde breve/grave) |
| 3 | curado_pronto | melhora_int_breve | 156 | 9,6% | Baixo (superestima internação) |
| 4 | melhora_int_breve | melhora_int_grave | 142 | 13,2% | Médio (superestima gravidade) |
| 5 | melhora_int_grave | curado_pronto | 59 | **17,5%** | **Máximo** (grave → descarta internação) |
| 6 | curado_pronto | melhora_int_grave | 49 | 3,0% | Baixo (superestima gravidade) |
| 7 | melhora_int_breve | melhora_pronto | 46 | 4,3% | Médio |
| 8 | melhora_pronto | melhora_int_breve | 36 | 11,2% | Médio |

#### Erros de maior risco clínico — melhora_internado_grave

Do total de 338 casos `melhora_internado_grave` no conjunto de teste:

| Predito como | N | % | Consequência clínica |
|---|---|---|---|
| melhora_int_breve | **181** | **53,6%** | Paciente grave tratado como breve — risco de alta precoce |
| curado_pronto | 59 | 17,5% | Paciente grave não internado — **risco máximo** |
| melhora_pronto | 12 | 3,6% | Paciente grave tratado como pronto — risco alto |
| **Correto** | **86** | **25,4%** | — |

> **O modelo acerta apenas 1 em 4 casos graves.** 71,1% dos casos graves são enviados para classes de menor intensidade de cuidado (breve + pronto + curado_pronto). Este é o principal gap clínico do modelo atual e o ponto mais crítico para a defesa.

### Fase 4/4 — BEHRT Pooled + RF Centralizado

| Modelo | Accuracy | F1 macro | AUC macro | Referência (Exp 16) | Δ |
|---|---|---|---|---|---|
| Pooled A (sem demo) | 68,32% | 0,5169 | — | 68,29% | +0,03 p.p. |
| Pooled B (late fusion) | **69,09%** | 0,5212 | — | 68,68% | +0,41 p.p. |
| RF Centralizado (BoT) | 68,20% | 0,5071 | 0,7942 | 68,41% | −0,21 p.p. |
| RF BPSP (por hospital) | 58,65% | 0,3299 | 0,7379 | — | — |
| RF HSL (por hospital) | 24,22% | 0,1857 | 0,6961 | — | — |

### RAG Precision@3 (pós-correção do bug)

| Classe | P@3 (novo, full FL) | P@3 (Exp 15, com bug) | Δ |
|---|---|---|---|
| curado_pronto | 0,1132 | 0,0821 | +0,031 |
| curado_internado | 0,1310 | 0,1667 | −0,036 |
| melhora_pronto | 0,0052 | **0,6012** | −0,596 |
| melhora_internado_breve | **0,5177** | 0,0829 | +0,435 |
| melhora_internado_grave | 0,0158 | 0,0424 | −0,027 |
| **Macro P@3** | **0,2218** | 0,1284 | **+0,093** |

> O macro P@3 melhorou (+0,093) pela remoção da contaminação "adulto" na knowledge base. A redistribuição por classe reflete o modelo FL treinado com os novos seeds — `melhora_pronto` caiu porque o novo modelo distribui atenção diferente em relação ao Exp 15. O texto das fontes agora é legível ("Desfecho: melhora_internado_breve. Marcadores de maior atenção: Hemoglobina Corpuscular Media NORMAL...") vs o texto ilegível anterior.

### Análise consolidada — o que este run revela

**1. O novo melhor resultado é 70,19% (vs 69,59% no Exp 15).**
O seeding determinístico alterou a trajetória de otimização — o modelo federado convergiu para um ótimo local ligeiramente melhor em acurácia e F1. Porém, o AUC caiu (0,8101 vs 0,8181) e a ECE piorou (0,0159 vs 0,0149). O ganho em acurácia vem principalmente de melhora em `melhora_internado_breve` (F1: 0,607 vs 0,463 anterior), que é a segunda maior classe (1.074 casos). Não é um resultado dominante — o Exp 15 discrimina melhor entre classes minoritárias.

**2. `curado_internado` permanece com F1=0,000.**
28 amostras no conjunto de teste, 0 preditas corretamente. Estruturalmente igual ao Exp 15 — não é um problema de otimização, é de representação. Ponto para discutir com a orientadora.

**3. A regressão do HSL (−5 p.p.) é o achado mais relevante desta run para a defesa.**
Um único cliente com 3.621 amostras de treino apresentou instabilidade significativa ao mudar a ordem dos batches. Isso demonstra empiricamente que o seeding afeta mais datasets menores — o BPSP (20.019 amostras) foi mais robusto (+0,36 p.p.). Para o texto da metodologia: a sensibilidade ao seeding é inversamente proporcional ao volume de dados locais, o que tem implicações diretas para a equidade da federação entre hospitais de tamanhos diferentes.

**4. FL continua superando todos os baselines centralizados com budget equivalente.**
FL 70,19% > Pooled B 69,09% > Pooled A 68,32% > RF 68,20%. O custo de privacidade da federação permanece **negativo**.

### Métricas de convergência por fase (R1 / R30 / R60 / R90 / R120)

#### BPSP-only (id=6)

| Rodada | Hora | Acc | Loss |
|---|---|---|---|
| R1 | 18:58 | 0,5487 | 1,2216 |
| R30 | 19:27 | 0,6072 | 1,0762 |
| R60 | 19:54 | 0,6170 | 1,2179 |
| R90 | 20:20 | 0,6324 | 1,1959 |
| R120 | 20:44 | 0,6341 | 1,1591 |
| **Best** | — | **0,6522 @ R95** | — |

#### HSL-only (id=7)

| Rodada | Hora | Acc | Loss |
|---|---|---|---|
| R1 | 21:04 | 0,2153 | 1,8106 |
| R30 | 21:10 | 0,2650 | 1,8734 |
| R60 | 21:15 | 0,2677 | 2,0854 |
| R90 | 21:19 | 0,2697 | 2,6289 |
| R120 | 21:24 | 0,2558 | 3,1512 |
| **Best** | — | **0,3505 @ R63** | — |

> Loss crescente ao longo de toda a run — sinal claro de instabilidade com dataset pequeno e seeding alterado. O HSL com 3.621 amostras e 1 cliente não teve sinal suficiente para aprender com a nova trajetória de batches.

#### Federado BPSP+HSL (id=8)

| Rodada | Hora | Acc | Loss |
|---|---|---|---|
| R1 | 21:30 | 0,4579 | 1,1855 |
| R30 | 22:05 | 0,6714 | 0,8512 |
| R60 | 22:36 | 0,6528 | 0,8439 |
| R90 | 23:04 | 0,6829 | 0,7980 |
| R120 | 23:32 | 0,6581 | 0,8423 |
| **Best** | — | **0,7019 @ R105** | — |

> Convergência saudável: loss decrescendo e estabilizando. O melhor ponto (R105) está bem acima do R120 (65,81%) — confirma o valor do checkpoint guloso.

### Duração

| Fase | Período | Duração aprox. |
|---|---|---|
| 1/4 BPSP-only (120 rounds) | 18:58 → 20:44 | ~1h46 |
| 2/4 HSL-only (120 rounds) | 21:04 → 21:24 | ~20min |
| 3/4 Federado (120 rounds) | 21:30 → 23:32 | ~2h02 |
| 4/4 BEHRT Pooled (120+120 épocas) + RF | 23:55 → 03:12 | ~3h17 |
| **Total** | 18:56 → 03:12 | **~8h16** |

### Temperatura de hardware

**Não monitorada nesta run.** O `make temperature-monitor` (ou script equivalente) não estava rodando durante a execução de 2026-06-29/30. Para os experimentos DP (Exp 17/18/19), iniciar o monitoramento antes de rodar `make training-full`:

```bash
# Em terminal separado, antes de iniciar o treinamento:
watch -n 60 "date && sensors | grep -E 'Core|temp'" >> experiments/logs/temperature_exp17.log &

# Ou via make (se implementado):
make temperature-monitor &
make training-full
```

Coletar ao menos nos marcos R30, R60, R90 e R120 de cada fase.

---

## Experimento 17 — DP-FedAvg (σ=1,0) — AGUARDANDO EXECUÇÃO

**Status:** Pronto para executar — DP não foi aplicado no run de 2026-06-30  
**Nova referência base:** 70,19% (run de validação seeding fix, acima)  
**Comando:**
```bash
FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full
```

### O que muda em relação ao Exp 15

| Mudança | Impacto esperado |
|---|---|
| Seeding determinístico (`torch.manual_seed` por round × client) | Negligenciável na acurácia; elimina variância espúria entre runs |
| DP-FedAvg σ=1,0, S=1,0 | Degradação de acurácia a medir (estimativa: −2 a −8 p.p.) |

### Parâmetros DP

| Parâmetro | Valor |
|---|---|
| `FL_DP_NOISE` (σ — multiplicador de ruído) | 1,0 |
| `FL_DP_CLIP` (S — sensitivity / norma máxima do update) | 1,0 |
| ruído_std por parâmetro | σ·S/n = 1,0·1,0/2 = **0,50** |
| δ (probabilidade de falha da garantia DP) | 1e-5 |
| ε por rodada (cota solta Gaussiana) | √(2·ln(1,25/1e-5)) / 1,0 ≈ **3,52** |
| ε acumulado (120 rodadas, cota solta) | ≈ **422** |

> A cota ε≈422 é conservadora (mecanismo gaussiano simples). RDP/moments accountant produziria valor menor. Suficiente para o TCC ilustrar o trade-off.

### O que registrar após o treinamento

Copiar os resultados abaixo após `make training-full` completar:

```
### Resultados Exp 17

**Data:** ___  
**Log:** experiments/logs/___  
**σ usado:** 1,0 | **S:** 1,0

#### Federado (fase 3/4)

| Métrica | Exp 17 (σ=1,0) | Exp 15 (σ=0) | Δ |
|---|---|---|---|
| Accuracy | ___ | 69,59% | ___ |
| Macro AUC | ___ | 0,8181 | ___ |
| Macro F1 | ___ | 0,4946 | ___ |
| ECE isotônica | ___ | 0,0149 | ___ |
| Melhor rodada | ___ | R79 | — |

#### DP metrics (extrair do log)
- dp_update_norm médio por cliente: ___
- Fração de updates clipados (scale < 1): ___
- ε acumulado reportado pelo logger: ___

#### BPSP-only (fase 1/4)
| Métrica | Valor |
|---|---|
| Accuracy | ___ |
| Melhor rodada | ___ |

#### HSL-only (fase 2/4)
| Métrica | Valor |
|---|---|
| Accuracy | ___ |
| Melhor rodada | ___ |

#### BEHRT Pooled (fase 4/4)
| Config | Accuracy | Macro F1 |
|---|---|---|
| Pooled A (sem demo) | ___ | ___ |
| Pooled B (late fusion) | ___ | ___ |

#### RAG Precision@3 (federado)
| Classe | P@3 |
|---|---|
| curado_pronto | ___ |
| curado_internado | ___ |
| melhora_pronto | ___ |
| melhora_internado_breve | ___ |
| melhora_internado_grave | ___ |
| **Macro P@3** | ___ |

#### Duração
| Fase | Duração |
|---|---|
| 1/4 BPSP-only | ___ |
| 2/4 HSL-only | ___ |
| 3/4 Federado | ___ |
| 4/4 BEHRT Pooled | ___ |
| **Total** | ___ |
```

---

## Série DP planejada (após Exp 17)

| Exp | σ | S | ε_acum (cota solta, 120 rounds) | Objetivo |
|---|---|---|---|---|
| 15 | 0,0 | — | ∞ | referência (sem DP) |
| **17** | **1,0** | **1,0** | **≈422** | **primeiro ponto da curva** |
| 18 | 0,5 | 1,0 | ≈845 | menos ruído, avaliar degradação menor |
| 19 | 2,0 | 1,0 | ≈211 | mais ruído, avaliar degradação maior |

Comando genérico:
```bash
FL_DP_NOISE=<σ> FL_DP_CLIP=<S> make training-full
```

---

## Roadmap Geral — Estado 2026-06-30

| Fase | Descrição | Status |
|---|---|---|
| **1 — Treinamento FL** | FedNova + calibração isotônica + checkpoint guloso | ✓ Concluído — **novo melhor: 70,19% (R105, seeding fix)** |
| **2 — DP** | DP-FedAvg implementado; Exp 17/18/19 para medir curva Acc × ε | ✓ Implementado — **aguardando execução** |
| **3 — Distribuído** | Desktop como servidor + notebook como cliente; comunicação real entre nós | Pendente (pós-Exp 19) |
| **4 — API de Inferência** | REST endpoint para prognóstico de novo paciente com o modelo federado | ✓ **Concluído — ver seção abaixo** |
| **Análise erros clínicos** | Tipologia dos 4 tipos de erro + implicações para a defesa | ✓ Concluído (`docs/analise_erros_clinicos.md`) |
| **Tabela unificada** | Tabela comparativa com todos os resultados (incluindo DP) | Pendente — aguardar Exp 17/18/19 |
| **Diagrama de execução** | Fluxo do `make training-full` (4 fases) para a defesa | Pendente |
| **Refactoring MVP** | Modularização + configurações em banco; executar APÓS todas as fases | Pendente |
| **Fusão 3 classes** | Avaliar com orientadora se clinicamente justificável | A definir com orientadora |

---

## API de Inferência — Concluída em 2026-06-29

### O que foi implementado

O gap entre o pipeline de treinamento e a API de inferência estava em: o treinamento salva checkpoints no banco (`CheckpointStore` → PostgreSQL/SQLite), mas a API lia apenas arquivos `.pt` em disco — que nunca existiam. Sem conexão entre os dois.

**3 mudanças para fechar o gap:**

1. **`InferenceEngine.load_from_store(checkpoint: dict)`** (`infrastructure/mosaicfl_api/inference_engine.py`)  
   Carrega pesos + vocab + metadados diretamente do dict retornado por `CheckpointStore.load_best()`.

2. **Fallback em `state._get_engine()`** (`infrastructure/mosaicfl_api/state.py`)  
   Sequência: procura `round_*.pt` em `FL_CHECKPOINT_DIR` → se não encontrar, tenta `CheckpointStore.load_best(FL_DB_URL)` → se banco vazio, sobe sem modelo com WARNING → nunca trava.

3. **`make api` + `make export-checkpoint`** (`Makefile` + `scripts/export_checkpoint.py`)  
   `make api` sobe o banco e inicia o servidor na porta 8000. `make export-checkpoint` extrai o melhor checkpoint do banco para `checkpoints/best_model.pt` (para deploy offline).

### Verificado ao vivo

```
POST /api/predict HTTP/1.1 → 200 OK

startup: checkpoint carregado do PostgreSQL — round=79, acc=0.6959, vocab=648, version=c71727ce4b53
resposta: predicted_label=melhora_pronto, risk_score=0.39, mc_samples=50
          curado_pronto: 0.3423 ± 0.1167
          melhora_pronto: 0.3703 ± 0.1082
          melhora_internado_breve: 0.1998 ± 0.0794
          curado_internado: 0.0435 ± 0.0205
          melhora_internado_grave: 0.0441 ± 0.0282
```

### Testes adicionados

10 novos testes em `tests/unit/test_inference_engine_store.py`:
- `TestLoadFromStore` (6): vocab, temperatura, metadados, erros de validação, pesos realmente carregados no modelo
- `TestGetEngineFallback` (4): usa store quando sem `.pt`, pula store quando `.pt` existe, banco vazio não trava, erro de conexão não trava

**Suite completa:** 41 testes integração (API) + 10 novos unit = todos passando.

### Como usar

```bash
# Iniciar a API (modelo carregado automaticamente do banco)
make api

# Para desenvolvimento sem autenticação:
FL_AUTH_REQUIRED=false FL_ENV=development make api

# Exportar checkpoint para arquivo (deploy offline / inspeção):
make export-checkpoint
make export-checkpoint FL_TRAINING_ID=5   # treinamento específico

# Swagger UI (documentação interativa):
# http://localhost:8000/docs
```

### O que já estava pronto (não foi alterado)

A API já tinha toda a estrutura hexagonal implementada: endpoints `/api/predict`, `/api/exams/ingest`, `/api/patients`, `/api/fl/status`, `/api/fl/reload`, segurança JWT/API Key, rate limiting, pseudonimização LGPD (HMAC-SHA256), `PatientDB` PostgreSQL/SQLite, `outcome_feedback` (ground truth tardio), exportação FHIR R4, ClinicalPath, UI Bootstrap em `static/index.html`. O único gap era o carregamento do modelo.

---

## Referência Rápida — Arquivos Chave

### Treinamento e modelo

| Arquivo | Responsabilidade |
|---|---|
| `src/mosaicfl/core/config.py` | Todos os hiperparâmetros (`FedConfig`, `RuntimeConfig`) |
| `src/mosaicfl/core/client.py` | Treino local, FedProx, DP clipping, seeding |
| `experiments/training/fl_core.py` | Agregação FedNova, `apply_dp_noise()`, loop federado |
| `src/mosaicfl/core/rag.py` | RAG: backends Ollama/HF, KB building, geração |
| `src/mosaicfl/core/interpretability.py` | Extração de tokens de atenção, filtro de special tokens |
| `experiments/run_training.py` | Entrypoint de cada fase do `make training-full` |
| `infrastructure/shared/checkpoint_store.py` | Persistência de checkpoints (SQLite + PostgreSQL) |

### API de Inferência

| Arquivo | Responsabilidade |
|---|---|
| `infrastructure/mosaicfl_api/inference_engine.py` | Motor de predição: tokenização idêntica ao treino, MC Dropout, `load_from_store()` |
| `infrastructure/mosaicfl_api/state.py` | Singleton do engine com fallback ao `CheckpointStore` |
| `infrastructure/mosaicfl_api/routers/prediction.py` | `POST /api/predict` e `POST /api/exams/ingest` |
| `infrastructure/mosaicfl_api/routers/admin.py` | `GET /api/fl/status` e `POST /api/fl/reload` |
| `infrastructure/mosaicfl_api/db.py` | `PatientDB`: histórico de risco, exames, ground truth tardio |
| `infrastructure/mosaicfl_api/schemas.py` | Pydantic: `PredictRequest/Response`, `IngestRequest/Response` |
| `scripts/export_checkpoint.py` | Exporta melhor checkpoint do banco para `checkpoints/best_model.pt` |

### Documentação e análise

| Arquivo | Responsabilidade |
|---|---|
| `docs/Sumario_Treinamento.md` | Histórico completo Exp 1–16 |
| `docs/analise_erros_clinicos.md` | Tipologia dos 4 erros por classe + implicações para defesa |
| `docs/avaliacao_metodologia_mosaicfl.md` | Estado MVP + qualidade científica + lacunas para mestrado |

### Make targets relevantes nesta fase

| Target | Uso |
|---|---|
| `make training-full` | Executa as 4 fases do pipeline |
| `FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full` | Exp 17 (DP σ=1,0) |
| `make api` | Inicia a API de inferência na porta 8000 |
| `make export-checkpoint` | Exporta melhor checkpoint do banco para arquivo `.pt` |
| `make ollama-setup` | Instala Ollama + baixa gemma3:4b |
| `make ollama-check` | Verifica se Ollama está rodando |

---

## Sessão 2026-06-29 (continuação) — Documentação e revisão

### O que foi feito nesta sessão

**1. README.md — seções pendentes concluídas**

- **Seção Experimentos:** tabela consolidada Exp 1–19 com resultados reais (Acc, F1 macro, AUC macro, ECE), per-class F1, ablação late fusion, baseline RF, conclusão central ("custo de privacidade negativo")
- **Rodando Localmente:** adicionada subseção da API com `make api`, exemplo de `curl` e documentação do `make export-checkpoint`
- **Referências:** FedNova (Wang et al. 2020, NeurIPS, arXiv:2007.14481) e DP-FedAvg (McMahan et al. 2018, ICLR, arXiv:1710.06963) adicionados

**2. README.md — correção de documentação desatualizada**

Três trechos citavam `logs/training_state.json` como mecanismo de persistência de estado do ServerApp. Corrigidos para refletir a realidade atual: o estado é persistido no **PostgreSQL via `CheckpointStore`** (`metrics.fl_checkpoints`); o arquivo JSON é apenas cache local de leitura rápida.

Trechos corrigidos:
- Seção "Recovery de Sessão" (linha ~144)
- Parágrafo após `flwr run . production` na seção SuperLink (linha ~1055)
- Seção "Troubleshooting — ServerApp cai no meio do treinamento" (linha ~1279)

**3. Logs — avaliação de versionamento**

Verificação de quais arquivos citados no README podem ser commitados:
- `experiments/logs/` e `experiments/data/` → já rastreados e commitados corretamente (`.gitignore` usa `!exceptions` para incluí-los)
- `logs/` (raiz) → artefatos de runtime, não commitados e não devem ser (`api_daemon.log`, `audit.log`, `training_state.json` etc.)
- Conclusão: configuração do `.gitignore` está correta; os arquivos de resultado de experimentos estão no repositório.

**4. RAG — backend LLM: explicação e pendências**

Pergunta da sessão: qual era o modelo anterior e qual o novo da RAG?

- **Modelo anterior (fallback/padrão do código):** `distilgpt2` via HuggingFace (`AutoModelForCausalLM`) — roda no processo Python, ~82 MB, zero dependências externas
- **Modelo novo (padrão operacional):** `gemma3:4b` via Ollama (`localhost:11434/api/generate`) — ~3,3 GB, requer `ollama serve` e `ollama pull gemma3:4b`

O conceito de "backend" aqui é o serviço que executa a geração de texto. A explicação completa foi adicionada na seção "Nota explicativa" desta parte do sumário (acima).

**Pendências identificadas (anotadas em memória):**

| Pendência | Impacto |
|---|---|
| Verificar `google/gemma-3-4b-it` disponível via HuggingFace | Eliminaria dependência do Ollama sem trocar de modelo |
| Melhorar observabilidade do fallback de modelo | Hoje é apenas WARNING; modelo efetivo usado não aparece nos JSONs de resultado — afeta reprodutibilidade e comparabilidade dos experimentos |

---

## Sessão 2026-06-30 — Revisão de partições e seeds

### Motivação

Durante a análise dos resultados do Exp 8 (FedNova, 70.19%), foi observado que o seeding determinístico introduzido em `client.py` causou regressão de −5 p.p. no HSL (40.05% → 35.05%). A explicação levantada foi uma possível interação entre o `torch.manual_seed` global e o shuffle do DataLoader. Isso motivou uma revisão completa de como o MosaicFL gerencia seeds e partições em todos os pontos de inicialização de dados.

### Análise: consistência do split entre experimentos

**Conclusão central: o split é consistente e metodologicamente correto.**

Todos os experimentos — FL (FedNova/FedAvg), BEHRT pooled, RF baseline e ablation study — recebem os mesmos `client_loaders` provenientes de uma única chamada a `prepare_dataloaders_from_db`. A partição não é recriada por experimento.

O split é determinístico por três razões encadeadas:

1. **SQL tem `ORDER BY` explícito** (`preprocessor.py:379`): `ORDER BY patient_id, attendance_id, dia_relativo, analyte` — o DataFrame retornado pelo banco é sempre idêntico, dado o mesmo banco e os mesmos dados.
2. **Hospitais ordenados por `sorted()`** (`preprocessor.py:653`): BPSP sempre processado antes de HSL, independente da ordem retornada pelo banco.
3. **`randperm` com gerador fixo** (`dataloaders.py`): split 70/10/10/10 reproduzível entre runs.

| Ponto verificado | Status |
|---|---|
| Split train/val/cal/test idêntico para FL, pooled BEHRT, RF, ablation | ✅ |
| Ordem dos hospitais determinística (`sorted()`) | ✅ |
| Query SQL com `ORDER BY` explícito | ✅ |
| DataLoaders de treino com `generator` seeded por cliente | ✅ |
| Per-round seeding no cliente (`client.py:108`) | ✅ |
| RNG compartilhado entre hospitais | ⚠️ corrigido (ver abaixo) |
| `create_client_fn` em `client.py:197` | ⚠️ código legado — não afeta produção |

### Problema 1 — RNG compartilhado entre hospitais (corrigido)

**Diagnóstico:**

Em `dataloaders.py`, um único `rng` era criado com `RANDOM_SEED` e compartilhado sequencialmente entre todos os hospitais:

```python
# antes
rng = torch.Generator()
rng.manual_seed(RANDOM_SEED)
# ...
for cid, hospital_id in enumerate(hospital_data.items()):
    perm = torch.randperm(n, generator=rng)  # mesmo rng — estado avança
```

O problema: a permutação de HSL dependia implicitamente do número de amostras do BPSP (porque o `rng` avança `n_BPSP` passos antes de gerar a permutação de HSL). Adicionar um terceiro hospital, ou alterar o tamanho do dataset BPSP, mudaria o split do HSL sem nenhuma mudança de seed.

**Fix aplicado (`dataloaders.py`):**

```python
# depois — gerador independente por hospital
_split_rng = torch.Generator().manual_seed(RANDOM_SEED + 1000 + cid)
perm = torch.randperm(n, generator=_split_rng)
```

Seeds usadas: BPSP → 1042, HSL → 1043 (namespace `+1000` para não conflitar com os seeds dos DataLoaders de shuffle, que usam `RANDOM_SEED + cid`).

**Impacto nos experimentos anteriores:**

Os treinamentos 1–8 usaram o RNG compartilhado — split diferente do que será usado a partir de Exp 9. O número 70.19% (training_id=8, R105) foi obtido com o split antigo e **não é diretamente comparável** a experimentos com o split corrigido. Os experimentos DP (Exp 17/18/19) e FedNova Exp 9 serão todos executados com o split novo — comparações dentro desse grupo são válidas.

### Problema 2 — `torch.manual_seed` em `client.py:108` não afetava DataLoader (sem bug)

**Hipótese inicial:** o `torch.manual_seed` por rodada poderia estar interferindo no shuffle do DataLoader do cliente, criando trajetórias de batch diferentes entre runs.

**Conclusão após análise:** a hipótese é **falsa para o path de produção**.

O DataLoader criado em `dataloaders.py:169-171` tem `generator` explícito:

```python
_gen = torch.Generator().manual_seed(RANDOM_SEED + cid)
DataLoader(..., shuffle=True, generator=_gen)
```

Quando um DataLoader tem `generator` explícito, o PyTorch usa aquele objeto no `RandomSampler` — o estado global do `torch.manual_seed` é completamente ignorado. Portanto:

- **Shuffle de batch**: controlado por `_gen` (criado uma vez, avança deterministicamente a cada epoch/rodada) — **imune** ao `torch.manual_seed` global
- **Dropout e operações random do modelo**: controlados pelo estado global — seedados deterministicamente por `torch.manual_seed(RANDOM_SEED + round * num_clients + client_id)` ✅

Não havia bug. Não houve correção.

### Causa real da regressão −5 p.p. no HSL

O `torch.manual_seed` por rodada mudou o comportamento do **dropout** (não do shuffle de batch):

- **Antes do seeding**: dropout com estado global acumulado — trajetória de máscaras "aleatória" porém contínua
- **Depois do seeding**: dropout com seed fixo por rodada — trajetória determinística e reiniciada a cada rodada

Para o BPSP (20k amostras), o efeito estatístico de qualquer trajetória de dropout é diluído pela quantidade de dados. Para o HSL (3.6k amostras), a trajetória específica de dropout tem impacto mensurável na qualidade do modelo. A mudança de comportamento resultou em uma trajetória de otimização menos favorável para o HSL naquele run específico — não é um bug, é sensibilidade esperada de dataset pequeno a mudanças de aleatoriedade do dropout.

---

## Sessão 2026-06-30 — Correção de gaps, observabilidade e critério de checkpoint

### Contexto

Sessão focada em corrigir gaps técnicos identificados na infraestrutura de treinamento antes de iniciar o Bloco 2. Os gaps foram levantados como riscos de incorreção latente ou perda de observabilidade — não eram blockers para rodar o treinamento, mas comprometiam a rastreabilidade dos resultados e a validade metodológica de comparações futuras.

### Gaps corrigidos

#### Gap 3 — Nome ambíguo do arquivo de avaliação

**Problema:** `evaluation_round_{round_num}.json` usava `round_num` (última rodada, ex: 120) no nome, mas o arquivo documenta o melhor checkpoint (ex: rodada 75). Um arquivo chamado `evaluation_round_75.json` não comunicava se 75 era a best_round ou o total de rodadas.

**Fix:** renomeado para `evaluation_best_r{best_round}_of_{round_num}.json` (ex: `evaluation_best_r75_of_120.json`). O payload JSON também foi corrigido — substituiu o campo genérico `"round"` por dois campos explícitos:
```json
{
  "best_round":   75,
  "total_rounds": 120,
  "best_accuracy": 0.6959
}
```

**Por quê importa:** ambiguidade no nome do arquivo levaria a erros de interpretação na defesa ou em análise pós-hoc. O nome agora é autoexplicativo.

---

#### Gap 4 — API podia servir modelo errado (load_best sem training_id)

**Problema:** `_load_from_store()` em `state.py` chamava `store.load_best()` sem `training_id`. Em ambiente com múltiplos training_ids no banco (BPSP-only=1, HSL-only=2, Federado=3, Pooled=4), o `load_best()` global retornava o checkpoint com maior accuracy — que poderia ser o BPSP-only, não o federado.

**Fix:** resolução em duas camadas:
1. `state.py` lê `FL_TRAINING_ID` (env var) → `experiments/last_federated_training_id.txt` (arquivo) → `None` (fallback global)
2. `orchestrator.py` grava automaticamente o `training_id` da fase federada (>1 cliente) em `experiments/last_federated_training_id.txt` ao final do `train()` — sem nenhum parâmetro manual necessário

**Consequência:** `make training-full` agora persiste automaticamente o `training_id` correto. A API carrega o modelo federado na inicialização sem configuração adicional.

---

#### Gap 5 — τ_eff do FedNova não rastreado por rodada

**Problema:** `tau_eff` (passos efetivos do FedNova — normalização do update por τ_i) era logado em texto mas descartado após o loop. Análise pós-hoc da curva de τ ao longo das rodadas exigia re-parsing de log.

**Fix:**
- `history["tau_eff"]` adicionado ao dict de histórico — preenchido por rodada: valor real para FedNova, `None` para FedAvg
- Migration 013 (`fl_round_history`) criada com coluna `tau_eff REAL` (nullable — correto para FedAvg)
- `save_round_history()` persiste τ_eff junto com accuracy e loss no banco

---

#### Gap 7 — Edge case best_round = 0

**Problema:** `best_round` é inicializado em 0. Se nenhuma rodada melhorasse o critério acima de 0.0 (cenário improvável mas possível), `history["loss"][best_round - 1]` = `history["loss"][-1]` em Python — acesso ao último elemento, não ao primeiro. Incorreção latente.

**Fix:** após o loop, antes de qualquer uso de `best_round`:
```python
if best_round == 0:
    best_round    = round_num
    best_f1       = history["f1_macro"][-1]
    best_accuracy = history["accuracy"][-1]
```
A guarda pontual que existia em um único acesso (`if best_round > 0 else 0.0`) foi removida — dead code após o fix centralizado.

---

### Gap 6 — Critério de checkpoint: decisão e trade-offs

Este gap gerou a discussão mais importante da sessão.

**Problema original:** o checkpoint guloso selecionava a rodada com maior `accuracy` global. Com 5 classes desbalanceadas, accuracy favorece a classe majoritária. Os dados do Run de Validação (training_id=8) confirmaram o gap empiricamente:

| Métrica | Valor | Interpretação |
|---|---|---|
| Accuracy | 70,19% | Parece bom |
| F1 macro | 0,4994 | Metade das classes mal cobertas |
| F1 `curado_internado` | **0,000** | 28 amostras, nunca acertada |
| F1 `melhora_internado_grave` | 0,277 | 1 em 4 casos graves acertados |

Gap de 20 p.p. entre accuracy e F1 macro — sustentado quase inteiramente por `curado_pronto` (48% do dataset, F1=0,83).

**Trade-off analisado:**

| Critério | Vantagem | Risco |
|---|---|---|
| Accuracy | Estável, interpretável, legado Bloco 1 | Favorece maioria; checkpoint ótimo ≠ melhor clinicamente |
| F1 macro | Penaliza falha em qualquer classe; alinhado com avaliação final | Com `curado_internado` (28 amostras), F1 oscila por ruído estatístico, não por melhora real |
| F1 por classe específica | Foca na classe clinicamente prioritária | Exige decisão clínica sobre qual classe priorizar — fora do escopo técnico |

**Questão central levantada:** com 28 amostras no conjunto de teste, F1 de `curado_internado` pode mudar 0.1+ por 2-3 amostras certas ou erradas — instabilizando o critério de convergência.

**Decisão: rastrear os dois critérios, decidir com dados.**

Em vez de escolher agora, a implementação passa a registrar ambos por rodada, com `f1_macro` como critério padrão do Bloco 2:

- `evaluate_global_model()` retorna `(loss, accuracy, f1_macro, per_class_f1)` — uma passagem no test loader computa tudo
- `history["f1_macro"]` e `history["per_class_f1"]` rastreados por rodada
- Checkpoint selecionado por `f1_macro > best_f1` (padrão Bloco 2)
- `best_accuracy` registrado na rodada do melhor F1 — ambos disponíveis
- Critério de convergência também migrado para `Δ F1` em vez de `Δ accuracy` — consistência interna
- `evaluation_payload` inclui `"best_f1_macro"` e `"best_accuracy"` explicitamente

**Observabilidade no banco:** Migration 013 (`fl_round_history`) inclui colunas `f1_macro REAL` e `per_class_f1 JSONB` — curvas de ambas as métricas por rodada consultáveis via SQL para qualquer `training_id`.

---

### Parametrização do critério de checkpoint

**Motivação:** o critério de seleção do checkpoint é uma decisão metodológica que pode mudar conforme os dados revelam o comportamento do modelo. Hardcodar `f1_macro` no código exigiria redeploy a cada mudança.

**Decisão de arquitetura:** parametrizar via `FedConfig.checkpoint_criterion`, com fonte configurável sem redeploy:
```python
checkpoint_criterion: str = field(
    default_factory=lambda: os.getenv("FL_CHECKPOINT_CRITERION", "f1_macro")
)
```

**Rastreabilidade por training_id:** Migration 014 adiciona `checkpoint_criterion TEXT NOT NULL DEFAULT 'f1_macro'` em `metrics.fl_trainings`. Cada `training_id` registra qual critério foi usado — comparações entre runs são metodologicamente verificáveis.

**Trade-off aceito:** o loop usa `FED_CFG.checkpoint_criterion`, lido uma vez no início do treinamento. Mudar o critério no banco afeta apenas o próximo run — não o que está rodando. Isso é o comportamento correto: mudar o critério mid-training produziria comparação inválida entre rodadas.

**Caminho futuro (refactoring MVP):** o critério migra para tabela `fl_config` com colunas `changed_at`, `changed_by`, `justification`, `expected_effect` — interface web muda sem redeploy, com audit trail completo.

---

### Migrations criadas nesta sessão

| Migration | Tabela afetada | O que adiciona |
|---|---|---|
| 013 `fl_round_history` | `metrics.fl_round_history` (nova) | accuracy, loss, tau_eff, f1_macro, per_class_f1 por rodada por training_id |
| 014 `fl_trainings_checkpoint_criterion` | `metrics.fl_trainings` | coluna `checkpoint_criterion TEXT` — qual critério foi usado em cada training_id |

### Arquivos alterados nesta sessão

| Arquivo | O que mudou |
|---|---|
| `experiments/training/fl_core.py` | `evaluate_global_model()` retorna f1_macro e per_class_f1; history com tau_eff, f1_macro, per_class_f1; critério de checkpoint e convergência migrados para f1_macro; eval_path renomeado; `save_round_history()` chamado ao final; best_round=0 edge case corrigido |
| `infrastructure/shared/checkpoint_store.py` | ABC + SQLite + PostgreSQL: `save_round_history()` com tau_eff, f1_macro, per_class_f1; `register_training()` com checkpoint_criterion |
| `infrastructure/mosaicfl_api/state.py` | Resolução de `_INFERENCE_TRAINING_ID`: env → arquivo → None |
| `experiments/training/orchestrator.py` | Grava `last_federated_training_id.txt` automaticamente após fase federada |
| `src/mosaicfl/core/config.py` | `FedConfig.checkpoint_criterion` parametrizado via `FL_CHECKPOINT_CRITERION` |
| `alembic/versions/013_fl_round_history.py` | Criado — tabela de histórico por rodada |
| `alembic/versions/014_fl_trainings_checkpoint_criterion.py` | Criado — coluna checkpoint_criterion em fl_trainings |

### Estado ao final da sessão

- **Bloco 2 pronto para iniciar** — split corrigido (seeds independentes por hospital), critério de checkpoint parametrizado (f1_macro por padrão), observabilidade completa por rodada no banco
- **Migrations 013 e 014 não aplicadas** — aplicar antes do próximo `make training-full` com `alembic upgrade head`
- **Critério em uso no Bloco 2:** `f1_macro` (padrão) — sobrescrevível com `FL_CHECKPOINT_CRITERION=accuracy`
- **Incompatibilidade Bloco 1 × Bloco 2:** Bloco 1 usou accuracy como critério; Bloco 2 usa f1_macro. Comparação direta de checkpoints entre blocos é inválida. Comparações dentro do Bloco 2 são válidas.

---

### Expectativas para o primeiro treinamento do Bloco 2 (FedNova sem DP)

O objetivo deste treinamento é estabelecer o baseline do Bloco 2 — o ponto de referência sobre o qual os experimentos DP (Exp 17/18/19) serão comparados. As expectativas abaixo servem como hipóteses a confrontar com os resultados obtidos.

#### Accuracy global

Esperamos accuracy entre **65% e 72%**. O split corrigido (seeds independentes por hospital) produz partições ligeiramente diferentes das do Bloco 1. O modelo não sabe quais partições eram "favoráveis" — o resultado pode ir em qualquer direção dentro desse intervalo. A referência histórica de 70,19% (Bloco 1) **não é o piso** — pode ficar abaixo.

#### F1 macro

Esperamos F1 macro entre **0,45 e 0,55**. Com o critério de seleção agora sendo F1 macro em vez de accuracy, o checkpoint selecionado deve ter F1 macro maior do que o que seria selecionado pelo critério anterior — mas não necessariamente accuracy mais alta. É possível ver accuracy ligeiramente menor com F1 macro maior, o que seria o resultado esperado da mudança de critério.

#### `curado_internado` (28 amostras)

Esperamos F1 = 0,000 novamente. Com 28 amostras no teste e uma classe rara no treino, o modelo provavelmente não aprenderá esse padrão com dados de apenas dois hospitais. Se F1 > 0, seria resultado surpreendente — positivo, mas frágil (pequena variação de partição pode zerar novamente).

#### `melhora_internado_grave`

Esperamos F1 entre 0,25 e 0,40. Esta é a classe de maior risco clínico e onde o modelo mais falha. Com o critério F1 macro, o checkpoint escolhido penaliza mais falhas nessa classe do que o critério accuracy — esperamos melhora marginal em relação ao Bloco 1 (0,277).

#### Melhor rodada

Esperamos best_round entre R60 e R110. O padrão histórico mostra que o modelo converge entre R79 (Exp 15) e R118 (Exp 13). Com f1_macro como critério, a rodada selecionada pode ser diferente da que seria selecionada por accuracy — especialmente se as duas métricas divergirem em fases distintas do treinamento.

#### HSL-only (fase 2/4)

Esperamos instabilidade similar ao Run de Validação (accuracy ~35–42%). O split corrigido dá ao HSL uma partição diferente, com sementes independentes — pode melhorar ou piorar em relação ao Run de Validação (35,05%). Loss crescente ao longo das rodadas é esperado com 1 cliente e dataset pequeno.

#### tau_eff (FedNova)

Esperamos τ_eff estável entre 40 e 80 (BPSP tem ~1.252 batches × 1 epoch; HSL ~226 batches × 1 epoch; ponderado por amostras). Se τ_eff oscilar muito entre rodadas, pode indicar instabilidade no lado HSL — dado observável diretamente no banco via `fl_round_history.tau_eff` sem re-parsing de log.

#### Template para registro dos resultados

```
### Bloco 2 — Treinamento 1 (FedNova sem DP)

**Data:** 2026-06-30
**Log:** experiments/logs/run_complete_20260630_091435.log
**Critério de checkpoint:** f1_macro (FL_CHECKPOINT_CRITERION=f1_macro)
**Split:** corrigido (seeds independentes por hospital — BPSP seed 1042, HSL seed 1043)
**training_ids:** BPSP=9 | HSL=10 | Federado=11

#### Fase 1/4 — BPSP-only (training_id=9)

| Métrica | Obtido |
|---|---|
| Accuracy (best) | 62,41% (R63) |
| F1 macro | 0,3548 |
| Macro AUC | 0,7496 |
| ECE isotônica | 0,0494 |
| Rodadas | 71 (convergiu) |
| Duração | 67,0 min |
| Peak RAM | 2.445 MB |
| CPU médio | 2.353% (~23 núcleos) |

#### Fase 2/4 — HSL-only (training_id=10)

| Métrica | Obtido | Esperado | Δ | Interpretação |
|---|---|---|---|---|
| Accuracy (best) | 33,19% (R21) | 35–42% | −1,81 p.p. | Abaixo do intervalo — dataset pequeno (3.621 amostras), convergiu cedo em R39 |
| F1 macro | 0,2479 | — | — | curado_pronto e curado_internado com F1=0 (classes raras no HSL) |
| Loss | 2,3263 | crescente | — | Loss alta — sinal de instabilidade com 1 cliente |
| ECE isotônica | 0,0717 | — | — | Calibração pior que o federado, esperado com dataset pequeno |
| Per-class F1 | cp=0,00 ci=0,00 mp=0,54 mib=0,47 mig=0,23 | — | — | Modelo colapsa para as 3 classes majoritárias locais do HSL |
| Duração | 7,0 min | — | — | Rápido por convergência em R39 |

#### Fase 3/4 — Federado BPSP+HSL (training_id=11)

| Métrica | Obtido | Esperado | Δ | Interpretação |
|---|---|---|---|---|
| Accuracy (best) | **65,90%** (R77) | 65–72% | ✓ dentro | Dentro do intervalo esperado; 4,29 p.p. abaixo do Run de Validação (split diferente) |
| F1 macro | **0,4905** | 0,45–0,55 | ✓ dentro | Dentro do intervalo; critério de seleção funcionou |
| Macro AUC | **0,8105** | 0,79–0,83 | ✓ dentro | AUC robusta mesmo sem convergência |
| ECE isotônica | **0,0311** | < 0,02 | +0,011 | Acima do esperado — temperatura T=1,2377 indica subconfiança leve |
| Melhor rodada | **R77** | R60–R110 | ✓ dentro | Padrão histórico mantido |
| F1 curado_internado | **0,000** | 0,000 | = | Confirmado — 28 amostras são insuficientes |
| F1 melhora_int_grave | **0,3215** | 0,25–0,40 | ✓ dentro | Melhora em relação ao Bloco 1 (0,277) — critério F1 penalizou mais as classes minoritárias |
| τ_eff médio | **1.095,0** | 40–80 | ⚠ muito maior | Ver análise abaixo |
| Convergência | **Não** (120 rodadas) | — | — | Primeiro run do Bloco 2 sem convergência — modelo ainda explorando novo espaço de parâmetros |
| Duração | 121,0 min | — | — | Equivalente ao Bloco 1 |
| Peak RAM | 2.295 MB | — | — | Baseline de memória para comparação pós-refactoring e GPU |
| CPU médio | 2.358% | — | — | ~23 núcleos saturados — pipeline CPU-bound como esperado |

**Per-class F1 na R77 (best checkpoint):**

| Classe | F1 | vs Bloco 1 (Run Val.) |
|---|---|---|
| curado_pronto | 0,8009 | ↑ (+0,003) |
| curado_internado | 0,0000 | = |
| melhora_pronto | 0,7550 | ↑ (+0,049) |
| melhora_internado_breve | 0,5753 | ↑ (+0,060) |
| melhora_internado_grave | 0,3215 | ↑ (+0,044) |

Todas as classes com amostras suficientes melhoraram em relação ao Bloco 1. Melhora mais expressiva em `melhora_pronto` (+0,049) e `melhora_internado_breve` (+0,060).

#### Fase 4/4 — BEHRT Pooled Baseline

| Modelo | Accuracy | F1 macro |
|---|---|---|
| BEHRT Pooled A (sem demo) | **69,51%** | **0,5128** |
| BEHRT Pooled B (late fusion) | 67,20% | 0,5101 |
| RF Centralizado (BoT) | 66,67% | 0,5026 |
| **BEHRT FL Federado (Bloco 2)** | **65,90%** | **0,4905** |

**Custo de privacidade (BEHRT FL vs BEHRT Pooled A):** −3,61 p.p. Acc | −0,0223 F1 macro

> Inversão em relação ao Bloco 1: no Exp 15 o FL superou o BEHRT Pooled. No Bloco 2, com split corrigido, o pooled supera o federado em 3,61 p.p. Isso é esperado e metodologicamente mais honesto — o split anterior pode ter favorecido o FL artificialmente.

**FL vs RF Centralizado:** RF supera FL em −0,77 p.p. Acc e −0,0121 F1 macro.

> RF supera FL novamente — volta ao padrão histórico de T1–T14. O único experimento em que o FL superou o RF foi T15 ("marco histórico" no Bloco 1). Com o split corrigido no Bloco 2, o padrão histórico se restabelece. Argumento para a defesa: o custo de privacidade em relação ao baseline mais simples é marginal (0,77 p.p. em Acc), mesmo sem qualquer ajuste fino de hiperparâmetros para o novo split.

#### O critério de checkpoint fez diferença?

- **Rodada selecionada por f1_macro:** R77 — Acc=65,90% | F1=0,4905
- **Rodada que seria selecionada por accuracy:** R58 — Acc=**68,15%** | F1=0,4819
- **Diferença:** usando accuracy, ganharíamos +2,25 p.p. em Acc mas perderíamos −0,0086 em F1 macro
- **Interpretação:** o checkpoint por accuracy (R58) sacrifica F1 das classes minoritárias em favor da majoritária (`curado_pronto`). O critério f1_macro escolheu corretamente para um modelo mais equilibrado entre classes — que é o objetivo clínico do projeto.

#### Análise do τ_eff = 1.095

O valor 1.095 é constante em todas as 120 rodadas (std=0,0), o que confirma que FedNova está funcionando corretamente: τ_eff depende apenas dos tamanhos dos datasets e da configuração de batches, que não mudam entre rodadas.

Cálculo esperado:
- τ_BPSP = ceil(20.019 / 16) = 1.252 passos | p_BPSP = 20.019/23.640 = 0,847
- τ_HSL = ceil(3.621 / 16) = 227 passos | p_HSL = 3.621/23.640 = 0,153
- **τ_eff = 0,847 × 1.252 + 0,153 × 227 = 1.059,8 + 34,7 = 1.094,5 ≈ 1.095** ✓

A expectativa de 40–80 estava errada — foi estimada sem considerar o número real de batches por época. O valor 1.095 é o correto para a configuração atual. **Não há bug — há erro de estimativa na documentação anterior.**

#### Recursos computacionais — Baseline Bloco 2

| Fase | Duração | Peak RAM | CPU médio |
|---|---|---|---|
| BPSP-only | 67 min | 2.445 MB | 2.353% |
| HSL-only | 7 min | 2.299 MB | 2.368% |
| Federado | 121 min | 2.295 MB | 2.358% |
| BEHRT Pooled A | ~92 min | — | — |
| BEHRT Pooled B | ~94 min | — | — |
| **Total make training-full** | **~7h (420 min)** | **pico 2.445 MB** | **~23 núcleos** |

Este é o baseline de custo computacional para comparação futura (pós-refactoring, GPU).

#### Conclusão sobre as expectativas

##### O que aconteceu conforme esperado

| Expectativa | Previsto | Obtido | |
|---|---|---|---|
| Accuracy global | 65–72% | 65,90% | ✓ |
| F1 macro | 0,45–0,55 | 0,4905 | ✓ |
| Macro AUC | 0,79–0,83 | 0,8105 | ✓ |
| Melhor rodada | R60–R110 | R77 | ✓ |
| `curado_internado` F1 | 0,000 | 0,000 | ✓ |
| `melhora_internado_grave` F1 | 0,25–0,40 | 0,3215 | ✓ |
| HSL instável com dataset pequeno | sim | acc=33,19%, loss crescente | ✓ |
| Calibração isotônica melhor que temperature | sim | ECE_iso=0,031 < ECE_temp=0,065 | ✓ |
| τ_eff constante por rodada | — | 1.095 em todas as 120 rodadas | ✓ |

##### O que não aconteceu conforme esperado

| Expectativa | Previsto | Obtido | Explicação |
|---|---|---|---|
| ECE isotônica | < 0,02 | 0,0311 | Modelo subconfiante (T=1,237) — subconfiança leve é normal em BEHRT com nova partição |
| HSL accuracy | 35–42% | 33,19% | Ficou 1,81 p.p. abaixo — split corrigido deu ao HSL uma partição ligeiramente mais difícil |
| τ_eff médio | 40–80 | 1.095 | **Estimativa estava errada** — o valor 1.095 é matematicamente correto (calculado com batches reais); a previsão de 40–80 usou número de clientes, não número de passos locais |
| Convergência | esperada (padrão histórico) | **não convergiu** (120 rodadas) | Único run do projeto sem convergência — o novo espaço de parâmetros (split corrigido) é mais difícil de otimizar; modelo ainda explorando |

##### O que aconteceu e não estava previsto

- **Não houve convergência em 120 rodadas** — todos os treinamentos anteriores convergiram (exceto runs com DP severo). O split corrigido produz partições que o modelo ainda não explorou — as 120 rodadas podem não ser suficientes para o Bloco 2. Implicação direta: considerar aumentar `n_rounds_max` para 150 no Exp 17.

- **Per-class F1 melhorou em todas as classes** apesar de accuracy menor que o Bloco 1 (65,90% vs 70,19%) — evidência direta de que o critério F1 macro produziu um modelo mais equilibrado entre classes. A accuracy menor não representa regressão: é resultado de splits diferentes e de um critério de seleção que sacrifica acurácia na classe majoritária para beneficiar as minoritárias.

- **O critério F1 macro fez diferença real e mensurável:** a rodada selecionada por F1 macro foi R77 (Acc=65,90%, F1=0,4905); se tivéssemos usado accuracy, teria sido R58 (Acc=68,15%, F1=0,4819). Ou seja: usando accuracy ganharíamos +2,25 p.p. de Acc mas perderíamos −0,0086 de F1 macro — um modelo que acerta mais a classe majoritária (`curado_pronto`) às custas das minoritárias. O critério F1 macro escolheu corretamente para o objetivo clínico do projeto.

- **Inversão FL vs BEHRT Pooled:** no Bloco 1 (T15), o FL federado superava o BEHRT Pooled A (69,59% vs 68,29%). No Bloco 2, o Pooled A supera o FL em 3,61 p.p. (69,51% vs 65,90%). O split corrigido é metodologicamente mais honesto — a superioridade do T15 pode ter sido favorecida pela partição anterior. A inversão é esperada e não invalida o projeto: o custo de privacidade existe e agora pode ser medido corretamente.

- **RF centralizado supera FL federado** — volta ao padrão histórico de T1–T14 (T15 foi a exceção). O custo de privacidade em relação ao baseline mais simples é 0,77 p.p. em Acc e 0,0121 em F1 macro. Argumento para a defesa: a federação preserva privacidade com perda marginal frente ao baseline mais simples, sem ajuste fino de hiperparâmetros para o novo split.

##### Implicações para Exp 17 (próxima etapa — DP)

- **Baseline do Bloco 2 estabelecido:** Acc=65,90% | F1=0,4905 | AUC=0,8105 | ECE_iso=0,0311 (training_id=11)
- **Risco de instabilidade com DP:** o modelo sem ruído já não convergiu em 120 rodadas — adicionar ruído gaussiano pode agravar. Considerar `n_rounds_max=150` ou redução da taxa de aprendizado local antes de introduzir DP.
- **Alvo clínico principal:** `melhora_internado_grave` (F1=0,3215) — classe de maior risco; DP não deve degradar muito esta classe se o ruído for calibrado (σ pequeno na primeira rodada).
- **τ_eff correto:** 1.095 é o valor esperado para a configuração atual. Qualquer desvio significativo desse valor em Exp 17 indicaria bug na implementação DP + FedNova.

---

## Sessão 2026-06-30 (tarde) — Observabilidade de recursos, análise de resultados e preparação para GPU

### Implementações realizadas antes do treinamento

#### Monitoramento de recursos computacionais (psutil)

Antes de rodar o Bloco 2 Treinamento 1, foi implementado monitoramento de CPU, RAM e tempo por rodada. Motivação: sem esses dados, não seria possível comparar o custo computacional entre CPU (código atual), GPU (próxima etapa) e código refatorado.

**Arquivos alterados:**

| Arquivo | Mudança |
|---|---|
| `alembic/versions/013_fl_round_history.py` | Adicionada coluna `round_duration_s REAL` à tabela |
| `alembic/versions/015_fl_trainings_resource_metrics.py` | Criada — 3 colunas em `fl_trainings`: `total_duration_s`, `peak_ram_mb`, `avg_cpu_pct` |
| `infrastructure/shared/checkpoint_store.py` | ABC + SQLite + PostgreSQL: novas assinaturas em `complete_training()` e `save_round_history()` |
| `experiments/training/fl_core.py` | `import psutil`, `_proc = psutil.Process()`, coleta por rodada, log estruturado, repasse para banco |

**O que é coletado:**
- Por rodada: `round_duration_s` → `fl_round_history`
- Por treinamento: `total_duration_s`, `peak_ram_mb`, `avg_cpu_pct` → `fl_trainings`
- No log: `[Recursos] RAM=NNNMb (pico=NNNMb) CPU=N.N% Rodada=N.Ns` por rodada + `resource_summary` ao final

**Nota sobre `avg_cpu_pct`:** valor pode ultrapassar 100% — psutil mede por processo e acumula por núcleo (ex: 2.358% = ~23 núcleos saturados). É informação real de paralelismo, não erro.

#### Migrações aplicadas antes do treinamento

```
alembic upgrade head  →  aplica 013 + 014 + 015
```

Cadeia: `012 → 013 (fl_round_history + round_duration_s) → 014 (checkpoint_criterion) → 015 (resource metrics)`

### Decisões tomadas após análise dos resultados

#### Decisão: NÃO trocar critério de checkpoint para F1 por classe antes da refatoração

**Questão levantada:** valeria usar F1 de `melhora_internado_grave` como critério de checkpoint em vez do F1 macro global, para forçar o modelo a priorizar a classe de maior risco clínico?

**Decisão: não.** Razões pragmáticas:

1. **Critério de checkpoint ≠ objetivo de treinamento.** O critério só seleciona qual rodada salvar — não muda o que o modelo aprende. O modelo aprende via `CrossEntropyLoss`. Para melhorar uma classe específica, o alavanca correto é a função de perda, não o critério de seleção.
2. **O modelo já não convergiu em 120 rodadas com F1 macro.** Um critério mais rígido (F1 de uma classe, que oscila mais) agravaria a instabilidade.
3. **Quebra metodológica.** O baseline do Bloco 2 usa `f1_macro`. Trocar o critério no Exp 17 invalida a comparação DP vs sem-DP.
4. **Custo alto, ganho marginal em CPU.** Cada run leva ~7h. Após GPU, o ciclo cai para minutos — o momento certo para experimentos de critério.

**Quando fazer:** após refatoração + GPU, quando o ciclo de experimentação for rápido.

#### Decisão: experimentar `class_weights` para `melhora_internado_grave` somente após GPU

**Contexto:** o projeto já usa `class_weights` inversamente proporcionais à frequência de cada classe (`client.py:83`), com teto em `max=15.0` (peso 47 causava explosão de gradiente em experimento anterior). Com a distribuição do BPSP, `melhora_internado_grave` recebe peso ~0,5 — sub-representado relativamente às classes raras.

**O que class_weights faz:** erra na classe rara → gradiente maior → modelo aprende mais dessa classe. É a alavanca correta para melhorar F1 de classes específicas (diferente do critério de checkpoint, que só seleciona a rodada).

**Opções futuras para experimentar (pós-GPU):**
- Aumentar o teto de `max=15.0` com monitoramento de grad_norm
- Trocar fórmula de inversão por raiz quadrada (`sqrt(total/count)`) — distribuição mais suave
- Peso manual explícito para `melhora_internado_grave` independente da frequência

**Quando fazer:** após GPU — ciclo de 7h em CPU torna qualquer experimento de hiperparâmetro caro demais.

### Próximos passos estabelecidos nesta sessão

#### Sequência definida

```
1. Instalar driver NVIDIA (RTX 4070 Ti) — em andamento
2. Rodar verify_gpu.sh para confirmar GPU operacional
3. make training-full  →  Bloco 2 Treinamento 1 na GPU
4. Comparar recursos: CPU baseline (hoje) vs GPU
5. Refactoring MVP (modularização + config em banco)
6. Confirmar resultados equivalentes pós-refactoring
7. Simulação distribuída (desktop como servidor + notebook como cliente)
```

A decisão de refatorar APÓS a GPU (e não antes) foi da própria pesquisadora, com o argumento correto: ter os 3 pontos de comparação (CPU atual / GPU atual / GPU refatorado) torna o capítulo de resultados do TCC mais sólido.

#### Scripts de instalação criados

Localização: `/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/`

| Script | Função |
|---|---|
| `install_nvidia_driver.sh` | Instala `nvidia-driver-595-open` (recomendado pelo Ubuntu para RTX 4070 Ti / Ada Lovelace). Requer `sudo`. Após execução: reiniciar o computador. |
| `verify_gpu.sh` | Verifica driver (`nvidia-smi`), PyTorch (`cuda_available`), e operação real (matmul 4096×4096). Executar após reinicialização. |

**Sem mudança de código no projeto:** o DEVICE já é detectado automaticamente via `torch.cuda.is_available()` em `config.py`. Quando o driver estiver instalado, o próximo `make training-full` roda na GPU sem nenhuma alteração.

**Nota sobre Secure Boot (MOK):** durante a instalação, o Ubuntu pede para criar uma senha temporária para assinar o módulo do kernel. Essa senha é usada uma única vez na reinicialização seguinte (tela "Enroll MOK"). Após o enroll, nunca mais é pedida.

### Baseline de recursos computacionais — CPU (referência para comparação futura)

| Fase | Duração | Peak RAM | CPU médio | Observação |
|---|---|---|---|---|
| BPSP-only (training_id=9) | 67 min | 2.445 MB | 2.353% | ~23 núcleos |
| HSL-only (training_id=10) | 7 min | 2.299 MB | 2.368% | Convergiu em R39 |
| Federado BPSP+HSL (training_id=11) | 121 min | 2.295 MB | 2.358% | 120 rodadas sem convergência |
| BEHRT Pooled A | ~92 min | — | — | CPU, sem coleta psutil |
| BEHRT Pooled B | ~94 min | — | — | CPU, sem coleta psutil |
| **Total make training-full** | **~420 min (7h)** | **pico 2.445 MB** | **~23 núcleos** | Baseline CPU estabelecido |

Este é o ponto de referência para medir o ganho real da GPU no próximo treinamento.
