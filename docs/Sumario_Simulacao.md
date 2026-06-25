# Sumário de Simulações — MOSAIC-FL

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico  
**Autora:** Jacqueline Abreu | ICMC/USP  
**Atualizado em:** 2026-06-25

Este documento registra cada execução de treinamento com dados reais FAPESP, preservando condições, distribuição dos dados, hiperparâmetros, pesos de classe e resultados completos. O objetivo é permitir rastreabilidade total de cada experimento para o TCC.

> **Nota sobre terminologia:** o termo "simulação" refere-se à infraestrutura FL rodando em uma única máquina (hospitais simulados localmente). Os dados e o treinamento do modelo são reais.

---

## Configuração Comum (fixada em todos os experimentos)

### Dados — FAPESP

| Hospital | ID cliente | Atendimentos totais | Sequências (treino) | Sequências (val) |
|---|---|---|---|---|
| BPSP | 0 | 28.599 | 22.879 | 2.859 |
| HSL | 1 | 5.174 | 4.139 | 517 |
| **Teste global** | — | — | — | **3.379** |

**Vocabulário:** 649 tokens (construído sobre o pool completo BPSP+HSL)  
**max_seq_len:** 128  
**Query:** `4.145.906 linhas` → pipeline de agregação em 21.2s

#### Distribuição de classes por hospital

Mapeamento de classes:

| Índice | Rótulo | Definição |
|---|---|---|
| 0 | `curado_pronto` | Alta sem internação, desfecho curado |
| 1 | `curado_internado` | Alta após internação, desfecho curado |
| 2 | `melhora_pronto` | Alta sem internação, desfecho melhora |
| 3 | `melhora_internado_breve` | Alta após internação breve, desfecho melhora |
| 4 | `melhora_internado_grave` | Alta após internação grave, desfecho melhora |

**BPSP — distribuição total:**

| Classe | N | % |
|---|---|---|
| curado_pronto (0) | 15.892 | 55,6% |
| curado_internado (1) | 318 | 1,1% |
| melhora_pronto (2) | 120 | **0,4%** |
| melhora_internado_breve (3) | 9.448 | 33,0% |
| melhora_internado_grave (4) | 2.821 | 9,9% |

**HSL — distribuição total:**

| Classe | N | % |
|---|---|---|
| curado_pronto (0) | 67 | **1,3%** |
| curado_internado (1) | 45 | 0,9% |
| melhora_pronto (2) | 3.182 | **61,5%** |
| melhora_internado_breve (3) | 1.280 | 24,7% |
| melhora_internado_grave (4) | 600 | 11,6% |

**Observação non-IID crítica:** `melhora_pronto` representa 61,5% do HSL e apenas 0,4% do BPSP. `curado_pronto` representa 55,6% do BPSP e apenas 1,3% do HSL. Esta é a principal fonte de heterogeneidade nos experimentos.

**Distribuição do conjunto de teste global:**

| Classe | N | % |
|---|---|---|
| curado_pronto | 1.620 | 48,0% |
| melhora_internado_breve | 1.073 | 31,8% |
| melhora_internado_grave | 338 | 10,0% |
| melhora_pronto | 321 | 9,5% |
| curado_internado | 27 | 0,8% |

#### Pesos de classe (calculados por cliente, fixos em todos os experimentos)

Os pesos são calculados via `compute_class_weight('balanced', ...)` sobre o conjunto de treino de cada cliente.

**Cliente 0 — BPSP:**

| Classe | Peso | N treino |
|---|---|---|
| curado_pronto (0) | 0,361 | 12.686 |
| curado_internado (1) | 17,398 | 263 |
| melhora_pronto (2) | **47,173** | 97 |
| melhora_internado_breve (3) | 0,605 | 7.565 |
| melhora_internado_grave (4) | 2,018 | 2.268 |

**Cliente 1 — HSL:**

| Classe | Peso | N treino |
|---|---|---|
| curado_pronto (0) | 15,051 | 55 |
| curado_internado (1) | 24,347 | 34 |
| melhora_pronto (2) | 0,324 | 2.557 |
| melhora_internado_breve (3) | 0,817 | 1.013 |
| melhora_internado_grave (4) | 1,725 | 480 |

> O peso 47,173 para `melhora_pronto` no BPSP é consequência da raridade extrema (97 amostras). Em experimentos futuros considera-se clipar pesos em `max_weight=15,0`.

### Modelo — SimplifiedBEHRT

| Parâmetro | Valor |
|---|---|
| embed_dim | 64 |
| num_layers | 2 |
| num_heads | 4 |
| vocab_size | 649 |
| demo_dim | 2 (idade normalizada + sexo binário) |
| fusão demográfica | late fusion (concatenação pré-classificador) |

### Infraestrutura FL

| Parâmetro | Valor |
|---|---|
| Algoritmo | FedProx (manual, sem Ray) |
| Número de clientes | 2 (BPSP, HSL) |
| Critério de convergência | 3 rodadas consecutivas com Δacc < 0,005 |
| Tráfego por agregação | ~10–11 MB por rodada |
| Ambiente | `FL_ENV=production` |

---

## Experimento 1

**Data:** 2026-06-25  
**Início:** 06:18:17 | **Fim:** 07:16:13  
**Duração:** 57,4 min (3.443,93s)  
**Log:** `experiments/logs/run_complete_1.log`  
**Avaliação:** `experiments/logs/evaluation_round_20.json`

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Loss global | Acurácia | Δacc | Conv. |
|---|---|---|---|---|
| 1 | 1,2326 | 51,3% | — | — |
| 2 | 1,2647 | 51,0% | 0,00266 | 1/3 |
| 3 | 1,2860 | 53,3% | 0,02249 | reset |
| 4 | 1,1518 | 53,8% | 0,00562 | reset |
| 5 | 1,1538 | 54,0% | 0,00207 | 1/3 |
| 6 | 1,2160 | 49,9% | 0,04114 | reset |
| 7 | 1,2071 | 52,7% | 0,02782 | reset |
| 8 | 1,1190 | 58,1% | 0,05357 | reset |
| 9 | 1,1466 | 57,8% | 0,00296 | 1/3 |
| 10 | 1,1227 | 51,9% | 0,05860 | reset |
| 11 | 1,2157 | 53,9% | 0,02042 | reset |
| 12 | 1,1271 | 56,9% | 0,02959 | reset |
| 13 | 1,1552 | 52,9% | 0,03995 | reset |
| 14 | 1,1005 | 56,1% | 0,03137 | reset |
| 15 | 1,0780 | 56,5% | 0,00474 | 1/3 |
| 16 | 1,1002 | 57,8% | 0,01302 | reset |
| 17 | 1,1614 | 53,9% | 0,03877 | reset |
| 18 | 1,1279 | 55,9% | 0,01983 | reset |
| 19 | 1,1158 | 58,7% | 0,02782 | reset |
| 20 | 1,1485 | 58,0% | 0,00710 | reset |

**Convergência:** Não atingida  
**Tráfego total FL:** 217,17 MB  
**Checkpoint:** sha256=`4efd52ad5238` | T=1,1769 | round=20

### Avaliação final (rodada 20)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Recall | Precision | N |
|---|---|---|---|---|---|
| curado_pronto | 0,853 | 0,764 | 0,784 | 0,746 | 1620 |
| curado_internado | 0,667 | 0,083 | 0,074 | 0,095 | 27 |
| melhora_pronto | 0,663 | 0,083 | 0,053 | 0,191 | 321 |
| melhora_internado_breve | 0,715 | 0,479 | 0,465 | 0,494 | 1073 |
| melhora_internado_grave | 0,803 | 0,385 | 0,509 | 0,310 | 338 |
| **Macro** | **0,740** | **0,359** | — | — | 3379 |

**Acurácia:** 58,0%  
**ECE pré-calibração:** 0,0590 (MCE=0,6107)  
**ECE pós-calibração:** 0,0978 (MCE=0,4050) — temperature scaling **piorou** a calibração  
**Temperatura ajustada:** T=1,1769

**Matriz de confusão (pré-calibração):**

```
                curado_p  curado_i  melhora_p  melhora_ib  melhora_ig
curado_p (1620)     1270         3        50         202          95
curado_i (27)          6         2         0          11           8
melhora_p (321)       90         5        17         200           9
melhora_ib (1073)    276        10        17         499         271
melhora_ig (338)      61         1         5          99         172
```

**Diagnóstico de confiabilidade (pré-calibração):**

| Faixa Conf | Acc | Gap | N | Status |
|---|---|---|---|---|
| 0,262 | 0,000 | 0,262 | 1 | OVERCONF |
| 0,314–0,499 | 0,350–0,543 | 0,030–0,044 | 1817 | Adequado |
| 0,562–0,760 | 0,653–0,867 | 0,060–0,107 | 1271 | UNDERCONF |
| 0,821 | 0,859 | 0,038 | 85 | Adequado |
| 0,896–0,972 | 0,286–0,854 | 0,118–0,611 | 55 | OVERCONF |

### Ocorrências e incidentes

- **Crash pós-simulação:** O script crashou após concluir as 20 rodadas com `AttributeError: 'list' object has no attribute 'get'` na linha 1739 de `run_experiments_simulation.py`. A função `run_federated_learning_manual()` retorna `history` como dict de listas (`{rounds:[], accuracy:[], ...}`), mas o código em `main()` esperava dict de dicts por round. Etapas 4 (RAG) e 5 (Baseline RF) não foram executadas.  
- **Correção aplicada:** Detecção do formato flat-list via `"rounds" in history` antes de extrair métricas. Corrigido em `run_experiments_simulation.py` linhas 1733–1748.

---

## Experimento 2

**Data:** 2026-06-25  
**Início:** 07:51:40 | **Fim parcial:** 08:12:32 (ablation em andamento)  
**Log:** `experiments/logs/run_complete_1_correcao1.log`  
**Avaliação:** `experiments/logs/evaluation_round_7.json`

### Hiperparâmetros

*(idênticos ao Experimento 1)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Loss global | Acurácia | Δacc | Conv. |
|---|---|---|---|---|
| 1 | 1,2706 | 54,4% | — | — |
| 2 | 1,2939 | 52,5% | 0,01894 | reset |
| 3 | 1,3096 | 54,9% | 0,02368 | reset |
| 4 | 1,1574 | 53,2% | 0,01657 | reset |
| 5 | 1,1204 | 53,1% | 0,00178 | 1/3 |
| 6 | 1,1428 | 52,6% | 0,00444 | 2/3 |
| 7 | 1,1287 | 52,5% | 0,00148 | **3/3 — CONVERGIU** |

**Convergência:** Atingida na rodada 7  
**Tráfego total FL:** 76,01 MB  
**Checkpoint:** sha256=`75c0d9a5b8bb` | T=1,1269 | round=7

### Avaliação final (rodada 7)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Recall | Precision | N |
|---|---|---|---|---|---|
| curado_pronto | 0,827 | 0,726 | 0,812 | 0,656 | 1620 |
| curado_internado | 0,618 | 0,000 | 0,000 | 0,000 | 27 |
| melhora_pronto | 0,851 | 0,048 | 0,028 | 0,177 | 321 |
| melhora_internado_breve | 0,746 | 0,310 | 0,223 | 0,510 | 1073 |
| melhora_internado_grave | 0,791 | 0,352 | 0,618 | 0,246 | 338 |
| **Macro** | **0,767** | **0,287** | — | — | 3379 |

**Acurácia:** 52,5%  
**ECE pré-calibração:** 0,0613 (MCE=0,1209)  
**ECE pós-calibração:** 0,0637 (MCE=0,3011) — temperature scaling marginalmente piorou  
**Temperatura ajustada:** T=1,1269

**Matriz de confusão (pré-calibração):**

```
                curado_p  curado_i  melhora_p  melhora_ib  melhora_ig
curado_p (1620)     1316         0        36          94         174
curado_i (27)         14         0         0           4           9
melhora_p (321)      217         0         9          85          10
melhora_ib (1073)    377         3         5         239         449
melhora_ig (338)      81         0         1          47         209
```

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | 0,0029 |
| curado_internado | 0,1975 |
| melhora_pronto | **0,6075** |
| melhora_internado_breve | 0,1911 |
| melhora_internado_grave | 0,1272 |
| **Macro** | **0,1341** |

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,1%** | **0,786** | **0,504** | 0,063 |
| RF Hospital 0 (BPSP isolado) | 59,5% | 0,735 | 0,337 | 0,061 |
| RF Hospital 1 (HSL isolado) | 28,0% | 0,720 | 0,204 | 0,201 |

**Ablation late fusion demográfica:** em andamento no momento deste registro (épocas locais=10, seed=42, Config A: demo_dim=0 vs Config B: demo_dim=2).

---

## Tabela Comparativa dos Experimentos

| Atributo | Experimento 1 | Experimento 2 |
|---|---|---|
| Log | `run_complete_1.log` | `run_complete_1_correcao1.log` |
| Rodadas executadas | 20 | 7 |
| Convergência | Não | **Sim (rodada 7)** |
| Acurácia final | 58,0% | 52,5% |
| Macro AUC (pré-cal) | 0,740 | **0,767** |
| Macro F1 (pré-cal) | **0,359** | 0,287 |
| ECE pré-calibração | **0,059** | 0,061 |
| ECE pós-calibração | 0,098 (piorou) | 0,064 (marginal) |
| Temperatura T | 1,177 | 1,127 |
| Tráfego FL total | 217 MB | 76 MB |
| Etapas pós-FL | Crash (bug corrigido) | RAG + RF concluídos |
| Baseline RF (Acc) | — | 68,1% |

### Comparativo BEHRT-FL vs Baseline RF (Experimento 2)

| Modelo | Accuracy | AUC | F1 Macro |
|---|---|---|---|
| RF Centralizado (BoT) — teto sem privacidade | 68,1% | 0,786 | 0,504 |
| RF BPSP isolado — local sem FL | 59,5% | 0,735 | 0,337 |
| RF HSL isolado — local sem FL | 28,0% | 0,720 | 0,204 |
| **SimplifiedBEHRT FL (Exp 2, round 7)** | **52,5%** | **0,767** | **0,287** |
| **SimplifiedBEHRT FL (Exp 1, round 20)** | **58,0%** | **0,740** | **0,359** |

---

## Diagnóstico Consolidado

### Problema 1 — Acurácia abaixo do baseline RF

O RF centralizado (68,1%) supera o SimplifiedBEHRT FL (52,5–58,0%). As causas principais são:

1. **Non-IID extremo:** `melhora_pronto` é 61,5% do HSL mas 0,4% do BPSP. FedAvg pondera por volume de dados — BPSP tem 5,5× mais amostras e domina a agregação, fazendo o modelo global "esquecer" o que HSL aprende sobre essa classe.
2. **Peso de classe desestabilizador:** peso 47,173 para `melhora_pronto` no BPSP gera gradientes instáveis sem aprendizado real (apenas 97 amostras de treino nessa classe nesse cliente).
3. **Convergência prematura (Exp 2):** convergiu na rodada 7 com acurácia 52,5%, antes de alcançar os patamares do Experimento 1 (melhor: 58,7% na rodada 19).
4. **Client drift:** µ=0,01 no FedProx é insuficiente para o grau de heterogeneidade — a loss oscila (sobe e desce) ao longo das rodadas em vez de descer monotonicamente.

### Problema 2 — Temperature scaling piora a calibração

Em ambos os experimentos T>1 aumentou o ECE pós-calibração. Causa:
- O `test_loader` é reutilizado como calibration set (limitação acadêmica explicitada no código)
- O padrão de calibração é misto: bins de média confiança são underconfident, bins extremos são overconfident — temperatura única não resolve

### Classes não aprendidas

| Classe | Causa | Evidência |
|---|---|---|
| `melhora_pronto` | Quasi-exclusiva do HSL; BPSP domina FedAvg | F1=0,05–0,08 em ambos experimentos |
| `curado_internado` | Raridade extrema em ambos os clientes (N=27 no teste) | F1=0,00–0,08; zero acertos no Exp 2 |

### Ações corretivas planejadas

| Ação | Impacto esperado | Prioridade |
|---|---|---|
| Clipar pesos de classe em `max_weight=15,0` | Reduz instabilidade de gradiente no BPSP | Alta |
| Aumentar µ FedProx 0,01 → 0,1 | Reduz client drift, loss mais estável | Alta |
| Reduzir local epochs 2 → 1 | Reduz divergência entre clientes por rodada | Média |
| Separar calibration set (10% do val) | Torna temperature scaling válido metodologicamente | Média |
| Avaliar fusão para 3 classes | Resolve non-IID estrutural se clinicamente justificável | A definir com orientadora |

---

## Arquivos de Referência

| Arquivo | Conteúdo |
|---|---|
| `experiments/logs/run_complete_1.log` | Log completo do Experimento 1 (20 rodadas) |
| `experiments/logs/evaluation_round_20.json` | Avaliação detalhada por classe — Experimento 1 |
| `experiments/logs/run_complete_1_correcao1.log` | Log completo do Experimento 2 (7 rodadas + pós-FL) |
| `experiments/logs/evaluation_round_7.json` | Avaliação detalhada por classe — Experimento 2 |
| `experiments/data/baseline_rf_20260625_081232.json` | Resultado baseline RF — Experimento 2 |
| `experiments/data/history_20260625_071610.json` | Histórico de loss/acc por rodada — Experimento 1 |
| `AVALIACAO_PROJETO.md` | Avaliação acadêmica e clínica do projeto |
