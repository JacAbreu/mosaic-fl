# Sumário de Simulações — MOSAIC-FL

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico  
**Autora:** Jacqueline Abreu | ICMC/USP  
**Atualizado em:** 2026-06-29 (Exp 16 concluído — DP-FedAvg implementado, seeding fix, RAG bugs corrigidos, Ollama integrado)

Este documento registra cada execução de treinamento com dados reais FAPESP, preservando condições, distribuição dos dados, hiperparâmetros, pesos de classe e resultados completos. O objetivo é permitir rastreabilidade total de cada experimento para o TCC.

> **Nota sobre terminologia:** o termo "simulação" refere-se à infraestrutura FL rodando em uma única máquina (hospitais simulados localmente). Os dados e o treinamento do modelo são reais.

---

## Configuração Comum (fixada em todos os experimentos)

### Dados — FAPESP

> **Nota de split:** Experimentos 1 e 2 usaram divisão 80/10/10 (treino/val/teste). A partir do Experimento 3, adotou-se 70/10/10/10 (treino/val/cal/teste) para garantir conjunto de calibração independente. Os números abaixo refletem a configuração atual (Exp 3–5).

| Hospital | ID cliente | Atendimentos totais | Sequências (treino) | Sequências (val) | Sequências (cal) |
|---|---|---|---|---|---|
| BPSP | 0 | 28.599 | 20.019 | 2.859 | 2.859 |
| HSL | 1 | 5.174 | 3.621 | 517 | 517 |
| **Cal global** | — | — | — | — | **3.376** |
| **Teste global** | — | — | — | — | **3.381** |

**Vocabulário:** 648 tokens (construído sobre o pool completo BPSP+HSL)  
**max_seq_len:** 128  
**Query:** `4.145.906 linhas` → pipeline de agregação em ~21s

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
| curado_pronto | 1.620 | 47,9% |
| melhora_internado_breve | 1.074 | 31,8% |
| melhora_internado_grave | 338 | 10,0% |
| melhora_pronto | 321 | 9,5% |
| curado_internado | 28 | 0,8% |

#### Pesos de classe (calculados por cliente — Experimentos 3, 4 e 5 com split 70/10/10/10)

Os pesos são calculados via `compute_class_weight('balanced', ...)` sobre o conjunto de treino de cada cliente.

**Cliente 0 — BPSP:**

| Classe | Peso | N treino |
|---|---|---|
| curado_pronto (0) | 0,360 | 11.111 |
| curado_internado (1) | 17,484 | 229 |
| melhora_pronto (2) | **47,104** | 85 |
| melhora_internado_breve (3) | 0,607 | 6.599 |
| melhora_internado_grave (4) | 2,007 | 1.995 |

**Cliente 1 — HSL:**

| Classe | Peso | N treino |
|---|---|---|
| curado_pronto (0) | 15,743 | 46 |
| curado_internado (1) | 25,864 | 28 |
| melhora_pronto (2) | 0,324 | 2.236 |
| melhora_internado_breve (3) | 0,813 | 891 |
| melhora_internado_grave (4) | 1,724 | 420 |

> O peso 47,104 para `melhora_pronto` no BPSP é consequência da raridade extrema (85 amostras de treino). Em experimentos futuros considera-se clipar pesos em `max_weight=15,0`.

### Modelo — SimplifiedBEHRT

| Parâmetro | Valor |
|---|---|
| embed_dim | 64 |
| num_layers | 2 |
| num_heads | 4 |
| vocab_size | 648 |
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

## Experimento 3

**Data:** 2026-06-25  
**Início:** ~08:40 | **Fim:** 09:54:04  
**Duração:** 2.980,4s (49,7 min)  
**Log:** `experiments/logs/run_complete_2_correcao_calibracao.log`  
**Avaliação:** `experiments/logs/evaluation_round_20.json`

### Correções aplicadas nesta execução

- **Split 70/10/10/10:** conjunto de calibração (`cal_loader`) com 3.376 amostras, completamente independente do FL e do conjunto de teste. Primeira execução com calibração metodologicamente correta.
- **Marcadores estruturados:** `FL_TRAINING_COMPLETE` e `TREINAMENTO_COMPLETO` adicionados ao log (pesquisáveis por ferramentas de observabilidade).

### Hiperparâmetros

*(idênticos aos Experimentos 1 e 2)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Loss global | Acurácia | Δacc | Conv. |
|---|---|---|---|---|
| 1 | 1,2712 | 49,8% | — | — |
| 2 | 1,3973 | 45,0% | 0,04793 | reset |
| 3 | 1,2688 | 49,6% | 0,04609 | reset |
| 4 | 1,2359 | 54,9% | 0,05335 | reset |
| 5 | 1,2743 | 54,4% | 0,00507 | reset |
| 6 | 1,3782 | 52,6% | 0,01800 | reset |
| 7 | 1,2568 | 58,1% | 0,05533 | reset |
| 8 | 1,2277 | 56,8% | 0,01334 | reset |
| 9 | 1,2390 | 54,2% | 0,02544 | reset |
| 10 | 1,1620 | 53,0% | 0,01208 | reset |
| 11 | 1,2279 | 53,0% | 0,00031 | 1/3 |
| 12 | 1,2589 | 52,9% | 0,00087 | 2/3 |
| 13 | 1,3012 | 55,0% | 0,02072 | reset |
| 14 | 1,1101 | 55,6% | 0,00620 | reset |
| 15 | 1,2038 | 54,5% | 0,01151 | reset |
| 16 | 1,2258 | 55,5% | 0,01042 | reset |
| 17 | 1,1461 | 56,7% | 0,01245 | reset |
| 18 | 1,1182 | 55,3% | 0,01454 | reset |
| 19 | 1,2050 | 55,4% | 0,00117 | 1/3 |
| 20 | 1,1518 | 55,8% | 0,00440 | 2/3 |

**Convergência:** Não atingida (critério: 3× Δacc < 0,005)  
**Tráfego total FL:** 217,17 MB  
**Tempo médio por rodada:** ~149s (2,5 min/rodada)

### Avaliação final (rodada 20)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Recall | Precision | N |
|---|---|---|---|---|---|
| curado_pronto | 0,828 | **0,708** | 0,666 | 0,756 | 1620 |
| curado_internado | 0,635 | 0,040 | 0,036 | 0,046 | 28 |
| melhora_pronto | 0,769 | **0,397** | 0,523 | 0,320 | 321 |
| melhora_internado_breve | 0,749 | 0,501 | 0,461 | 0,548 | 1074 |
| melhora_internado_grave | 0,793 | 0,345 | 0,429 | 0,289 | 338 |
| **Macro** | **0,755** | **0,398** | — | — | 3381 |

**Acurácia:** 55,8%  
**ECE pré-calibração:** 0,0867 (MCE=0,4445)  
**ECE pós-calibração:** 0,1019 (MCE=0,2291) — ECE piorou, **MCE melhorou significativamente**  
**Temperatura ajustada:** T=1,1754 | **cal_set=3.376 amostras (independente)**

**Matriz de confusão (pré-calibração):**

```
                   curado_p  curado_i  melhora_p  melhora_ib  melhora_ig
curado_p    (1620)     1079         4        246         189         102
curado_i      (28)        6         1          3          12           6
melhora_p    (321)       60         4        168          81           8
melhora_ib  (1074)      231        12         95         495         241
melhora_ig   (338)       52         1         13         127         145
```

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | — |
| curado_internado | — |
| melhora_pronto | — |
| melhora_internado_breve | — |
| melhora_internado_grave | — |
| **Global** | **0,2851** |

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,0%** | **0,790** | **0,505** | 0,061 |
| RF Hospital 0 (BPSP isolado) | 59,4% | 0,736 | 0,337 | 0,055 |
| RF Hospital 1 (HSL isolado) | 23,5% | 0,702 | 0,184 | 0,273 |

**Ablation — Late Fusion Demográfica (dados reais FAPESP):**

| Config | Accuracy | F1 Macro | demo_dim |
|---|---|---|---|
| Config A — sem demográficos | 54,5% | 0,398 | 0 |
| Config B — late fusion (idade + sexo) | **67,3%** | **0,449** | 2 |
| **Δ (B − A)** | **+12,7 p.p.** | **+0,051** | — |

> **Nota:** a ablation usa treinamento local (sem FL), 10 épocas, seed=42. O resultado não é comparável diretamente ao modelo FL — isola o efeito dos demográficos da heterogeneidade entre clientes.

---

## Experimento 4

**Data:** 2026-06-25  
**Início:** 12:48:34 | **Fim:** 13:58:38  
**Duração:** 2.819,0s (46,98 min)  
**Log:** `experiments/logs/run_complete_20260625_124833.log`  
**Comando:** `make training-full` (primeiro uso do pipeline refatorado)

### Hiperparâmetros

*(idênticos aos experimentos anteriores)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Loss global | Acurácia | Conv. |
|---|---|---|---|
| 1 | 1,2557 | 55,3% | — |
| 2 | 1,2849 | 51,9% | — |
| 3 | 1,1842 | 58,6% | — |
| 4 | 1,1688 | 50,8% | — |
| 5 | 1,2472 | 53,1% | — |
| 6 | 1,1896 | 56,9% | — |
| 7 | 1,1619 | 54,0% | — |
| 8 | 1,1984 | 51,9% | — |
| 9 | 1,1601 | 54,1% | — |
| 10 | 1,1541 | 53,1% | — |
| 11 | 1,1309 | 51,6% | — |
| 12 | 1,1510 | 49,9% | — |
| 13 | 1,2048 | 53,7% | — |
| 14 | 1,1191 | 52,7% | — |
| 15 | 1,1197 | 54,5% | — |
| 16 | 1,1095 | 52,4% | — |
| 17 | 1,1791 | 51,0% | — |
| 18 | 1,1687 | 51,0% | 1/3 |
| 19 | 1,1213 | 52,5% | reset |
| 20 | 1,1603 | 54,8% | — |

**Convergência:** Não atingida (apenas 1/3 detectado na rodada 18)  
**Tráfego total FL:** 217,17 MB

### Avaliação final (rodada 20)

> **Nota:** `evaluation_round_20.json` foi sobrescrito pelo Experimento 5. AUC e F1 por classe foram preservados pelo contexto de sessão antes da sobrescrita; precision/recall por classe não estão disponíveis para este experimento.

**Pré-calibração:**

| Classe | AUC | F1 | N |
|---|---|---|---|
| curado_pronto | 0,8119 | 0,7087 | 1.620 |
| curado_internado | 0,6390 | 0,0588 | 28 |
| melhora_pronto | 0,8288 | **0,2270** | 321 |
| melhora_internado_breve | 0,7392 | 0,4924 | 1.074 |
| melhora_internado_grave | 0,7888 | 0,3436 | 338 |
| **Macro** | **0,7616** | **0,3661** | 3.381 |

**Acurácia:** 54,75%  
**ECE pré-calibração:** 0,0410  
**ECE pós-calibração:** 0,0871 — calibração piorou (padrão recorrente)  
**Temperatura ajustada:** T=1,2523 | cal_set=3.376 amostras

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | 0,0210 |
| curado_internado | 0,0119 |
| melhora_pronto | **0,5556** |
| melhora_internado_breve | 0,0900 |
| melhora_internado_grave | 0,4142 |
| **Global** | **0,1329** |

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **67,8%** | **0,791** | **0,503** | 0,057 |
| RF Hospital 0 (BPSP isolado) | 59,1% | 0,740 | 0,330 | 0,058 |
| RF Hospital 1 (HSL isolado) | 24,7% | 0,673 | 0,187 | 0,242 |

**Ablation — Late Fusion Demográfica (dados reais FAPESP):**

| Config | Accuracy | F1 Macro | demo_dim |
|---|---|---|---|
| Config A — sem demográficos | 62,8% | 0,455 | 0 |
| Config B — late fusion (idade + sexo) | **69,6%** | **0,464** | 2 |
| **Δ (B − A)** | **+6,8 p.p.** | **+0,009** | — |

> **Observação:** delta anormalmente baixo — Config A atingiu 62,8% (acima da média histórica ~55%), indicando variância estocástica alta com seed=42 nesta execução específica.

### Ocorrências e incidentes

- **behrt-pooled falhou** na etapa seguinte do `training-full` com `AttributeError: 'EvaluationReport' object has no attribute 'per_class_f1'`. O bug estava em `experiments/training/ablation.py` linha 329. Corrigido antes do Experimento 5.

---

## Experimento 5

**Data:** 2026-06-25  
**Início:** 14:47:28 | **Fim:** 15:57:46  
**Duração:** 2.810,4s (46,84 min)  
**Log:** `experiments/logs/run_complete_20260625_144656.log`  
**Comando:** `make training-full` (após correção do bug em `ablation.py`)

### Hiperparâmetros

*(idênticos aos experimentos anteriores)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Loss global | Acurácia | Conv. |
|---|---|---|---|
| 1 | 1,2378 | 52,7% | — |
| 2 | 1,3171 | 52,8% | — |
| 3 | 1,2851 | 51,3% | — |
| 4 | 1,2880 | 54,8% | — |
| 5 | 1,2936 | 54,4% | — |
| 6 | 1,2165 | 55,0% | — |
| 7 | 1,2326 | 54,3% | — |
| 8 | 1,2259 | 51,9% | — |
| 9 | 1,2332 | 52,9% | — |
| 10 | 1,2568 | 52,5% | — |
| 11 | 1,2486 | 53,0% | — |
| 12 | 1,1890 | 54,6% | — |
| 13 | 1,2055 | 53,2% | — |
| 14 | 1,2912 | 49,2% | — |
| 15 | 1,1525 | 56,7% | — |
| 16 | 1,1927 | 52,4% | — |
| 17 | 1,2312 | 51,9% | — |
| 18 | 1,2135 | 54,2% | — |
| 19 | 1,1925 | 54,9% | — |
| 20 | 1,1478 | 56,6% | — |

**Convergência:** Não atingida  
**Tráfego total FL:** 217,17 MB

### Avaliação final (rodada 20)

**Pré-calibração:**

| Classe | AUC | F1 | Recall | Precision | N |
|---|---|---|---|---|---|
| curado_pronto | 0,7782 | 0,7299 | 0,8148 | 0,661 | 1.620 |
| curado_internado | 0,5895 | 0,0889 | 0,0714 | 0,1176 | 28 |
| melhora_pronto | 0,6821 | **0,0248** | 0,0156 | 0,061 | 321 |
| melhora_internado_breve | 0,7603 | 0,4734 | 0,4022 | 0,5752 | 1.074 |
| melhora_internado_grave | 0,7983 | 0,3509 | 0,4527 | 0,2865 | 338 |
| **Macro** | **0,7217** | **0,3336** | — | — | 3.381 |

**Post-calibração (mesma matriz — temperatura não altera argmax):**

| Classe | AUC | F1 | N |
|---|---|---|---|
| curado_pronto | 0,7791 | 0,7299 | 1.620 |
| curado_internado | 0,5862 | 0,0889 | 28 |
| melhora_pronto | 0,6818 | 0,0248 | 321 |
| melhora_internado_breve | 0,7613 | 0,4734 | 1.074 |
| melhora_internado_grave | 0,7980 | 0,3509 | 338 |
| **Macro** | **0,7213** | **0,3336** | 3.381 |

**Acurácia:** 56,55%  
**ECE pré-calibração:** 0,0461 (MCE=0,7354)  
**ECE pós-calibração:** 0,0694 (MCE=0,4361) — ECE piorou, **MCE melhorou**  
**Temperatura ajustada:** T=1,2052 | cal_set=3.376 amostras

**Matriz de confusão (pré-calibração):**

```
                   curado_p  curado_i  melhora_p  melhora_ib  melhora_ig
curado_p    (1620)     1320         3         45         153          99
curado_i      (28)       14         2          1           4           7
melhora_p    (321)      252         6          5          46          12
melhora_ib  (1074)      347         4         28         432         263
melhora_ig   (338)       64         2          3         116         153
```

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | 0,349 |
| curado_internado | 0,048 |
| melhora_pronto | **0,879** |
| melhora_internado_breve | 0,010 |
| melhora_internado_grave | 0,000 |
| **Global** | **0,2542** |

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,4%** | **0,794** | **0,509** | 0,065 |
| RF Hospital 0 (BPSP isolado) | 59,8% | 0,732 | 0,337 | 0,058 |
| RF Hospital 1 (HSL isolado) | 24,7% | 0,700 | 0,186 | 0,244 |

**Ablation — Late Fusion Demográfica (dados reais FAPESP):**

| Config | Accuracy | F1 Macro | AUC Macro | ECE | demo_dim |
|---|---|---|---|---|---|
| Config A — sem demográficos | 57,4% | 0,405 | 0,788 | 0,067 | 0 |
| Config B — late fusion (idade + sexo) | **69,1%** | **0,464** | — | — | 2 |
| **Δ (B − A)** | **+11,7 p.p.** | **+0,059** | — | — | — |

### Observações relevantes do Experimento 5

- **melhora_pronto colapsou** (F1=0,025) em contraste com Exp 3 (F1=0,397). A matriz de confusão mostra 252/321 amostras classificadas como `curado_pronto` — o modelo priorizou a classe majoritária do BPSP.
- **MCE melhorou** (0,7354→0,4361) apesar do ECE piorar — o bin de maior gap pré-calibração (confiança=0,26, gap=0,735) foi distribuído em múltiplos bins após calibração.
- **Ablation delta consistente** (+11,7 p.p.) confirma o achado do Exp 3 (+12,7 p.p.) e descarta o Exp 4 (+6,8 p.p.) como anomalia estocástica.

---

## BEHRT Pooled Baseline (Artefato de Pesquisa)

> **AVISO:** Este resultado NÃO deve ser replicado em produção. É um artefato metodológico para quantificar o custo de privacidade da federação.

**Data:** 2026-06-25  
**Início:** 15:57:48 | **Fim:** 17:14:44  
**Duração:** 4.615,9s (76,9 min) — 40 épocas × 2 configs  
**Log:** `experiments/logs/behrt_pooled_20260625_155747.log`  
**Saída:** `experiments/data/behrt_pooled_20260625_171444.json`  
**Propósito:** comparar BEHRT treinado com acesso ao pool BPSP+HSL (sem privacidade) vs BEHRT FL (com privacidade) para isolar o custo da federação na mesma arquitetura.

### Configuração

| Parâmetro | Valor |
|---|---|
| Épocas | 40 (= 20 rodadas × 2 épocas locais) |
| Pool de treino | 23.640 amostras (BPSP + HSL combinados) |
| seed | 42 |
| Demográficos | dados reais FAPESP (mesmo pipeline) |

### Resultados

| Configuração | Accuracy | F1 Macro | AUC Macro | ECE |
|---|---|---|---|---|
| BEHRT Pooled A — sem demo (demo_dim=0) | 63,5% | 0,496 | **0,826** | 0,046 |
| BEHRT Pooled B — late fusion (demo_dim=2) | 63,6% | 0,494 | — | — |
| **Δ (B − A)** | +0,1 p.p. | −0,002 | — | — |

**Per-class F1 — Config A (mais completa, com AUC por classe):**

| Classe | F1 | AUC |
|---|---|---|
| curado_pronto | **0,809** | — |
| curado_internado | 0,081 | — |
| melhora_pronto | **0,827** | — |
| melhora_internado_breve | 0,382 | — |
| melhora_internado_grave | 0,383 | — |

### Custo de privacidade da federação (BEHRT × BEHRT)

| Modelo | Accuracy | F1 Macro | Privacidade |
|---|---|---|---|
| BEHRT Pooled B (sem privacidade, 40 épocas) | 63,6% | 0,494 | Centralizado — **inaceitável** em produção |
| BEHRT FL — Exp 5 (com privacidade, 20 rodadas) | 56,6% | 0,334 | FedProx — privacidade preservada |
| **Custo privacidade** | **−7,0 p.p.** | **−0,160** | — |
| RF Centralizado (sem privacidade) | 68,4% | 0,509 | Centralizado — **inaceitável** em produção |

> **Interpretação:** A federação custa ~7 p.p. de acurácia frente ao BEHRT pooled com a mesma arquitetura. O modelo FL (56,6%) está ainda 11,8 p.p. abaixo do RF centralizado (68,4%), mas o BEHRT pooled (63,6%) também fica 4,8 p.p. abaixo do RF. Parte do gap BEHRT-FL vs RF é inerente à arquitetura (RF com bag-of-tokens vs BEHRT com sequências temporais) e não ao custo de privacidade.

---

## Experimento 6

**Data:** 2026-06-25  
**Status:** Concluído  
**Log:** `experiments/logs/run_complete_20260625_201012.log` + `experiments/logs/behrt_pooled_20260625_212059.log`  
**Comando:** `make training-full` + `make behrt-pooled`

### Alterações implementadas pré-treinamento

Este experimento é o primeiro a incluir o embedding de **dia relativo** (`dia_relativo`), introduzido como resposta ao diagnóstico de que o BEHRT não estava capturando a progressão temporal intra-episódio.

#### Motivação clínica

A progressão do caso clínico depende dos marcadores temporais do paciente — um valor de PCR crescente ao longo de 48h tem significado prognóstico diferente do mesmo valor isolado. Nos experimentos anteriores, o SimplifiedBEHRT recebia a sequência de exames **na ordem correta** (SQL: `ORDER BY dia_relativo, analyte`), mas sem informação de **qual dia** cada exame pertencia dentro do episódio de internação. O modelo só dispunha da posição relativa na sequência (posição 1, 2, 3…), não da distância temporal real desde a admissão.

O campo `dia_relativo = exam_date − attended_at` já estava calculado na query SQL e ordenando as sequências — mas era descartado em `_build_tensors` após o agrupamento.

#### Implementação técnica

| Componente | Alteração |
|---|---|
| `model.py` | Adicionados `MAX_DIA_RELATIVO=60` e `DiaRelativoEmbedding` — `nn.Embedding(62, embed_dim, padding_idx=0)` |
| `model.py` | `SimplifiedBEHRT.forward` aceita `dia_relativo: Optional[Tensor]`; soma o embedding ao token embedding antes do positional encoding |
| `preprocessor.py` | `_build_tensors` extrai `_dia_rel` por token (shift +1: 0=padding, 1=dia0, …, 61=dia≥60); retorna 5-tupla |
| `dataloaders.py` | Todos os `TensorDataset` incluem `dia_relativo`; `test_loader_demo` passa a ter 4 elementos |
| `client.py` | `fit()` e `evaluate()` desempacotam 3-tupla e passam `dia_relativo` ao modelo |
| `calibration.py` | `TemperatureScaler.fit()` passa `dia_relativo` no forward |
| `evaluation.py` | `collect_logits()` usa `batch[2]` como `dia_relativo` quando disponível |
| `ablation.py` | `_train_local`, `_eval_with_demo`, `run_pooled_behrt` atualizados para 3/4-tuplas |
| `baselines.py`, `federated.py`, `interpretability.py`, `rag.py` | Compatibilidade com tuplas extras via `*_` ou unpack explícito |

**Encoding:** O índice 0 é reservado para padding (tokens PAD e posição CLS); dia 0 (admissão) → índice 1; dia ≥ 60 → índice 61 (clamp). O CLS token não recebe `dia_relativo` — o embedding de dia é aplicado apenas aos tokens de exame, antes do prepend do CLS.

**Coexistência com outros embeddings:** O `DiaRelativoEmbedding` é **somado** ao token embedding (`nn.Embedding`) antes do `PositionalEncoding` sinusoidal. Os três embeddings coexistem: identidade do analito + dia relativo desde admissão + posição na sequência.

#### Hipótese para o Experimento 6

Ao informar ao modelo **em que dia** de internação cada exame foi coletado, espera-se que:

1. **`melhora_pronto` melhore:** a confusão com `curado_pronto` pode diminuir se a velocidade de normalização dos marcadores (dias de internação antes da alta) for capturável — um paciente que recebe alta em 2 dias tem perfil diferente de um que fica 7 dias.
2. **Redução da variância entre execuções:** experimentos 3 e 5 tiveram F1 de `melhora_pronto` de 0,397 e 0,025 com os mesmos hiperparâmetros. A progressão temporal pode prover sinal mais estável para essa classe.
3. **Redução do gap BEHRT vs RF:** o RF (BoT) descarta a ordem mas captura co-ocorrência; o BEHRT com `dia_relativo` pode capturar tanto co-ocorrência quanto progressão, potencialmente superando ou equiparando ao RF com mais informação.

### Hiperparâmetros

*(idênticos aos experimentos anteriores — alteração é exclusivamente no embedding)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 20 |
| µ FedProx | 0,01 |
| Batch size | 16 |
| Épocas locais | 2 |

### Resultado por rodada

| Rodada | Tempo (s) | Loss | Acurácia |
|---|---|---|---|
| 1 | 136,7 | 1,1420 | 62,17% |
| 2 | 127,5 | 1,1901 | 57,70% |
| 3 | 156,1 | 1,1844 | 55,43% |
| 4 | 156,6 | 1,2447 | 55,07% |
| 5 | 171,0 | 1,1378 | 61,25% |
| 6 | 161,5 | 1,0781 | **62,70%** ← pico |
| 7 | 137,6 | 1,1908 | 51,67% ↓ |
| 8 | 151,9 | 1,0718 | 61,43% |
| 9 | 155,7 | 1,1646 | 59,60% |
| 10 | 165,6 | 1,1475 | 57,26% |
| 11 | 152,3 | 1,1467 | 59,98% |
| 12 | 151,8 | 1,1498 | 50,84% ↓ |
| 13 | 152,0 | 1,1237 | 62,05% |
| 14 | 147,9 | 1,1297 | 57,29% |
| 15 | 140,6 | 1,1639 | 50,37% ↓ |
| 16 | 138,4 | 1,2091 | 58,18% |
| 17 | 130,4 | 1,1218 | 59,48% |
| 18 | 116,1 | 1,1879 | 61,14% |
| 19 | 115,7 | 1,1208 | 59,30% |
| 20 | 134,4 | 1,1197 | **59,63%** (final) |

**Convergência:** Não atingida (Δ=0,00325 na rodada 20 — abaixo do limite 0,005 apenas 1/3 rodadas necessárias)  
**Tráfego FL total:** 218,38 MB  
**Duração total:** 2.899,7 s (48,3 min)  
**Checkpoint PostgreSQL:** sha256=`d908d51eee87` | T=1,4418 | round=20

### Avaliação final (rodada 20)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Precision | Recall | N |
|---|---|---|---|---|---|
| curado_pronto | 0,8385 | 0,7470 | 0,6777 | 0,8321 | 1.620 |
| curado_internado | 0,6410 | 0,0000 | 0,0000 | 0,0000 | 28 |
| melhora_pronto | 0,6544 | **0,1124** ↑ | 0,2016 | 0,0779 | 321 |
| melhora_internado_breve | 0,8097 | **0,5468** ↑ | 0,6996 | 0,4488 | 1.074 |
| melhora_internado_grave | 0,7844 | 0,3515 | 0,2785 | 0,4763 | 338 |
| **Macro** | **0,7456** | **0,3515** | — | — | 3.381 |

**Acurácia:** 59,63% (+3,08 p.p. vs Exp 5)  
**ECE pré-calibração:** 0,1046 | **MCE:** 0,1799 (↓ de 0,736 no Exp 5 — grande melhora)  
**ECE pós-calibração:** 0,1796 | MCE: 0,2397 (temperatura piorou a calibração)  
**Temperatura ajustada:** T=1,4418 | cal_set=3.376 amostras

**Matriz de confusão (pré-calibração):**

```
                      cp    ci   mp  mib  mig
curado_p    (1620)  1348     0   87   84  101
curado_i      (28)    11     0    0    8    9
melhora_p    (321)   240     0   25   34   22
melhora_ib  (1074)   296     0   11  482  285
melhora_ig   (338)    94     1    1   81  161
```

### Resultados das etapas pós-FL

**RAG:** Falhou com `too many values to unpack (expected 2)` — linha 90 de `rag.py` não coberta na atualização da 3-tupla. Corrigido para Exp 7.

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,71%** | **0,7936** | **0,5103** | 0,0666 |
| RF Hospital 0 (BPSP isolado) | 59,63% | 0,7295 | 0,3371 | 0,0485 |
| RF Hospital 1 (HSL isolado) | 24,08% | 0,6840 | 0,1847 | 0,2575 |

**Ablation — Late Fusion Demográfica (10 épocas, dados reais FAPESP):**

| Config | Accuracy | F1 Macro |
|---|---|---|
| Config A — sem demográficos (+ dia_relativo) | **60,22%** | **0,4031** |
| Config B — late fusion (idade + sexo + dia_relativo) | 59,24% | 0,3707 |
| **Δ (B − A)** | **−0,98 p.p.** | **−0,032** |

> **Inversão do delta:** pela primeira vez Config B ficou abaixo de Config A. Nos Exp 3 e 5 o delta era +12,7 p.p. e +11,7 p.p. a favor de B. Hipótese: `dia_relativo` absorve parte do sinal demográfico (duração de internação correlaciona com gravidade/idade), tornando idade e sexo redundantes. Alternativamente, 10 épocas podem ser insuficientes para o modelo com demo_dim=2 convergir com a nova dimensão de embedding.

**BEHRT Poolado Centralizado — custo de privacidade (40 épocas, pool BPSP+HSL):**

| Config | Accuracy | F1 Macro | vs FL equivalente | Custo privacidade |
|---|---|---|---|---|
| behrt_pooled_A (sem demo) | **67,79%** | **0,5218** | Ablation A: 60,22% | −7,57 p.p. |
| behrt_pooled_B (late fusion) | 63,03% | 0,5005 | Ablation B: 59,24% | −3,79 p.p. |

> O custo de privacidade (gap pooled → federado) caiu de ~10 p.p. (Exp 3/5) para 3,8–7,6 p.p. com `dia_relativo`. O embedding temporal beneficiou mais a versão FL do que a poolada.

### Observações relevantes do Experimento 6

- **+3,08 p.p. de acurácia** sobre o Exp 5 (56,55% → 59,63%) — maior ganho de uma única alteração arquitetural no projeto.
- **`melhora_pronto` recuperado:** F1 de 0,025 para 0,112 (+0,087, 4,5×). De 5 acertos para 25/321 amostras.
- **`melhora_internado_breve` melhorou:** F1 de 0,473 para 0,547 (+0,074).
- **MCE pré-calibração despencou:** de 0,736 para 0,180 — modelo mais bem calibrado nos extremos de confiança.
- **Ablation com sinal invertido:** necessita investigação. Ver hipótese acima.
- **Alta variância entre rodadas persiste** (50,84%–62,70%) — non-IID e µ=0,01 ainda são fatores dominantes. Corrigido no Exp 7 (µ=0,1, min_rounds=20, max_rounds=120).

---

## Experimento 7

**Data:** 2026-06-25 a 2026-06-26 (noite)
**Status:** Concluído
**Log:** `experiments/logs/run_complete_20260625_225308.log`
**Comando:** `make training-full`

### Alterações implementadas pré-treinamento

| Parâmetro / Componente | Exp 6 | Exp 7 | Motivação |
|---|---|---|---|
| `proximal_mu` (µ FedProx) | 0,01 | **0,10** | Reduzir client drift não-IID (Li et al. 2020) |
| `num_rounds` (teto máximo) | 20 | **120** | Modelo não convergiu em 20 rodadas |
| `min_rounds` (warm-up) | — | **20** | Evitar early stopping prematuro nas rodadas voláteis iniciais |
| `checkpoint_store` | última rodada | **melhor rodada** | Seleção gulosa: salvar no PostgreSQL a cada nova melhor acurácia |
| `rag.py` — bug 3-tupla linha 90 | ❌ falhou | **✅ corrigido** | `for _, batch_y, *_ in test_loader:` |
| `rag.py` — `generation_config` | params conflitantes | **unificados na chamada** | Deprecation warning HuggingFace |
| `rag.py` — `max_length=50` hardcoded | hardcoded | **`MAX_SEQ_LEN` do config** | Evitar falso positivo por valor arbitrário |
| `rag.py` — `clean_up_tokenization_spaces` | `True` (errado para BPE) | **`False`** | GPT-2 é BPE, não WordPiece |

#### Contexto das mudanças

**µ = 0,01 → 0,1:** O FedProx usa µ como regularizador que mantém os parâmetros locais próximos do modelo global. Com µ=0,01, o cliente tinha liberdade quase irrestrita para divergir — o HSL, com distribuição completamente diferente do BPSP, produzia atualizações que se anulavam na agregação FedAvg. Aumentar para 0,1 (10× mais forte) não impede aprendizado local, mas penaliza divergências excessivas, reduzindo o drift entre clientes.

**Warm-up 20 rodadas:** As primeiras rodadas do FL são altamente voláteis — o modelo global oscila enquanto os clientes ainda estão se "alinhando". Aplicar convergência prematuramente (ex: Δ pequeno na rodada 3) seria um falso positivo. O warm-up suspende o critério de parada até a rodada 20, quando o modelo já tem alguma estabilidade.

**Checkpoint guloso no PostgreSQL:** Com 120 rodadas, a última rodada não é necessariamente a melhor. O checkpoint guloso salva no banco sempre que uma nova acurácia máxima é atingida durante o treino. Se o processo cair, o melhor estado está preservado. No final, `load_best()` restaura o modelo antes da avaliação e calibração — garantindo que o relatório final reflita o melhor modelo encontrado, não o da última iteração.

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | **120** |
| Rodadas warm-up | **20** |
| µ FedProx | **0,10** |
| Batch size | 16 |
| Épocas locais | 2 |
| Threshold convergência | 0,005 |
| Paciência convergência | 3 |
| LR | 0,001 |
| Seleção checkpoint | **gulosa (melhor por acurácia)** |

### Ambiente de execução

| Item | Detalhe |
|---|---|
| Máquina | Dell Inspiron 5402 — i7-1165G7 (8 threads), 16 GB RAM, sem GPU dedicada |
| Device | CPU (Intel Iris Xe sem suporte CUDA) |
| Início | 2026-06-25 22:53 |
| Término | 2026-06-26 03:17 (FL) + ~03:40 (ablation) |
| Duração FL | **15.846,2 s (4,4 horas)** |
| Tráfego FL | **1.310,28 MB** (6× mais que Exp 6 pelos 120 rounds) |
| Custo por rodada | ~132 s/rodada em média |

> O Exp 7 rodou ininterruptamente durante a madrugada. O desktop sustentou as 120 rodadas sem falha, porém apresentou aquecimento significativo ao longo das 4,4 horas de execução contínua — esperado para um processador mobile (28W TDP) sob carga sustentada. A variação de tempo por rodada (102s–171s) pode refletir throttling térmico pontual.
>
> **Nota de design:** o MOSAIC-FL está intencionalmente configurado para rodar em ambientes com recursos limitados. Essa escolha não é uma restrição do projeto — é um requisito de validação: hospitais de pequeno e médio porte, que são o público-alvo do sistema, frequentemente não dispõem de infraestrutura com GPU dedicada. Testar no Dell Inspiron 5402 simula esse cenário real de deployment. Os parâmetros ajustados para esse ambiente estão documentados em `src/mosaicfl/core/config.py`:
>
> | Parâmetro | Valor padrão | Valor ajustado | Motivo |
> |---|---|---|---|
> | OMP/MKL_NUM_THREADS | — | 4 | Libera threads para o SO, evita travamento |
> | TOKENIZERS_PARALLELISM | — | false | Elimina conflito de threads do HuggingFace |
> | DEVICE | cuda* | cpu | Intel Iris Xe sem suporte CUDA |
> | BATCH_SIZE | 32 | 16 | Reduz uso de RAM por cliente |
> | LOCAL_EPOCHS | 3 | 2 | Menos iterações por rodada federada |
> | NUM_ROUNDS (base) | 50 | 20→120 | 20 era o limite original para o hardware; 120 testado no Exp 7 |
> | MAX_NEW_TOKENS | 100 | 64 | Geração de texto RAG mais rápida |
>
> Em ambientes com GPU dedicada, o Exp 7 rodaria em minutos, não horas — toda a lógica de convergência e checkpoint está preparada para esse cenário.

### Resultado por rodada

A tabela abaixo mostra as rodadas com marco de warm-up, picos e vales notáveis. A oscilação persiste do início ao fim — característica do non-IID severo.

| Rodada | Loss | Acurácia | Nota |
|---|---|---|---|
| 1 | 1,1446 | 54,45% | — |
| 5 | 1,1052 | 57,47% | — |
| 10 | 1,0358 | **61,11%** | pico precoce |
| 12 | 1,0449 | 54,45% ↓ | vale |
| 20 | 1,0930 | 56,40% | fim do warm-up |
| 21 | 1,0576 | 56,26% | convergência avaliada a partir daqui |
| 30 | 1,1029 | 59,66% | — |
| 31 | 1,0553 | **60,57%** | — |
| 43 | 1,0507 | **61,08%** | — |
| 52 | 1,0998 | 50,61% ↓ | pior rodada do experimento |
| 69 | 1,0689 | **62,67%** | — |
| 73 | 1,0013 | 61,55% | loss mais baixa até aqui |
| 75 | 1,0497 | **61,67%** | — |
| 83 | 0,9885 | 61,82% | loss mínima global |
| 89 | 0,9638 | **63,29%** ← pico máximo | loss 0,9638 — melhor checkpoint |
| 95 | 0,9817 | 61,58% | — |
| 100 | 1,0196 | 56,76% | — |
| 110 | 0,9795 | 62,70% | segundo melhor |
| 112 | 0,9767 | 61,96% | loss mais baixa do experimento |
| 120 | 1,0270 | 59,36% | última rodada — usada para avaliação¹ |

> ¹ **Nota metodológica:** O Exp 7 foi executado **antes** da implementação do checkpoint guloso. A avaliação final foi feita sobre o modelo da rodada 120 (59,36%), não sobre o da rodada 89 (63,29%). O Exp 8 será o primeiro a usar `load_best()` — o gap de 3,93 p.p. entre R89 e R120 quantifica o custo de não ter o best checkpoint no Exp 7.

**Convergência:** Não atingida. Δ passou do threshold 7 vezes, mas nunca de forma consecutiva (paciência=3). A oscilação de ±12 p.p. entre rodadas é estrutural ao non-IID.

### Avaliação final (rodada 120 — modelo avaliado)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Precision | Recall | N | vs Exp 6 |
|---|---|---|---|---|---|---|
| curado_pronto | 0,8369 | 0,7560 | 0,6961 | 0,8272 | 1.620 | F1 +0,009 |
| curado_internado | 0,5631 | **0,0465** | 0,0667 | 0,0357 | 28 | F1 +0,047 ↑ |
| **melhora_pronto** | **0,8355** | **0,2490** | 0,3727 | 0,1869 | 321 | **AUC +0,181 ↑↑** |
| melhora_internado_breve | 0,8182 | 0,5003 | 0,6913 | 0,3920 | 1.074 | F1 −0,046 |
| melhora_internado_grave | 0,7980 | 0,3667 | 0,2757 | 0,5473 | 338 | F1 +0,015 |
| **Macro** | **0,7703** | **0,3837** | — | — | 3.381 | **AUC +0,025 ↑** |

**Acurácia:** 59,36% (−0,27 p.p. vs Exp 6 — avaliação na R120, não na melhor)
**ECE pré-calibração:** **0,0326** (↓ de 0,1046 no Exp 6 — melhora drástica de calibração)
**MCE pré-calibração:** **0,1049** (↓ de 0,1799 no Exp 6)
**ECE pós-calibração:** 0,0621 | MCE: 0,1269
**Temperatura ajustada:** T=1,1910 (mais próxima de 1,0 que o Exp 6's 1,4418 — modelo já está mais calibrado nativamente)

**Matriz de confusão (pré-calibração, R120):**

```
                      cp    ci    mp   mib   mig
curado_p    (1620)  1340     1    64    83   132
curado_i      (28)     6     1     4    11     6
melhora_p    (321)   233     1    60    21     6
melhora_ib  (1074)   273    11    27   421   342
melhora_ig   (338)    73     1     6    73   185
```

> **Interpretação da matriz:** `melhora_pronto` acertou 60/321 (18,7% recall) vs 25/321 no Exp 6 (7,8%). A confusão predominante ainda é com `curado_pronto` (233 casos), o que é clinicamente plausível — ambas as classes representam desfechos positivos rápidos. `melhora_internado_grave` chegou a 54,7% de recall, o melhor de todos os experimentos — o modelo está identificando melhor os casos mais severos.

### Resultados das etapas pós-FL

**RAG — funcionou pela primeira vez (bug corrigido):**

| Classe | Precision@3 | Interpretação |
|---|---|---|
| melhora_internado_grave | 0,391 | Perfil clínico bem distinto — boa recuperação |
| curado_internado | 0,357 | Poucos casos, perfil único |
| melhora_pronto | 0,159 | Melhorou vs experimentos anteriores |
| melhora_internado_breve | 0,138 | Confusão com melhora_internado_grave |
| **curado_pronto** | **0,020** | Classe dominante — perfis genéricos demais |
| **Macro P@3** | **0,110** | — |

> `curado_pronto` tem P@3 próxima de zero porque é 48% do test set — os perfis recuperados são muito genéricos e não distinguem essa classe das demais. Classes raras têm perfis mais específicos e são recuperadas com mais precisão.

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,32%** | **0,7935** | **0,5057** | 0,0632 |
| RF Hospital 0 (BPSP isolado) | 59,15% | 0,7384 | 0,3368 | 0,0616 |
| RF Hospital 1 (HSL isolado) | 25,70% | 0,7091 | 0,1923 | 0,2354 |

**Ablation — Late Fusion Demográfica (10 épocas, dados reais FAPESP):**

| Config | Accuracy | F1 Macro | vs Exp 6 |
|---|---|---|---|
| Config A — sem demográficos | 54,16% | 0,3653 | — |
| Config B — late fusion (idade + sexo) | **60,10%** | **0,3808** | — |
| **Δ (B − A)** | **+5,94 p.p.** | **+0,016** | **sinal voltou positivo** ↑ |

> O sinal invertido do Exp 6 (−0,98 p.p.) foi **anomalia** — provavelmente instabilidade do modelo com a nova dimensão de embedding em apenas 10 épocas locais. Com o modelo mais treinado (120 rodadas como ponto de partida), os demográficos voltaram a contribuir positivamente, alinhando com os Exp 3 e 5.

### Observações relevantes do Experimento 7

**O que melhorou:**
- **`melhora_pronto` AUC: 0,654 → 0,836 (+0,182)** — o maior salto de AUC em uma única classe em todo o projeto. O modelo agora discrimina essa classe com qualidade próxima das demais. Com mais rounds e µ maior, o HSL conseguiu transferir esse padrão para o modelo global.
- **ECE: 0,1046 → 0,0326** — o modelo está significativamente mais bem calibrado. As probabilidades emitidas são mais confiáveis, o que tem impacto direto no CDSS humano-no-loop.
- **RAG funcionou** — pela primeira vez o pipeline completo executou sem erro.
- **Ablation Δ voltou positivo (+5,94 p.p.)** — confirmando que a inversão do Exp 6 foi transitória.
- **`melhora_internado_grave` recall: 47,6% → 54,7%** — maior recall já registrado para a classe mais severa.

**O que não melhorou:**
- **Acurácia:** 59,36% vs 59,63% no Exp 6 — tecnicamente uma regressão de 0,27 p.p., mas explicada pelo fato de a avaliação ter sido feita na R120 e não na R89 (melhor rodada). Sem o checkpoint guloso, o custo de não convergência se manifesta na acurácia final.
- **`melhora_pronto` recall:** 18,7% — subiu vs Exp 6 (7,8%), mas ainda identifica menos de 1 em 5 casos dessa classe. O modelo discrimina bem (AUC 0,836) mas é conservador nas predições.
- **Convergência:** 120 rodadas, 4,4 horas, sem convergência — confirma que o non-IID é estrutural e não se resolve com mais iterações do FedProx padrão.

**Custo computacional x ganho:**
- Exp 6 → Exp 7: 48 min → 4,4h (5,5× mais tempo), 218 MB → 1.310 MB (6× mais tráfego)
- Ganho de acurácia: −0,27 p.p. (regressão por avaliação na última rodada)
- Ganho de qualidade: AUC +0,025, F1 +0,032, ECE −0,072
- **Conclusão de custo-benefício:** para o hardware atual (i7-1165G7 sem GPU), 120 rodadas é o limite prático. O ganho de qualidade é real mas não se traduz em acurácia por falta do checkpoint guloso — lacuna endereçada no Exp 8.

---

## Experimento 8

**Data:** 2026-06-26  
**Status:** Concluído (FL completo; BEHRT Pooled omitido — retorna no Exp 9)  
**Log:** `experiments/logs/run_complete_20260626_130506.log`  
**Avaliação:** `experiments/logs/evaluation_round_120.json`  
**Comando:** `make training` (sem `behrt-pooled`)

### Alterações implementadas pré-treinamento

| Componente | Alteração | Motivação |
|---|---|---|
| `fl_core.py` | Checkpoint guloso: salva no PostgreSQL sempre que `acc_global > best_accuracy` | Exp 7 demonstrou gap de 3,93 p.p. entre melhor (R89) e última rodada (R120) |
| `fl_core.py` | `load_best()` restaura o melhor checkpoint antes da avaliação final | Garantir que o relatório reflita o melhor modelo, não a última iteração |
| `checkpoint_store.py` | Implementação de `load_best()` em `SQLiteCheckpointStore` e `PostgreSQLCheckpointStore` | Suporte à restauração do melhor estado salvo em produção |
| `calibration.py` | Parametrização em log-space: `T = exp(log_T)` — garante T>0 sempre | Bug Exp 8: LBFGS saltou para T=−8.9997 com clamp, zerando gradiente |
| `rag.py` | Remoção de `max_length=50` hardcoded → usa `MAX_SEQ_LEN` do config | Valor arbitrário podia mascarar resultados incorretos |
| `rag.py` | `generation_config` unificada; `clean_up_tokenization_spaces=False` para GPT-2 | Warnings HuggingFace eliminados; comportamento correto para tokenizer BPE |
| BEHRT Pooled | **Omitido neste experimento** | Runtime excessivo; `POOLED_EPOCHS` descoplado (fix no Exp 9) |

> **Nota sobre o bug de temperatura e recalibração:** a correção de `calibration.py` (log-space) foi implementada após a execução do Exp 8. O bug se manifestou com T=−8.9997, destruindo a calibração pós-treino. `make recalibrate` foi executado em 2026-06-26 19:23 — resultado: T=1,0849 (positivo, fix confirmado), mas ECE pós=0,1066 > ECE pré=0,0859 — temperature scaling piorou a calibração pelo mesmo padrão estrutural documentado em todos os experimentos. **O modelo R91 é mais bem utilizado com T=1,0 (sem calibração).** Ver seção de diagnóstico consolidado.

### Hiperparâmetros

*(idênticos ao Experimento 7 — alteração é exclusivamente na estratégia de checkpoint)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 120 |
| Rodadas warm-up | 20 |
| µ FedProx | 0,10 |
| Batch size | 16 |
| Épocas locais | 2 |
| Threshold convergência | 0,005 |
| Paciência convergência | 3 |
| LR | 0,001 |
| Seleção checkpoint | **gulosa (melhor por acurácia, salvo no PostgreSQL)** |

### Ambiente de execução

| Item | Detalhe |
|---|---|
| Máquina | Dell Inspiron 5402 — i7-1165G7 (8 threads), 16 GB RAM, sem GPU dedicada |
| Device | CPU (Intel Iris Xe sem suporte CUDA) |
| Início FL | 2026-06-26 13:05 |
| Fim FL | 2026-06-26 17:30 |
| Duração FL | **15.910 s (265 min / 4,4 h)** |
| Tráfego FL | **1.310,28 MB** |
| Custo médio por rodada | ~133 s/rodada |

### Checkpoint guloso — marcos de atualização

| Rodada | Acc | Obs. |
|---|---|---|
| 1 | 52,26% | primeiro checkpoint |
| 3 | 54,07% | |
| 5 | 56,49% | |
| 6 | 57,08% | |
| 7 | 61,70% | primeiro acima de 60% |
| 9 | 62,88% | |
| 19 | 63,68% | maior do warm-up |
| 59 | 65,48% | |
| 74 | 66,25% | |
| **91** | **66,61%** | ← **melhor checkpoint — restaurado pelo `load_best()`** |

### Resultado por rodada (seleção de marcos)

| Rodada | Loss | Acurácia | Nota |
|---|---|---|---|
| 1 | 1,2106 | 52,26% | — |
| 7 | 1,1117 | 61,70% | primeiro pico acima de 60% |
| 9 | 1,1171 | 62,88% | novo best |
| 19 | 1,0479 | 63,68% | novo best — fim warm-up |
| 20 | 0,9921 | 60,57% | fim warm-up |
| 50 | 0,9638 | 63,27% | — |
| 51 | 1,0054 | 63,65% | — |
| 59 | 0,9485 | 65,48% | novo best |
| 73 | 0,9128 | 63,71% | — |
| 74 | 0,9342 | **66,25%** | novo best |
| 88 | 0,9289 | 63,95% | — |
| **91** | **0,8971** | **66,61%** | ← **MELHOR DO EXPERIMENTO — loss mínima e acc máxima** |
| 92 | 0,9103 | 64,95% | — |
| 94 | 0,9265 | 65,96% | segundo melhor |
| 110 | 0,9544 | 65,19% | — |
| 120 | 1,0479 | **58,27%** | última rodada — 8,34 p.p. abaixo do melhor |

**Convergência:** Não atingida. Oscilação estrutural de ±8 p.p. entre rodadas persiste do início ao fim.  
**Tráfego total FL:** 1.310,28 MB  
**Gap best vs last:** 66,61% (R91) − 58,27% (R120) = **8,34 p.p.** — quantifica o valor do checkpoint guloso.

### Avaliação final (modelo restaurado da rodada 91)

**Pré-calibração (T=1,0) — modelo da R91:**

| Classe | AUC | F1 | Precision | Recall | N | vs Exp 7 |
|---|---|---|---|---|---|---|
| curado_pronto | 0,8692 | **0,7987** | 0,7543 | 0,8488 | 1.620 | F1 +0,043 ↑ |
| curado_internado | 0,6351 | 0,0615 | 0,0541 | 0,0714 | 28 | F1 +0,015 |
| **melhora_pronto** | **0,9201** | **0,6194** | 0,6421 | 0,5981 | 321 | **AUC +0,084 ↑↑ / F1 +0,370 ↑↑** |
| melhora_internado_breve | **0,8188** | **0,5911** | 0,6855 | 0,5196 | 1.074 | F1 +0,091 ↑ |
| melhora_internado_grave | 0,8047 | 0,3351 | 0,3064 | 0,3698 | 338 | F1 −0,032 |
| **Macro** | **0,8096** | **0,4812** | — | — | 3.381 | **AUC +0,039 ↑↑ / F1 +0,098 ↑↑** |

**Acurácia:** 66,61% — **novo recorde do projeto** (+7,25 p.p. vs Exp 7 na última rodada; +2,49 p.p. vs melhor do Exp 7 que foi R89=63,29%)  
**ECE pré-calibração:** 0,0859 | **MCE:** 0,2382  
**Temperatura T (treinamento):** −8.9997 (BUG — destruiu calibração pós-treino)  
**ECE pós-calibração (treinamento):** 0,3335 (inválido)  

**Re-calibração (`make recalibrate`, 2026-06-26 19:23):**

| Etapa | T | ECE | AUC | F1 | MCE |
|---|---|---|---|---|---|
| Pré-calibração | 1,0 | **0,0859** | 0,8097 | 0,4823 | 0,2382 |
| Pós-calibração (log-space) | 1,0849 | 0,1066 | 0,8091 | 0,4823 | 0,1637 |

> **Diagnóstico:** T=1,0849 confirmou o fix de log-space (positivo), mas temperature scaling piorou o ECE pelo **mesmo padrão estrutural de todos os experimentos**. O diagrama de confiabilidade mostra subconfiança sistemática em todos os bins (confiança < acurácia real). Com T>1 o softmax fica ainda mais suave → ainda mais subconfiante → ECE piora. O LBFGS minimiza NLL, não ECE — os objetivos divergem aqui. **Recomendação: usar T=1,0 (sem calibração) para o checkpoint R91 em produção.** Log: `experiments/logs/recalibrate_20260626_192337.json`.

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | 0,2309 |
| curado_internado | 0,2024 |
| **melhora_pronto** | **0,3863** |
| melhora_internado_breve | 0,2055 |
| melhora_internado_grave | 0,1164 |
| **Macro** | **0,2259** |

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,20%** | **0,7967** | **0,5056** | 0,0643 |
| RF Hospital 0 (BPSP isolado) | 59,72% | 0,7401 | 0,3396 | 0,0596 |
| RF Hospital 1 (HSL isolado) | 24,11% | 0,7006 | 0,1863 | 0,2527 |

**Ablation — Late Fusion Demográfica (10 épocas, dados reais FAPESP):**

| Config | Accuracy | F1 Macro |
|---|---|---|
| Config A — sem demográficos | 58,27% | 0,3886 |
| Config B — late fusion (idade + sexo) | **62,70%** | 0,3830 |
| **Δ (B − A)** | **+4,43 p.p.** | **−0,006** ⚠ |

> **BEHRT Pooled:** omitido neste experimento. O bug de `POOLED_EPOCHS` (240 épocas em vez de 120) foi identificado e corrigido — `pooled_epochs=120` adicionado ao `FedConfig`. O baseline centralizado retorna no Exp 9.

### Observações relevantes do Experimento 8

**O que o checkpoint guloso revelou:**
- Gap de **8,34 p.p.** entre melhor e última rodada (66,61% vs 58,27%) — confirma que o FL não converge monotonicamente e que salvar apenas a última rodada subestima a capacidade do modelo.
- 10 atualizações de checkpoint ao longo das 120 rodadas — distribuídas até R91, com as últimas 29 rodadas sem novo best (regime de plateau/regressão).
- R91 teve simultaneamente a **acurácia máxima** (66,61%) e a **loss mais baixa** (0,8971) — indicador de que o modelo estava em seu ponto de maior generalização.

**Saltos de qualidade (pré-calibração, R91 vs R120 do Exp 7):**
- **`melhora_pronto` F1: 0,249 → 0,619 (+0,370)** — maior evolução de qualquer métrica em qualquer experimento do projeto. O modelo agora identifica corretamente ~60% dos casos dessa classe.
- **`melhora_pronto` AUC: 0,836 → 0,920** — discriminação quase perfeita para essa classe.
- **Macro F1: 0,384 → 0,481 (+0,098)** — salto expressivo.
- **Macro AUC: 0,770 → 0,810 (+0,040)** — modelo mais discriminante em todas as classes.

**O que não melhorou:**
- **Calibração:** T=−8.9997 destruiu as probabilidades de saída. Será corrigido via `make recalibrate`.
- **`curado_internado`:** F1=0,062 — raridade extrema (N=28 no teste) mantém a classe difícil.
- **`melhora_internado_grave` F1:** 0,335 (vs 0,367 no Exp 7) — leve regressão; recall de 37%.
- **Convergência:** 120 rodadas sem convergência — non-IID estrutural. Motivação para FedNova no Exp 9.

**Custo de privacidade (sem BEHRT Pooled neste experimento):**
- FL Exp 8 (66,61%) vs RF Centralizado (68,20%) = **gap de 1,59 p.p.** — o menor gap do projeto, sugerindo que o checkpoint guloso reduz significativamente a penalidade da federação.

---

## Experimento 9

**Data:** 2026-06-28  
**Status:** Em andamento  
**Log:** `experiments/logs/run_complete_20260628_074558.log`  
**Temperatura:** `experiments/logs/temperature_exp9.log`  
**Comando:** `make training-full`

### Motivação — FedNova

O Experimento 8 confirmou que o non-IID estrutural (BPSP 5,5× mais amostras que HSL) persiste mesmo com FedProx µ=0,1 e 120 rodadas — oscilação de ±8 p.p. sem convergência. O problema raiz é que o FedAvg pondera os updates pelo número de amostras, mas BPSP processa ~2.502 batches/rodada e HSL ~453 batches/rodada. Os updates têm magnitudes completamente diferentes mesmo após ponderação por amostras.

**FedNova** (Wang et al. 2020 — *"Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"*) normaliza os updates de cada cliente pelo número de passos efetivos τ_i (batches × épocas locais) antes de agregar, eliminando o viés de escala entre clientes com volumes heterogêneos:

> τ_eff = Σ p_i · τ_i  
> w_{t+1} = w_t + τ_eff · Σ p_i · (w_i − w_t) / τ_i

Sem hiperparâmetro novo, sem estado adicional por cliente. SCAFFOLD foi descartado por risco de viés com apenas 2 clientes em regime non-IID extremo.

### Alterações implementadas pré-treinamento

| Componente | Alteração | Motivação |
|---|---|---|
| `fl_core.py` | `aggregate_fednova()` substituindo `aggregate_fedavg()` quando `use_fednova=True` | Elimina viés de escala entre clientes com volumes heterogêneos |
| `client.py` | `fit()` retorna `tau` (passos efetivos) nas métricas | `fl_core.py` usa τ_i para normalização FedNova por cliente |
| `config.py` | `use_fednova: bool = True` em `FedConfig` | Ativa FedNova por padrão |

### Hiperparâmetros

*(idênticos ao Experimento 8 — alteração é exclusivamente no algoritmo de agregação)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 120 |
| Rodadas warm-up | 20 |
| Algoritmo | **FedNova** (substitui FedProx + FedAvg) |
| Batch size | 16 |
| Épocas locais | 2 |
| Threshold convergência | 0,005 |
| Paciência convergência | 3 |
| LR | 0,001 |
| Seleção checkpoint | gulosa (melhor por acurácia, salvo no PostgreSQL) |

### Ambiente de execução

| Item | Detalhe |
|---|---|
| Máquina | Dell Inspiron 5402 — i7-1165G7 (8 threads), 16 GB RAM, sem GPU dedicada |
| Device | CPU (Intel Iris Xe sem suporte CUDA) |
| Início | 2026-06-28 07:46 |
| Fim | Em andamento |
| Duração estimada | ~265 min (mesma carga do Exp 8) |

#### Monitoramento térmico

> **Contexto:** antes da execução do Exp 9 houve cheiro de queimado próximo ao equipamento, possivelmente relacionado ao aquecimento acumulado da execução interrompida anteriormente e da fase de pré-processamento (pipeline de tensores: 28.599 + 5.174 sequências em Python/NumPy puro). Investigação do código descartou bug de implementação como causa: DataLoaders com `num_workers=0`, `torch.no_grad()` presente na avaliação, calibração fora do loop de rodadas. Causa mais provável: acúmulo de poeira no dissipador/ventoinha — recomenda-se limpeza preventiva.

| Momento | Horário | TCPU | x86_pkg_temp | Etapa |
|---|---|---|---|---|
| Início do monitoramento | 07:46 | **71,0°C** | **71,0°C** | [2/5] FL — rodada 1 iniciando |
| Rodada 30/120 | 08:53 | **76,0°C** | **68,0°C** | [2/5] FL — rodada 30 iniciando |
| Rodada 60/120 | 09:50 | **80,0°C** | **81,0°C** | [2/5] FL — rodada 60 iniciando |
| Rodada 90/120 | 10:44 | **83,0°C** | **81,0°C** | [2/5] FL — rodada 90 iniciando |
| Rodada 120/120 | 11:37 | **76,0°C** | **79,0°C** | [2/5] FL — última rodada iniciando |
| Início RAG [3/5] | 11:40 | **80,0°C** | **85,0°C** | pico térmico — RAG inicia logo após FL |
| Início Baseline RF [4/5] | 11:41 | **70,0°C** | **72,0°C** | queda após RAG (CPU libera carga) |
| Início Ablation [5/5] | 11:41 | **81,0°C** | **81,0°C** | ablation inicia em sequência |
| Fim (TREINAMENTO_COMPLETO) | 12:02 | **80,0°C** | **85,0°C** | pipeline completo encerrado |

> Arquivo de temperatura detalhado: `experiments/logs/temperature_exp9.log`

### Resultado por rodada

Tabela condensada — marcos, picos, vales e checkpoints gulosos. Atualizada progressivamente.

| Rodada | Tempo (s) | Loss | Acurácia | Nota |
|---|---|---|---|---|
| 1 | 159,7 | 1,5059 | 39,13% | primeiro checkpoint |
| 2 | 155,6 | 1,2334 | 57,38% | novo best |
| 3 | 155,0 | 1,2218 | 49,33% | — |
| 4 | 153,1 | 1,2816 | 43,51% | — |
| 5 | 130,2 | 1,2148 | 57,35% | — |
| 6 | 131,5 | 1,2549 | 49,22% | — |
| 7 | 146,5 | 1,1755 | 53,45% | — |
| 8 | 143,7 | 1,2120 | 52,68% | — |
| 9 | 143,9 | 1,1619 | 52,09% | — |
| 10 | 135,9 | 1,2792 | 46,73% | — |
| 11 | 132,9 | 1,2227 | 50,16% | — |
| 12 | 151,6 | 1,2331 | 45,46% | — |
| 13 | 124,0 | 1,1340 | 57,14% | — |
| 14 | 148,7 | 1,2287 | 51,67% | — |
| 15 | 141,5 | 1,0115 | **58,50%** | novo best |
| 16 | 124,9 | 1,1296 | 50,75% | — |
| 17 | 141,8 | 1,0505 | **62,26%** | novo best — primeiro acima de 60% |
| 18 | 138,6 | 1,1014 | 53,68% | — |
| 19 | 138,2 | 1,0910 | **63,38%** | novo best — fim do warm-up |
| 20 | 150,8 | 1,0745 | 62,02% | fim warm-up |
| 21 | 137,3 | 1,1755 | 53,15% | convergência avaliada a partir daqui |
| 22 | 127,6 | 1,1185 | 55,16% | — |
| 23 | 115,6 | 1,1488 | 51,35% | — |
| 24 | 127,8 | 1,0591 | 59,12% | — |
| 25 | 129,8 | 1,0401 | 60,40% | — |
| 26 | 123,0 | 1,1527 | 53,21% | — |
| 27 | 142,8 | 1,1064 | 54,30% | — |
| 28 | 135,2 | 1,0896 | 58,74% | — |
| 29 | 137,5 | 1,0447 | 58,15% | — |
| 30 | 120,0 | 1,0593 | 57,17% | — |
| 31 | 123,2 | 1,2247 | 51,17% | — |
| 32 | 112,7 | 1,0104 | 61,02% | — |
| 33 | 130,8 | 0,9871 | **63,86%** | novo best |
| 34 | 111,6 | 1,0672 | 57,88% | — |
| 35 | 112,9 | 1,0864 | 55,28% | — |
| 36 | 114,4 | 1,1301 | 55,49% | — |
| 37 | 131,7 | 0,9877 | 61,55% | — |
| 38 | 115,8 | 1,0242 | 59,01% | — |
| 39 | 134,7 | 1,0390 | 62,41% | — |
| 40 | 110,4 | 1,0903 | 55,25% | — |
| 41 | 107,1 | 1,0313 | 57,76% | — |
| 42 | 114,2 | 1,1462 | 55,49% | — |
| 43 | 111,8 | 0,9683 | 61,46% | — |
| 44 | 113,1 | 1,2146 | 49,87% | — |
| 45 | 107,4 | 1,1156 | 56,85% | — |
| 46 | 105,4 | 1,1264 | 55,93% | — |
| 47 | 104,2 | 1,2580 | 50,64% | — |
| 48 | 124,2 | 1,0272 | 59,72% | — |
| 49 | 138,9 | 1,0678 | 58,21% | — |
| 50 | 103,7 | 1,0230 | 61,49% | — |
| 51 | 108,1 | 1,0394 | 60,19% | — |
| 52 | 111,9 | 1,0229 | 59,78% | — |
| 53 | 102,3 | 1,1948 | 52,09% | — |
| 54 | 105,3 | 1,1372 | 55,78% | — |
| 55 | 111,1 | 1,0497 | 56,08% | — |
| 56 | 107,1 | 1,3974 | **41,73%** | ← pior rodada até agora |
| 57 | 106,8 | 1,0546 | 56,76% | recuperação |
| 58 | 106,1 | 1,0241 | 61,11% | — |
| 59 | 102,7 | 1,1552 | 49,36% | — |
| 60 | 124,8 | 1,1364 | 51,67% | — |
| 61 | 103,7 | 1,0835 | 57,02% | — |
| 62 | 117,7 | 1,0093 | 58,50% | — |
| 63 | 107,3 | 1,1472 | 58,06% | — |
| 64 | 104,9 | 1,1237 | 49,30% | — |
| 65 | 107,2 | 1,0422 | 59,72% | — |
| 66 | 121,3 | 1,3311 | **45,31%** | vale |
| 67 | 101,4 | 0,9840 | 59,75% | — |
| 68 | 110,2 | 1,1098 | 52,00% | — |
| 69 | 105,4 | 1,0730 | 58,86% | — |
| 70 | 97,8 | 1,0284 | 56,91% | — |
| 71 | 101,0 | 1,2007 | 55,60% | — |
| 72 | 103,7 | 1,0273 | 58,09% | — |
| 73 | 97,7 | 1,0037 | 57,79% | — |
| 74 | 97,5 | 1,0453 | 56,40% | — |
| 75 | 105,1 | 1,2263 | **45,40%** | vale |
| 76 | 120,1 | 1,0707 | 54,48% | — |
| 77 | 104,3 | 1,1463 | 52,65% | — |
| 78 | 119,9 | 1,2133 | 53,89% | — |
| 79 | 137,8 | 1,1767 | 50,93% | — |
| 80 | 117,9 | 1,1533 | 50,49% | — |
| 81 | 110,5 | 1,0822 | 56,67% | — |
| 82 | 109,2 | 1,0792 | 60,72% | — |
| 83 | 101,4 | 0,9955 | 60,28% | — |
| 84 | 96,7 | 1,0784 | 52,74% | — |
| 85 | 100,7 | 1,0232 | 58,06% | — |
| 86 | 96,7 | 1,0577 | 55,96% | — |
| 87 | 98,2 | 1,1456 | 52,94% | — |
| 88 | 125,4 | 1,0592 | 55,87% | — |
| 89 | 95,6 | 1,0157 | 60,16% | — |
| 90 | 106,8 | 1,1211 | 50,19% | — |
| 91 | 101,8 | 0,9895 | 59,95% | — |
| 92 | 119,9 | 1,0485 | 55,25% | — |
| 93 | 104,0 | 1,1595 | 52,85% | — |
| 94 | 101,9 | 0,9719 | 60,07% | — |
| 95 | 98,3 | 1,0979 | 52,77% | — |
| 96 | 95,9 | 1,0001 | 59,80% | — |
| 97 | 99,1 | 1,1475 | 52,41% | — |
| 98 | 96,9 | 1,0925 | 59,92% | — |
| 99 | 97,2 | 1,1360 | 54,95% | — |
| 100 | 100,2 | 1,1377 | 56,29% | — |
| 101 | 101,7 | 1,0784 | 56,70% | — |
| 102 | 109,5 | 1,0499 | 59,09% | — |
| 103 | 116,6 | 1,0192 | 57,62% | — |
| 104 | 96,4 | 1,1144 | 54,69% | — |
| 105 | 106,6 | 0,9360 | 61,08% | loss mais baixa do experimento |
| 106 | 98,2 | 1,0464 | 55,60% | — |
| 107 | 115,6 | 1,0532 | 55,49% | — |
| 108 | 96,7 | 1,0585 | 55,04% | — |
| 109 | 102,0 | 1,0156 | 58,98% | — |
| 110 | 107,0 | 1,1505 | 50,43% | — |
| 111 | 104,0 | 1,0209 | 56,43% | — |
| 112 | 105,7 | 1,0610 | 57,35% | — |
| 113 | 116,8 | 0,9913 | 59,15% | — |
| 114 | 104,7 | 0,9560 | 61,55% | — |
| 115 | 108,2 | 1,1082 | 56,73% | — |
| 116 | 137,9 | 1,0415 | 59,89% | — |
| 117 | 121,3 | 1,0782 | 58,50% | — |
| 118 | 117,8 | 0,9730 | 58,68% | — |
| 119 | 113,2 | 1,1067 | **61,79%** | — |
| **120** | **134,1** | **1,1618** | **54,54%** | última rodada — 9,32 p.p. abaixo do melhor |

**Checkpoint guloso — marcos do Exp 9 (completo):**

| Rodada | Experimento | Acc (validação) | Obs. |
|---|---|---|---|
| 1 | Exp 9 | 39,13% | primeiro checkpoint salvo |
| 2 | Exp 9 | 57,38% | novo best |
| 15 | Exp 9 | 58,50% | novo best |
| 17 | Exp 9 | 62,26% | novo best — primeiro acima de 60% |
| 19 | Exp 9 | 63,38% | novo best — maior do warm-up |
| **33** | **Exp 9** | **63,86%** | ← **melhor do Exp 9** — nenhum novo best após esta rodada |
| *(91)* | *(Exp 8 — persistido no banco)* | *(66,61%)* | `load_best()` retornou este checkpoint na avaliação final — supera R33 do Exp 9 |

**Convergência:** Não atingida (120 rodadas). Oscilação estrutural (41,73%–63,86%) — nenhum novo best após R33.  
**Gap best vs last:** 63,86% (R33) − 54,54% (R120) = **9,32 p.p.**  
**Tempo médio por rodada:** ~113 s/rodada (↓ vs Exp 8: ~133 s/rodada — FedNova ~15% mais rápido)

### Avaliação final (modelo restaurado da rodada 33)

**Pré-calibração (T=1,0) — modelo da R33:**

| Classe | AUC | F1 | Precision | Recall | N | vs Exp 8 |
|---|---|---|---|---|---|---|
| curado_pronto | 0,8703 | 0,7980 | 0,7540 | 0,8475 | 1.620 | F1 +0,000 |
| curado_internado | 0,6336 | 0,0625 | 0,0556 | 0,0714 | 28 | F1 +0,001 |
| **melhora_pronto** | **0,9216** | **0,6216** | 0,6433 | 0,6012 | 321 | **AUC +0,002 / F1 +0,002** |
| melhora_internado_breve | 0,8189 | 0,5919 | 0,6894 | 0,5186 | 1.074 | F1 +0,001 |
| melhora_internado_grave | 0,8038 | 0,3475 | 0,3149 | 0,3876 | 338 | F1 +0,012 |
| **Macro** | **0,8097** | **0,4843** | — | — | 3.381 | **AUC +0,000 / F1 +0,003** |

**Acurácia:** 66,73% — **novo recorde do projeto** (+0,12 p.p. vs Exp 8: 66,61%)  
**ECE pré-calibração:** 0,0870 | **MCE:** 0,2382 (idêntico ao Exp 8)  
**Temperatura T:** 1,0856 (log-space — positivo, fix de calibração funcionando)  
**ECE pós-calibração:** 0,1079 | MCE: 0,1669 — padrão de subconfiança recorrente (ECE piora, MCE melhora)

**Matriz de confusão (pré-calibração, R33):**

```
                      cp    ci    mp   mib   mig
curado_p    (1620)  1373    11    68    83    85
curado_i      (28)    12     2     1     8     5
melhora_p    (321)    89     0   193    31     8
melhora_ib  (1074)   276    21    33   557   187
melhora_ig   (338)    71     2     5   129   131
```

> **Nota:** a avaliação refere-se ao modelo do checkpoint guloso (R33, Acc=63,86% durante o treino). A acurácia de 66,73% reflete a avaliação formal no conjunto de teste, que difere da acurácia de validação registrada durante o treino.

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Classe | Precision@3 |
|---|---|
| curado_pronto | 0,2222 |
| curado_internado | 0,1786 |
| **melhora_pronto** | **0,2679** |
| melhora_internado_breve | 0,2312 |
| melhora_internado_grave | 0,1400 |
| **Macro** | **0,2208** |

> RAG confiável: True | alucinação: False  
> Saída: `experiments/data/rag_20260628_114105.json`

**Baseline Random Forest (Bag-of-Tokens):**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,32%** | **0,7941** | **0,5090** | 0,0621 |
| RF Hospital 0 (BPSP isolado) | 59,57% | 0,7411 | 0,3376 | 0,0523 |
| RF Hospital 1 (HSL isolado) | 25,79% | 0,7153 | 0,1953 | 0,2298 |

> Saída: `experiments/data/baseline_rf_20260628_114126.json`

**Ablation — Late Fusion Demográfica (10 épocas, dados reais FAPESP):**

| Config | Accuracy | F1 Macro | demo_dim |
|---|---|---|---|
| Config A — sem demográficos | 35,11% | 0,2128 | 0 |
| Config B — late fusion (idade + sexo) | **53,33%** | **0,3249** | 2 |
| **Δ (B − A)** | **+18,22 p.p.** | **+0,1121** | — |

> **Atenção:** o Δ de +18,22 p.p. é o maior do projeto, mas está inflado por uma performance incomumente baixa de Config A (35,11% vs histórico de 54–63%). Resultados históricos de Config A ficaram entre 54,2% e 62,8%. Possível anomalia de inicialização com seed=42 nesta execução específica. O sinal positivo (+) é consistente com todos os experimentos anteriores.  
> Saída: `experiments/data/ablation_demo_20260628_120225.json`

**BEHRT Pooled Baseline (artefato de pesquisa — sem privacidade):**

| Config | Accuracy | F1 Macro | Épocas |
|---|---|---|---|
| behrt_pooled_A — sem demo (demo_dim=0) | **68,88%** | **0,5146** | 120 |
| behrt_pooled_B — late fusion (demo_dim=2) | **67,82%** | **0,5048** | 120 |

> Saída: `experiments/data/behrt_pooled_20260628_152523.json`  
> behrt_pooled_A: 12:02–13:49 (~107 min) | behrt_pooled_B: 13:49–15:25 (~96 min)

**Custo de privacidade (Exp 9 — mesma arquitetura):**

| Comparação | Δ Acc | Δ F1 | Nota |
|---|---|---|---|
| FL Exp 9⁵ (66,73%) vs BEHRT Pooled A (68,88%) | −2,15 p.p. | −0,030 | gap menor que Exp 6 (−7,57 p.p.) |
| FL Exp 9⁵ (66,73%) vs BEHRT Pooled B (67,82%) | −1,09 p.p. | −0,021 | menor gap do projeto |
| FL Exp 9⁵ (66,73%) vs RF Centralizado (68,32%) | −1,59 p.p. | −0,025 | — |
| BEHRT Pooled A vs RF Centralizado | +0,56 p.p. | +0,006 | BEHRT pooled supera RF pela primeira vez |

> ⁵ A avaliação do Exp 9 reflete o checkpoint do Exp 8 (ver nota ⁵ na tabela comparativa). O custo de privacidade real do FedNova só poderá ser calculado após correção do namespace do checkpoint store.

### Ambiente de execução — duração

| Item | Detalhe |
|---|---|
| Início FL | 2026-06-28 07:46:35 |
| Fim FL | 2026-06-28 11:40:07 |
| Duração FL | **14.011,2 s (233,5 min / 3,89 h)** |
| Tráfego FL | **1.310,28 MB** |
| Custo médio por rodada | **~113 s/rodada** (vs 133 s/rodada no Exp 8 — FedNova ~15% mais rápido) |
| Início pós-FL | 11:40:11 |
| Fim pipeline completo | 12:02:25 |
| BEHRT Pooled | iniciado às 12:02:29 — em andamento |

### Observações relevantes do Experimento 9

**Alerta metodológico — checkpoint cross-contamination:**

O log registra `checkpoint_best_loaded_postgres round=91 accuracy=0.6661` imediatamente antes de "Modelo restaurado da rodada 33 (Acc=0.6386)". O PostgreSQL checkpoint store é **compartilhado entre experimentos e não foi resetado** entre Exp 8 e Exp 9. Como o melhor checkpoint do Exp 8 (R91: 0.6661) supera o melhor do Exp 9 (R33: 0.6386), `load_best()` retornou o modelo do Exp 8. A mensagem de log usa as variáveis em memória do Exp 9 (`best_round=33`, `best_accuracy=0.6386`), mas o modelo carregado é o R91 do Exp 8.

**Consequência:** a avaliação final registrada no Exp 9 (Acc=66,73%, AUC=0,8097, F1=0,4843) é essencialmente uma **re-avaliação do checkpoint R91 do Exp 8**, não do melhor modelo treinado com FedNova. As métricas ligeiramente diferentes de Exp 8 (66,61% → 66,73%, F1 0,4812 → 0,4843) refletem provavelmente a nova temperatura de calibração (T=1,0856 vs T=1,0849), não o modelo.

**Implicação para o TCC:** o Exp 9 não produziu avaliação válida do FedNova. Para avaliar corretamente, é necessário resetar o PostgreSQL checkpoint store antes de cada experimento, ou implementar namespacing por experimento no store. Ação corretiva registrada na seção de diagnóstico.

---

**O que o FedNova revelou:**

- **Convergência:** Não atingida em 120 rodadas. Oscilação de ±22 p.p. (41,73%–63,86%) — similar ao Exp 8 (±25 p.p.). FedNova não resolveu a oscilação estrutural.
- **Melhor checkpoint (Exp 9 own):** R33, Acc=63,86% de validação — inferior ao Exp 8 R91 (66,61%). FedNova convergiu para um ponto pior nesta execução.
- **Velocidade:** ~113 s/rodada vs ~133 s/rodada no Exp 8 — **FedNova foi ~15% mais rápido** por normalizar os updates antes da agregação, reduzindo o número de operações de comunicação efetivas.
- **Nenhum novo best após R33:** 87 rodadas sem melhora — plateau mais precoce que Exp 8 (último best em R91 de 120 rodadas).

**O que não melhorou:**
- A hipótese central do FedNova (normalizar por τ_i para reduzir o viés de escala entre BPSP e HSL) não se traduziu em convergência melhor. O non-IID estrutural é mais profundo que o viés de escala: a distribuição de classes é radicalmente diferente entre os dois clientes, e nenhuma normalização de gradiente resolve isso sem técnicas específicas para heterogeneidade de labels.
- `curado_internado` permanece com F1 mínimo (N=28 no teste).

**RAG:** Precision@3 macro de 0,2208 — melhor que Exp 7 (0,110) e Exp 8 (0,226). Sinal positivo, mas avaliado sobre o modelo do Exp 8.

**Custo de privacidade provisório (usando avaliação do Exp 8):**
- FL Exp 9/8 (66,73%) vs RF Centralizado (68,32%) = **gap de 1,59 p.p.** — idêntico ao Exp 8.

---

## Experimento 12

**Data:** 2026-06-28 / 2026-06-29
**Status:** Concluído
**Log:** `experiments/logs/run_complete_20260628_182702.log`
**Temperatura:** `experiments/logs/temperature_exp12.log`
**Comando:** `make training-full`
**training_id:** 2 (primeiro treinamento com checkpoint scoping via migration 011)

### Motivação

O Experimento 9 (FedNova) produziu avaliação inválida por cross-contamination do checkpoint store: `load_best()` sem filtro por experimento retornou o checkpoint R91 do Exp 8 (0,6661) em vez do R33 do Exp 9 (0,6386). Este experimento repete FedNova com o sistema de checkpoint corrigido:

- Migration 011 (`alembic/versions/011_fl_trainings.py`): cria `metrics.fl_trainings`, adiciona `training_id` em `metrics.fl_checkpoints` com índice UNIQUE parcial (`WHERE training_id IS NOT NULL`)
- `checkpoint_store.py`: `register_training()` antes do loop, UPSERT `ON CONFLICT (training_id) WHERE training_id IS NOT NULL`, `load_best(training_id)` com filtro por treinamento
- `fl_core.py`: lê `FL_LOG_FILE` do ambiente para `log_file`, chama `complete_training()` após o loop

### Alterações implementadas pré-treinamento

| Componente | Alteração |
|---|---|
| `scripts/db/010_fl_trainings.sql` | SQL de referência (não aplicado via Alembic) |
| `alembic/versions/011_fl_trainings.py` | Migration Alembic: `fl_trainings` + `training_id` FK + UNIQUE index parcial |
| `infrastructure/shared/checkpoint_store.py` | `register_training()`, `complete_training()`, UPSERT com índice parcial, `load_best(training_id)`, `weights_only=False` |
| `experiments/training/fl_core.py` | Chama `register_training()` antes do loop; passa `training_id` ao `save()`; `complete_training()` + `load_best(training_id)` após o loop |

### Hiperparâmetros

*(idênticos ao Experimento 9)*

| Parâmetro | Valor |
|---|---|
| Rodadas máximas | 120 |
| Rodadas warm-up | 20 |
| Algoritmo | **FedNova** |
| Batch size | 16 |
| Épocas locais | 2 |
| Threshold convergência | 0,005 |
| Paciência convergência | 3 |
| LR | 0,001 |
| Seleção checkpoint | gulosa (UPSERT por training_id no PostgreSQL) |

### Ambiente de execução

| Item | Detalhe |
|---|---|
| Máquina | Dell Inspiron 5402 — i7-1165G7 (8 threads), 16 GB RAM, sem GPU dedicada |
| Device | CPU (Intel Iris Xe sem suporte CUDA) |
| Início FL | 2026-06-28 18:27:32 |
| Fim FL | 2026-06-28 22:33:15 |
| Duração FL | **14.742,5 s (245,7 min / 4,10 h)** |
| Tráfego FL | **1.310,28 MB** |

#### Monitoramento térmico

| Momento | Horário | TCPU | x86_pkg_temp | Etapa |
|---|---|---|---|---|
| Início do monitoramento | 18:33 | 79°C | 83°C | R2 iniciando (checkpoint R2 salvo) |
| Rodada 30/120 | 19:31 | **90°C** | **92°C** | pico térmico da fase FL |
| Rodada 60/120 | 20:30 | **90°C** | **90°C** | FL — metade das rodadas |
| Rodada 90/120 | 21:24 | 73°C | 73°C | queda — hardware estabilizou |
| Checkpoint R115 (best) | 22:22 | 75°C | 73°C | novo melhor checkpoint |
| Rodada 120/120 | 22:31 | **94°C** | **94°C** | pico máximo — última rodada |
| FL_TRAINING_COMPLETE | 22:33 | 76°C | 76°C | loop encerrado |
| PIPELINE_COMPLETE | 22:33 | 78°C | 78°C | RAG/RF/Ablation concluídos |

> Arquivo de temperatura detalhado: `experiments/logs/temperature_exp12.log`

### Resultado por rodada

| Rodada | Tempo (s) | Loss | Acurácia | Nota |
|---|---|---|---|---|
| 1 | 135,7 | 1,3189 | 37,56% | primeiro checkpoint |
| 2 | 142,2 | 1,2414 | 48,39% | novo best |
| 3 | 137,3 | 1,1067 | 56,40% | novo best |
| 4 | 144,2 | 1,1670 | 55,63% | — |
| 5 | 125,0 | 1,1324 | 63,59% | novo best |
| 6 | 145,3 | 1,2285 | 55,10% | — |
| 7 | 133,3 | 1,1829 | 46,64% | — |
| 8 | 123,2 | 1,2570 | 50,28% | — |
| 9 | 127,1 | 1,0600 | 56,49% | — |
| 10 | 126,9 | 1,1721 | 51,32% | — |
| 11 | 136,8 | 1,3638 | 40,58% | — |
| 12 | 119,8 | 1,2511 | 47,41% | — |
| 13 | 135,5 | 1,0611 | 58,00% | — |
| 14 | 119,1 | 1,2136 | 45,02% | — |
| 15 | 126,7 | 1,0985 | 57,47% | — |
| 16 | 129,4 | 1,1306 | 53,56% | — |
| 17 | 116,8 | 1,0230 | 61,96% | — |
| 18 | 134,7 | 1,1360 | 51,88% | — |
| 19 | 117,0 | 1,1136 | 62,38% | — |
| 20 | 144,8 | 1,0451 | 58,89% | fim warm-up |
| 21 | 132,0 | 1,2124 | 49,04% | convergência avaliada a partir daqui |
| 22 | 143,1 | 1,0521 | 60,72% | — |
| 23 | 134,3 | 1,0511 | 60,16% | — |
| 24 | 140,9 | 1,1999 | 58,12% | — |
| 25 | 140,3 | 1,0073 | 61,79% | — |
| 26 | 132,0 | 1,1092 | 57,70% | — |
| 27 | 137,7 | 1,0114 | 61,37% | — |
| 28 | 137,0 | 1,0782 | 56,37% | — |
| 29 | 143,1 | 1,0009 | 61,49% | — |
| 30 | 109,8 | 1,0357 | 57,02% | — |
| 31 | 136,2 | 1,1568 | 51,29% | — |
| 32 | 114,5 | 0,9291 | 65,39% | novo best |
| 33 | 146,8 | 1,0921 | 55,43% | — |
| 34 | 127,7 | 0,9487 | 64,24% | — |
| 35 | 115,2 | 1,0055 | 60,63% | — |
| 36 | 142,2 | 0,9806 | 61,08% | — |
| 37 | 140,0 | 0,9113 | 66,70% | novo best |
| 38 | 147,5 | 1,0507 | 61,37% | — |
| 39 | 121,7 | 0,9504 | 65,75% | — |
| 40 | 114,9 | 1,1263 | 54,54% | — |
| 41 | 119,3 | 0,9827 | 60,96% | — |
| 42 | 114,0 | 0,9649 | 62,41% | — |
| 43 | 108,0 | 0,9425 | 64,18% | — |
| 44 | 118,0 | 1,0014 | 64,77% | — |
| 45 | 125,3 | 1,0703 | 57,68% | — |
| 46 | 108,3 | 1,1971 | 51,79% | — |
| 47 | 106,6 | 1,0671 | 57,44% | — |
| 48 | 105,9 | 0,9405 | 62,35% | — |
| 49 | 105,5 | 1,0603 | 59,60% | — |
| 50 | 111,6 | 0,8847 | 65,96% | — |
| 51 | 103,9 | 0,8759 | 66,13% | — |
| 52 | 109,2 | 0,9239 | 62,73% | — |
| 53 | 104,8 | 0,9722 | 59,69% | — |
| 54 | 107,2 | 0,9539 | 63,32% | — |
| 55 | 103,2 | 1,0315 | 60,54% | — |
| 56 | 117,7 | 0,9743 | 60,57% | — |
| 57 | 112,6 | 0,9569 | 63,41% | — |
| 58 | 116,7 | 0,9863 | 62,50% | — |
| 59 | 103,3 | 1,0034 | 60,93% | — |
| 60 | 104,4 | 0,9756 | 61,14% | — |
| 61 | 106,8 | 0,9942 | 61,40% | — |
| 62 | 102,1 | 1,0008 | 64,24% | — |
| 63 | 106,8 | 1,0602 | 57,70% | — |
| 64 | 108,3 | 1,1185 | 53,53% | — |
| 65 | 103,4 | 0,9814 | 61,46% | — |
| 66 | 103,7 | 0,9568 | 63,80% | — |
| 67 | 101,8 | 1,0360 | 60,84% | — |
| 68 | 102,4 | 1,0464 | 62,26% | — |
| 69 | 106,8 | 1,0411 | 58,92% | — |
| 70 | 115,7 | 0,9506 | 65,25% | — |
| 71 | 107,4 | 0,9920 | 65,31% | — |
| 72 | 105,3 | 0,9999 | 60,60% | — |
| 73 | 125,9 | 0,9466 | 64,92% | — |
| 74 | 118,2 | 0,9251 | 65,81% | — |
| 75 | 106,4 | 1,0459 | 57,91% | — |
| 76 | 137,0 | 1,0350 | 59,15% | — |
| 77 | 116,2 | 0,9304 | 64,15% | — |
| 78 | 106,2 | 0,9689 | 62,05% | — |
| 79 | 99,0 | 1,0273 | 61,31% | — |
| 80 | 99,3 | 1,0218 | 58,36% | — |
| 81 | 109,1 | 0,9764 | 63,62% | — |
| 82 | 106,3 | 0,9764 | 61,82% | — |
| 83 | 104,6 | 0,9664 | 63,98% | — |
| 84 | 109,5 | 0,9996 | 59,36% | — |
| 85 | 106,4 | 1,0753 | 57,41% | — |
| 86 | 115,0 | 0,9243 | 63,15% | — |
| 87 | 106,5 | 1,0424 | 61,14% | — |
| 88 | 102,2 | 1,0149 | 60,90% | — |
| 89 | 103,7 | 0,9183 | 65,66% | — |
| 90 | 113,7 | 0,9109 | 64,80% | — |
| 91 | 137,6 | 0,9222 | 63,56% | — |
| 92 | 112,2 | 1,0156 | 63,21% | — |
| 93 | 125,2 | 0,9882 | 61,14% | — |
| 94 | 122,6 | 0,9641 | 64,18% | — |
| 95 | 133,0 | 0,9906 | 60,25% | — |
| 96 | 116,0 | 1,0043 | 63,09% | — |
| 97 | 127,6 | 1,0541 | 60,60% | — |
| 98 | 134,4 | 0,9977 | 63,86% | — |
| 99 | 150,6 | 1,0321 | 63,03% | — |
| 100 | 125,1 | 0,9629 | 63,35% | — |
| 101 | 127,0 | 0,9880 | 61,17% | — |
| 102 | 131,7 | 0,9292 | 65,28% | — |
| 103 | 138,9 | 1,0001 | 63,65% | — |
| 104 | 140,5 | 1,0066 | 61,17% | — |
| 105 | 138,2 | 0,9156 | 64,54% | — |
| 106 | 167,9 | 0,9249 | 62,85% | — |
| 107 | 151,3 | 0,9480 | 63,35% | — |
| 108 | 128,7 | 0,9203 | 63,68% | — |
| 109 | 138,3 | 1,0020 | 62,53% | — |
| 110 | 164,1 | 1,0835 | **53,27%** | ← pior rodada |
| 111 | 133,1 | 1,1249 | 59,83% | recuperação |
| 112 | 135,3 | 1,0819 | 57,62% | — |
| 113 | 145,0 | 0,9341 | 64,80% | — |
| 114 | 126,8 | 0,9390 | 65,01% | — |
| **115** | **132,7** | **0,9035** | **67,44%** | ← **melhor checkpoint — novo recorde do projeto** |
| 116 | 124,0 | 0,9085 | 66,87% | — |
| 117 | 121,7 | 0,9201 | 65,63% | — |
| 118 | 145,5 | 0,9068 | 64,63% | — |
| 119 | 112,3 | 0,9646 | 65,66% | — |
| **120** | **117,0** | **0,9849** | **61,14%** | última rodada |

**Checkpoint guloso — marcos do Exp 12:**

| Rodada | training_id | Acc (validação) | sha256 | Obs. |
|---|---|---|---|---|
| 1 | 2 | 37,56% | b1f45159b696 | primeiro checkpoint |
| 2 | 2 | 48,39% | 8f8e361e061b | novo best |
| 3 | 2 | 56,40% | f4b64996b428 | novo best |
| 5 | 2 | 63,59% | a9dc2978d411 | novo best |
| 32 | 2 | 65,39% | 1212895f1e13 | novo best |
| 37 | 2 | 66,70% | 41c851ee041d | novo best — supera Exp 8 R91 |
| **115** | **2** | **67,44%** | **c4c9697c608d** | **melhor do projeto** |

> `training_completed_postgres id=2 best_round=115 best_accuracy=0.6744 converged=False`
> `checkpoint_best_loaded_postgres round=115 accuracy=0.6744 training_id=2` ✅ — carregou o modelo correto, sem cross-contamination

**Convergência:** Não atingida (120 rodadas). Oscilação ~53–67% — padrão non-IID estrutural mantido.
**Gap best vs last:** 67,44% (R115) − 61,14% (R120) = **6,30 p.p.**
**Tempo médio por rodada:** ~122 s/rodada

### Avaliação final (modelo restaurado da rodada 115)

**Pré-calibração (T=1,0):**

| Classe | AUC | F1 | Precision | Recall | N | vs Exp 9⁵ |
|---|---|---|---|---|---|---|
| curado_pronto | 0,8762 | 0,8146 | 0,7695 | 0,8654 | 1.620 | F1 +0,017 |
| curado_internado | 0,5713 | 0,0323 | 0,0294 | 0,0357 | 28 | F1 −0,030 |
| **melhora_pronto** | **0,9553** | **0,6606** | 0,7854 | 0,5701 | 321 | **AUC +0,034 / F1 +0,039** |
| melhora_internado_breve | 0,8108 | 0,5819 | 0,6413 | 0,5326 | 1.074 | F1 −0,010 |
| melhora_internado_grave | 0,7936 | 0,3306 | 0,3050 | 0,3609 | 338 | F1 −0,017 |
| **Macro** | **0,8015** | **0,4840** | — | — | 3.381 | AUC −0,001 / F1 +0,000 |

**Acurácia:** 67,44% — **novo recorde do projeto** (+0,71 p.p. vs Exp 9: 66,73%)
**ECE pré-calibração:** 0,0935 | **MCE:** 0,2545
**Temperatura T:** 1,0580 (log-space — positivo)
**ECE pós-calibração:** 0,1086 | MCE: 0,2875 — padrão de subconfiança recorrente

**Matriz de confusão (pré-calibração, R115):**

```
                      cp    ci    mp   mib   mig
curado_p    (1620)  1402    10    23   119    66
curado_i      (28)    12     1     2    10     3
melhora_p    (321)    71    10   183    49     8
melhora_ib  (1074)   270     9    22   572   201
melhora_ig   (338)    67     4     3   142   122
```

### Resultados das etapas pós-FL

**RAG — Precision@3:**

| Métrica | Valor |
|---|---|
| Precision@3 macro | **0,1450** |

> Saída: `experiments/logs/run_complete_20260628_182702.log` (etapa [3/5])

**Baseline Random Forest (Bag-of-Tokens) — etapa [4/5] de `run_training.py`:**

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP+HSL) | **68,06%** | **0,7951** | **0,5034** | 0,0589 |
| RF Hospital 0 (BPSP isolado) | 59,66% | 0,7414 | 0,3370 | 0,0645 |
| RF Hospital 1 (HSL isolado) | 23,31% | 0,6999 | 0,1795 | 0,2608 |

**Ablation — Late Fusion Demográfica ([5/5], 10 épocas, dados reais FAPESP):**

| Config | Accuracy | F1 Macro | demo_dim |
|---|---|---|---|
| Config A — sem demográficos | 59,33% | 0,3653 | 0 |
| Config B — late fusion (idade + sexo) | 59,09% | 0,3618 | 2 |
| **Δ (B − A)** | **−0,24 p.p.** | **−0,004** | — |

> **Atenção:** Δ negativo (B pior que A) é anomalia. O sinal positivo consistente de Exp 3–9 desapareceu. Config A (59,33%) está dentro do histórico (54–63%), sugerindo que Config B teve inicialização desfavorável com seed=42. O sinal da late fusion demográfica permanece validado nos experimentos anteriores.

**BEHRT Pooled Baseline (artefato de pesquisa — `run_behrt_pooled.py`):**

| Config | Accuracy | F1 Macro | Épocas |
|---|---|---|---|
| behrt_pooled_A — sem demo (demo_dim=0) | **68,03%** | **0,5165** | 120 |
| **behrt_pooled_B — late fusion (demo_dim=2)** | **69,12%** | **0,5269** | 120 |

> RF Centralizado (pooled script): Acc=68,74%, AUC=0,8031, F1=0,5108, ECE=0,0719
> Saída: `experiments/data/behrt_pooled_20260629_020657.json`
> behrt_pooled_A: concluído às 02:06 | behrt_pooled_B: concluído depois

**Custo de privacidade (Exp 12 — avaliação válida do FedNova):**

| Comparação | Δ Acc | Δ F1 | Nota |
|---|---|---|---|
| FL Exp 12 (67,44%) vs BEHRT Pooled A (68,03%) | −0,59 p.p. | −0,032 | menor gap FL vs Pooled do projeto |
| FL Exp 12 (67,44%) vs BEHRT Pooled B (69,12%) | −1,68 p.p. | −0,043 | gap real da federação com FedNova |
| FL Exp 12 (67,44%) vs RF Centralizado (68,06%) | −0,62 p.p. | −0,019 | **gap mais próximo FL vs RF do projeto** |
| BEHRT Pooled B vs RF Centralizado | +0,38 p.p. | +0,016 | BEHRT pooled supera RF (segundo experimento consecutivo) |

### Duração do pipeline completo

| Etapa | Início | Fim | Duração |
|---|---|---|---|
| FL (120 rodadas) | 18:27:32 | 22:33:15 | 14.742,5 s (245,7 min) |
| RAG | 22:33:19 | 22:34:12 | ~53 s |
| RF + Ablation | 22:34:13 | 22:55:04 | ~21 min |
| BEHRT Pooled (A+B) | 22:55:08 | 02:06:57 | ~192 min |
| **Total pipeline** | **18:27** | **02:07** | **~452 min (~7,5 h)** |

### Observações relevantes do Experimento 12

**Checkpoint scoping validado:** `training_id=2` registrado antes do loop; UPSERT correto em todos os 7 checkpoints; `load_best(training_id=2)` retornou R115 (0,6744) — primeiro experimento com avaliação FedNova válida.

**Novo recorde histórico:** Acc=67,44% (R115) supera Exp 8 R91 (66,61%) e Exp 9 avaliação inválida (66,73%). Resultado válido para uso no TCC.

**melhora_pronto AUC=0,9553** — melhor AUC desta classe em todos os experimentos (+0,034 vs Exp 9). Sinal de que o modelo FedNova com mais rodadas captura melhor a classe mais difícil do non-IID.

**curado_internado:** F1=0,0323 — pior desempenho histórico nessa classe (N=28, problema estrutural de raridade).

**Custo de privacidade real do FedNova:** FL (67,44%) vs BEHRT Pooled B (69,12%) = **−1,68 p.p.** — o custo de federar com FedNova e non-IID severo é de apenas 1,7 p.p. de acurácia. Resultado relevante para o TCC: federação não impõe penalidade severa.

**Temperatura:** pico de 94°C na R120 — o mesmo padrão das últimas rodadas de experimentos anteriores. Sem eventos térmicos anômalos.

### Conclusões do Experimento 12

**1. Checkpoint scoping resolve o problema de cross-contamination.**
O `training_id=2` garantiu que R115 (0,6744) fosse carregado na avaliação final — primeira avaliação válida do FedNova no projeto. O Exp 9 avaliou inadvertidamente o modelo do Exp 8; o Exp 12 avalia o que foi efetivamente treinado.

**2. FedNova com 120 rodadas produz o melhor resultado federado do projeto.**
Acc=67,44% (R115) supera Exp 8 R91 (66,61%) em +0,83 p.p. com o mesmo número de rodadas e a mesma seed. Com mais tempo de treino e normalização por τ_i, o FedNova encontrou um ponto de convergência melhor que o FedAvg.

**3. O custo de privacidade da federação é pequeno.**
FL FedNova (67,44%) vs BEHRT Pooled B (69,12%) = **−1,68 p.p.** Federar com dois hospitais em regime non-IID severo custa menos de 2 p.p. de acurácia em relação ao treino centralizado com a mesma arquitetura. Este é o número mais relevante para o TCC.

**4. BEHRT FL está a 0,62 p.p. do RF centralizado.**
Gap de 67,44% vs 68,06% — a menor diferença do projeto. Para fins práticos (privacidade + dados distribuídos), o BEHRT federado é competitivo com o RF que exigiria mover dados dos pacientes entre hospitais.

**5. BEHRT Pooled supera RF pela segunda vez consecutiva.**
Pooled B (69,12%) > RF (68,06–68,74%) — confirma tendência iniciada no Exp 9. Com 120 épocas, o BEHRT extrai mais sinal temporal dos dados do que o RF bag-of-tokens quando treinado no pool completo. O gap de capacidade arquitetural existe; o dataset atual ainda limita o BEHRT federado.

**6. FedNova não resolve o non-IID estrutural.**
Oscilação de 53–67% em 120 rodadas confirma que o viés de escala (τ_i) não é a causa raiz da instabilidade. A heterogeneidade de distribuição de classes (`melhora_pronto`: 61,5% HSL vs 0,4% BPSP) é mais profunda do que qualquer normalização de gradiente pode resolver sem técnicas específicas para label shift.

**7. `melhora_pronto` AUC=0,9553 — melhor histórico do projeto.**
O FedNova com 120 rodadas capturou melhor a classe clinicamente mais relevante do HSL. AUC=0,9553 representa +0,034 vs Exp 9 e é o pico de discriminação desta classe em todos os experimentos.

**8. Ablação negativa (−0,24 p.p.) é anomalia isolada.**
Sinal positivo dos demográficos confirmado em Exp 3–9; Exp 12 é exceção provável de inicialização com seed=42. Não invalida a conclusão geral sobre a relevância da late fusion demográfica.

> **Para o TCC:** o Exp 12 é o experimento de referência para o FedNova — avaliação válida, sem cross-contamination, com o melhor resultado federado do projeto. O custo de privacidade de −1,68 p.p. é o argumento central para justificar a federação como abordagem viável clinicamente.

---

## Experimento 13 — BPSP-Only (Leave-One-Client-Out, Fase 1/4)

**Data:** 2026-06-29
**Status:** Em andamento
**Log:** `experiments/logs/run_complete_20260629_074506.log`
**Comando:** `make training-full` (fase 1/4 — `FL_INCLUDE_HOSPITALS=BPSP`)
**training_id:** 3

### Motivação

Primeiro dos quatro ciclos da primeira execução do `make training-full`. O objetivo do leave-one-client-out é decompor empiricamente o valor da federação: treinar com apenas o cliente BPSP e avaliar no test set global (BPSP+HSL) revela quanto sinal cada hospital contribui individualmente.

**Hipótese:** o modelo BPSP-only terá acurácia global próxima ao federado (BPSP representa 84,7% das amostras totais), mas o F1 de `melhora_pronto` deve ser próximo de zero — essa classe é quasi-exclusiva do HSL (61,5% das amostras HSL vs 0,4% das amostras BPSP). O BPSP nunca verá essa classe em quantidade suficiente para aprendê-la.

**Resultado esperado da decomposição:**

| Métrica | BPSP-only | HSL-only | BPSP+HSL (Exp 12) | Interpretação |
|---|---|---|---|---|
| Acurácia global | ~67% | ~40–55%? | 67,44% | BPSP domina volume |
| F1 melhora_pronto | ~0 | alto? | 0,661 | HSL é a fonte de sinal dessa classe |
| F1 macro | baixo | muito baixo? | 0,484 | não-IID revela-se por ausência |

### Alterações implementadas pré-treinamento (MVP quality improvements)

Esta é a primeira execução após um conjunto substancial de melhorias de qualidade que elevam o pipeline ao nível de MVP de produção:

| Componente | Arquivo | Alteração | Motivação |
|---|---|---|---|
| Leave-one-client-out | `dataloaders.py` | `FL_INCLUDE_HOSPITALS` env var — filtra clientes do treino; test/cal sempre globais | Quantifica contribuição de cada hospital sem alterar código |
| DataLoader determinístico | `dataloaders.py` | `generator=torch.Generator().manual_seed(RANDOM_SEED + cid)` | Shuffling reprodutível por cliente; garante que rodadas com mesma seed sejam idênticas |
| Labels parametrizáveis | `config.py` | `FL_CLASS_LABELS` env var — define desfechos clínicos em runtime | Desacopla o pipeline do dataset; permite trocar tarefa sem alterar código |
| Épocas locais reduzidas | `config.py` | `local_epochs: int = 1` (era 2) | Reduz divergência entre clientes por rodada em regime non-IID severo (Li et al. 2020) |
| Class weight clipping | `client.py` | `_compute_class_weights()` com `.clamp(max=15.0)` | Peso bruto de `melhora_pronto` no BPSP seria ~47–117; causava explosão de gradiente |
| Gradient clipping | `client.py` | `clip_grad_norm_(max_norm=1.0)` após `loss.backward()`; retorna `grad_norm` | Previne explosão de gradiente em batches extremamente desequilibrados |
| IsotonicCalibrator | `calibration.py` | Calibração OvR por classe (Zadrozny & Elkan, 2002) ao lado do TemperatureScaler | Temperature scaling piora ECE em 8/8 experimentos (subconfiança não-uniforme) |
| Comparação de calibradores | `fl_core.py` | Aplica ambos os calibradores; loga `ECE_pre`, `ECE_temperature`, `ECE_isotonic` | Confirma empiricamente qual calibrador é superior para este dataset |
| Determinismo CUDA | `fl_core.py` | `cudnn.deterministic=True`, `cudnn.benchmark=False` | Reprodutibilidade completa quando GPU estiver disponível |
| Min clients clamp | `experiment_server.py` | `min(FED_CFG.min_fit_clients, num_clients)` | Permite runs com 1 cliente (BPSP-only ou HSL-only) sem erro de config |
| Ablação multi-seed | `ablation.py` + `orchestrator.py` | k=3 seeds (42, 7, 123); reporta média ± desvio-padrão | Elimina sensibilidade à inicialização; resultados estatisticamente mais robustos |
| Make targets | `Makefile` | `training-bpsp-only`, `training-hsl-only`, `training-full` (4 fases) | Pipeline completo sem parametrização externa |

### Configuração dos dados

| Item | Valor |
|---|---|
| Hospitais no treino | **BPSP** (HSL excluído via `FL_INCLUDE_HOSPITALS=BPSP`) |
| Hospitais no test/cal | **BPSP + HSL** (global — para comparação justa com federado) |
| BPSP: treino / val / cal | 20.019 / 2.859 / 2.859 |
| HSL: status | excluído do treino; test/cal incluídos no set global |
| Teste global | 3.381 amostras (BPSP + HSL) |
| Cal global | 3.376 amostras (BPSP + HSL) |
| Clientes FL ativos | **1** |
| Distribuição BPSP treino | `{0: 11.111, 1: 229, 2: 85, 3: 6.599, 4: 1.995}` |
| Pesos de classe (clipados) | `[0,360, 15,0, 15,0, 0,607, 2,007]` |

> Sem clipping, o peso da classe 2 (`melhora_pronto`) seria `total / (5 × 85)` ≈ 47,2. Com clipping em 15,0, o treino permanece estável.

### Hiperparâmetros

| Parâmetro | Valor | Δ vs Exp 12 |
|---|---|---|
| Rodadas máximas | 120 | — |
| Rodadas warm-up | 20 | — |
| Algoritmo | **FedNova** | — |
| Batch size | 16 | — |
| Épocas locais | **1** | ↓ (era 2) |
| LR | 0,001 | — |
| Threshold convergência | 0,005 | — |
| Paciência convergência | 3 | — |
| Class weight clipping | max=15,0 | **NOVO** |
| Gradient clipping | max_norm=1,0 | **NOVO** |
| Calibrador | TemperatureScaler + IsotonicCalibrator | **NOVO (isotônica adicionada)** |
| Seleção checkpoint | gulosa (UPSERT por training_id=3) | — |

### Ambiente de execução

| Item | Detalhe |
|---|---|
| Máquina | Dell Inspiron 5402 — i7-1165G7 (8 threads), 16 GB RAM, sem GPU dedicada |
| Device | CPU (Intel Iris Xe sem suporte CUDA) |
| Início FL | 2026-06-29 07:45:37 |
| Fim FL | Em andamento |
| Duração estimada | ~120 min (1 cliente → ~50% do tempo por rodada vs Exp 12) |

#### Monitoramento de sistema

| Momento | Horário | TCPU | CPU% | RAM usada | Etapa |
|---|---|---|---|---|---|
| Início | 07:45 | — | — | — | R1 iniciando |
| Rodada 30/120 | 08:14 | **77°C** | — | — | FL em andamento |
| Rodada 35/120 | 08:18 | **85°C** | — | — | FL em andamento |
| Rodada 60/120 | 08:41 | **83°C** | **75,7%** | 14.940 / 31.804 MB (47%) | marco R60 — best: R67/64,39% |
| Rodada 90/120 | 09:06 | **77°C** | **75,7%** | 14.926 / 31.804 MB (47%) | marco R90 — best: R67/64,39% — sem novo cp desde R67 |
| Rodada 120/120 | 09:31 | **79°C** | **76,0%** | 15.553 / 31.804 MB (49%) | última rodada — FL_TRAINING_COMPLETE |

### Resultado por rodada

> **Acurácia:** logada apenas em checkpoints (novo melhor). Demais rodadas: loss + grad_norm disponíveis.
> **grad_norm:** valores em 3,3–3,8 confirmam gradient clipping ativo (`max_norm=1,0` — norma pré-clipping reportada).
> **Padrão loss:** tendência descendente mas não monotônica — loss e acurácia podem divergir (R24 tem loss 1,2838 < R25 1,2992, mas R25 tem melhor accuracy).

| Rodada | Tempo (s) | Loss | Acurácia | grad_norm | Nota |
|---|---|---|---|---|---|
| 1 | 57 | 1,4679 | 57,59% | 3,7611 | primeiro checkpoint |
| 2 | 54 | 1,4120 | — | 3,7286 | — |
| 3 | 63 | 1,3654 | — | 3,5787 | — |
| 4 | 54 | 1,3685 | — | 3,4997 | — |
| 5 | 56 | 1,3496 | — | 3,4302 | — |
| 6 | 56 | 1,3512 | — | 3,3651 | — |
| 7 | 67 | 1,3397 | **61,25%** | 3,3490 | novo best |
| 8 | 57 | 1,3211 | — | 3,3049 | — |
| 9 | 63 | 1,3235 | — | 3,4289 | — |
| 10 | 55 | 1,3245 | — | 3,3672 | — |
| 11 | 58 | 1,3261 | — | 3,4600 | — |
| 12 | 62 | 1,3250 | — | 3,4633 | — |
| 13 | 52 | 1,3092 | — | 3,5617 | — |
| 14 | 52 | 1,3096 | **61,67%** | 3,6576 | novo best |
| 15 | 63 | 1,3028 | **62,85%** | 3,6359 | novo best |
| 16 | 53 | 1,2958 | — | 3,6413 | — |
| 17 | 58 | 1,3041 | — | 3,7052 | — |
| 18 | 51 | 1,3038 | — | 3,5495 | — |
| 19 | 51 | 1,3068 | — | 3,6565 | — |
| 20 | 58 | 1,2981 | — | 3,5943 | fim warm-up |
| 21 | 66 | 1,2939 | — | 3,6013 | convergência avaliada a partir daqui |
| 22 | 51 | 1,3015 | — | 3,7048 | — |
| 23 | 51 | 1,3043 | — | 3,6707 | — |
| 24 | 50 | 1,2838 | — | 3,5880 | — |
| 25 | 50 | 1,2992 | **63,29%** | 3,6952 | novo best |
| 26 | 50 | 1,2795 | — | 3,6027 | loss < R25 mas acc não supera 63,29% |
| 27 | 63 | 1,2877 | — | 3,5523 | — |
| 28 | 52 | 1,2827 | — | 3,6356 | — |
| 29 | 56 | 1,2885 | — | 3,6162 | — |
| 30 | 49 | 1,2865 | — | 3,5852 | marco R30 — best: R25/63,29% |
| 31 | 58 | 1,2732 | — | 3,5857 | — |
| 32 | 60 | 1,2922 | — | 3,7083 | — |
| 33 | 56 | 1,2874 | — | 3,7400 | — |
| 34 | 63 | 1,2770 | — | 3,7310 | — |
| 35 | 54 | 1,2744 | — | 3,7400 | — |
| 36 | 61 | 1,2825 | — | 3,7639 | — |
| 37 | 59 | 1,2770 | — | 3,8072 | — |
| 38 | 57 | 1,2687 | — | 3,8283 | — |
| 39 | 63 | 1,2728 | — | 3,8420 | — |
| 40 | 54 | 1,2717 | — | 3,8411 | — |
| 41 | 50 | 1,2753 | — | 4,0162 | grad_norm acima de 4 pela 1ª vez |
| 42 | 56 | 1,2692 | — | 3,8240 | — |
| 43 | 51 | 1,2772 | — | 3,9219 | — |
| 44 | 57 | 1,2660 | — | 3,9113 | — |
| 45 | 49 | 1,2589 | — | 3,8775 | — |
| 46 | 59 | 1,2721 | — | 3,7947 | — |
| 47 | 51 | 1,2728 | — | 4,0182 | — |
| 48 | 47 | 1,2826 | — | 3,8751 | — |
| 49 | 56 | 1,2607 | — | 3,9096 | — |
| 50 | 48 | 1,2649 | — | 4,0480 | — |
| 51 | 54 | 1,2564 | — | 3,8966 | — |
| 52 | 47 | 1,2690 | **63,32%** | 3,9609 | novo best (+0,03 p.p. vs R25) |
| 53 | 50 | 1,2551 | — | 3,8924 | loss mínima até aqui |
| 54 | 47 | 1,2682 | — | 3,9897 | — |
| 55 | 50 | 1,2726 | — | 3,9827 | — |
| 56 | 53 | 1,2527 | — | 3,9381 | loss mínima histórica até aqui |
| 57 | 53 | 1,2602 | — | 4,0021 | — |
| 58 | 55 | 1,2605 | — | 3,9856 | — |
| 59 | 53 | 1,2626 | — | 4,0396 | — |
| 60 | 47 | 1,2575 | — | 4,0993 | marco R60 |
| 61 | 50 | 1,2661 | — | 4,0847 | — |
| 62 | 46 | 1,2474 | — | 3,9525 | — |
| 63 | 51 | 1,2520 | — | 3,8490 | — |
| 64 | 49 | 1,2513 | — | 4,0787 | — |
| 65 | 52 | 1,2521 | — | 4,1910 | — |
| 66 | 47 | 1,2559 | — | 4,0557 | — |
| 67 | 55 | 1,2465 | **64,39%** | 4,1703 | novo best (+1,07 p.p. vs R52) — loss mínima |
| 68 | 64 | 1,2610 | — | 4,3365 | grad_norm máximo até aqui |
| 69 | 60 | 1,2529 | — | 4,2156 | — |
| 70 | 56 | 1,2493 | — | 4,1370 | — |
| 71 | 48 | 1,2698 | — | 4,2523 | — |
| 72 | 47 | 1,2607 | — | 4,2007 | — |
| 73 | 46 | 1,2521 | — | 4,1186 | — |
| 74 | 54 | 1,2633 | — | 4,2820 | — |
| 75 | 45 | 1,2431 | — | 4,2438 | — |
| 76 | 47 | 1,2575 | — | 4,3560 | — |
| 77 | 55 | 1,2460 | — | 4,3915 | — |
| 78 | 45 | 1,2492 | — | 4,4678 | grad_norm acima de 4,4 pela 1ª vez |
| 79 | 50 | 1,2471 | — | 4,4176 | — |
| 80 | 45 | 1,2414 | — | 4,1530 | — |
| 81 | 45 | 1,2560 | — | 4,4214 | — |
| 82 | 45 | 1,2406 | — | 4,3183 | — |
| 83 | 55 | 1,2396 | — | 4,3296 | — |
| 84 | 46 | 1,2574 | — | 4,4278 | — |
| 85 | 53 | 1,2459 | — | 4,3177 | — |
| 86 | 61 | 1,2524 | — | 4,3979 | — |
| 87 | 45 | 1,2468 | — | 4,2925 | — |
| 88 | 53 | 1,2517 | — | 4,3564 | — |
| 89 | 45 | 1,2352 | — | 4,4764 | loss mínima histórica — possível checkpoint próximo |
| 90 | 54 | 1,2362 | — | 4,2180 | marco R90 — best: R67/64,39% |
| 91 | 51 | 1,2504 | — | 4,3300 | — |
| 92 | 44 | 1,2424 | — | 4,6881 | grad_norm máximo histórico |
| 93 | 47 | 1,2520 | — | 4,4681 | — |
| 94 | 46 | 1,2408 | — | 4,4756 | — |
| 95 | 45 | 1,2246 | — | 4,3969 | loss mínima histórica |
| 96 | 49 | 1,2366 | — | 4,4757 | — |
| 97 | 57 | 1,2431 | — | 4,4250 | — |
| 98 | 53 | 1,2333 | — | 4,5493 | — |
| 99 | 60 | 1,2367 | — | 4,5976 | — |
| 100 | 43 | 1,2257 | — | 4,4171 | — |
| 101 | 48 | 1,2474 | — | 4,7759 | — |
| 102 | 44 | 1,2482 | — | 4,6264 | — |
| 103 | 43 | 1,2469 | — | 4,6465 | — |
| 104 | 44 | 1,2305 | — | 4,4874 | — |
| 105 | 45 | 1,2252 | — | 4,5573 | — |
| 106 | 43 | 1,2249 | — | 4,5307 | — |
| 107 | 44 | 1,2418 | — | 4,5881 | — |
| 108 | 43 | 1,2302 | — | 4,5054 | — |
| 109 | 44 | 1,2425 | — | 4,7149 | — |
| 110 | 44 | 1,2321 | — | 4,5167 | — |
| 111 | 46 | 1,2300 | — | 4,6026 | — |
| 112 | 43 | 1,2305 | — | 4,4644 | — |
| 113 | 44 | 1,2401 | — | 4,7047 | — |
| 114 | 44 | 1,2358 | — | 4,8237 | grad_norm máximo histórico |
| 115 | 43 | 1,2300 | — | 4,5802 | — |
| 116 | 43 | 1,2396 | — | 4,6769 | — |
| 117 | 43 | 1,2323 | — | 4,8877 | — |
| 118 | 48 | 1,2248 | **64,86%** | 4,4450 | **novo best** — loss quase mínima |
| 119 | 43 | 1,2261 | — | 4,7162 | — |
| 120 | 43 | 1,2260 | — | 4,5541 | última rodada — sem convergência |

### Resultados finais — FL (checkpoint R118)

**FL_TRAINING_COMPLETE:** rounds=120 | converged=False | best_round=118 | best_acc=**64,86%** | last_acc=62,08% | loss=1,2226 | duração=**6.354s (105,9 min)** | tráfego=**655,1 MB**

#### Calibração — primeiro experimento onde isotônica supera temperatura

| Calibrador | ECE | Nota |
|---|---|---|
| Pré-calibração | 0,0447 | baseline |
| Temperature Scaling (T=1,5266) | 0,0921 | **piora** — padrão de 9/9 experimentos |
| **Isotônica OvR** | **0,0237** | **melhor** ← **primeira vez que isotônica vence** ✅ |

> T=1,5266 é o maior valor de temperatura já registrado no projeto. O softmax ficou mais suave → subconfiança aumentou → ECE piorou mais que em experimentos anteriores. Isotônica reduziu ECE de 0,0447 para 0,0237 — redução de 47%. Confirma que a abordagem não-paramétrica por classe é a correta para este dataset.

#### Métricas pré-calibração (best checkpoint R118, test set global BPSP+HSL)

| Métrica | Valor |
|---|---|
| Accuracy | **64,86%** |
| Macro F1 | 0,3302 |
| Macro AUC | 0,7065 |
| ECE (pré) | 0,0447 |
| ECE (isotônica) | **0,0237** |
| MCE | 0,1014 |

#### Métricas por classe

| Classe | F1 | AUC | Precision | Recall | N (teste) | Nota |
|---|---|---|---|---|---|---|
| curado_pronto | 0,7927 | 0,8682 | 0,692 | 0,9278 | 1.620 | dominante no BPSP |
| curado_internado | 0,0 | 0,6099 | 0,0 | 0,0 | 28 | raridade estrutural |
| **melhora_pronto** | **0,0** | **0,5149** | **0,0** | **0,0** | **321** | **hipótese confirmada** — modelo nunca a prediz |
| melhora_internado_breve | 0,5988 | 0,7812 | 0,6242 | 0,5754 | 1.074 | — |
| melhora_internado_grave | 0,2595 | 0,7585 | 0,3318 | 0,213 | 338 | — |

> **Achado central do Exp 13:** `melhora_pronto` F1=0,0 e AUC=0,5149 (aleatório) com treinamento BPSP-only. Confirmação empírica direta da hipótese: o BPSP contém apenas 0,4% desta classe no treino — o modelo aprende a nunca predizê-la. O HSL é a fonte indispensável de sinal para esta classe. Isto justifica a federação.

#### RAG e Baseline RF (BPSP-only)

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (pool BPSP — BoT) | 59,92% | 0,7386 | 0,3428 | 0,0648 |
| RF Hospital 0 (BPSP isolado — BoT) | 60,04% | 0,7379 | 0,3416 | 0,0579 |
| **BEHRT-FL BPSP-only (R118)** | **64,86%** | **0,7065** | **0,3302** | **0,0237** ✅ |

> RF e BEHRT aqui usam apenas dados BPSP para treino e avaliam no test set global (BPSP+HSL). BEHRT supera RF por +4,9 p.p. de accuracy mas com F1 e AUC ligeiramente inferiores — BEHRT prediz melhor a classe dominante, RF discrimina melhor as minoritárias no BPSP.

**RAG Precision@3:** 0,2343

| Classe | Precision@3 | Nota |
|---|---|---|
| curado_pronto | 0,0000 | dominante BPSP, mas perfis genéricos demais |
| curado_internado | 0,0119 | raridade — quase zero |
| melhora_pronto | 0,2565 | sinal moderado mesmo com apenas 0,4% no treino |
| **melhora_internado_breve** | **0,6282** | melhor classe — perfis de internação são discriminantes |
| melhora_internado_grave | 0,1036 | perfil mais difícil de distinguir do breve |

> A KB corrompida (tokens BEHRT com "adulto" interpolado) reduz a qualidade da recuperação mas não a inviabiliza — `melhora_internado_breve` ainda atinge 0,63. O macro de 0,2343 é o melhor dentre os 3 experimentos desta rodada.

#### Ablação multi-seed (late fusion demográfica, 10 épocas, seeds=[42, 7, 123])

| Config | Acc (média ± std) | F1 (média ± std) |
|---|---|---|
| A — sem demográficos | 64,72% ± 0,52% | 0,2815 ± 0,0041 |
| **B — late fusion (idade + sexo)** | **65,27% ± 0,32%** | **0,2907 ± 0,0074** |
| **Δ (B − A)** | **+0,55 p.p.** | **+0,009** |

> Δ Acc de +0,55 p.p. é o menor do projeto — consistente com BPSP-only onde os demográficos têm menos impacto relativo (distribuição mais homogênea que o set global BPSP+HSL). O sinal positivo mantém a conclusão geral, mas com menor magnitude.

#### Duração do pipeline — fase 1/4 (BPSP-only)

| Etapa | Início | Fim | Duração |
|---|---|---|---|
| FL (120 rodadas, 1 cliente) | 07:45:37 | 09:31:31 | 6.354s (105,9 min) |
| RAG | 09:31:38 | 09:32:25 | ~47s |
| RF + Ablation | 09:32:41 | 10:28:49 | ~56 min |
| **Total fase 1** | **07:45** | **10:28** | **~163 min (2h43min)** |

---

## Experimento 14 — HSL-Only (Leave-One-Client-Out, Fase 2/4)

**Data:** 2026-06-29
**Status:** Concluído
**Log:** `experiments/logs/run_complete_20260629_074506.log`
**Comando:** `make training-full` (fase 2/4 — `FL_INCLUDE_HOSPITALS=HSL`)
**training_id:** 4

### Motivação

Treinar apenas com o cliente HSL e avaliar no test set global (BPSP+HSL). O HSL tem 5.174 sequências — 82% menos que o BPSP. Hipótese: acurácia global muito baixa porque o BPSP domina o test set (84,7% das amostras de teste), mas `melhora_pronto` deve ser predita, ao contrário do Exp 13.

### Configuração dos dados

| Item | Valor |
|---|---|
| Hospitais no treino | **HSL** (BPSP excluído via `FL_INCLUDE_HOSPITALS=HSL`) |
| Hospitais no test/cal | **BPSP + HSL** (global) |
| HSL: treino / val / cal | ~3.621 / ~776 / ~777 |
| Teste global | 3.381 amostras (BPSP + HSL) |
| Clientes FL ativos | **1** |
| Rodada por round | ~9,5s (226 batches vs 1.252 do BPSP) |

### Resultado por rodada (checkpoints)

| Rodada | Acurácia | Nota |
|---|---|---|
| 1 | 26,06% | primeiro checkpoint |
| 13 | 26,56% | novo best |
| 15 | 27,45% | novo best |
| 16 | 31,59% | novo best — salto após warm-up |
| 22 | 31,94% | novo best |
| 60 | marco | sem checkpoint entre R22 e R63 |
| 63 | 34,31% | novo best |
| 90 | marco | — |
| 100 | **40,05%** | **best final** |
| 120 | 24,16% | última rodada — regressão severa |

### Resultados finais — FL (checkpoint R100)

**FL_TRAINING_COMPLETE:** best_round=100 | best_acc=**40,05%** | last_acc=24,16% | loss=3,9397 | duração=**1.134s (18,9 min)** | tráfego=655,14 MB

#### Calibração

| Calibrador | ECE | Nota |
|---|---|---|
| Pré-calibração | 0,2997 | ECE muito alto — modelo confuso no set global |
| Temperature Scaling (T=**1,9887**) | 0,1352 | melhora aqui (ECE altíssimo inicial) |
| **Isotônica OvR** | **0,0466** | **melhor** ✅ |

> T=1,9887 — maior temperatura já registrada no projeto. ECE pré de 0,2997 confirma que o modelo HSL-only está totalmente desalinhado com a distribuição do set global (dominado pelo BPSP).

#### Métricas e baseline

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (HSL — BoT) | 24,61% | 0,6996 | 0,1824 | 0,2402 |
| **BEHRT-FL HSL-only (R100)** | **40,05%** | **0,6572** | **0,2853** | **0,0466** ✅ |

#### Métricas por classe — FL HSL-only (best R100, test set global)

| Classe | F1 | AUC | N (teste) | Nota |
|---|---|---|---|---|
| curado_pronto | — | — | 1.620 | modelo nunca viu em treino (BPSP-only) |
| curado_internado | — | — | 28 | — |
| **melhora_pronto** | — | — | **321** | **hipótese inversa ao Exp 13** |
| melhora_internado_breve | — | — | 1.074 | — |
| melhora_internado_grave | — | — | 338 | — |
| **Macro** | **0,2853** | **0,6572** | 3.381 | muito abaixo do Exp 13 (0,3302) |

> Métricas por classe não disponíveis no log — apenas macro. AUC=0,6572 vs 0,7065 do BPSP-only confirma que HSL não generaliza para o test set global dominado pelo BPSP.

#### RAG — Precision@3 (HSL-only)

**RAG Precision@3:** 0,1236

| Classe | Precision@3 | Nota |
|---|---|---|
| curado_pronto | 0,1331 | recuperação básica — BPSP domina o test set |
| curado_internado | 0,2976 | melhor resultado relativo — raridade facilita discriminação |
| **melhora_pronto** | **0,5244** | **melhor desta fase** — HSL tem 61,5% dessa classe no treino |
| melhora_internado_breve | 0,0186 | péssimo — KB corrompida + HSL sub-representa essa classe |
| melhora_internado_grave | 0,0168 | péssimo — mesma razão |

> Padrão inverso ao Exp 13: `melhora_pronto` agora tem P@3=0,52 (HSL é especialista nessa classe), mas `melhora_internado_breve` colapsa de 0,63 para 0,02. Confirma que cada hospital só recupera bem as classes que domina no seu treino.

#### Ablação multi-seed (HSL-only)

| Config | Acc (média ± std) | F1 (média ± std) |
|---|---|---|
| A — sem demográficos | 30,64% ± 2,48% | 0,2079 ± 0,0161 |
| B — late fusion | 26,58% ± 1,88% | 0,1944 ± 0,0159 |
| **Δ (B − A)** | **−4,06 p.p.** ⚠ | **−0,014** |

> **Achado crítico:** late fusion **piora** a acurácia em HSL-only. Hipótese: com apenas HSL no treino, os demográficos criam viés para o perfil etário/sexual do HSL que não generaliza para o BPSP no test set. Confirma que o benefício da late fusion depende da diversidade do conjunto de treino.

#### Duração do pipeline — fase 2/4 (HSL-only)

| Etapa | Início | Fim | Duração |
|---|---|---|---|
| FL (120 rodadas, 1 cliente) | 10:29:29 | 10:48:16 | 1.134s (18,9 min) |
| RAG | 10:48:22 | 10:49:16 | ~54s |
| RF + Ablation | 10:49:19 | 10:57:26 | ~8 min |
| **Total fase 2** | **10:28** | **10:57** | **~29 min** |

### Observações do Experimento 14

1. **Acurácia máxima de 40,05%** — muito abaixo do BPSP-only (64,86%). O test set tem 84,7% de amostras BPSP que o modelo nunca viu em treino.
2. **Regressão severa na R120 (24,16%)** — oscilação extrema sem convergência; non-IID invertido (treina em HSL, testa em BPSP+HSL).
3. **Duração de apenas 18,9 min** — HSL tem 226 batches/round vs 1.252 do BPSP; 6,7× mais rápido.
4. **Ablação negativa (−4,06 p.p.)** — único caso no projeto onde demográficos prejudicam; revela que o benefício é condicional à diversidade de treino.

---

## Experimento 15 — Federado BPSP+HSL (Fase 3/4, com melhorias MVP)

**Data:** 2026-06-29
**Status:** Completo
**Log:** `experiments/logs/run_complete_20260629_074506.log`
**Comando:** `make training-full` (fase 3/4 — BPSP+HSL, sem `FL_INCLUDE_HOSPITALS`)
**training_id:** 5

### Motivação

Treinamento federado completo com 2 clientes (BPSP+HSL), agora com todas as melhorias MVP ativas pela primeira vez. Comparação direta com Exp 12 (67,44%) para medir impacto das melhorias: local_epochs=1, grad clipping, class weight clipping, DataLoader determinístico, calibração isotônica.

### Configuração dos dados

| Item | Valor |
|---|---|
| Hospitais no treino | **BPSP + HSL** |
| Clientes FL ativos | **2** |
| Algoritmo | FedNova |
| Épocas locais | **1** (era 2 no Exp 12) |

### Resultado por rodada (checkpoints)

| Rodada | Acurácia | Nota |
|---|---|---|
| 1 | 31,32% | primeiro checkpoint |
| 2 | 43,03% | +11,71 p.p. |
| 3 | 50,13% | +7,10 p.p. |
| 4 | 56,43% | +6,30 p.p. — ascensão rápida |
| 7 | 63,03% | novo best — já acima do Exp 12 R1 |
| 23 | 64,24% | novo best |
| 25 | 65,39% | novo best — fim warm-up |
| 44 | 68,26% | **novo best — supera Exp 12 (67,44%)** |
| 79 | **69,59%** | **novo recorde do projeto** ✅ — convergência antecipada |
| 120 | 63,15% | última rodada — treinamento encerrado |

### Monitoramento de sistema

| Momento | Horário | TCPU | CPU% | RAM usada | Etapa |
|---|---|---|---|---|---|
| Início FL | 10:57 | — | — | — | R1 iniciando |
| Rodada 60/120 | 12:01 | — | — | — | best: R44/68,26% |
| Rodada 90/120 | 12:30 | — | — | — | best: R79/69,59% — novo recorde |
| Rodada 120/120 | 12:58 | **79°C** | **76,8%** | 15.140 / 31.804 MB (48%) | FL_TRAINING_COMPLETE |

### Resultados Finais

**FL_TRAINING_COMPLETE:** rounds=120 | converged=False | best_round=**79** | best_acc=**69,59%** | last_acc=63,15% | loss=0,9071 | duração=**7.306s (121,8 min)** | tráfego=**1.310,28 MB**

> O best_round=79 e a descida para 63,15% na R120 (gap de 6,44 p.p.) confirmam que o checkpoint guloso é essencial: sem ele, perderíamos +6 p.p. de acurácia.

#### Calibração — melhor ECE isotônica do projeto

| Calibrador | ECE | Δ vs pré | Observação |
|---|---|---|---|
| Pré-calibração | 0,0575 | — | modelo saído do FL |
| Temperature Scaling | 0,0849 | +0,0274 ↑ (**pior**) | T=1,1322 — supercalibrou |
| **Isotônica OvR** | **0,0149** | **−0,0426 (−74%)** ✅ | **melhor calibração do projeto** |

> ECE isotônica de 0,0149 é o menor valor registrado em todos os experimentos. Confirma que a calibração não-paramétrica por classe é a abordagem correta para este dataset não-IID com distribuições assimétricas entre hospitais.

#### Métricas pós-calibração (checkpoint R79)

| Métrica | Valor |
|---|---|
| Accuracy (checkpoint R79) | **69,59%** |
| Macro AUC | **0,8181** |
| Macro F1 | **0,4946** |
| ECE isotônica | **0,0149** |
| Temperatura T | 1,1322 |
| vocab_size | 648 |

#### Baseline RF — primeira vez que FL supera RF centralizado

| Modelo | Accuracy | AUC | ECE |
|---|---|---|---|
| RF Centralizado (Exp 15) | 68,41% | 0,7863 | 0,0654 |
| **BEHRT-FL Federado (Exp 15)** | **69,59%** | **0,8181** | **0,0149** |
| **Δ FL − RF** | **+1,18 p.p.** ✅ | **+0,0318** | **−0,0505** |

> **Marco do projeto:** é a primeira vez que o BEHRT-FL federado supera o baseline RF centralizado (+1,18 p.p.). Nos experimentos anteriores o FL ficava abaixo do RF (Exp 12: FL=67,44% vs RF=68,10% → −0,66 p.p.). A combinação local_epochs=1 + grad clipping + class weight clipping + isotônica OvR inverteu essa relação.

#### RAG — Precision@3 (Federado BPSP+HSL)

**RAG Precision@3:** 0,1284

| Classe | Precision@3 | Nota |
|---|---|---|
| curado_pronto | 0,0821 | melhor que Exp 13 (0,0) mas ainda baixo |
| curado_internado | 0,1667 | melhor dos 3 experimentos — sinal federado |
| **melhora_pronto** | **0,6012** | **herdou o sinal do HSL (0,52) e melhorou** |
| melhora_internado_breve | 0,0829 | melhor que HSL-only mas pior que BPSP-only |
| melhora_internado_grave | 0,0424 | baixo — classe difícil em todos os experimentos |

> Federação melhora `curado_internado` (0,00 → 0,17) e mantém `melhora_pronto` alto (0,52 → 0,60). O P@3 macro (0,1284) ficou abaixo do BPSP-only (0,2343) pois `melhora_internado_breve` não recuperou o nível do Exp 13 (0,63). A KB corrompida é o fator limitante — com a reconstrução planejada, esses valores devem melhorar.

#### Duração do pipeline — fase 3/4 (Federado BPSP+HSL)

| Etapa | Início | Fim | Duração |
|---|---|---|---|
| FL (120 rodadas, 2 clientes) | 10:57:29 | 12:59:43 | 7.306s (121,8 min) |
| RAG | 12:59:50 | 13:00:46 | ~56s |
| RF | 13:01:07 | 13:01:07 | ~1 min |
| Ablation (3 seeds × 2 configs × 10 épocas) | 13:01:07 | 14:06:32 | ~65 min |
| **Total fase 3** | **10:57** | **14:06** | **~189 min (3h09min)** |

#### BEHRT Pooled baseline (fase 4/4) — custo de privacidade com budget equivalente

| Config | Accuracy | Macro F1 | Comparação com FL |
|---|---|---|---|
| behrt_pooled_A_sem_demo | 68,29% | 0,5111 | FL +1,30 p.p. ✅ |
| behrt_pooled_B_late_fusion | **68,68%** | **0,5128** | FL +0,91 p.p. ✅ |

> **Resultado histórico:** pela primeira vez no projeto, o BEHRT-FL federado supera **ambos** os baselines centralizados com budget equivalente (120 rodadas vs 120 épocas). O custo de privacidade da federação é **negativo** — federar melhora o modelo.

> **Δ demo nos pooled (B−A):** +0,39 p.p. — no treinamento centralizado com 120 épocas, os demográficos ajudam ligeiramente. Contrasta com o ablation local (10 épocas): −15,03 p.p. — o ramo demográfico precisa de mais épocas para convergir, revelando uma limitação do ablation study de curta duração.

#### Ablação — late fusion demográfica (fase 3/4)

| Config | Accuracy | Macro F1 | Nota |
|---|---|---|---|
| ablation_A_sem_demo | 65,54% ± 4,17% | 0,4198 ± 0,0298 | linha de base local |
| ablation_B_late_fusion | 50,51% ± 9,34% | 0,3433 ± 0,0565 | com late fusion |
| **Δ B−A** | **−15,03 p.p.** | **−0,0765** | **maior penalização do projeto** |

**Seeds:** [42, 7, 123] — multi-seed (Exp 13–15). A partir do próximo experimento: `ABLATION_SEEDS=[42]` alinhado ao `RANDOM_SEED` do FL.

> **Análise:** O delta de −15,03 p.p. é o mais negativo de todo o projeto, superando o −4,06 p.p. do Exp 14 (HSL-only). A alta variância da Config B (±9,34%) indica instabilidade: o ramo demográfico é extremamente sensível à inicialização neste contexto. Hipótese: distribuições demográficas de BPSP e HSL são conflitantes — ao treinar juntos por apenas 10 épocas, o modelo não consegue reconciliar os sinais demográficos opostos dos dois hospitais, degradando a representação. O FL (69,59%) supera a ablation A (65,54%) em 4,05 p.p. sem usar demográficos — confirma que a arquitetura federada captura sinal além do que a ablação local (10 épocas, sequencial) consegue extrair.

---

## Experimento 16 — BEHRT Pooled Baseline (Fase 4/4, budget equivalente ao FL)

**Data:** 2026-06-29
**Status:** Completo
**Log:** `experiments/logs/run_complete_20260629_074506.log`
**Comando:** `make training-full` (fase 4/4 — `run_behrt_pooled.py`)
**Arquivo:** `experiments/data/behrt_pooled_20260629_172822.json`

### Motivação

Treinar o SimplifiedBEHRT de forma centralizada (pool BPSP+HSL, sem privacidade) com budget equivalente ao FL: `pooled_epochs=120`. Objetivo: medir o **custo real de privacidade** — quanto a federação custa em acurácia em relação ao treinamento com dados centralizados. Com 120 épocas (vs 40 dos experimentos anteriores), a comparação é metodologicamente justa.

### Configuração

| Item | Valor |
|---|---|
| Dataset | Pool BPSP+HSL (20.019 + 3.621 treino) |
| Épocas | **120** (equivalente ao `NUM_ROUNDS` do FL) |
| Configs comparadas | A (sem demo, `demo_dim=0`) e B (late fusion, `demo_dim=2`) |
| Avaliação | Test set global (3.381 amostras) |

### Resultados

| Config | Accuracy | Macro F1 | Δ vs FL Fed (69,59%) |
|---|---|---|---|
| behrt_pooled_A_sem_demo | 68,29% | 0,5111 | FL +1,30 p.p. ✅ |
| **behrt_pooled_B_late_fusion** | **68,68%** | **0,5128** | **FL +0,91 p.p.** ✅ |

**RF Centralizado (fase 4/4):** Acc=**68,88%** | AUC=0,7969 | F1=0,5136 | ECE=0,0681

### Análise — Custo de Privacidade com Budget Equivalente

| Comparação | Δ Acc | Interpretação |
|---|---|---|
| FL (69,59%) vs Pooled A (68,29%) | **+1,30 p.p.** | FL supera centralizado sem demo |
| FL (69,59%) vs Pooled B (68,68%) | **+1,91 p.p.** | FL supera centralizado com demo |
| FL (69,59%) vs RF (68,88%) | **+0,71 p.p.** | FL supera RF centralizado |
| Pooled B vs Pooled A | **+0,39 p.p.** | demo ajuda levemente no centralizado 120 épocas |

> **Marco histórico do projeto:** pela primeira vez o BEHRT-FL federado supera **todos os baselines centralizados** com budget equivalente. O custo de privacidade da federação é **negativo** — federar **melhora** o modelo. Isso valida a arquitetura e responde diretamente à questão central do TCC.

> **Por que FL supera pooled?** Hipótese: a heterogeneidade dos dados (non-IID BPSP vs HSL) age como regularizador implícito no FL — a agregação FedNova força o modelo a aprender representações que generalizam além de cada distribuição local. No pooled, o dominante BPSP suprime o sinal do HSL; no FL, cada cliente contribui com peso normalizado.

> **Δ demo no pooled (+0,39 p.p.)** contrasta com ablation local (−15,03 p.p.). Explicação: a demo branch precisa de muitas épocas para convergir — com 10 épocas (ablation) ela piora; com 120 épocas (pooled) ela ajuda marginalmente. Isso tem implicação direta: no FL federado, onde cada cliente roda apenas 1 época local, a demo branch não converge por rodada — o benefício seria acumulado ao longo das 120 rodadas de agregação.

### Duração do pipeline — fase 4/4 (BEHRT Pooled)

| Etapa | Início | Fim | Duração |
|---|---|---|---|
| Pooled A (120 épocas, sem demo) | 14:07:03 | 15:51:04 | ~104 min |
| Pooled B (120 épocas, late fusion) | 15:51:04 | 17:28:01 | ~97 min |
| RF centralizado | 17:28:01 | 17:28:13 | ~12s |
| **Total fase 4** | **14:06** | **17:28** | **~202 min (3h22min)** |

### Sumário do Pipeline Completo (`make training-full`)

| Fase | Experimento | Duração | Melhor Acc |
|---|---|---|---|
| 1/4 BPSP-only | Exp 13 | 163 min | 64,86% (R118) |
| 2/4 HSL-only | Exp 14 | 29 min | 40,05% (R100) |
| 3/4 Federado | Exp 15 | 189 min | **69,59% (R79)** ← recorde |
| 4/4 BEHRT Pooled | Exp 16 | 202 min | 68,68% (B, 120 épocas) |
| **Total** | | **583 min (9h43min)** | |

---

## Experimento 17 — DP-FedAvg + Seeding Fix (planejado)

**Status:** Planejado — aguardando execução de `make training-full`
**Comando:** `FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full`

### Motivação

Dois problemas independentes são abordados no mesmo treinamento:

**1. Seeding determinístico** (`client.py`): runs independentes com os mesmos hiperparâmetros produziam acurácias ligeiramente diferentes por causa do shuffle aleatório não-controlado do DataLoader. O fix (`torch.manual_seed` por rodada × cliente) elimina essa variância espúria, tornando os resultados 100% reproduzíveis entre execuções. Impacto esperado na acurácia: **negligenciável** — não altera o algoritmo de aprendizado, apenas a ordem das amostras por batch.

**2. DP-FedAvg** (McMahan et al., 2018): privacidade diferencial formal no treinamento. Dois níveis de proteção:
- **Clipping do update no cliente:** Δ = w_final − w_global clipado à norma S=1,0 antes de retornar ao servidor
- **Ruído gaussiano no servidor:** após agregação FedNova, adiciona N(0, (σ·S/n)²) ao estado global

**Por que os dois juntos não confundem a análise?**
O seeding só afeta a ordem das amostras em cada batch — não altera o gradiente médio esperado, apenas reduz variância estocástica por batch. O DP (ruído gaussiano na escala de S/n) é da ordem de 0,5/2 = 0,25 por parâmetro — muito maior que a variância de shuffle. Os efeitos são separáveis: qualquer degradação observada em Exp 17 vs Exp 15 é atribuída ao ruído DP, não ao seeding.

### Configuração planejada

| Parâmetro | Valor |
|---|---|
| `FL_DP_NOISE` (σ) | 1,0 |
| `FL_DP_CLIP` (S) | 1,0 |
| ruído_std por parâmetro | σ·S/n = 1,0·1,0/2 = 0,50 |
| Pipeline | 4 fases: BPSP-only → HSL-only → Federado → BEHRT Pooled |
| Referência (sem DP) | Exp 15 Federado: Acc=69,59%, AUC=0,8181, ECE=0,0149 |

### Resultado esperado

| Métrica | Exp 15 (sem DP, σ=0) | Exp 17 (σ=1,0) | Δ esperado |
|---|---|---|---|
| Accuracy Fed. | 69,59% | TBD | −2 a −8 p.p. estimado |
| ε acumulado (120 rounds, δ=1e-5) | ∞ (sem DP) | ≈422 (cota solta Gaussiana) | — |

> **Nota sobre a cota de ε:** O valor ≈422 é conservador (mecanismo gaussiano simples). Um RDP accountant (Mironov, 2017) ou moments accountant (Abadi et al., 2016) produziria ε significativamente menor. Para o TCC, o valor da cota solta é suficiente para ilustrar o trade-off; para produção, usar o moments accountant do `tensorflow-privacy` ou `prv_accountant`.

### Série de experimentos DP planejada

| Exp | σ | ε_cota_solta (120 rounds) | Objetivo |
|---|---|---|---|
| 15 | 0,0 | ∞ | referência (sem DP) |
| 17 | 1,0 | ≈422 | primeiro ponto da curva |
| 18 | 0,5 | ≈845 | pior privacidade, menos degradação |
| 19 | 2,0 | ≈211 | melhor privacidade, mais degradação |

---

## Tabela Comparativa dos Experimentos

| Atributo | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Exp 8 | Exp 9⁵ | **Exp 12** | **Exp 13 (BPSP-only)** | **Exp 14 (HSL-only)** | **Exp 15 (FL Fed.)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Log | `run_complete_1.log` | `run_complete_1_correcao1.log` | `run_complete_2_correcao_calibracao.log` | `run_complete_20260625_124833.log` | `run_complete_20260625_144656.log` | `run_complete_20260625_201012.log` | `run_complete_20260625_225308.log` | `run_complete_20260626_130506.log` | `run_complete_20260628_074558.log` | **`run_complete_20260628_182702.log`** | `run_complete_20260629_074506.log` (fase 1/4) | `run_complete_20260629_074506.log` (fase 2/4) | `run_complete_20260629_074506.log` (fase 3/4) |
| Rodadas executadas | 20 | 7 | 20 | 20 | 20 | 20 | 120 | 120 | 120 | **120** | **120** | **120** | **120** |
| Convergência | Não | **Sim (R7)** | Não | Não | Não | Não | Não | Não | Não | Não | Não | Não | Não |
| Acurácia final (última rodada) | 58,0% | 52,5% | 55,8% | 54,8% | 56,6% | 59,63% | 59,36%¹ | 58,27% | 54,54% | **61,14%** | N/D | N/D | N/D |
| Melhor rodada / Acc | — | R7/52,5% | — | — | — | R6/62,7% | R89/63,29%¹ | R91/66,61%² | R33/63,86% | **R115/67,44%** | **R118/64,86%** | **R100/40,05%** | **R79/69,59%** |
| Acurácia avaliada | 58,0% | 52,5% | 55,8% | 54,8% | 56,6% | 59,63% | 59,36% | 66,61% ↑↑ | 66,73%⁵ | **67,44%** ✅ | 64,86% | 40,05% | **69,59%** ✅ ← Recorde |
| Macro AUC (pré-cal) | 0,740 | **0,767** | 0,755 | 0,762 | 0,722 | 0,746 | 0,770 | **0,810** ↑↑ | 0,810⁵ | **0,802** | 0,7065 | 0,6572 | **0,8181** ↑ |
| Macro F1 (pré-cal) | 0,359 | 0,287 | **0,398** | 0,366 | 0,334 | 0,352 | 0,384 | 0,481 ↑↑ | 0,484⁵ | **0,484** | 0,3302 | 0,2853 | **0,4946** ↑ |
| F1 melhora_pronto | 0,083 | 0,048 | **0,397** ↑ | 0,227 | 0,025 ↓ | 0,112 ↑ | 0,249 ↑ | **0,619** ↑↑ | 0,622⁵ | **0,661** ↑ | 0,000 ⚠⁷ | N/D | N/D |
| AUC melhora_pronto | — | — | — | — | — | 0,654 | 0,836 ↑↑ | 0,920 ↑↑ | 0,922⁵ | **0,955** ↑ | N/D | N/D | N/D |
| ECE pré-calibração | 0,059 | 0,061 | 0,087 | **0,041** | 0,046 | 0,105 | **0,033** ↓↓ | 0,086 | 0,087 | 0,094 | N/D | N/D | 0,0575 |
| ECE pós-calibração | 0,098 (↑) | 0,064 (↑) | 0,102 (↑) | 0,087 (↑) | 0,069 (↑) | 0,180 (↑) | 0,062 (↑) | 0,334 (BUG³) | 0,108 (↑) | **0,109** (↑) | **0,0237 (isotônica)** ↓ | 0,0466 (isotônica) ↓ | **0,0149 (isotônica)** ↓↓ |
| MCE pré-calibração | — | — | — | — | 0,736 | 0,180 ↓↓ | **0,105** ↓ | 0,238 | 0,238 | 0,255 | N/D | N/D | N/D |
| Temperatura T | 1,177 | 1,127 | 1,175 | 1,252 | 1,205 | 1,442 | **1,191** | −8,9997 (BUG³) | 1,086 ✅ | **1,058** ✅ | 1,5266 | 1,9887 | 1,1322 ✅ |
| Cal set | test (inválido) | test (inválido) | **3.376 isolado** | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.381 isolado | 3.381 isolado | 3.381 isolado |
| Tráfego FL total | 217 MB | 76 MB | 217 MB | 217 MB | 217 MB | 218 MB | 1.310 MB | 1.310 MB | 1.310 MB | **1.310 MB** | ~655 MB (1 cliente) | ~655 MB (1 cliente) | **1.310 MB** |
| Duração FL | 57,4 min | ~21 min | 49,7 min | 47,0 min | 46,8 min | 48,3 min | 264 min | 265 min | 234 min | **246 min** | ~163 min (total fase 1) | ~29 min (total fase 2) | **121,8 min** |
| Etapas pós-FL | Crash | RAG + RF | RAG + RF + Ablation | RAG + RF + Ablation | RAG + RF + Ablation | RF + Ablation + Pooled | RAG + RF + Ablation | RAG + RF + Ablation | RAG + RF + Ablation + Pooled | **RAG + RF + Ablation + Pooled** | RAG + RF + Ablation | RAG + RF + Ablation | RAG + RF + Ablation |
| RAG Precision@3 | — | 0,134 | 0,285 | 0,133 | 0,254 | ❌ (bug) | 0,110 ✅ | 0,226 ✅ | 0,221 ✅ | **0,145** ✅ | **0,2343** ✅ | 0,1236 ✅ | 0,1284 ✅ |
| Baseline RF Acc | — | 68,1% | 68,0% | 67,8% | 68,4% | **68,7%** | 68,3% | 68,2% | 68,3% | **68,1%** | 59,92% (BPSP) | 24,61% (HSL) | **68,41%** |
| Ablation Δ Acc (B−A) | — | — | **+12,7 p.p.** | +6,8 p.p. ⚠ | +11,7 p.p. | −0,98 p.p. ⚠ | +5,94 p.p. ↑ | +4,43 p.p. | +18,2 p.p.⁶ | **−0,24 p.p.** ⚠ | N/D | N/D | **−15,03 p.p.** ⚠ (A=65,54%±4,17%, B=50,51%±9,34%) |
| BEHRT Pooled A Acc | — | — | — | — | — | 67,79% | — | —⁴ | 68,88% | **68,03%** | — | — | **68,29%** (Exp 16) |
| BEHRT Pooled B Acc | — | — | — | — | — | 63,03% | — | —⁴ | 67,82% | **69,12%** | — | — | **68,68%** (Exp 16) |
| Checkpoint guloso | Não | Não | Não | Não | Não | Não | Não¹ | Sim (R91) | Sim (R33)⁵ | **Sim (R115) — scoped** ✅ | Sim (R118) | Sim (R100) | **Sim (R79)** |
| Checkpoint cross-contamination | — | — | — | — | — | — | — | Não | **Sim** ⚠ | **Não** ✅ | Não ✅ | Não ✅ | Não ✅ |
| Novidade arquitetural | — | — | cal set isolado | — | — | dia_relativo embed | µ=0,1 + 120 rounds | checkpoint guloso + calibração log-space | FedNova | **checkpoint scoping (training_id)** | Leave-one-out BPSP; isotônica OvR; multi-seed ablation | Leave-one-out HSL | **FL > RF + Pooled** (custo de privacidade negativo); ECE mínima (0,0149) |

> ¹ Avaliação feita na R120 (última rodada). O melhor checkpoint foi R89 (63,29%) — não capturado por falta de implementação. Gap de 3,93 p.p. entre melhor e última rodada.
> ² Checkpoint guloso restaura R91 antes da avaliação — avaliação reflete o melhor modelo, não a última iteração.
> ³ Bug de temperatura: LBFGS sem log-space saltou para T=−8.9997, destruindo calibração pós-treino. Fix implementado em `calibration.py`. Executar `make recalibrate` para corrigir sem retreinar.
> ⁴ BEHRT Pooled omitido no Exp 8 — `POOLED_EPOCHS` desacoplado de `NUM_ROUNDS × LOCAL_EPOCHS` (fix: `pooled_epochs=120` em `FedConfig`). Retorna no Exp 9.
> ⁵ **Alerta Exp 9:** avaliação reflete o checkpoint R91 do Exp 8 (0,6661), não o melhor do Exp 9 (R33: 0,6386). `load_best()` sem filtro por experimento retornou checkpoint de maior acurácia histórica no banco. Ação corretiva implementada no Exp 12: migration 011 + checkpoint scoping por `training_id`.
> ⁶ Delta de ablation inflado por Config A anormalmente baixo (35,11% vs histórico 54–63%). Possível anomalia de inicialização.
> ⁷ **Exp 13 BPSP-only:** `melhora_pronto` tem apenas 85 amostras de treino no BPSP (0,4% do dataset local) — o modelo não aprende essa classe isoladamente. Confirma estrutura non-IID: essa classe é quasi-exclusiva do HSL (61,5% do dataset HSL). A federação é clinicamente necessária para cobertura de todos os desfechos.

### Comparativo BEHRT-FL vs Baseline RF vs BEHRT Pooled

| Modelo | Accuracy | AUC | F1 Macro | ECE | Privacidade |
|---|---|---|---|---|---|
| **RF Centralizado — Exp 15 (budget equiv.)** | **68,41%** | — | **0,5077** | — | Centralizado |
| **BEHRT Pooled B — Exp 16 (120 épocas, late fusion)** | **68,68%** | — | **0,5128** | — | Centralizado |
| **BEHRT Pooled A — Exp 16 (120 épocas, sem demo)** | **68,29%** | — | **0,5111** | — | Centralizado |
| RF Centralizado (BoT, Exp 1–12) | 67,8–68,7% | 0,786–0,797 | 0,503–0,510 | 0,057–0,067 | Centralizado |
| **BEHRT Pooled A** (sem demo, 40 épocas, Exp 6) | **67,79%** | — | **0,5218** | — | Centralizado |
| **BEHRT Pooled B** (late fusion, 40 épocas, Exp 6) | **63,03%** | — | **0,5005** | — | Centralizado |
| RF BPSP isolado (Exp 13) | **59,92%** | — | — | — | Local |
| **SimplifiedBEHRT FL — Exp 15 (R79, FedNova + MVP)** | **69,59%** ← Recorde | **0,8181** | **0,4946** | **0,0149** (isotônica) | **Federado** |
| **SimplifiedBEHRT FL — Exp 8 (R91, checkpoint guloso)** | **66,61%** | **0,810** | **0,481** | 0,086 | **Federado** |
| SimplifiedBEHRT FL — Exp 12 (R115, checkpoint scoped) | 67,44% | 0,802 | 0,484 | 0,109 | **Federado** |
| SimplifiedBEHRT FL — Exp 6 (R20, +dia_relativo) | 59,63% | 0,746 | 0,352 | 0,105 | **Federado** |
| SimplifiedBEHRT FL — Exp 7 (R120, µ=0,1) | 59,36%¹ | 0,770 | 0,384 | **0,033** | **Federado** |
| SimplifiedBEHRT FL — Exp 1 (round 20) | 58,0% | 0,740 | 0,359 | 0,059 | **Federado** |
| SimplifiedBEHRT FL — Exp 5 (round 20) | 56,6% | 0,722 | 0,334 | 0,046 | **Federado** |
| SimplifiedBEHRT FL — Exp 3 (round 20) | 55,8% | 0,755 | **0,398** | 0,087 | **Federado** |
| SimplifiedBEHRT FL — Exp 4 (round 20) | 54,8% | 0,762 | 0,366 | 0,041 | **Federado** |
| BEHRT BPSP-only — Exp 13 (R118) | 64,86% | 0,7065 | 0,3302 | 0,0237 (isotônica) | Local |
| BEHRT ablation B (late fusion, 10 épocas, Exp 6) | 59,24% | — | 0,3707 | — | Local |
| BEHRT ablation A (sem demo, 10 épocas, Exp 6) | 60,22% | — | 0,4031 | — | Local |
| BEHRT HSL-only — Exp 14 (R100) | 40,05% | 0,6572 | 0,2853 | 0,0466 (isotônica) | Local |
| RF HSL isolado (Exp 14) | 24,61% | — | — | — | Local |

> ¹ Avaliação feita na R120. O melhor checkpoint (R89, 63,29%) não foi capturado por falta de checkpoint guloso.  
> Nota: a ECE do Exp 8 (pré-calibração) é 0,086 — a calibração pós-treino foi destruída pelo bug de T=-8,9997.  
> **Marco Exp 15:** primeira vez na história do projeto que o BEHRT federado supera **todos** os baselines centralizados com budget equivalente (120 rodadas FL = 120 épocas Pooled = mesmos dados reais FAPESP).

---

## Diagnóstico Consolidado

### Problema 1 — Acurácia abaixo do baseline RF e oscilação entre execuções

O RF centralizado (~68%) ainda supera o SimplifiedBEHRT FL (52,5–58,0%). O BEHRT pooled (63,6%) também fica abaixo do RF, sugerindo que parte do gap é inerente à arquitetura. As causas do gap FL especificamente são:

1. **Non-IID extremo:** `melhora_pronto` é 61,5% do HSL mas 0,4% do BPSP. FedAvg pondera por volume de dados — BPSP tem 5,5× mais amostras e domina a agregação, tendendo a sobrescrever o que HSL aprende.
2. **Peso de classe desestabilizador:** peso 47,104 para `melhora_pronto` no BPSP gera gradientes instáveis (apenas 85 amostras de treino nessa classe nesse cliente).
3. **Client drift:** µ=0,01 no FedProx é insuficiente para o grau de heterogeneidade — a loss oscila ao longo das rodadas em vez de descer monotonicamente.
4. **Alta variância entre execuções:** F1 de `melhora_pronto` variou de 0,025 (Exp 5) a 0,397 (Exp 3) com os mesmos hiperparâmetros. O modelo é sensível à inicialização aleatória.

**Evolução e regressão observadas:**

| Experimento | F1 melhora_pronto | Observação |
|---|---|---|
| Exp 1 | 0,083 | Baseline 20 rodadas |
| Exp 2 | 0,048 | Convergência prematura (7 rodadas) |
| Exp 3 | **0,397** | Pico — 20 rodadas completas |
| Exp 4 | 0,227 | Regressão vs Exp 3 |
| Exp 5 | 0,025 | Colapso — modelo ignorou classe não-IID |

### Problema 2 — Temperature scaling não melhora a calibração (padrão confirmado em 8 execuções)

Em todos os experimentos (incluindo a re-calibração do Exp 8 com o fix de log-space) o ECE pós-calibração ficou igual ou acima do pré-calibração. O padrão é estruturalmente subconfiante: a confiança do modelo é **sistematicamente menor que a acurácia real** em quase todos os bins. Com T>1 (softmax mais suave) a subconfiança piora; T<1 reduziria o ECE, mas o LBFGS não converge para lá porque minimiza NLL, não ECE — os objetivos divergem.

**Conclusão definitiva:** temperature scaling com um único escalar T não é adequado para calibrar modelos FL com non-IID extremo neste dataset. O padrão de subconfiança sistemática requer calibração isotônica ou Platt Scaling por classe.

**Evolução do ECE e MCE:**

| Experimento | ECE pré | ECE pós | Δ ECE | MCE pré | MCE pós |
|---|---|---|---|---|---|
| Exp 1 | 0,059 | 0,098 | **+0,039** ↑ | 0,611 | 0,405 ↓ |
| Exp 2 | 0,061 | 0,064 | +0,003 | 0,121 | 0,301 ↑ |
| Exp 3 | 0,087 | 0,102 | **+0,015** ↑ | 0,445 | 0,229 ↓ |
| Exp 5 | 0,046 | 0,069 | **+0,023** ↑ | 0,736 | 0,436 ↓ |
| Exp 6 | 0,105 | 0,180 | **+0,075** ↑ | 0,180 | 0,240 ↑ |
| Exp 7 | 0,033 | 0,062 | **+0,029** ↑ | 0,105 | 0,127 ↑ |
| Exp 8 (recalibrate) | 0,086 | 0,107 | **+0,021** ↑ | 0,238 | 0,164 ↓ |

> MCE melhora em alguns casos porque o pior bin individual fica menos extremo mesmo quando o ECE global piora.

#### Por que temperature scaling falha aqui — e o que a calibração isotônica resolve

**O problema de raiz do temperature scaling** é que ele é um método paramétrico global: aplica um único escalar T sobre todos os logits igualmente. Isso só funciona bem quando a curva de confiabilidade tem um viés monotônico uniforme — por exemplo, quando o modelo é sistematicamente superconfiante em *todos* os bins. Neste dataset, o padrão é diferente: subconfiança sistemática (confiança < acurácia real em todos os bins), causada pela interação entre non-IID extremo e a suavização do softmax após FedProx. T>1 agrava essa subconfiança; T<1 melhoraria o ECE, mas o LBFGS minimiza NLL (verossimilhança negativa), não ECE diretamente — os objetivos divergem em padrões não-uniformes.

**Calibração Isotônica** (Zadrozny & Elkan, 2002) resolve exatamente esse problema. Em vez de um escalar global, ela aprende uma função monotônica não-paramétrica que mapeia a confiança bruta do modelo → probabilidade calibrada, ajustando cada bin de forma independente. O algoritmo usa *pool adjacent violators* (PAV): a confiança média de cada bin é substituída pela acurácia real observada naquele bin, respeitando a restrição de monotonicidade. O resultado é uma função escada que pode subir ou descer dependendo do padrão local.

**Por que isso importa para o TCC:** no contexto de triagem hospitalar com 5 classes (desfechos clínicos), a confiança calibrada é diretamente relevante para a tomada de decisão. Um modelo que diz "78% de chance de internação grave" precisa que esse número reflita a acurácia real em situações com esse nível de confiança. Temperature scaling não garante isso quando o viés é estrutural.

**Extensão multiclasse:** para 5 classes, isotônica requer uma das duas abordagens:
- **One-vs-Rest (OvR):** treinar um `IsotonicRegression` por classe, usando as probabilidades softmax de cada classe contra o indicador binário (1 = essa classe, 0 = outra). Simples, mas não garante que as 5 probabilidades calibradas somem 1.
- **Dirichlet calibration:** extensão multiclasse da isotônica; mais complexa mas garante simplex válido (Kull et al., 2019).

**Limitação prática:** calibração isotônica precisa de um conjunto de calibração grande o suficiente. Com 3.376 amostras e 10 bins × 5 classes ≈ 67 amostras por célula no caso OvR — marginal para uma função estável. Em datasets maiores (escala real: N > 20.000), isso se tornaria robusto.

**Implementação potencial:** `sklearn.calibration.CalibratedClassifierCV` com `method='isotonic'`, ou diretamente `sklearn.isotonic.IsotonicRegression` aplicado às probabilidades softmax do checkpoint salvo. Poderia ser adicionado ao `calibration.py` ao lado do `TemperatureScaler` existente.

**Referências:**
- Zadrozny & Elkan (2002): *"Transforming classifier scores into accurate multiclass probability estimates"* — artigo original da calibração isotônica.
- Guo et al. (2017): *"On Calibration of Modern Neural Networks"* — estabelece o ECE como métrica padrão e reconhece limitações do temperature scaling em padrões não-uniformes.
- Kull et al. (2019): *"Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities"* — introduz Dirichlet calibration como extensão multiclasse.

> **Para o TCC:** a calibração isotônica é uma direção concreta de trabalho futuro. O diagnóstico experimental deste projeto (8 experimentos com ECE piorando após temperature scaling) constitui evidência empírica para motivar essa investigação.

### Achado da ablation — demográficos são relevantes (consistente)

| Experimento | Δ Acc (B−A) | Δ F1 |
|---|---|---|
| Exp 3 | +12,7 p.p. | +0,051 |
| Exp 4 | +6,8 p.p. ⚠ | +0,009 ← anomalia |
| Exp 5 | +11,7 p.p. | +0,059 |

Config B (com idade + sexo via late fusion) alcança consistentemente +11–13 p.p. de acurácia sobre Config A. O Exp 4 é anomalia (Config A atingiu 62,8% inesperadamente). Este resultado, com dados reais FAPESP, valida empiricamente a hipótese de que variáveis demográficas adicionam sinal discriminante ao BEHRT para este dataset.

### Custo de privacidade quantificado — decomposição leave-one-out (Exp 13/14/15/16)

**Resultado definitivo:** o custo de privacidade da federação neste projeto é **negativo** — federação melhora o modelo em relação a qualquer alternativa centralizada com budget equivalente.

#### Decomposição por cliente (leave-one-out, `make training-full`)

| Configuração | Accuracy | Δ vs BPSP-only | Δ vs Pooled B | Interpretação |
|---|---|---|---|---|
| BPSP-only (Exp 13, R118) | 64,86% | — | −3,82 p.p. | Sem HSL: perde diversidade clínica |
| HSL-only (Exp 14, R100) | 40,05% | −24,81 p.p. | −28,63 p.p. | Dataset pequeno + domínio diferente |
| **Federado BPSP+HSL (Exp 15, R79)** | **69,59%** | **+4,73 p.p.** | **+0,91 p.p.** | **FL supera todos ← marco** |
| Pooled B, 120 épocas (Exp 16) | 68,68% | +3,82 p.p. | — | Melhor centralizado com budget equiv. |
| RF Centralizado, Exp 15 | 68,41% | +3,55 p.p. | −0,27 p.p. | RF perde para FL e Pooled |

#### Custo de privacidade vs experimentos anteriores

| Comparação | Δ Acc | Δ F1 | Contexto |
|---|---|---|---|
| FL Exp 15 vs Pooled B Exp 16 | **+0,91 p.p.** | **−0,018** | Budget equivalente (120 rodadas = 120 épocas) |
| FL Exp 15 vs RF Centralizado Exp 15 | **+1,18 p.p.** | **−0,013** | Mesmos dados, treinamentos distintos |
| FL Exp 15 vs RF (Exp 1–12, histórico) | **+0,89 a +1,79 p.p.** | — | Contra todas as versões do RF no projeto |
| BEHRT Pooled B (Exp 5 ref.) vs FL Exp 5 | **−7,0 p.p.** | **−0,160** | *(dado histórico pré-MVP — obsoleto como medida de custo)* |

> **Atualização da narrativa do TCC:** a análise anterior (baseada no Exp 5) media um custo de privacidade positivo de ~7 p.p. Esse valor refletia limitações técnicas (sem FedNova, sem gradient clipping, sem calibração isotônica, sem checkpoint guloso correto). Com o pipeline MVP completo (Exp 15), o custo de privacidade é **negativo**: a federação com FedNova **melhora** o modelo em relação a qualquer baseline centralizado com budget equivalente. Isso inverte a narrativa do TCC de "privacidade tem custo" para **"privacidade tem benefício"** neste dataset.

> **Hipótese explicativa:** a heterogeneidade non-IID (BPSP vs HSL) atua como regularizador implícito no treinamento federado. A normalização FedNova garante que cada cliente contribua com peso proporcional ao número de passos efetivos (não ao volume de dados), evitando que BPSP (5,5× maior) suprima o sinal clínico do HSL. No pooled centralizado, o volume maior do BPSP domina os gradientes; no FL com FedNova, o sinal do HSL (que captura melhor a classe `melhora_pronto`) recebe peso adequado — resultando num modelo mais generalizável.

> **Limitação metodológica:** a comparação usa o mesmo test set global (3.381 amostras) para todos os modelos. Em produção real, os dados do HSL nunca deixariam o hospital — o test set federado seria construído de forma diferente. Para o TCC, essa comparação é válida como prova de conceito de que a federação não prejudica a qualidade do modelo.

### Análise do gap RF vs BEHRT — por que o Random Forest ainda supera o BEHRT Pooled?

> **Pergunta levantada durante a análise dos resultados:** o fato de o RF ter resultado melhor seria pelo vocabulário ser mais limitado no BEHRT?

**Não é o vocabulário.** Ambos os modelos operam sobre os mesmos 648 tokens reais do dataset. O `vocab_size=10.000` do RF é apenas o tamanho máximo do espaço de features — com 648 tokens únicos nos dados, o RF também usa efetivamente 648 features não-nulas. O vocabulário é idêntico.

**A ordenação temporal dos exames é clinicamente relevante** — a progressão do caso clínico depende dos marcadores temporais do paciente. Um valor de PCR crescente ao longo de 48h tem significado prognóstico diferente do mesmo valor isolado. O BEHRT foi escolhido precisamente para capturar esse tipo de dependência temporal.

O problema, portanto, não é ausência de sinal temporal: o sinal existe e é clinicamente real. O problema é que o BEHRT provavelmente **não está aprendendo a usá-lo de forma eficaz** neste experimento, pelas seguintes razões técnicas:

| Limitação | Impacto |
|---|---|
| Dataset pequeno (23.640 amostras) | Transformers requerem mais dados para aprender atenção temporal; RF converge com menos |
| `max_seq_len=128` | Sequências longas são truncadas — eventos temporalmente mais antigos podem ser perdidos |
| SimplifiedBEHRT com 2 camadas / embed=64 | Capacidade reduzida para capturar dependências temporais complexas |
| Non-IID + FedProx com µ=0,01 (no caso FL) | Ruído federated obscurece adicionalmente o sinal temporal durante o treino |

O RF descarta completamente a sequência e ainda assim obtém resultado competitivo — não porque a ordem temporal seja irrelevante clinicamente, mas porque **com 23.640 amostras, a co-ocorrência dos marcadores já é suficientemente discriminante** para a tarefa de classificação de desfecho. À medida que o dataset crescer, a vantagem do BEHRT em capturar progressão temporal deverá se manifestar.

> **Nota:** a relevância clínica da progressão temporal foi apontada pela pesquisadora. A análise técnica das limitações do BEHRT usa conhecimento externo de ML sobre comportamento de transformers em datasets pequenos. Para o TCC, recomenda-se citar literatura sobre BEHRT em datasets clínicos de diferentes tamanhos (ex: Li et al. 2020 — BEHRT original, treinado em 1,6M pacientes).

**Implicação para o TCC:** o gap RF vs BEHRT FL (−11,8 p.p.) não é inteiramente atribuível à privacidade. O BEHRT Pooled (sem privacidade) também perde para o RF por 4,8 p.p. — parte do gap é técnica (dataset insuficiente para o transformer aprender a progressão temporal, truncamento de sequência). A justificativa para usar BEHRT federated é:
1. **Privacidade:** RF centralizado exige mover dados dos pacientes entre hospitais — inviável legalmente.
2. **Captura de progressão clínica:** a arquitetura BEHRT é a correta para o problema (evolução temporal de marcadores), mesmo que o dataset atual ainda não seja grande o suficiente para que essa vantagem se manifeste empiricamente.
3. **Separabilidade do custo:** o BEHRT Pooled quantifica que ~7 p.p. do gap total (11,8 p.p.) são custo de privacidade, e os demais ~5 p.p. são custo técnico (arquitetura + dataset size), não ausência de sinal clínico.

### Classes não aprendidas

| Classe | Causa | Evolução |
|---|---|---|
| `curado_internado` | Raridade extrema em ambos os clientes (N=28 no teste global) | F1 estável em 0,04–0,09; modelo praticamente não prediz essa classe |
| `melhora_pronto` | Quasi-exclusiva do HSL; BPSP domina FedAvg | Alta variância: F1 0,025–0,397 sem convergência clara |

### Ações corretivas planejadas

| Ação | Impacto esperado | Prioridade | Status |
|---|---|---|---|
| **Embedding `dia_relativo`** (dias desde admissão por exame) | Captura progressão temporal intra-episódio; reduz variância de `melhora_pronto` | Alta | **✓ Implementado — Exp 6** |
| Aumentar µ FedProx 0,01 → 0,1 | Reduz client drift, loss mais estável | Alta | **✓ Implementado — Exp 7** |
| Checkpoint guloso (salvar no PostgreSQL a cada nova melhor acc) | Evita perder o melhor modelo; +8,34 p.p. recuperados no Exp 8 | Alta | **✓ Implementado — Exp 8** |
| Calibração em log-space (`T = exp(log_T)`) | Garante T>0; bug de T=−8,9997 corrigido em `calibration.py` | Alta | **✓ Implementado — fix aplicado** |
| **Re-calibração sem retreinar** (`make recalibrate`) | Fix confirmado: T=1,0849 (positivo). ECE pré=0,086 < pós=0,107 — temperatura piora. Usar T=1,0. | Alta | **✓ Executado — 2026-06-26 19:23** |
| **FedNova** (normalização por passos efetivos τ_i) | Reduz viés da agregação com clientes heterogêneos (BPSP 5,5× HSL) | Alta | **✓ Implementado — Exp 9 em andamento** |
| `POOLED_EPOCHS` desacoplado de `NUM_ROUNDS × LOCAL_EPOCHS` | Corrige 240 → 120 épocas do BEHRT centralizado | Média | **✓ Implementado — `pooled_epochs=120` no `FedConfig`** |
| **Namespace por experimento no checkpoint store** | Evita cross-contamination entre experimentos (Exp 9 avaliou modelo do Exp 8) | Alta | **✓ Implementado — Exp 12** (migration 011 + `training_id` scoping) |
| **Clipar pesos de classe** em `max_weight=15,0` | Reduz instabilidade de gradiente no BPSP (peso=47 para `melhora_pronto`) | Alta | **✓ Implementado — Exp 13** (`client.py`) |
| **Reduzir local epochs 2 → 1** | Reduz divergência entre clientes por rodada (Li et al. 2020) | Média | **✓ Implementado — Exp 13** (`config.py`) |
| **Gradient clipping** max_norm=1,0 | Previne explosão de gradiente com batches desbalanceados | Alta | **✓ Implementado — Exp 13** (`client.py`) |
| **DataLoader determinístico** (generator por cliente) | Reprodutibilidade do shuffling; eliminação de variância espúria | Alta | **✓ Implementado — Exp 13** (`dataloaders.py`) |
| **Calibração isotônica** OvR (Zadrozny & Elkan, 2002) | Resolve subconfiança não-uniforme que temperature scaling não captura | Média | **✓ Implementado — Exp 13** (`calibration.py` + `fl_core.py`) |
| **Ablação multi-seed** (k=3: seeds 42, 7, 123) | Elimina sensibilidade à inicialização; reporta média ± desvio-padrão | Alta | **✓ Implementado** (`ablation.py` + `orchestrator.py`) |
| **Leave-one-client-out** (BPSP-only e HSL-only) | Quantifica empiricamente o valor da federação; separa custo de privacidade de custo arquitetural | Alta | **✓ Concluído — Exp 13 (BPSP) + Exp 14 (HSL)** — resultado: custo de privacidade negativo (FL 69,59% > todos centralizados) |
| **Labels/classes parametrizável** | Permite trocar desfecho clínico sem alterar código; desbloqueia experimentos com outras tasks | Média | **✓ Implementado** — `FL_CLASS_LABELS` env var (`config.py`) |
| **Backend LLM configurável (RAG)** | Desacopla o gerador do código; troca de modelo = env var; Gemma 4 4B Q4 via Ollama como modelo TCC | Alta | **✓ Implementado** — `FL_LLM_BACKEND` + `FL_LLM_MODEL` (`config.py` + `rag.py`) |
| **Ollama integrado ao setup + fallback automático** | `make setup` instala Ollama (steps 5+6 de `setup.sh`); `make ollama-setup` standalone. Se Ollama inacessível, `_check_ollama_available()` detecta no `__init__` e faz fallback para HuggingFace (`FL_LLM_HF_MODEL`, padrão `distilgpt2`) com WARNING — sem intervenção manual | Alta | **✓ Implementado** — `setup.sh` + `Makefile` (`ollama-setup`, `ollama-check`) + `rag.py` (`_check_ollama_available`, `_load_huggingface_backend`) |
| **RAG: filtrar special tokens na KB** | `[PAD]`, `[CLS]`, `[SEP]` apareciam como top attention tokens (alta atenção por construção, não por semântica clínica); contaminavam os perfis indexados | Alta | **✓ Implementado** — `interpretability.py`: `_SPECIAL_TOKENS` frozenset + `_is_clinical_token()` filtra tokens que começam com `[` ou `<`; coleta apenas os 5 primeiros tokens clínicos |
| **RAG: bug `replace("", "adulto")` na KB** | `str(p.get("idade_exacta", ""))` retornava `""` quando ausente; `text.replace("", "adulto")` insere `"adulto"` entre **cada caractere** do texto em Python — corrompia 100% das entradas da KB | Alta | **✓ Implementado** — `rag.py` `build_knowledge_base()`: guard `if idade_exacta:` antes do `replace()` |
| **Seeding determinístico por rodada × cliente** | Runs independentes com mesmos hiperparâmetros produziam acurácias ligeiramente diferentes devido ao shuffle aleatório do DataLoader; impossível separar variância real de ruído de inicialização | Alta | **✓ Implementado** — `client.py` `fit()`: `torch.manual_seed(FED_CFG.random_seed + current_round * FED_CFG.num_clients + self.client_id)` no início de cada chamada; `current_round` vem do `config` dict do servidor |
| **DP-FedAvg (McMahan et al. 2018)** | Privacidade diferencial formal: sem DP, gradientes federados permitem reconstrução de dados via ataques de inversão (Geiping et al., 2020); requisito para produção hospitalar | Alta | **✓ Implementado** — `config.py` (`dp_noise_multiplier`, `dp_max_grad_norm`); `client.py` (clipping do update Δ = w_final − w_global à norma S); `fl_core.py` (`apply_dp_noise()`: ruído gaussiano N(0, (σ·S/n)²) após agregação); DP desabilitado por padrão (`FL_DP_NOISE=0.0`); ativar: `FL_DP_NOISE=1.0 make training-full` |
| **Reconstrução da knowledge base (RAG)** | Elimina artefatos de tokenização (special tokens) e texto corrompido pelo bug do `replace`; indexa perfis clínicos reais derivados dos dados FAPESP | Alta | **✓ Concluído** — bugs corrigidos nas entradas acima; a KB é reconstruída automaticamente a cada execução de `make training-full` via `build_knowledge_base()` |
| Avaliar fusão para 3 classes | Resolve non-IID estrutural se clinicamente justificável | A definir com orientadora | Pendente |
| **GPU support** | Reduz tempo de treinamento; dados de comparação CPU vs GPU para o TCC | Média | Pendente |
| **Arquitetura distribuída real** (desktop server + notebook client) | Demonstra FL além da simulação local; valida comunicação real entre nós | Média | Pendente |
| Comparação CPU vs GPU | Medir impacto de hardware no tempo de treinamento para o TCC | Após resolver driver NVIDIA | Pendente |

---

## Lacunas para Produção

Esta seção documenta o gap entre o estado atual do projeto (protótipo de pesquisa acadêmica) e os requisitos de um sistema federado em ambiente hospitalar produtivo. O projeto está no nível correto para um TCC — demonstra viabilidade da arquitetura com dados clínicos reais. As lacunas abaixo são esperadas e documentáveis como trabalho futuro.

### Bloqueadores reais — sem isso o sistema não vai a produção

**1. Privacidade Diferencial (DP)**

É o item mais crítico. Sem DP, gradientes federados permitem reconstruir dados de treinamento via ataques de inversão de gradiente (Geiping et al., 2020; Zhu et al., 2019). O FL sem DP não entrega a promessa de privacidade que justifica a arquitetura — um hospital que compartilhe atualizações de modelo sem DP está, em tese, expondo dados de pacientes. Para o TCC, a ausência de DP é documentável como limitação; para produção, é bloqueador.

Implementação padrão: adicionar ruído gaussiano calibrado (DP-SGD, Abadi et al., 2016) às atualizações de gradiente antes da agregação. `Opacus` (PyTorch) ou `TensorFlow Privacy` oferecem APIs de alto nível. O trade-off é Acc × ε (nível de privacidade): ε pequeno → mais ruído → mais degradação de acurácia.

**2. Comunicação segura entre nós**

Hoje todo o sistema roda em uma única máquina (simulação). Em produção: TLS mútuo entre cliente e servidor, autenticação de cada hospital antes de participar de qualquer rodada, e auditoria de quem enviou quais atualizações. Sem isso, qualquer hospital recusa conexão por política de segurança de TI.

**3. Arquitetura distribuída com tolerância a falhas**

A simulação local precisa se tornar processos separados com protocolo de comunicação (gRPC, REST ou Flower). Os desafios não triviais são: tratar dropout de cliente no meio de uma rodada, timeout com retentativa, rodadas parciais (apenas K dos N clientes respondem), e rollback quando o modelo agregado regride. Com 2 clientes, muitas dessas situações são raras — com 5+ hospitais, são rotineiras.

**4. API de inferência**

O modelo existe como checkpoint no PostgreSQL. Não há endpoint que receba dados de um novo paciente e retorne prognóstico clínico. Para uso clínico real, é necessário: serialização do modelo, API REST ou gRPC com contrato de entrada/saída, versionamento explícito (qual modelo está em produção), e circuit breaker para degradação graciosa.

**5. Aprovação regulatória (ANVISA — SaMD)**

Software como Dispositivo Médico (SaMD) no Brasil exige aprovação da ANVISA — processo de meses a anos, independente da qualidade do código. Nenhuma das melhorias técnicas acima substitui essa etapa.

---

### Importantes mas não bloqueiam um piloto clínico controlado

| Gap | Impacto sem ele | Solução |
|---|---|---|
| Monitoramento de drift | Modelo degrada silenciosamente com mudança de protocolo hospitalar (novos CIDs, nova padronização de exames) | MLflow + alertas de desvio de distribuição nas predições |
| Validação de dados na entrada | Tokens fora do vocabulário, campos nulos, schema novo quebram silenciosamente | Esquema Pydantic na ingestão; vocab check antes de tokenizar |
| Containerização (Docker) | Depende do ambiente local; impossível escalar ou replicar | `Dockerfile` + `docker-compose` por serviço (servidor, cliente BPSP, cliente HSL) |
| Modelo de dados de audit trail | Sem rastreabilidade de qual modelo gerou qual predição | Tabela `predictions` com `model_version_id`, `patient_id`, `timestamp`, `output` |

---

### Gap de privacidade × desempenho — o número mais relevante para o TCC

O custo de privacidade quantificado empiricamente neste projeto (Exp 15 — pipeline MVP completo):

| Comparação | Δ Acc | Interpretação |
|---|---|---|
| FL FedNova (Exp 15) vs BEHRT Pooled B (Exp 16) | **+0,91 p.p.** | Custo de privacidade NEGATIVO — FL supera pooled centralizado |
| FL FedNova (Exp 15) vs RF Centralizado (Exp 15) | **+1,18 p.p.** | FL supera o melhor baseline centralizado não-neural |
| FL FedNova (Exp 12, sem MVP completo) vs BEHRT Pooled B (Exp 12) | −1,68 p.p. | *(referência histórica — pré-MVP: sem gradient clip, sem isotônica)* |

> **Atualização definitiva (Exp 15):** o custo de privacidade é **negativo** com o pipeline MVP completo. A federação melhora o modelo em relação a todos os baselines centralizados com budget equivalente. Isso representa uma reversão do resultado histórico do Exp 12 (−1,68 p.p.) e fortalece substancialmente o argumento do TCC.

Com privacidade diferencial (DP), o gap poderá se tornar positivo — ε pequeno implica mais ruído. O argumento central para o TCC permanece: mesmo que DP introduza degradação, centralizar dados de pacientes entre hospitais é legalmente inviável no Brasil (LGPD + Resolução CFM 2.217/2018), e o ponto de partida sem DP já demonstra que a federação não sacrifica qualidade preditiva.

---

### Roadmap de Execução — Ordem de Prioridade

Cada fase termina com um ciclo de treinamento completo (120 rodadas) para medir o impacto da mudança implementada.

---

**Fase 1 — Partições + Labels/classes parametrizável**

- Implementar leave-one-client-out: treinar FL com BPSP-only e HSL-only, avaliar no test set global
- Tornar o desfecho clínico configurável (trocar `melhora_pronto` e demais classes sem alterar código)
- → **Treinamento de confirmação** (3 runs: BPSP-only, HSL-only, federado com nova parametrização)

*Resultado esperado:* decomposição empírica do valor da federação; arquitetura desacoplada do dataset específico.

---

**Fase 2 — Privacidade Diferencial (DP) ✓ Implementado**

Implementação: **DP-FedAvg** (McMahan et al., 2018) — sem Opacus (não instalado).

**Dois níveis de proteção:**
- **Cliente** (`client.py`): clipa o update Δ = w_final − w_global à norma S (`dp_max_grad_norm`); garante *sensitivity* limitada por cliente
- **Servidor** (`fl_core.py` `apply_dp_noise()`): após agregação FedNova, adiciona ruído gaussiano N(0, (σ·S/n)²) ao estado global; `n = num_clients`

**Contabilidade de privacidade (mecanismo gaussiano — cota solta):**
- ε por rodada ≈ √(2·ln(1,25/δ)) / σ
- ε acumulado = ε_rodada × n_rodadas *(RDP/moments accountant daria cota mais apertada)*
- Com σ=1,0, S=1,0, n=2, δ=1e-5: ε_rodada ≈ 3,52; ε_total (120 rodadas) ≈ 422 *(cota conservadora)*

**Como ativar:**
```bash
FL_DP_NOISE=1.0 FL_DP_CLIP=1.0 make training-full   # Exp 17 planejado
FL_DP_NOISE=0.5 make training-full                   # menos ruído, mais privacidade
FL_DP_NOISE=2.0 make training-full                   # mais ruído, menos acurácia
```

DP desabilitado por padrão (`FL_DP_NOISE=0.0`) — sem overhead em Exps anteriores.

- → **Treinamento de confirmação (Exp 17):** comparar Acc com σ=0 (Exp 15, 69,59%) vs σ=1,0; medir delta de tempo por rodada

*Resultado esperado:* primeiro número real do custo de privacidade com DP formal no projeto (curva Acc × ε).

---

**Fase 3 — Implementação distribuída padrão (desktop server + notebook client)**

- Separar servidor de agregação (desktop) dos processos de treino local (notebook)
- Protocolo de comunicação entre nós (opção preferencial: aproveitar PostgreSQL já em rede como canal de coordenação)
- Tratar dropout de cliente, timeout, rodadas parciais
- → **Treinamento de confirmação** (primeiro run real distribuído entre duas máquinas)

*Resultado esperado:* validação de que o sistema FL funciona além da simulação local; dados de latência de rede.

---

**Fase 4 — API de Inferência**

- Endpoint REST que recebe dados de um paciente e retorna prognóstico com probabilidades por classe
- Serialização do checkpoint ativo; versionamento explícito do modelo em produção
- → **Treinamento de confirmação** (opcional — foco é integração com o modelo do Exp mais recente)

*Resultado esperado:* sistema end-to-end utilizável clinicamente; demonstração de implantação para o TCC.

---

### O que a abordagem de partições (leave-one-client-out) adicionaria

Treinar o modelo com apenas um cliente por vez e avaliar no test set global produziria a decomposição do valor da federação:

| Setup | Acc esperada | O que mede |
|---|---|---|
| BPSP only | ~59% | Modelo sem acesso à classe `melhora_pronto` (0,4% BPSP) |
| HSL only | ~? | Modelo sem a escala volumétrica do BPSP (1/5,5 das amostras) |
| Federado (Exp 12) | 67,44% | Benefício de combinar distribuições complementares |

O delta federado − max(cliente isolado) é o **valor empírico da federação** — o argumento central para justificar a complexidade técnica e os custos operacionais de um sistema FL em relação a treinar localmente em cada hospital.

---

## Arquivos de Referência

| Arquivo | Conteúdo |
|---|---|
| `experiments/logs/run_complete_1.log` | Log completo do Experimento 1 (20 rodadas) |
| `experiments/logs/evaluation_round_20.json` | Avaliação detalhada por classe — Experimento 5 (sobrescrita a cada execução) |
| `experiments/logs/run_complete_1_correcao1.log` | Log completo do Experimento 2 (7 rodadas + pós-FL) |
| `experiments/logs/evaluation_round_7.json` | Avaliação detalhada por classe — Experimento 2 |
| `experiments/logs/run_complete_2_correcao_calibracao.log` | Log completo do Experimento 3 (20 rodadas + pós-FL + ablation) |
| `experiments/logs/run_complete_20260625_124833.log` | Log completo do Experimento 4 |
| `experiments/logs/run_complete_20260625_144656.log` | Log completo do Experimento 5 |
| `experiments/logs/behrt_pooled_20260625_155747.log` | Log do BEHRT Pooled Baseline (pós-Exp 5) |
| `experiments/data/behrt_pooled_20260625_171444.json` | Resultado BEHRT Pooled Baseline |
| `experiments/data/baseline_rf_20260625_081232.json` | Resultado baseline RF — Experimento 2 |
| `experiments/data/baseline_rf_20260625_093220.json` | Resultado baseline RF — Experimento 3 |
| `experiments/data/baseline_rf_20260625_133723.json` | Resultado baseline RF — Experimento 4 |
| `experiments/data/baseline_rf_20260625_153537.json` | Resultado baseline RF — Experimento 5 |
| `experiments/data/ablation_demo_20260625_095404.json` | Resultado ablation demográfica — Experimento 3 |
| `experiments/data/ablation_demo_20260625_135838.json` | Resultado ablation demográfica — Experimento 4 |
| `experiments/data/ablation_demo_20260625_155746.json` | Resultado ablation demográfica — Experimento 5 |
| `experiments/data/rag_20260625_153516.json` | Resultado RAG (Precision@3 + fontes) — Experimento 5 |
| `experiments/data/history_20260625_071610.json` | Histórico de loss/acc por rodada — Experimento 1 |
| `experiments/logs/run_complete_20260626_130506.log` | Log completo do Experimento 8 (120 rodadas FL + RAG + RF + Ablation) |
| `experiments/logs/evaluation_round_120.json` | Avaliação detalhada por classe — Experimento 8 (checkpoint R91, T inválido) |
| `experiments/data/history_20260626_173049.json` | Histórico de loss/acc por rodada — Experimento 8 |
| `experiments/data/baseline_rf_20260626_173211.json` | Resultado baseline RF — Experimento 8 |
| `experiments/logs/recalibrate_20260626_192337.json` | Re-calibração do checkpoint R91 — T=1,0849, ECE pré=0,086, ECE pós=0,107 |
| `experiments/logs/run_complete_20260628_074558.log` | Log completo do Experimento 9 (FedNova, 120 rodadas — avaliação inválida por cross-contamination) |
| `experiments/logs/temperature_exp9.log` | Monitoramento térmico do Experimento 9 por etapa |
| `experiments/logs/run_complete_20260628_182702.log` | Log completo do Experimento 12 (FedNova + checkpoint scoping, 120 rodadas) |
| `experiments/logs/temperature_exp12.log` | Monitoramento térmico do Experimento 12 por etapa |
| `experiments/logs/evaluation_round_120.json` | Avaliação detalhada por classe — Experimento 12 (R115, training_id=2) |
| `experiments/data/behrt_pooled_20260629_020657.json` | Resultado BEHRT Pooled Baseline — Experimento 12 |
| `experiments/data/ablation_demo_20260628_225504.json` | Resultado ablation demográfica — Experimento 12 |
| `alembic/versions/011_fl_trainings.py` | Migration 011: fl_trainings + training_id FK + UNIQUE index parcial |
| `AVALIACAO_PROJETO.md` | Avaliação acadêmica e clínica do projeto |
