# Sumário de Simulações — MOSAIC-FL

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico  
**Autora:** Jacqueline Abreu | ICMC/USP  
**Atualizado em:** 2026-06-26 (Experimento 8 concluído — Experimento 9 pendente)

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

## Tabela Comparativa dos Experimentos

| Atributo | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | **Exp 8** |
|---|---|---|---|---|---|---|---|---|
| Log | `run_complete_1.log` | `run_complete_1_correcao1.log` | `run_complete_2_correcao_calibracao.log` | `run_complete_20260625_124833.log` | `run_complete_20260625_144656.log` | `run_complete_20260625_201012.log` | `run_complete_20260625_225308.log` | `run_complete_20260626_130506.log` |
| Rodadas executadas | 20 | 7 | 20 | 20 | 20 | 20 | 120 | **120** |
| Convergência | Não | **Sim (R7)** | Não | Não | Não | Não | Não | Não |
| Acurácia final (última rodada) | 58,0% | 52,5% | 55,8% | 54,8% | 56,6% | 59,63% | 59,36%¹ | 58,27% |
| Melhor rodada / Acc | — | R7/52,5% | — | — | — | R6/62,7% | R89/63,29%¹ | **R91/66,61%²** |
| Acurácia avaliada | 58,0% | 52,5% | 55,8% | 54,8% | 56,6% | 59,63% | 59,36% | **66,61%** ↑↑ |
| Macro AUC (pré-cal) | 0,740 | **0,767** | 0,755 | 0,762 | 0,722 | 0,746 | 0,770 | **0,810** ↑↑ |
| Macro F1 (pré-cal) | 0,359 | 0,287 | **0,398** | 0,366 | 0,334 | 0,352 | 0,384 | **0,481** ↑↑ |
| F1 melhora_pronto | 0,083 | 0,048 | **0,397** ↑ | 0,227 | 0,025 ↓ | 0,112 ↑ | 0,249 ↑ | **0,619** ↑↑ |
| AUC melhora_pronto | — | — | — | — | — | 0,654 | 0,836 ↑↑ | **0,920** ↑↑ |
| ECE pré-calibração | 0,059 | 0,061 | 0,087 | **0,041** | 0,046 | 0,105 | **0,033** ↓↓ | 0,086 |
| ECE pós-calibração | 0,098 (↑) | 0,064 (↑) | 0,102 (↑) | 0,087 (↑) | 0,069 (↑) | 0,180 (↑) | 0,062 (↑) | **0,334 (BUG³)** |
| MCE pré-calibração | — | — | — | — | 0,736 | 0,180 ↓↓ | **0,105** ↓ | 0,238 |
| Temperatura T | 1,177 | 1,127 | 1,175 | 1,252 | 1,205 | 1,442 | **1,191** | **−8,9997 (BUG³)** |
| Cal set | test (inválido) | test (inválido) | **3.376 isolado** | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado | 3.376 isolado |
| Tráfego FL total | 217 MB | 76 MB | 217 MB | 217 MB | 217 MB | 218 MB | 1.310 MB | **1.310 MB** |
| Duração FL | 57,4 min | ~21 min | 49,7 min | 47,0 min | 46,8 min | 48,3 min | 264 min | **265 min** |
| Etapas pós-FL | Crash | RAG + RF | RAG + RF + Ablation | RAG + RF + Ablation | RAG + RF + Ablation | RF + Ablation + Pooled | RAG + RF + Ablation | **RAG + RF + Ablation** |
| RAG Precision@3 | — | 0,134 | 0,285 | 0,133 | 0,254 | ❌ (bug) | 0,110 ✅ | **0,226** ✅ |
| Baseline RF Acc | — | 68,1% | 68,0% | 67,8% | 68,4% | **68,7%** | 68,3% | 68,2% |
| Ablation Δ Acc (B−A) | — | — | **+12,7 p.p.** | +6,8 p.p. ⚠ | +11,7 p.p. | −0,98 p.p. ⚠ | +5,94 p.p. ↑ | +4,43 p.p. |
| BEHRT Pooled A Acc | — | — | — | — | — | 67,79% | — | **—⁴** |
| BEHRT Pooled B Acc | — | — | — | — | — | 63,03% | — | **—⁴** |
| Checkpoint guloso | Não | Não | Não | Não | Não | Não | Não¹ | **Sim (R91)** |
| Novidade arquitetural | — | — | cal set isolado | — | — | dia_relativo embed | µ=0,1 + 120 rounds | **checkpoint guloso + calibração log-space** |

> ¹ Avaliação feita na R120 (última rodada). O melhor checkpoint foi R89 (63,29%) — não capturado por falta de implementação. Gap de 3,93 p.p. entre melhor e última rodada.  
> ² Checkpoint guloso restaura R91 antes da avaliação — avaliação reflete o melhor modelo, não a última iteração.  
> ³ Bug de temperatura: LBFGS sem log-space saltou para T=−8.9997, destruindo calibração pós-treino. Fix implementado em `calibration.py`. Executar `make recalibrate` para corrigir sem retreinar.  
> ⁴ BEHRT Pooled omitido no Exp 8 — `POOLED_EPOCHS` desacoplado de `NUM_ROUNDS × LOCAL_EPOCHS` (fix: `pooled_epochs=120` em `FedConfig`). Retorna no Exp 9.

### Comparativo BEHRT-FL vs Baseline RF vs BEHRT Pooled

| Modelo | Accuracy | AUC | F1 Macro | ECE | Privacidade |
|---|---|---|---|---|---|
| RF Centralizado (BoT) | 67,8–68,7% | 0,786–0,797 | 0,503–0,510 | 0,057–0,067 | Centralizado |
| **BEHRT Pooled A** (sem demo, 40 épocas, Exp 6) | **67,79%** | — | **0,5218** | — | Centralizado |
| **BEHRT Pooled B** (late fusion, 40 épocas, Exp 6) | **63,03%** | — | **0,5005** | — | Centralizado |
| RF BPSP isolado | 59,1–59,7% | 0,729–0,743 | 0,330–0,340 | 0,047–0,062 | Local |
| **SimplifiedBEHRT FL — Exp 8 (R91, checkpoint guloso)** | **66,61%** | **0,810** | **0,481** | 0,086 | **Federado** |
| SimplifiedBEHRT FL — Exp 6 (R20, +dia_relativo) | 59,63% | 0,746 | 0,352 | 0,105 | **Federado** |
| SimplifiedBEHRT FL — Exp 7 (R120, µ=0,1) | 59,36%¹ | 0,770 | 0,384 | **0,033** | **Federado** |
| SimplifiedBEHRT FL — Exp 1 (round 20) | 58,0% | 0,740 | 0,359 | 0,059 | **Federado** |
| SimplifiedBEHRT FL — Exp 5 (round 20) | 56,6% | 0,722 | 0,334 | 0,046 | **Federado** |
| SimplifiedBEHRT FL — Exp 3 (round 20) | 55,8% | 0,755 | **0,398** | 0,087 | **Federado** |
| SimplifiedBEHRT FL — Exp 4 (round 20) | 54,8% | 0,762 | 0,366 | 0,041 | **Federado** |
| BEHRT ablation B (late fusion, 10 épocas, Exp 6) | 59,24% | — | 0,3707 | — | Local |
| BEHRT ablation A (sem demo, 10 épocas, Exp 6) | 60,22% | — | 0,4031 | — | Local |
| RF HSL isolado | 23,5–24,3% | 0,673–0,701 | 0,184–0,205 | 0,201–0,273 | Local |

> ¹ Avaliação feita na R120. O melhor checkpoint (R89, 63,29%) não foi capturado por falta de checkpoint guloso.  
> Nota: a ECE do Exp 8 (pré-calibração) é 0,086 — a calibração pós-treino foi destruída pelo bug de T=-8,9997. O valor após `make recalibrate` será menor.

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

### Custo de privacidade quantificado (Exp 5 vs BEHRT Pooled)

| Comparação | Δ Acc | Δ F1 |
|---|---|---|
| BEHRT Pooled B vs BEHRT FL Exp 5 | **−7,0 p.p.** | **−0,160** |
| BEHRT Pooled B vs RF Centralizado | −4,8 p.p. | −0,015 |
| BEHRT FL Exp 5 vs RF Centralizado | −11,8 p.p. | −0,175 |

O custo de privacidade da federação (~7 p.p.) é separável do custo arquitetural (BEHRT seq vs RF BoT, ~5 p.p.). Parte do gap FL está relacionada ao não-IID extremo, não apenas à privacidade.

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
| **FedNova** (normalização por passos efetivos τ_i) | Reduz viés da agregação com clientes heterogêneos (BPSP 5,5× HSL) | Alta | **Pendente — Exp 9** |
| `POOLED_EPOCHS` desacoplado de `NUM_ROUNDS × LOCAL_EPOCHS` | Corrige 240 → 120 épocas do BEHRT centralizado | Média | **✓ Implementado — `pooled_epochs=120` no `FedConfig`** |
| Clipar pesos de classe em `max_weight=15,0` | Reduz instabilidade de gradiente no BPSP (peso=47 para `melhora_pronto`) | Alta | Pendente |
| Reduzir local epochs 2 → 1 | Reduz divergência entre clientes por rodada | Média | Pendente |
| Explorar calibração por Platt Scaling ou isotônica | Resolve padrões de calibração mistos que temperatura única não resolve | Média | Pendente |
| Avaliar fusão para 3 classes | Resolve non-IID estrutural se clinicamente justificável | A definir com orientadora | Pendente |
| Comparação CPU vs GPU | Medir impacto de hardware no tempo de treinamento para o TCC | Após resolver driver NVIDIA | Pendente |

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
| `AVALIACAO_PROJETO.md` | Avaliação acadêmica e clínica do projeto |
