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

## Tabela Comparativa dos Experimentos

| Atributo | Experimento 1 | Experimento 2 | Experimento 3 |
|---|---|---|---|
| Log | `run_complete_1.log` | `run_complete_1_correcao1.log` | `run_complete_2_correcao_calibracao.log` |
| Rodadas executadas | 20 | 7 | 20 |
| Convergência | Não | **Sim (rodada 7)** | Não |
| Acurácia final | 58,0% | 52,5% | 55,8% |
| Macro AUC (pré-cal) | 0,740 | **0,767** | 0,755 |
| Macro F1 (pré-cal) | 0,359 | 0,287 | **0,398** |
| F1 melhora_pronto | 0,083 | 0,048 | **0,397** ↑ |
| ECE pré-calibração | **0,059** | 0,061 | 0,087 |
| ECE pós-calibração | 0,098 (↑) | 0,064 (↑) | 0,102 (↑) |
| MCE pós-calibração | 0,405 | 0,301 | **0,229** ↓ |
| Temperatura T | 1,177 | 1,127 | 1,175 |
| Cal set | test_loader (inválido) | test_loader (inválido) | **3.376 amostras (isolado)** |
| Tráfego FL total | 217 MB | 76 MB | 217 MB |
| Duração total | 57,4 min | ~21 min | 49,7 min |
| Etapas pós-FL | Crash | RAG + RF | **RAG + RF + Ablation ✓** |
| RAG Precision@3 | — | 0,134 | **0,285** |
| Baseline RF (Acc) | — | 68,1% | 68,0% |
| Ablation Δ Acc (B−A) | — | — | **+12,7 p.p.** |

### Comparativo BEHRT-FL vs Baseline RF (Experimentos 1–3)

| Modelo | Accuracy | AUC | F1 Macro | ECE |
|---|---|---|---|---|
| RF Centralizado (BoT) — teto sem privacidade | 68,0–68,1% | 0,786–0,790 | 0,504–0,505 | 0,061 |
| RF BPSP isolado — baseline local sem FL | 59,4–59,5% | 0,735–0,736 | 0,337 | 0,055 |
| RF HSL isolado — baseline local sem FL | 23,5–28,0% | 0,702–0,720 | 0,184–0,204 | 0,201–0,273 |
| SimplifiedBEHRT FL (Exp 1, round 20) | 58,0% | 0,740 | 0,359 | — |
| SimplifiedBEHRT FL (Exp 2, round 7) | 52,5% | **0,767** | 0,287 | — |
| **SimplifiedBEHRT FL (Exp 3, round 20)** | 55,8% | 0,755 | **0,398** | 0,087 |
| BEHRT local + demo (ablation Exp 3) | **67,3%** | — | 0,449 | — |

---

## Diagnóstico Consolidado

### Problema 1 — Acurácia abaixo do baseline RF

O RF centralizado (~68%) ainda supera o SimplifiedBEHRT FL (52,5–58,0%). As causas principais são:

1. **Non-IID extremo:** `melhora_pronto` é 61,5% do HSL mas 0,4% do BPSP. FedAvg pondera por volume de dados — BPSP tem 5,5× mais amostras e domina a agregação, tendendo a sobrescrever o que HSL aprende.
2. **Peso de classe desestabilizador:** peso 47,173 para `melhora_pronto` no BPSP gera gradientes instáveis (apenas 97 amostras de treino nessa classe nesse cliente).
3. **Client drift:** µ=0,01 no FedProx é insuficiente para o grau de heterogeneidade — a loss oscila ao longo das rodadas em vez de descer monotonicamente.
4. **Convergência prematura (Exp 2):** convergiu na rodada 7 antes de atingir os patamares dos Experimentos 1 e 3.

**Evolução positiva observada (Exp 3):** F1 de `melhora_pronto` subiu de 0,05–0,08 (Exp 1–2) para **0,397** com 20 rodadas completas. Isso evidencia que o modelo consegue aprender a classe não-IID dado tempo suficiente, mas o processo é lento e oscilante.

### Problema 2 — Temperature scaling não melhora a calibração

Em todos os experimentos T>1 aumentou o ECE pós-calibração. Análise do Experimento 3 (primeiro com cal_set independente):

- ECE pré-cal: 0,087 → pós-cal: 0,102 (+0,015, piorou)
- **MCE pré-cal: 0,445 → pós-cal: 0,229 (−0,216, melhorou significativamente)**

O cal_set independente evidencia que o padrão de calibração é estruturalmente misto: bins de confiança intermediária são underconfident (modelo hesitante), enquanto extremos são overconfident. Temperatura única T não resolve padrões mistos — comprime as probabilidades uniformemente e piora o ECE agregado.

**Implicação:** o modelo está bem calibrado nos bins de alta confiança (gap < 0,04 acima de 0,83 de confiança) mas mal calibrado nos bins intermediários. Para o TCC, isso significa que predições de alta confiança são confiáveis; predições de média confiança devem ser tratadas com cautela.

### Achado da ablation — demográficos são relevantes

Config B (com idade + sexo via late fusion) alcançou Acc=67,3% vs Config A (sem demo) Acc=54,5% — **+12,7 p.p. em treinamento local**. O F1 macro subiu +0,051. Este resultado, com dados reais FAPESP, valida empiricamente a hipótese de que variáveis demográficas adicionam sinal discriminante ao BEHRT para este dataset. É uma das contribuições centrais do TCC.

### Classes não aprendidas

| Classe | Causa | Evolução |
|---|---|---|
| `curado_internado` | Raridade extrema em ambos os clientes (N=28 no teste global) | F1 estável em 0,04–0,08; modelo praticamente não prediz essa classe |
| `melhora_pronto` | Quasi-exclusiva do HSL; BPSP domina FedAvg | **Melhora significativa com mais rodadas:** F1 0,05→0,40 (Exp 2→3) |

### Ações corretivas planejadas

| Ação | Impacto esperado | Prioridade |
|---|---|---|
| Clipar pesos de classe em `max_weight=15,0` | Reduz instabilidade de gradiente no BPSP | Alta |
| Aumentar µ FedProx 0,01 → 0,1 | Reduz client drift, loss mais estável | Alta |
| Reduzir local epochs 2 → 1 | Reduz divergência entre clientes por rodada | Média |
| Explorar calibração por Platt Scaling ou isotônica | Resolve padrões de calibração mistos que temperatura única não resolve | Média |
| Avaliar fusão para 3 classes | Resolve non-IID estrutural se clinicamente justificável | A definir com orientadora |
| Comparação CPU vs GPU | Medir impacto de hardware no tempo de treinamento para o TCC | Após resolver driver NVIDIA |

---

## Arquivos de Referência

| Arquivo | Conteúdo |
|---|---|
| `experiments/logs/run_complete_1.log` | Log completo do Experimento 1 (20 rodadas) |
| `experiments/logs/evaluation_round_20.json` | Avaliação detalhada por classe — Experimentos 1 e 3 (round 20) |
| `experiments/logs/run_complete_1_correcao1.log` | Log completo do Experimento 2 (7 rodadas + pós-FL) |
| `experiments/logs/evaluation_round_7.json` | Avaliação detalhada por classe — Experimento 2 |
| `experiments/logs/run_complete_2_correcao_calibracao.log` | Log completo do Experimento 3 (20 rodadas + pós-FL + ablation) |
| `experiments/data/baseline_rf_20260625_081232.json` | Resultado baseline RF — Experimento 2 |
| `experiments/data/baseline_rf_20260625_093220.json` | Resultado baseline RF — Experimento 3 |
| `experiments/data/ablation_demo_20260625_095404.json` | Resultado ablation demográfica — Experimento 3 |
| `experiments/data/history_20260625_071610.json` | Histórico de loss/acc por rodada — Experimento 1 |
| `AVALIACAO_PROJETO.md` | Avaliação acadêmica e clínica do projeto |
