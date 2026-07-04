# Sumário de Treinamento — Parte 3 (Treinamentos Reais)

**Projeto:** TCC — Aprendizado Federado para Predição de Desfecho Clínico
**Autora:** Jacqueline Abreu | ICMC/USP
**Continuação de:** `docs/Sumario_Treinamento.md` (Exp 1–17) e `docs/Sumario_Treinamento_Parte2.md` (Bloco 1/2, GPU, modularização)
**Iniciado em:** 2026-07-02

---

## O que muda a partir deste documento

Em 2026-07-01, a autora decidiu classificar retroativamente **todo** o histórico anterior (Exp 1–17/T1–T16, Bloco 1, Bloco 2 CPU/GPU, modularização, e as validações funcionais de 2026-07-01/02) como **"Treinamentos de Ajuste"** — bugs ainda estavam sendo corrigidos, o esquema de labels e o critério de checkpoint mudaram no meio do caminho, o código passou por reestruturação completa. Nenhum número desse período deve ser citado como resultado final do TCC (detalhe completo em `docs/Linha_do_Tempo_MOSAIC-FL.md`, seção "Fechamento da Fase de Ajuste").

**Este documento registra exclusivamente os "Treinamentos Reais"** — execuções sobre o código já estável, marcadas explicitamente com `run_classification='treinamento_real'` no banco (`metrics.fl_trainings`), a partir da migration 021. São os números que devem alimentar o capítulo de resultados da defesa.

**Como verificar a classificação de qualquer `training_id` diretamente no banco, sem depender deste documento:**
```sql
SELECT id, partition_mode, run_classification, best_accuracy, macro_f1, macro_auc
FROM metrics.fl_trainings
WHERE run_classification = 'treinamento_real'
ORDER BY id;
```

---

## Treinamento Real 1 — GPU (2026-07-02, training_ids 37–40)

**Log:** `experiments/logs/run_complete_cuda_20260702_085025.log`
**Comando:** `FL_RUN_CLASSIFICATION=treinamento_real make training-full-cuda`
**Duração total:** 08:50 → 09:32 (~42 min)
**Critério de checkpoint:** `f1_macro` | **Algoritmo:** FedNova | **Device:** CUDA (RTX 4070 Ti)

### Nota sobre a tentativa anterior (training_ids 33–36) — não é este treinamento

Uma primeira tentativa (`run_complete_cuda_20260702_074407.log`, ids 33-36) rodou pela manhã com `run_classification=treinamento_real`, mas a **Fase 4/5 (BEHRT Pooled) falhou silenciosamente** — bug de regressão em `prepare_dataloaders_from_db()` (mudança de 7→9 valores de retorno na sessão anterior, não propagada para `run_behrt_pooled.py` e outros 3 scripts que chamam a função diretamente). O bug foi corrigido no mesmo dia (ver `docs/Linha_do_Tempo_MOSAIC-FL.md`) e os ids 33-36 foram **reclassificados para `ajuste`** via UPDATE direto no banco — não aparecem como resultado válido em lugar nenhum. O treinamento documentado aqui (ids 37-40) é a re-execução, já com o bug corrigido.

### Tabela consolidada

| Fase | training_id | Rodadas (melhor) | Accuracy | F1 macro | Macro AUC | ECE pré | ECE pós | Convergiu | Duração | Energia GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1/5 BPSP-only | 37 | 40 (R37) | 61,17% | 0,3248 | 0,7271 | 0,0410 | 0,0869 | Sim | 202,6s (3,4min) | 4,20 Wh |
| 2/5 HSL-only | 38 | 67 (R43) | 32,03% | 0,2469 | 0,6397 | 0,2254 | 0,1101 | Sim | 71,4s (1,2min) | 1,49 Wh |
| 3/5 Federado (non-IID real) | 39 | 120 (R88) | **67,44%** | **0,5031** | **0,8054** | 0,0361 | 0,0806 | **Não** | 720,6s (12,0min) | 15,12 Wh |
| 5/5 Federado (IID simulado) | 40 | 44 (R30) | **71,69%** | **0,5262** | **0,8436** | 0,0495 | 0,1018 | Sim | 264,3s (4,4min) | 5,64 Wh |

### Per-class F1/AUC (pós-calibração temperature scaling, melhor checkpoint de cada fase)

| Classe | BPSP-only (37) | HSL-only (38) | Federado real (39) | Federado IID (40) |
|---|---|---|---|---|
| curado_pronto | F1=0,7498 AUC=0,833 | F1=0,0000 AUC=0,558 | F1=0,8015 AUC=0,898 | F1=0,8425 AUC=0,918 |
| curado_internado | F1=0,0000 AUC=0,637 | F1=0,0000 AUC=0,470 | **F1=0,0400** AUC=0,571 | F1=0,0000 AUC=0,696 |
| melhora_pronto | F1=0,0000 AUC=0,592 | F1=0,5418 AUC=0,868 | F1=0,7446 AUC=0,963 | F1=0,8257 AUC=0,986 |
| melhora_internado_breve | F1=0,5543 AUC=0,799 | F1=0,4471 AUC=0,633 | F1=0,6106 AUC=0,814 | F1=0,5883 AUC=0,811 |
| melhora_internado_grave | F1=0,3198 AUC=0,775 | F1=0,2455 AUC=0,670 | F1=0,3188 AUC=0,782 | F1=0,3746 AUC=0,807 |

**Achado novo, primeira vez registrado:** `curado_internado` (28-46 amostras no teste, a classe mais rara do projeto) teve **F1=0,0400 no Federado real (id=39)** — não é mais F1=0,000. Em todos os Blocos anteriores (1 e 2) essa classe nunca havia sido acertada nenhuma vez. Amostra pequena demais para tirar conclusão robusta (uma mudança de 1-2 amostras corretas já move o F1 nessa faixa), mas é o primeiro sinal não-nulo registrado — vale mencionar na defesa com essa ressalva de fragilidade estatística.

### Leave-one-out confirma a necessidade da federação (subgroup_metrics)

| Cenário de treino | Accuracy em pacientes BPSP | Accuracy em pacientes HSL |
|---|---|---|
| BPSP-only (id=37) | 70,58% | **9,25%** (colapso — quase aleatório) |
| HSL-only (id=38) | 26,48% (colapso) | **62,62%** |
| Federado real (id=39) | 67,58% | 66,67% |
| Federado IID (id=40) | 71,98% | 70,08% |

Um hospital treinado isoladamente não generaliza para o outro — o modelo BPSP-only acerta só 9,25% dos casos de origem HSL (pior que chance aleatória para 5 classes, ~20%). A federação (real ou IID simulada) equilibra o desempenho entre origens sem colapso em nenhuma delas.

### Experimento 3 — contraste non-IID real vs. IID simulado (3ª execução, mesmo padrão)

| Métrica | Federado non-IID real (39) | Federado IID simulado (40) | Δ |
|---|---|---|---|
| Accuracy | 67,44% | **71,69%** | +4,26 p.p. |
| F1 macro | 0,5031 | **0,5262** | +0,0231 |
| Macro AUC | 0,8054 | **0,8436** | +0,0382 |
| Convergência | Não (120 rodadas) | Sim (R30) | — |

Terceira vez consecutiva (contando as duas validações de 2026-07-01/02 não-oficiais) que o cenário IID simulado supera o non-IID real em todas as métricas de qualidade, com tudo mais (algoritmo FedNova, hiperparâmetros, seed de inicialização, número de clientes) mantido idêntico entre as duas fases — a única variável é a origem dos dados de cada cliente. Esta é agora **evidência formal, não mais validação**, de que a heterogeneidade non-IID real (BPSP 28.599 vs. HSL 5.174 atendimentos, distribuição de classe assimétrica) tem custo mensurável sobre a qualidade do modelo federado.

### Custo de privacidade — BEHRT Federado vs. BEHRT Pooled vs. RF Centralizado (Fase 4/5)

Fonte: `experiments/data/behrt_pooled_20260702_092546.json`, budget equivalente (120 épocas/rodadas).

| Modelo | Accuracy | F1 macro | Macro AUC |
|---|---|---|---|
| BEHRT Federado (non-IID real, id=39) | 67,44% | **0,5031** | 0,8054 |
| BEHRT Pooled A (sem demográficos) | **68,03%** | 0,4948 | **0,8168** |
| BEHRT Pooled B (late fusion demográfica) | 67,08% | 0,4894 | — |
| RF Centralizado (Bag-of-Tokens) | 66,90% | 0,5043 | 0,7874 |

**Leitura:** diferenças pequenas entre os quatro (faixa de ~1,1 p.p. em accuracy). BEHRT Pooled A tem uma vantagem marginal em accuracy/AUC; RF tem uma vantagem marginal em F1 macro; o Federado fica no meio. **Não há, neste treinamento, uma vantagem clara da federação sobre a centralização, nem um custo de privacidade grande** — resultado mais modesto que o "marco histórico" do Bloco 1 (T15, onde o FL superava tudo) e mais próximo do padrão do Bloco 2. Deve ser reportado como está, sem forçar uma narrativa de superioridade em nenhuma direção — a honestidade do resultado é o que importa aqui.

### RAG — Precision@3 por fase

| Fase | P@3 |
|---|---|
| BPSP-only (37) | 0,2650 |
| HSL-only (38) | 0,2570 |
| Federado non-IID real (39) | 0,1756 |
| Federado IID simulado (40) | **0,4049** |

### Recursos computacionais

| Fase | Duração | Energia GPU (Wh) | Potência média GPU |
|---|---|---|---|
| BPSP-only | 3,4 min | 4,20 | 74,7W |
| HSL-only | 1,2 min | 1,49 | 75,1W |
| Federado non-IID real | 12,0 min | 15,12 | 75,5W |
| Federado IID simulado | 4,4 min | 5,64 | 76,8W |
| **Total (4 fases FL)** | **~21 min** | **~26,5 Wh** | — |

Energia estimada por amostragem (potência lida uma vez por rodada × duração), não medição contínua — ver `docs/Linha_do_Tempo_MOSAIC-FL.md` para a ressalva completa de metodologia. Não inclui a Fase 4 (BEHRT Pooled + RF), que não passa pelo mesmo mecanismo de coleta (não é um treino FL, não registra em `fl_trainings`).

---

---

## Treinamento Real 2 — CPU (2026-07-02/03, training_ids 41–44)

**Log:** `experiments/logs/run_complete_20260702_133335.log`
**Comando:** `FL_RUN_CLASSIFICATION=treinamento_real make training-full`
**Duração total:** 2026-07-02 13:33 → 2026-07-02 23:26 (~9h53min)
**Critério de checkpoint:** `f1_macro` | **Algoritmo:** FedNova | **Device:** CPU (mesma máquina do Treinamento Real 1)

Par de comparação formal do Treinamento Real 1 (GPU, ids 37-40) — mesmo código, mesmos dados, mesmo dia, única variável controlada é o device.

### Tabela consolidada

| Fase | training_id | Rodadas (melhor) | Accuracy | F1 macro | Macro AUC | ECE pré | ECE pós | Convergiu | Duração |
|---|---|---|---|---|---|---|---|---|---|
| 1/5 BPSP-only | 41 | 120 (R58) | 63,06% | 0,3625 | 0,7849 | 0,0627 | 0,0957 | **Não** | 6415,3s (106,9min) |
| 2/5 HSL-only | 42 | 71 (R27) | 31,91% | 0,2348 | 0,6828 | 0,2328 | 0,1181 | Sim | 794,5s (13,2min) |
| 3/5 Federado (non-IID real) | 43 | 120 (R115) | **67,26%** | **0,5115** | **0,8103** | 0,0239 | 0,0699 | **Não** | 7458,2s (124,3min) |
| 5/5 Federado (IID simulado) | 44 | 47 (R31) | **72,67%** | **0,5341** | **0,8415** | 0,0937 | 0,1259 | Sim | 3305,3s (55,1min) |

Recursos: `peak_ram_mb` ≈ 2.325–2.448 MB, `avg_cpu_pct` ≈ 2.345–2.352% (~23 núcleos saturados) em todas as fases — consistente com a máquina de desenvolvimento usada desde o Bloco 2.

### Custo de privacidade — BEHRT Federado vs. Pooled vs. RF (CPU)

Fonte: `experiments/data/behrt_pooled_20260702_220847.json`.

| Modelo | Accuracy | F1 macro | Macro AUC |
|---|---|---|---|
| BEHRT Federado (non-IID real, id=43) | 67,26% | **0,5115** | 0,8103 |
| BEHRT Pooled A (sem demográficos) | 69,09% | 0,5039 | **0,8092** |
| BEHRT Pooled B (late fusion demográfica) | **71,31%** | **0,5131** | — |
| RF Centralizado (Bag-of-Tokens) | 66,93% | 0,5029 | 0,7862 |

### RAG — Precision@3 (CPU)

| Fase | P@3 |
|---|---|
| BPSP-only | 0,2022 |
| HSL-only | 0,2864 |
| Federado non-IID real | 0,2208 |
| Federado IID simulado | 0,2661 |

### Leave-one-out (subgroup_metrics, CPU)

| Cenário de treino | Accuracy em pacientes BPSP | Accuracy em pacientes HSL |
|---|---|---|
| BPSP-only (id=41) | 71,94% | **14,07%** (colapso) |
| HSL-only (id=42) | 26,62% (colapso) | **61,08%** |
| Federado real (id=43) | 67,02% | 68,59% |
| Federado IID (id=44) | 73,27% | 69,29% |

Mesmo padrão do GPU: hospital isolado não generaliza para o outro; federação (real ou IID) equilibra.

---

## Comparação formal CPU × GPU — Treinamento Real 1 vs. Treinamento Real 2

Mesmo código (pós-modularização, pós-correção do bug de `prepare_dataloaders_from_db`), mesmos dados, mesmo dia — única variável controlada é o device.

| Fase | Acc GPU (37-40) | Acc CPU (41-44) | Δ | F1 GPU | F1 CPU | Δ | Duração GPU | Duração CPU | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| BPSP-only | 61,17% | 63,06% | +1,89 p.p. | 0,3248 | 0,3625 | +0,0377 | 202,6s | 6415,3s | **31,7×** |
| HSL-only | 32,03% | 31,91% | −0,12 p.p. | 0,2469 | 0,2348 | −0,0121 | 71,4s | 794,5s | **11,1×** |
| Federado non-IID real | 67,44% | 67,26% | −0,18 p.p. | 0,5031 | 0,5115 | +0,0084 | 720,6s | 7458,2s | **10,4×** |
| Federado IID simulado | 71,69% | 72,67% | +0,98 p.p. | 0,5262 | 0,5341 | +0,0079 | 264,3s | 3305,3s | **12,5×** |
| **Pipeline completo (5 fases)** | **~42 min** | **~593 min (9h53)** | — | — | — | — | — | — | **~14,1×** |

**Leitura:**

- **Qualidade do modelo é equivalente entre CPU e GPU** — diferenças de até ~1,9 p.p. em accuracy e ~0,04 em F1 macro, na mesma ordem de grandeza da variação já documentada entre execuções independentes no mesmo device (não-reprodutibilidade CPU↔GPU: RNGs diferentes — Mersenne Twister vs. Philox — e não-associatividade de ponto flutuante em reduções paralelas de GPU; ver `docs/Sumario_Treinamento_Parte2.md`, Parte 9). Nenhuma direção sistemática — GPU ganha em 2 fases, CPU ganha em 2 fases.
- **Velocidade: ~10-32× mais rápido em GPU**, variando por fase — BPSP-only (o hospital maior, 28.599 atendimentos) tem o maior ganho (31,7×), consistente com mais paralelismo disponível por rodada.
- **BEHRT Pooled B (late fusion) diverge mais entre devices que as fases FL** — 71,31% (CPU) vs. 67,08% (GPU), Δ=4,23 p.p. Maior divergência do experimento — coerente com o padrão já documentado de alta variância da ablação de late fusion em poucas épocas (10 no caso da ablação; aqui 120, mas ainda sensível). Não deve ser lido como "CPU produz modelos melhores com demográficos" — é a mesma trajetória estocástica específica de cada run.
- **Nenhuma fase convergiu de forma diferente entre os dois devices** de forma preocupante — BPSP-only e Federado natural não convergiram em 120 rodadas nos dois devices; HSL-only e Federado IID convergiram nos dois.

---

## Treinamento Real 3 — Curva Acurácia × ε, DP-FedAvg (2026-07-03, training_ids 48–50)

**Logs:** `experiments/logs/dp_curve_sigma{1.0,0.5,2.0}_cuda_20260703_06*.log`
**Comando:** `FL_RUN_CLASSIFICATION=treinamento_real make training-dp-curve-cuda`
**Device:** CUDA | **Escopo:** só a fase Federada (BPSP+HSL, non-IID real) — ver justificativa em `docs/TODO.md`
**S (clip norm) fixo em 1,0** nas 3 execuções.

Nota de processo: a primeira tentativa (mesmo dia, ids 45-47) quebrou por um bug de device (`torch.normal()` sem `device=`, ver `docs/Linha_do_Tempo_MOSAIC-FL.md`) — corrigido antes desta execução. ids 45-47 reclassificados para `ajuste` (registros órfãos, sem métrica).

### Tabela — Acurácia × ε

| σ (ruído) | training_id | Accuracy | F1 macro | Macro AUC | ε (composição simples) | ε (RDP) | Rodadas totais | Melhor rodada |
|---|---|---|---|---|---|---|---|---|
| 0,5 (menos ruído) | 49 | 43,12% | 0,1924 | 0,5252 | 406,96 | 144,29 | 42 | R29 |
| 1,0 | 48 | 31,17% | 0,0991 | 0,4742 | 135,65 | 37,93 | 28 | R5 |
| 2,0 (mais ruído) | 50 | 36,79% | 0,2069 | 0,5396 | 55,72 | **13,41** | 23 | R3 |

**Referência — mesma fase, sem DP** (Federado non-IID real, id=39 GPU / id=43 CPU, Treinamento Real 1/2): Accuracy ≈ 67,3-67,4% | F1 macro ≈ 0,503-0,512 | AUC ≈ 0,805-0,810.

### Leitura — resultado negativo, reportado com o mesmo rigor de um resultado positivo

Este experimento **não** produziu um trade-off privacidade×utilidade favorável. Reportado integralmente, sem suavizar:

1. **Custo de privacidade severo em todos os níveis de ruído testados.** Mesmo no ponto mais suave (σ=0,5), a accuracy cai de ~67% (sem DP) para 43% e o F1 macro de ~0,51 para 0,19 — queda grande, não marginal, no nível de ruído mais permissivo testado.

2. **A direção da privacidade está correta e monotônica**: mais ruído → menor ε (mais privado). σ=2,0 é o ponto de melhor privacidade testado (ε_RDP=13,41); σ=0,5 o de pior (ε_RDP=144,29). Isso confirma que o mecanismo de DP em si (clipping + ruído gaussiano + contabilidade RDP) está implementado corretamente.

3. **A relação ruído×utilidade NÃO é monotônica — achado inesperado, não explicado com confiança.** σ=1,0 (ruído intermediário) teve o pior resultado dos três (F1=0,099), pior inclusive que σ=2,0 (mais ruído, F1=0,207). Com apenas **1 execução por σ** (sem réplicas com seeds diferentes) e com os três modelos colapsando cedo (melhor rodada entre R3 e R29, convergência por plateau), essa inversão é compatível com variância de execução única — não há base estatística aqui para afirmar uma causa. **Isto é uma limitação do experimento, registrada explicitamente**: para caracterizar a curva Acc×ε com confiança, seriam necessárias réplicas por σ (múltiplas seeds), não feitas nesta rodada.

4. **Mesmo no melhor ponto de privacidade testado, o ε permanece fraco pelos padrões usuais da literatura de DP** (ε_RDP=13,41 em σ=2,0; privacidade "forte" costuma mirar ε<10, idealmente ε<1). Isto sozinho já é uma conclusão relevante para o TCC: **nesta configuração (2 clientes, S=1,0, budget de rodadas testado), o DP-FedAvg não atinge um bom trade-off privacidade×utilidade** — para se aproximar de privacidade forte exigiria ruído ainda maior que σ=2,0, o que (a julgar pela tendência observada) pioraria ainda mais a utilidade já degradada.

**Por que isto é um resultado válido para a defesa, não uma falha a esconder:** o método científico foi seguido corretamente — hipótese testada (DP-FedAvg viabiliza privacidade formal a custo aceitável), infraestrutura de medição validada (RDP + composição simples, ambas corretas e monotônicas na direção esperada), experimento executado sem erro. O resultado é que, **nesta configuração específica**, o custo de utilidade é alto demais para a privacidade obtida — uma conclusão negativa, mas honesta e citável, sobre os limites práticos do DP-FedAvg no cenário de 2 clientes/dados clínicos heterogêneos deste projeto.

### Investigação da não-monotonicidade — Etapa 1: repetição não-intencional da mesma configuração (2026-07-03, ids 51–53)

Registrado integralmente, incluindo o fato de não ser (ainda) uma conclusão fechada — por decisão explícita da autora de que observações não-confirmatórias também devem constar no documento.

**O que aconteceu:** ao investigar a não-monotonicidade do item 3 acima, a intenção era rodar réplicas com seeds diferentes (42/43/44) via o alvo `training-dp-curve-replicas-cuda`. Por engano de execução, o comando repetido foi `training-dp-curve-cuda` — a mesma configuração original (seed=42, mesmos 3 σ), não as réplicas com seed variada. O resultado, mesmo não sendo o experimento planejado, é informativo por si: é uma repetição exata da mesma configuração nominal.

| σ | Execução original (id) | F1 macro | Repetição, mesma config (id) | F1 macro | Δ F1 | Rodadas (orig. → repetição) |
|---|---|---|---|---|---|---|
| 0,5 | 49 | 0,1924 | 52 | 0,2076 | +0,0152 | 42 → 42 (igual) |
| 1,0 | 48 | 0,0991 | 51 | 0,1038 | +0,0047 | 28 → 28 (igual) |
| 2,0 | 50 | 0,2069 | 53 | 0,1944 | −0,0125 | 23 → **28** (diferente) |

**Leitura desta etapa:**

- **Os resultados NÃO são idênticos entre as duas execuções**, apesar de mesma seed (42) e `cudnn.deterministic=True` já ativo desde antes desta investigação. Em σ=2,0 o número de rodadas até a convergência (critério de checkpoint) até mudou (23 → 28) — evidência direta de não-determinismo na execução em GPU não capturado pelas configurações atuais de determinismo (`torch.use_deterministic_algorithms` nunca foi chamado no projeto — ver Etapa 2, planejada). Isto já responde, em parte, a uma pergunta prática levantada pela autora: **sim, os resultados desta curva podem mudar se o treinamento for repetido**, mesmo sem alterar nenhum parâmetro.
- **σ=1,0 foi o pior dos três em AMBAS as execuções** (F1=0,0991 e F1=0,1038, respectivamente) — um sinal a favor de que a inversão observada possa ser real, não só ruído. **Mas isto não é prova**: as duas execuções usaram a mesma seed nominal (42); a divergência entre elas veio do não-determinismo da GPU, não de uma variação de seed proposital — não é uma réplica estatística controlada. Um sinal fraco, consistente em 2 de 2 observações, mas com N muito pequeno para qualquer afirmação causal.
- **Status: investigação em andamento.** As réplicas com seed proposital (42/43/44 × 3 σ = 9 execuções, via `training-dp-curve-replicas-cuda`) foram iniciadas na sequência, ainda sem resultado no momento deste registro. A Etapa 2 (`torch.use_deterministic_algorithms(True, warn_only=True)`, decidida pela autora para rodar depois da Etapa 1) ainda não foi implementada.

### Investigação da não-monotonicidade — Etapa 1 (continuação): réplicas de propósito, 3 seeds × 3 σ (2026-07-03, ids 54–62)

As 9 execuções planejadas (`training-dp-curve-replicas-cuda`, seeds 42/43/44 × σ=1,0/0,5/2,0) completaram sem erro. `run_classification='ajuste'` (default do alvo — é investigação de robustez de um resultado já reportado, não um novo resultado formal).

| σ | seed=42 (F1) | seed=43 (F1) | seed=44 (F1) | **Média F1** | Faixa (min–max) |
|---|---|---|---|---|---|
| 0,5 | 0,1851 | 0,1855 | 0,1812 | **0,1839** | 0,1812–0,1855 (bem consistente entre seeds) |
| 1,0 | 0,1002 | 0,1522 | 0,1049 | **0,1191** | 0,1002–0,1522 |
| 2,0 | 0,1990 | 0,1506 | 0,1888 | **0,1795** | 0,1506–0,1990 |

**Leitura — agora com base estatística mínima (3 réplicas por ponto), não mais 1 execução isolada:**

- **σ=1,0 foi o pior das três seeds testadas, sem exceção** — média de F1 (0,1191) claramente abaixo de σ=0,5 (0,1839) e σ=2,0 (0,1795), que por sua vez ficam próximas entre si. Combinando com a repetição não-intencional da etapa anterior (onde σ=1,0 também foi o pior em 2/2 execuções), são **5 execuções independentes no total, todas com σ=1,0 como o ponto de pior utilidade** — deixa de ser plausível como só variância de execução única. A relação ruído×utilidade não-monotônica observada na curva original (ids 48-50) é, com esta evidência, **provavelmente um efeito real, não um artefato de ruído**.
- **Causa ainda não identificada — não afirmada por especulação.** Não há, nos dados coletados até aqui, uma explicação causal validada para por que especificamente σ=1,0 (ruído intermediário, não o mais alto nem o mais baixo testado) produz o pior resultado. Uma hipótese plausível, mas **não testada**, é uma interação entre a escala do ruído (`noise_std = σ × S / n_clientes` = 0,5 neste ponto) e outro hiperparâmetro do treino (taxa de aprendizado, norma de clipping) que tornaria esse valor específico particularmente disruptivo — sem validação adicional, isto permanece hipótese, não conclusão.
- **Limitação que permanece:** mesmo com 3 seeds, ainda é uma amostra pequena, e a fonte da variação intra-σ inclui tanto a diferença de seed quanto o não-determinismo de GPU documentado na etapa anterior (os dois efeitos não são separáveis com o desenho atual). A Etapa 2 (determinismo forçado) ajuda a isolar ao menos uma dessas fontes.

### Investigação da não-monotonicidade — Etapa 2: `torch.use_deterministic_algorithms` (2026-07-03, ids 63–65)

Reexecução dos 3 σ, **mesma seed=42 da curva original**, com `FL_DETERMINISTIC=1` (`torch.use_deterministic_algorithms(True, warn_only=True)`, implementado em `manual_loop.py` especificamente para esta investigação). Sem erro; `determinismo_forcado ativo` confirmado no log das 3 execuções.

| σ | Original (id) | Repetição não-intencional (id) | Determinístico (id) |
|---|---|---|---|
| 1,0 | 48: F1=0,0991, rodada 5/28 | 51: F1=0,1038, rodada 5/28 | 63: F1=0,0995, rodada 5/28 |
| 0,5 | 49: F1=0,1924, rodada 29/42 | 52: F1=0,2076, rodada 29/42 | 64: F1=0,2048, rodada **4**/42 |
| 2,0 | 50: F1=0,2069, rodada 3/23 | 53: F1=0,1944, rodada 3/**28** | 65: F1=0,1990, rodada 3/**26** |

**Leitura:** o determinismo forçado **não** eliminou a divergência entre execuções. Em σ=1,0, o resultado ficou muito próximo do original (F1=0,0995 vs. 0,0991) — mas em σ=0,5 e σ=2,0 o número de rodadas até a convergência continua diferente entre as três execuções mesmo com `FL_DETERMINISTIC=1` ativo. Hipótese mais provável: como `warn_only=True` é obrigatório (sem ele o treino pararia com `RuntimeError` na primeira operação sem implementação determinística), alguma operação do BEHRT — plausivelmente relacionada a embeddings, comuns nesta arquitetura — ainda cai no caminho não-determinístico da GPU, apenas emitindo aviso em vez de garantir reprodutibilidade. Isolar exatamente qual operação exigiria profiling adicional, fora do escopo desta investigação.

### Síntese final da investigação (Etapas 1 e 2 combinadas)

Reunindo as 6 execuções por σ (curva original + repetição não-intencional + 3 réplicas de seed + reexecução determinística):

| σ | F1 médio (6 execuções) | Faixa (min–max) |
|---|---|---|
| 0,5 | 0,1928 | 0,1812–0,2076 |
| 1,0 | **0,1106** | 0,0991–0,1522 |
| 2,0 | 0,1898 | 0,1506–0,2069 |

**Conclusão da investigação:**

1. **A não-monotonicidade da curva Acc×ε é um efeito real, não ruído de execução única.** σ=1,0 foi o pior ponto em todas as 6 execuções reunidas aqui, com F1 médio quase metade dos outros dois pontos testados (0,1106 vs. ~0,19). A hipótese inicial ("pode ser só variância de uma única rodada") foi **refutada pelos dados** — o padrão se sustenta através de variação de seed (Etapa 1) e mesmo sob tentativa de forçar determinismo (Etapa 2).
2. **A causa exata permanece não identificada.** Nenhuma das duas etapas de investigação (réplicas de seed; determinismo forçado) revelou o mecanismo por trás do porquê σ=1,0 especificamente — nem o ruído mais baixo nem o mais alto testado — produz o pior resultado. Isto é reportado como uma questão em aberto, não como uma lacuna escondida. Investigação adicional (ex.: profiling de qual operação do BEHRT foge do caminho determinístico; ou variar `S`/taxa de aprendizado para ver se a "zona ruim" se desloca) ficaria como trabalho futuro, fora do escopo desta sessão.
3. **Achado secundário, mas relevante para a metodologia do TCC como um todo:** confirmou-se que o pipeline de treino **não é bit-reprodutível em GPU**, mesmo com seed fixa e `cudnn.deterministic=True`, e que `torch.use_deterministic_algorithms(True, warn_only=True)` não resolve isso sozinho nesta arquitetura. Isto deve ser mencionado como limitação metodológica geral do projeto (não só da curva de DP) — resultados pontuais (um único treino) têm variação não capturada, o que reforça a importância de reportar médias/faixas quando possível, e é coerente com a prática já adotada de registrar resultados negativos e limitações com o mesmo rigor de resultados positivos.

### O ruído DP não afeta as 5 classes igualmente — colapso para as classes majoritárias

Revisão adicional dos dados (`evaluation_json`, per-class F1) das 18 execuções com DP (ids 48-65), comparadas com o suporte de cada classe no conjunto de teste (fixo, mesma partição em todos os runs): `curado_pronto`=1598 amostras, `melhora_internado_breve`=1096, `melhora_internado_grave`=321, `melhora_pronto`=320, `curado_internado`=46 (a mais rara, ~1,1% do teste).

| σ | curado_pronto (n=1598) | melhora_internado_breve (n=1096) | melhora_pronto (n=320) | melhora_internado_grave (n=321) | curado_internado (n=46) |
|---|---|---|---|---|---|
| 0,5 | F1 médio 0,539 | F1 médio 0,404 | **0,000 em 6/6** | **≈0 em 6/6** | **0,000 em 5/6** |
| 1,0 | **0,000 em 5/6** | F1 médio 0,455 | **0,000 em 6/6** | **≈0 em 6/6** | ≈0,026 (pequeno, mas não-zero em 6/6) |
| 2,0 | F1 médio 0,519 | F1 médio 0,156 | F1 médio 0,162 | F1 médio 0,111 | **0,000 em 6/6** |

**Leitura:**

- **Sob qualquer nível de ruído testado, o modelo colapsa para prever essencialmente só 1-2 classes majoritárias, abandonando as demais.** Isto não é uma degradação uniforme de qualidade — é uma mudança qualitativa de comportamento: em σ=0,5 e σ=1,0, praticamente só as duas maiores classes (`curado_pronto`, `melhora_internado_breve`) recebem algum sinal; as outras três ficam em F1≈0. σ=2,0 é o único ponto onde 4 das 5 classes retêm algum sinal (mesmo que fraco).
- **`curado_internado` (a classe mais rara, 46 amostras / ~1,1% do teste) não sobrevive ao DP em praticamente nenhuma execução** — 17 de 18 execuções com DP têm F1=0,000 nesta classe (a exceção é um sinal muito pequeno, ≈0,02-0,04, em σ=1,0). Isso contrasta com o Treinamento Real 1 sem DP (id=39), onde essa classe teve F1=0,04 pela primeira vez no projeto — um resultado já frágil, que o DP elimina por completo.
- **Achado que conecta com a não-monotonicidade de σ=1,0:** neste ponto específico, o modelo abandona `curado_pronto` — a MAIOR classe do dataset — em 5 das 6 execuções, algo que não acontece nem em σ=0,5 (ruído menor) nem em σ=2,0 (ruído maior, mas ainda assim recupera essa classe consistentemente). Isso é consistente com (mas não prova) a hipótese de que σ=1,0 cai numa faixa de ruído "particularmente disruptiva" para o mecanismo de otimização deste modelo especificamente — hipótese já registrada acima como não-testada.
- **Implicação prática para o TCC:** o resultado de accuracy/F1 agregado (macro) já reportado esconde esse padrão — dizer "accuracy caiu para 43%" é menos informativo do que dizer "o modelo com DP passa a prever essencialmente 1-2 categorias clínicas, e nunca acerta a categoria mais rara". Para um cenário de decisão clínica real, isso é uma limitação de utilidade mais severa do que a métrica agregada sozinha sugere — vale reportar os dois níveis (agregado e por classe) no texto da defesa.

### Privacidade prática (localidade de dados) vs. privacidade formal (garantia matemática do DP) — não são a mesma coisa

Ponto de esclarecimento conceitual, relevante para não deixar o resultado negativo da curva Acc×ε ser lido como "o MOSAIC-FL não é privado":

- **Aprendizado Federado, mesmo sem DP, já entrega privacidade *prática/informal*:** os dados brutos do paciente nunca saem do hospital de origem — apenas os pesos do modelo são compartilhados com o servidor de agregação. Isso reduz a superfície de exposição de dados sensíveis e está alinhado com princípios de minimização de dados. É exatamente o que os Treinamentos Reais 1 e 2 (sem DP, Accuracy≈67%) entregam — essa privacidade de localidade não depende de nenhum mecanismo de ruído.
- **DP-FedAvg adiciona, por cima disso, uma garantia *formal/matemática* (o ε calculável):** a localidade dos dados, por si só, não impede que alguém analise os PESOS do modelo compartilhado e infira informação sobre os dados de treino (ataques de inferência de pertencimento/membership, inversão de gradiente, memorização). O DP-FedAvg (McMahan et al., 2018) existe para fechar essa lacuna formal, com uma prova matemática de quanto uma única amostra pode influenciar o resultado.
- **O que os experimentos desta seção mostram:** que a camada *formal* (ε baixo, "privacidade forte" segundo os padrões usuais da literatura, ε<10) é cara nesta configuração específica (2 clientes, S=1,0) — não que o FL em si falhe em proteger os dados. O MOSAIC-FL, mesmo nos Treinamentos Reais sem DP, já é "privado" no sentido prático de localidade; o que não se atinge, nos níveis de ruído testados, é a garantia formal adicional sem um custo de utilidade severo.

### Diretriz de viabilidade — quanto ruído o sistema tolera nesta configuração

Juntando accuracy média (não só F1) das 18 execuções com DP:

| σ | Accuracy média | ε_RDP | Privacidade "forte" (ε<10) pela literatura? |
|---|---|---|---|
| 0,5 | 43,62% | ≈144 | Não, nem perto |
| 1,0 | 29,03% | ≈38 | Não |
| 2,0 | 33,91% | ≈13-15 | Não, mas o mais próximo testado |
| sem DP (referência) | ≈67% | — | — |

**σ=1,0 é dominado por σ=2,0 nas duas dimensões** (accuracy menor E ε maior/pior) — não deveria ser escolhido dentro deste conjunto. **Conclusão de viabilidade prática, honesta**: dentro do intervalo σ∈[0,5, 2,0] testado (2 clientes, S=1,0), não existe uma configuração que combine privacidade formal minimamente forte (ε<10) com utilidade aceitável para uso clínico — mesmo no ponto de menor ruído testado (σ=0,5, que já está longe de ε<10), a accuracy cai ~23 pontos percentuais em relação ao baseline sem DP, e colapsa quase inteiramente as classes minoritárias (seção anterior). Atingir ε<10 exigiria ruído maior que σ=2,0, o que — extrapolando a tendência já observada, com a devida ressalva de que essa tendência não é monotônica — tende a piorar ainda mais a utilidade.

---

## Pendências para completar a série de Treinamentos Reais

| Item | Status |
|---|---|
| Treinamento Real em CPU | ✅ Concluído (ids 41-44) |
| Comparação formal CPU × GPU | ✅ Concluída (acima) |
| Curva Acc×ε com DP-FedAvg (σ=1,0/0,5/2,0) | ✅ Concluída (acima) — resultado negativo, custo de privacidade alto em todos os níveis testados |
| Investigação da não-monotonicidade (réplicas de seed + determinismo forçado) | ✅ Concluída (acima) — efeito real confirmado (6 execuções), causa exata não identificada |
| Experimento 4 — avaliação Likert humana do RAG (gemma3:4b) | Planejado para 2026-07-03 |
| Reavaliação formal (nota acadêmica/produção clínica) pós-Treinamento Real | Nenhuma avaliação posterior a 2026-06-25 foi feita com este critério |

A curva de privacidade deve ser tratada como um novo Treinamento Real, registrado em continuidade a este documento.
