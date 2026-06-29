# Avaliação Crítica: Metodologia do Projeto MOSAIC-FL

**Fonte avaliada**: `Metodologia MOSAIC-FL - Final.pdf` (15 páginas, gerado por DeepSeek)  
**Avaliado por**: Claude Sonnet 4.6 com acesso integral ao repositório  
**Data**: 2026-06-29  

---

## Metodologia desta Avaliação

Cada afirmação do documento foi cruzada com: código-fonte (`src/mosaicfl/`), logs de experimentos (`experiments/logs/`), JSONs de avaliação (`experiments/data/`), e o arquivo `docs/dados_necessarios_texto_defesa.md` preenchido na sessão anterior. Nenhuma afirmação foi aceita apenas pela plausibilidade — apenas pelo código ou por resultado empírico rastreável.

---

## 1. O que está Correto ✓

### 1.1 Dados e Pré-processamento (Seção 2)
- As 4 tabelas usadas na query SQL estão corretas (`clinical.attendances`, `clinical.patients`, `metrics.clinical_outcomes`, `metrics.exam_records`) — verificado em `preprocessor.py`.
- Filtro `hospital_id IN ('HSL', 'BPSP')` e exclusão de HEI, HFL, HCSP — correto e verificado.
- Critérios de exclusão: `outcome_class NOT IN (2, 3, 4)` e `(co.outcome_at - a.attended_at) >= 0` — corretos.
- Threshold de 10 dias para `melhora_internado_breve/grave`: **verificado diretamente em `preprocessor.py`** (`return 3 if duration_days <= 10 else 4`).
- Distribuições de classes BPSP (N=28.599) e HSL (N=5.174) — corretas.
- Split 70/10/10/10 e volumes resultantes (BPSP: 20.019/2.859/2.859; HSL: 3.621/517/517; teste global: 3.381; cal global: 3.376) — confirmados pelo log `behrt_pooled_20260625_212059.log`.
- Formato de token `{analyte}_{classification}` e fallback para `NO_REF` — correto.
- Vocabulary de 9.997 tokens de capacidade, uso real de 648 tokens únicos — correto.
- Pesos de classe: `total / (n_classes × count_i)`, clamp(max=15.0) — correto e verificado em `client.py`.

### 1.2 Arquitetura SimplifiedBEHRT (Seção 3)
- Todos os valores de `ModelConfig` (vocab_size=10.000, embed_dim=64, max_seq_len=128, num_layers=2, num_heads=4, ff_dim=128, num_classes=5, dropout=0.1) — **verificados em `config.py`**.
- DiaRelativoEmbedding: deslocamento +1, CLS recebe `dia_relativo=0`, max_dia=60 — correto e verificado em `model.py`.
- PositionalEncoding sinusoidal pós-DiaRelativo — correto.
- BEHRTEncoderLayer: `need_weights=True, average_attn_weights=False`, shape `(batch, 4, seq, seq)` — **verificado diretamente em `model.py` linhas 110-114**.
- Post-LN (não Pre-LN), residual connections — correto.
- Late Fusion Demográfica: `[age_norm, sex_binary]` concatenados ao CLS antes do classifier — correto.
- Classifier head: `Linear(64+demo_dim, 64) → ReLU → Dropout(0.1) → Linear(64, 5)` — correto.
- Ganho de +3,08 p.p. com DiaRelativoEmbedding (Exp 6 vs Exp 5) — confirmado.

### 1.3 Protocolo Federado (Seção 4)
- `local_epochs=1` (reduzido de 2) — **verificado em `config.py`** com comentário "reduzido de 2→1: menos client drift em regime non-IID severo".
- FedProx: `μ=0.1` (aumentado de 0.01 no Exp 7) — **verificado em `config.py` e no comentário do código**.
- FedNova: fórmula `τ_eff = Σ p_i·τ_i`, `Δ_i = (w_i − w_global)/τ_i`, `w_{t+1} = w_global + τ_eff·Σ p_i·Δ_i` — **verificada em `fl_core.py`**.
- Proporção de batches BPSP:HSL ≈ 5,5× (BPSP ~1.251 batches/rodada com local_epochs=1 vs HSL ~226) — correto.
- Convergência: warm-up=20, patience=3, threshold=0.005, max=120 rodadas — **verificados em `config.py`**.
- Checkpoint scoping (Migration 011): `training_id`, UPSERT, `load_best(training_id)` — correto.
- Gap best vs last no Exp 8: 66,61% (R91) vs 58,27% (R120) = **8,34 p.p.** — confirmado.

### 1.4 Calibração (Seção 5)
- Temperature scaling falhou em todos os experimentos com temperatura (ECE sempre piorou) — **confirmado por `recalibrate_20260626_192337.json`**: ECE pré=0.0859, ECE pós temperature scaling=0.1066 (+0.021).
- Causa raiz (LBFGS minimiza NLL, não ECE; subconfiança sistemática piora com T>1) — correto.
- Calibração Isotônica OvR (Zadrozny & Elkan, 2002) com PAV — correto.

### 1.5 RAG (Seção 6)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim), `_PostgreSQLStore`/`_InMemoryStore`, `distilgpt2`, top-k=3 — **verificados em `rag.py`**.
- Detecção de alucinação: `probability < 0.6 AND "certeza" in justification.lower()` — **verificado em `rag.py` linha 236**.
- Limitação da base de conhecimento corrompida (tokenização BEHRT + "adulto" interpolado) — correto e documentado.

### 1.6 Segurança e Interoperabilidade (Seção 7)
- HMAC-SHA256 com `FL_PATIENT_ID_SECRET` local — **verificado em `security.py`**, LGPD Art. 13 §4° correto.
- JWT (HS256/RS256/RS512), `X-API-Key`, `FL_AUTH_REQUIRED=false` para dev — correto.
- Rate limiting: `_SlidingWindowLimiter(120/60s, 30/60s)` por IP — **verificado em `security.py`**.
- FHIR R4 `RiskAssessment` via `state._fhir_exporter.to_risk_assessment()` com `correlation_token` efêmero — correto.

### 1.7 Referência ao ClinicalPath (Seção 1.1)
- Verificado em `src/mosaicfl.egg-info/PKG-INFO`: "Extensão preditiva do ClinicalPath (Linhares et al., 2023)" e em `docs/CONTRIBUTING.md`: `clinical-path/ — exportador e watcher para ClinicalPath v2`. A menção é legítima.

---

## 2. O que está Incorreto ou Impreciso ✗

### 2.1 CRÍTICO: Números do "Custo de Privacidade" estão errados (Seção 8.3)

O documento afirma:
> "FL Exp 12 (67,44%) vs **BEHRT Pooled B (69,12%)**" → gap = -1,68 p.p.
> "FL Exp 12 (67,44%) vs **RF Centralizado (68,06%)**" → gap = -0,62 p.p.

**Os dados reais dos logs** (`behrt_pooled_20260625_212059.log` e `behrt_pooled_20260625_223649.json`):

| Modelo | Accuracy | F1 Macro | AUC | ECE |
|--------|----------|----------|-----|-----|
| BEHRT Pooled A (sem_demo) | **67,79%** | 0,5218 | 0,8354 | 0,0929 |
| BEHRT Pooled B (late_fusion) | **63,03%** | 0,5005 | — | — |
| RF Centralizado (BoT) | **68,35%** | 0,5116 | 0,7943 | 0,0638 |
| FL Exp 12 (resultado real) | **67,44%** | 0,484 | 0,8015 | 0,0935 |

**Conclusões corretas:**
- FL (67,44%) vs BEHRT Pooled **B** (63,03%): **FL supera em +4,41 p.p.** (o oposto do que o documento diz)
- FL (67,44%) vs BEHRT Pooled **A** (67,79%): gap = **-0,35 p.p.** (não -1,68 p.p.)
- FL (67,44%) vs RF Centralizado (68,35%): gap = **-0,91 p.p.** (não -0,62 p.p.)

**O 69,12% não existe nos logs de nenhuma das três execuções do pooled baseline.** Não é possível verificar a origem deste número.

**Agravante**: o pooled baseline foi treinado com **40 épocas**, não 120 (valor configurado em `FED_CFG.pooled_epochs=120`). O budget não é equivalente ao FL, então a comparação tem uma limitação metodológica que o documento não menciona.

**Impacto para o TCC**: a narrativa do "custo de privacidade de 1,68 p.p." precisa ser reescrita com os números reais. A conclusão mais honesta é que **FL com late fusion (Config B) supera o pooled treinado com 40 épocas**, e está a apenas -0,35 p.p. do pooled treinado sem demográficos. O custo real da privacidade exige rodar o pooled com 120 épocas para comparação justa.

---

### 2.2 CRÍTICO: Afirmação sobre dados sintéticos (Seção 9.2, item 1)

O documento afirma:
> "Todos os experimentos até o momento usam dados sintéticos ou dados estruturados já processados."

**Isso é factualmente errado.** Todos os experimentos usam **dados reais do FAPESP COVID-19 Data Sharing/BR**, acessados via PostgreSQL com queries sobre `clinical.attendances`, `metrics.exam_records`, etc. O log de 2026-06-25 confirma: "Hospital BPSP → cliente 0: 20019 treino | [...] Fonte demográficos: dados reais FAPESP".

A confusão pode ter origem na ausência de dados de validação clínica prospectiva (o sistema não foi testado em prática clínica real), mas isso é diferente de dizer que os dados de treinamento são sintéticos.

---

### 2.3 Número de testes automatizados (Seção 1.3)

O documento afirma **541 testes automatizados**. A contagem real no diretório `/tests`:
- **569 funções de teste** em **40 arquivos**

O número 541 não é verificável; 569 é o valor correto. A diferença pode indicar que o documento foi gerado antes da adição de alguns testes.

---

### 2.4 Atribuição imprecisa do experimento para `local_epochs=1` (Seção 4.1)

O documento diz "reduzido de 2 no **Experimento 13**". O `config.py` documenta a redução mas não cita o número do experimento. Não há evidência no repositório de que isso tenha sido Exp 13 especificamente. Recomendo verificar no git log ou no planejamento do desktop.

---

### 2.5 Ganho demográfico de +12,7 p.p. (Seção 3.2)

O documento cita "+12,7 p.p. (Experimento 3)". O Exp 3 na tabela 8.1 tem accuracy=55,8% e foi o split 70/10/10/10. A adição de demográficos não ocorreu no Exp 3 — ela pode ter ocorrido depois. Esse ganho específico de 12,7 p.p. não é verificável nos logs disponíveis. **Requer rastreamento no histórico de experimentos ou no planejamento do desktop.**

---

### 2.6 Tabela 8.1 incompleta

A tabela lista apenas Exp 1, 3, 6, 7, 8, 12. Não explica Exp 2, 4, 5, 9 (checkpoint contamination), 10, 11. O Exp 9 é especialmente relevante — foi o incidente de checkpoint cross-contamination — e merece uma linha na tabela com nota.

---

### 2.7 Resultados de calibração pós-isotônica ausentes

O documento diz que a calibração isotônica "resolve o padrão de subconfiança não-uniforme", mas **não apresenta o ECE pós-calibração isotônica**. O JSON de recalibração (`recalibrate_20260626_192337.json`) mostra que o ECE pós-temperature-scaling do modelo do Exp 8 (checkpoint R91) ficou em **0.1066** — mas não há dado comparável para a isotônica nesse arquivo (a estrutura tem apenas `pre_calibration` e `post_calibration` com os mesmos valores). **A ECE resultante da calibração isotônica ainda é uma lacuna real.**

---

## 3. O que Acrescento (Ausente no Documento)

### 3.1 MC Dropout para Incerteza na Inferência

Ausente do documento. O sistema executa **50 amostras de MC Dropout** em inferência, calculando média e desvio padrão por classe. Isso é distinto da calibração (que ajusta a escala das probabilidades) e é a fonte primária de **incerteza epistêmica** reportada no `RiskAssessment`. Deveria ter seção própria ou subseção dentro de 3.2 ou 5.

### 3.2 IsotonicCalibrator como Wrapper de Inferência

O documento descreve a isotônica como técnica de calibração mas não explica como ela é aplicada em inferência: um wrapper que substitui o softmax bruto pelas probabilidades calibradas por classe individualmente (OvR), e como isso se integra ao pipeline de predição na API.

### 3.3 Pipeline de 4 Fases (`make training-full`)

Ausente. O Makefile implementa um pipeline completo sem parametrização externa:
1. `training-bpsp-only` — BEHRT treinado só com dados BPSP (baseline isolado)
2. `training-hsl-only` — BEHRT treinado só com dados HSL
3. Treinamento federado completo
4. `behrt_pooled` + RF centralizado (baselines de comparação)

Esta é uma contribuição de **engenharia de reprodutibilidade** relevante para o TCC: qualquer rodada replicar o experimento completo com um único comando.

### 3.4 BEHRTPatternExtractor

Mencionado brevemente em 3.2 mas não explicado. Este módulo usa os pesos de atenção `(num_layers, batch, num_heads, seq, seq)` para:
- Identificar quais analitos cada cabeça de atenção foca por classe de desfecho
- Gerar heatmaps de co-ocorrência temporal (ex: "PCR elevado seguido de D-dímero em 3 dias")
- Fornecer interpretabilidade clínica além da predição

**Relevância**: em validação clínica, saber "por que o modelo previu X" é tão importante quanto a acurácia.

### 3.5 DataLoader Determinístico

Ausente. O `prepare_dataloaders_from_db()` usa `torch.Generator().manual_seed(RANDOM_SEED + cid)` para garantir que o split 70/10/10/10 seja **idêntico em toda re-execução por hospital**. Isso é parte da reprodutibilidade e deveria aparecer em 2.5.

### 3.6 ConvergenceTracker — Mecânica Interna

O documento menciona o `ConvergenceTracker` em 4.4 apenas como componente do `RoundDispatcher`. Falta descrever:
- Mantém histórico completo de accuracies por rodada
- Ao detectar convergência (Δacc < threshold por `patience` rodadas consecutivas após warm-up), sinaliza stop
- O "replay" do histórico garante que convergência não seja declarada por oscilação pontual

### 3.7 Análise Clínica das Classificações Incorretas

A matriz de confusão do Exp 12 mostra **67 dos 338 casos de `melhora_internado_grave` classificados como `curado_pronto`** = **19,8% de casos graves classificados como curados**. Este é o erro clínico mais crítico do sistema:
- `melhora_internado_grave` = paciente internado por mais de 10 dias com evolução de melhora lenta
- Classificar como `curado_pronto` pode levar a alta precoce ou subestimação de risco

Este dado existe mas não é analisado do ponto de vista clínico no documento. Para uma defesa de TCC com foco em impacto médico, essa análise é essencial.

### 3.8 Comparação Completa de Baselines (tabela corrigida)

Com base nos dados reais:

| Modelo | Acc | F1 Macro | AUC | ECE | n_epochs |
|--------|-----|----------|-----|-----|----------|
| FL Exp 12 (FedNova + Chk Scoped) | **67,44%** | 0,484 | 0,8015 | 0,0935 | 120 rounds |
| BEHRT Pooled A (sem_demo, 40 épocas) | 67,79% | 0,5218 | 0,8354 | 0,0929 | 40 |
| BEHRT Pooled B (late_fusion, 40 épocas) | 63,03% | 0,5005 | — | — | 40 |
| RF Centralizado (Bag-of-Tokens) | 68,35% | 0,5116 | 0,7943 | 0,0638 | — |
| RF Hospital 0 (BPSP isolado) | 59,45% | 0,337 | 0,7425 | 0,0467 | — |
| RF Hospital 1 (HSL isolado) | 24,25% | 0,184 | 0,6905 | 0,2541 | — |

**Observação crítica**: o BEHRT Pooled foi treinado com apenas 40 épocas (não 120, como especifica `FED_CFG.pooled_epochs`). Para comparação justa com o FL (120 rounds × 1 epoch efetiva), o pooled deveria ser rodado com 120 épocas.

### 3.9 Colapso do HSL Isolado

O RF Hospital 1 (HSL isolado) tem acc=**24,25%** — pior que chance aleatória para 5 classes (20%). Isso ocorre porque o HSL tem 61,5% de `melhora_pronto` enquanto o BPSP tem apenas 0,4% dessa classe. Um modelo treinado só no HSL não generaliza para a distribuição global do test set (que mistura BPSP+HSL). Este dado é poderoso para justificar a necessidade do aprendizado federado e está ausente do documento.

### 3.10 Ausência de Ablação Sistemática

O documento apresenta experimentos sequenciais mas não uma tabela de ablação formal que isole cada contribuição:

| Configuração | Acc | ΔAcc |
|-------------|-----|------|
| Baseline FL (FedAvg, sem demographics) | ~55-58% | — |
| + Split 70/10/10/10 | 55,8% | — |
| + DiaRelativoEmbedding | 59,63% | +3,08 p.p. |
| + μ=0,1 (FedProx mais restritivo) | 59,36%* | ~0 |
| + Checkpoint Guloso | 66,61% | +7 p.p. (best R91) |
| + FedNova + Checkpoint Scoped | 67,44% | +0,83 p.p. |
| + Demográficos (late fusion) | ? | pendente |

*Avaliado em R120 sem checkpoint guloso, sub-representando o real ganho.

---

## 4. O que Ainda Falta Obter

Mantendo os itens A-F da Seção 10 do documento, adiciono:

### G. Rodar BEHRT Pooled com 120 épocas (pendente de implementação)
**Por quê**: o pooled atual com 40 épocas sub-representa o baseline. Para o TCC, o argumento do custo de privacidade precisa de comparação com budget equivalente ao FL.  
**Como**: ajustar o pipeline `training-full` para usar `FED_CFG.pooled_epochs=120` (já está no config, mas a execução usou 40).

### H. ECE pós-calibração isotônica
**Por quê**: o documento afirma que a isotônica resolve a subconfiança, mas o arquivo de recalibração disponível não registra esse valor.  
**Como**: executar o script de calibração no checkpoint do Exp 12 e registrar ECE pré/pós para os dois métodos.

### I. Performance por hospital no test set
**Por quê**: o teste global mistura BPSP e HSL; a acurácia de 67,44% pode esconder desempenho muito diferente por hospital.  
**Como**: filtrar o test set por hospital_id e gerar métricas separadas.

### J. Ablação FL Config A vs Config B (sem vs com demográficos no FL)
**Por quê**: o log do pooled compara A vs B no treino centralizado, mas o FL federado foi sempre executado com `demo_dim=2`. Não há comparação FL sem demográficos.  
**Como**: executar 1 rodada de FL com `demo_dim=0` e comparar.

### K. Experimento 13 — resultados completos
**Por quê**: o config menciona reduções atribuídas ao Exp 13 (local_epochs=1 + isotônica), mas não há JSON de resultado nomeado como tal nos logs.  
**Como**: rastrear no planejamento do desktop.

### L. Distribuição temporal dos dados FAPESP
**Por quê**: dados de COVID-19 têm viés temporal forte (onda 1 vs onda 2 vs omicron). O período de coleta (item C da Seção 10) define a validade externa do modelo.  
**Como**: `SELECT MIN(co.outcome_at), MAX(co.outcome_at) FROM metrics.clinical_outcomes co JOIN clinical.attendances a ON co.attendance_id = a.attendance_id WHERE a.hospital_id IN ('HSL','BPSP')`.

### M. BEHRTPatternExtractor — alguma visualização real
**Por quê**: o documento cita interpretabilidade como contribuição, mas não apresenta nenhum exemplo. Mesmo um heatmap de atenção para 1 caso do Exp 12 seria suficiente para a defesa.

### N. Estatísticas demográficas por hospital e por classe
**Por quê**: age_mean e sex_M estão nos logs (BPSP: age_mean=0.51, sex_M=9591/20019≈48%; HSL: age_mean=0.52, sex_M=1981/3621≈55%), mas precisam ser estratificadas por classe de desfecho para caracterizar o dataset adequadamente.

---

## 5. Prioridade para a Defesa e Implementação

### Alta Prioridade (bloqueia a narrativa central do TCC)

| # | Item | Ação |
|---|------|------|
| 1 | Custo de privacidade com números reais | Reescrever Seção 8.3 com dados dos logs; rodar pooled com 120 épocas |
| 2 | Afirmação sobre "dados sintéticos" | Corrigir para "dados reais FAPESP COVID-19" |
| 3 | ECE pós-isotônica | Executar recalibração e registrar |
| 4 | Erro clínico de melhora_internado_grave | Adicionar análise em 8.2 |

### Média Prioridade (enriquece o texto)

| # | Item | Ação |
|---|------|------|
| 5 | MC Dropout — seção própria | Escrever subseção em 3 ou 5 |
| 6 | Performance por hospital | Query no test set dividido por hospital_id |
| 7 | Ablação sistemática | Montar tabela com dados disponíveis + executar Config A FL |
| 8 | Colapso HSL isolado (24,25%) | Incluir em 1.2 como motivação do FL |
| 9 | Pipeline make training-full | Descrever em nova seção de engenharia |

### Baixa Prioridade (para trabalhos futuros ou apêndice)

| # | Item | Ação |
|---|------|------|
| 10 | BEHRTPatternExtractor visualização | Gerar 1 heatmap para apêndice |
| 11 | Distribuição temporal dos dados | Query MIN/MAX dates |
| 12 | Top-20 analitos no vocabulário | Query em metrics.exam_records |
| 13 | Contagem exata de testes (569, não 541) | Atualizar Seção 1.3 |
| 14 | Pooled com 120 épocas completo | Execução longa (~8h) |

---

## 6. Resumo Executivo

O documento gerado pelo DeepSeek é **tecnicamente sólido em ~85% do conteúdo** — a arquitetura, o protocolo federado, a calibração e a segurança estão corretos e verificáveis no código. Funcionou bem como insumo inicial.

**Dois erros que não podem ir para o texto final:**

1. **O número 69,12% para BEHRT Pooled B não existe nos dados experimentais.** O valor real é 63,03% (BEHRT Pooled B, 40 épocas) ou 67,79% (BEHRT Pooled A, sem demográficos). A conclusão do custo de privacidade muda substancialmente: com arquitetura equivalente (Config B vs B), **o FL supera o pooled baseline**.

2. **Os dados NÃO são sintéticos.** Todos os experimentos usam o dataset FAPESP COVID-19 real.

**O que o documento mais deixa a desejar** é a análise de erros clínicos (19,8% de casos graves classificados como curados) e a ausência de uma tabela de ablação formal. Esses dois elementos dariam substância científica ao capítulo de resultados.
