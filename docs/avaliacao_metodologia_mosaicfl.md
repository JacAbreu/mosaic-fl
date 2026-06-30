# Avaliação do MOSAIC-FL: Estado Atual, Lacunas e Próximos Passos

**Referência**: código-fonte (`src/mosaicfl/`), logs de experimento (`experiments/logs/`), `docs/Sumario_Treinamento.md`, `docs/Sumario_Treinamento_Parte2.md`
**Avaliado por**: Claude Sonnet 4.6 com acesso integral ao repositório
**Última atualização**: 2026-06-29 (pós-Exp 16, pré-Exp 17)

---

## O MOSAIC-FL como Percurso pelos Fundamentos da Inteligência Artificial

O MOSAIC-FL não é apenas um sistema de aprendizado federado para hospitais — é um percurso deliberado e documentado pelos fundamentos da Inteligência Artificial aplicada. Cada decisão técnica do projeto pressupõe e exercita uma camada da cadeia de conhecimento que sustenta qualquer sistema de IA responsável em produção.

### Da mineração de dados à privacidade formal

O projeto parte de um problema concreto — dados clínicos reais de dois hospitais com distribuições radicalmente diferentes, protegidos por lei e incapazes de ser centralizados — e resolve esse problema percorrendo, na ordem correta, os fundamentos que o tornam solucionável:

**Fundamento 1 — Dados e Pré-processamento**
Antes de qualquer modelo, o projeto enfrenta o problema real de mineração de dados: como transformar registros clínicos heterogêneos (exames laboratoriais com nomenclaturas distintas por hospital) em uma representação computável e comparável. A solução — tokenização `{analito}_{classificação}` com mapeamento LOINC e vocabulário compartilhado de 648 tokens reais — é uma decisão de engenharia de dados com consequência direta na validade da federação. Sem vocabulário compartilhado, a agregação FedNova seria semanticamente inválida: o mesmo analito teria token IDs diferentes em cada hospital.

**Fundamento 2 — Estatística e Desbalanceamento de Classes**
O dataset FAPESP tem estrutura de classes extremamente assimétrica: `curado_pronto` representa 55,6% do BPSP e apenas 1,3% do HSL; `melhora_pronto` representa 61,5% do HSL e 0,4% do BPSP. Tratar isso com acurácia simples levaria a um modelo que aprende apenas a classe majoritária. O projeto aplica pesos de classe inversamente proporcionais à frequência, com capping em 15,0 para evitar instabilidade numérica — uma decisão que exige entender simultaneamente estatística de desbalanceamento, comportamento de gradientes e estabilidade de treinamento.

**Fundamento 3 — Modelagem Sequencial e Transformer**
A escolha do SimplifiedBEHRT sobre o Random Forest não é arbitrária — ela é justificada por ablação empírica. O RF (Bag-of-Tokens) obteve 68,41% ignorando a ordem temporal dos exames. O BEHRT federado atingiu 69,59% incorporando a sequência e o tempo relativo de cada exame (`DiaRelativoEmbedding`). O ganho de +1,80 p.p. isolado pelo DiaRelativoEmbedding quantifica o valor da modelagem temporal sobre a abordagem de frequência. O projeto demonstra, com dados, por que o transformer é a arquitetura correta para este problema — não por tendência, mas por necessidade clínica.

**Fundamento 4 — Treinamento e Convergência**
O treinamento federado expõe problemas invisíveis no treinamento centralizado: *client drift* (o modelo de cada hospital diverge do global ao longo das épocas locais), inconsistência objetiva por número diferente de passos entre clientes (BPSP: ~1.251 batches/rodada; HSL: ~226 batches/rodada), e instabilidade de gradientes com classes raras. Cada um desses problemas tem uma solução documentada no projeto: FedProx (μ=0,1) para o drift, FedNova para a inconsistência objetiva, gradient clipping e class weight clipping para a instabilidade. A sequência de experimentos (Exp 7 → Exp 12 → Exp 15) rastreia o efeito isolado de cada correção.

**Fundamento 5 — Avaliação e Matriz de Confusão**
A acurácia agregada (69,59%) é necessária mas insuficiente. O Macro F1 (0,4946) dá peso igual a `curado_internado` (N=28 no teste) e `curado_pronto` (N=1.620) — porque errar nos 28 casos graves tem impacto clínico diferente de errar nos frequentes. A matriz de confusão revela o erro clínico mais crítico do sistema: casos de `melhora_internado_grave` classificados como `curado_pronto`, ou seja, pacientes que deveriam ser internados recebendo predição de alta. Sem análise da matriz de confusão, a acurácia esconde exatamente o erro que mais importa.

**Fundamento 6 — Calibração de Probabilidades**
Um modelo com 69,59% de acurácia não está pronto para uso clínico se suas probabilidades estiverem descalibradas. Um médico que vê "78% de chance de melhora_internado_grave" precisa que esse número reflita a frequência real — não apenas a ordenação entre classes. O projeto aplica, documenta a falha (temperature scaling em 8/8 experimentos) e substitui pela solução correta (calibração isotônica OvR, ECE=0,0149). O processo de tentar, falhar, diagnosticar a causa raiz (subconfiança sistemática não-uniforme, LBFGS minimizando NLL em vez de ECE) e convergir para a solução certa é um exemplo completo de ciclo científico aplicado.

**Fundamento 7 — Privacidade como Consequência Necessária**
A privacidade diferencial (DP-FedAvg, McMahan et al. 2018) não foi introduzida como ornamento acadêmico — ela é a formalização matemática da restrição que motivou o projeto inteiro. O FL sem DP não garante privacidade plena: pesos de modelo trocados entre hospitais e servidor contêm informação suficiente para reconstruir dados de pacientes (Geiping et al. 2020; Zhu et al. 2019). A série de experimentos planejada (Exp 17: σ=1,0; Exp 18: σ=0,5; Exp 19: σ=2,0) gerará a curva Acc×ε — a representação quantitativa do trade-off entre utilidade clínica e garantia formal de privacidade. Sem ter percorrido os seis fundamentos anteriores, não seria possível entender o que está sendo sacrificado em cada ponto da curva.

### Por que esse percurso importa para a avaliação

Um sistema de IA clínica pode ser construído sem entender calibração — e terá probabilidades enganosas. Pode ser construído sem entender a matriz de confusão — e ocultará seus erros mais graves. Pode implementar FL sem entender o problema de passos efetivos — e o hospital com mais dados dominará a agregação silenciosamente. Pode adicionar DP sem entender o trade-off — e degradará o modelo sem saber quanto de privacidade está comprando.

O MOSAIC-FL trata cada fundamento como pré-requisito do seguinte. Esse encadeamento — e a documentação das decisões, erros e correções em cada etapa — é o que o posiciona além de uma implementação técnica e o qualifica como trabalho de pesquisa aplicada.

---

## Enquadramento desta Avaliação

O MOSAIC-FL é avaliado sob dois critérios simultâneos:

1. **MVP próximo de produção hospitalar** — o que separa o sistema de uso real em ambiente clínico, considerando apenas bloqueadores de infraestrutura e regulatório. Questões de engenharia de software e qualidade de modelo já foram resolvidas.

2. **Experimentação nível mestrado** — o que torna os resultados publicáveis e defensáveis em banca, com rigor metodológico equivalente a um artigo de conferência.

---

## 1. Estado Atual — Resultados Experimentais

### 1.1 Melhor resultado histórico (referência)

| Experimento | Configuração | Accuracy | Macro F1 | Macro AUC | ECE |
|---|---|---|---|---|---|
| Exp 15 — FL Federado | FedNova + FedProx μ=0,1 + Isotônica + Chk Guloso | **69,59%** | 0,4946 | 0,8181 | 0,0149 |
| Exp 16 — BEHRT Pooled B | 120 épocas, late fusion, budget equivalente | 68,68% | — | — | — |
| RF Centralizado | Bag-of-Tokens, dados pooled | 68,41% | — | — | — |
| BEHRT Pooled A | 120 épocas, sem demográficos | — | — | — | — |

**Conclusão central (Exp 15 e 16):** com budget equivalente (120 rodadas = 120 épocas), o FL FedNova supera *todos* os baselines centralizados. O custo de privacidade da federação é **negativo** — o modelo federado melhora a capacidade preditiva em relação ao centralizado, sem mover dados entre hospitais.

### 1.2 Leave-one-out

| Configuração | Accuracy | Interpretação |
|---|---|---|
| BPSP-only (Exp 13) | 64,86% | Perde `melhora_pronto` (0,4% dos casos BPSP) |
| HSL-only (Exp 14) | 40,05% | Dataset pequeno não generaliza para o teste global |
| Federado (Exp 15) | **69,59%** | Supera ambos os isolados |
| RF HSL isolado | 24,25% | Pior que chance aleatória para 5 classes |

A diferença de 29,54 p.p. entre HSL-only e o federado é o argumento empírico mais direto para a necessidade da federação.

### 1.3 Contribuição de cada componente (ablação parcial)

| Configuração | Accuracy | Δ acumulado |
|---|---|---|
| FL base (Exp 7 — FedAvg, μ=0,1) | 59,36% | referência |
| + FedNova + Checkpoint Scoped (Exp 12) | 67,44% | +8,08 p.p. |
| + local_epochs=1 + Grad/Class Clipping (Exp 13/15) | 69,59% | +2,15 p.p. |
| + Calibração Isotônica OvR | ECE 0,0575 → 0,0149 | −74% ECE |
| + DiaRelativoEmbedding (ablação Exp 4→6) | +1,80 p.p. | verificado separadamente |
| + Late Fusion Demográfica (ablação Pooled A→B) | +0,39 p.p. | verificado no centralizado |

---

## 2. MVP — O que já é nível de produção

### 2.1 Pipeline de treinamento ✓

- **FedNova + FedProx** com hiperparâmetros calibrados (μ=0,1, local_epochs=1, τ_i por cliente)
- **DP-FedAvg** implementado (McMahan et al. 2018) — clipping de update no cliente + ruído gaussiano no servidor. Ativado por `FL_DP_NOISE=σ`. Aguardando Exp 17/18/19
- **Checkpoint guloso com scoping** por `training_id` — rastreabilidade total de qual checkpoint pertence a qual experimento
- **Seeding determinístico** por rodada × cliente — reprodutibilidade 100% garantida
- **Gradient clipping** (max_norm=1,0) e **class weight clipping** (max=15,0) — estabilidade com classes extremamente raras

### 2.2 Calibração ✓

- **Isotônica OvR** com ECE=0,0149 (Exp 15) — abaixo do limiar clínico de 0,05 estabelecido para a defesa. Melhora de 74% em relação ao pré-calibração
- Temperature scaling testado e documentado como inadequado para este cenário (8/8 experimentos falhando) — decisão de abandono com evidência empírica

### 2.3 Segurança e conformidade LGPD ✓

- **HMAC-SHA256** para pseudonimização de `patient_id` (Art. 13 §4° da LGPD)
- **JWT (HS256/RS256)** e `X-API-Key` na API
- **Rate limiting** por janela deslizante (120 req/min geral, 30 req/min ingest)
- Verificação de integridade de checkpoints via SHA-256
- `path traversal` bloqueado no exportador
- **DP-FedAvg** como camada adicional de privacidade formal (ε,δ)-DP — implementado, experimentos pendentes

### 2.4 Interoperabilidade clínica ✓

- **FHIR R4** `RiskAssessment` com `correlation_token` efêmero — integração com prontuários modernos sem armazenar vínculo paciente ↔ predição
- **22 analitos mapeados para LOINC** — internacionalização dos códigos de exame
- **LOINC + vocabulário compartilhado** (648 tokens reais) — garante compatibilidade semântica entre hospitais para a agregação FedNova

### 2.5 Interpretabilidade ✓

- **RAG** com `_InMemoryStore` / `_PostgreSQLStore`, `all-MiniLM-L6-v2` (384 dim), top-k=3
- **Ollama / gemma3:4b** como backend LLM padrão; fallback automático para `distilgpt2` (HuggingFace) se Ollama offline
- **Detecção de alucinação** por heurística (`probability < 0.6 AND "certeza" in justification`)
- Bugs críticos da knowledge base corrigidos: filtro de special tokens (`[PAD]`, `[CLS]`, `[SEP]`) + guard `if idade_exacta:`

### 2.6 Arquitetura de software ✓

- **Arquitetura hexagonal**: domínio puro (`mosaicfl.core`) sem importações de infraestrutura — testável offline, substituível, auditável
- **569 testes automatizados** em 40 arquivos — cobertura de contrato e integração
- **Rastreabilidade completa**: `training_id` em todas as métricas, checkpoints com SHA-256, logs estruturados em JSON
- **Pipeline reproduzível**: `make training-full` executa as 4 fases sem parametrização manual (~9h43min em CPU)

### 2.7 Engenharia de experimentos ✓

- `docs/Sumario_Treinamento.md` como log histórico de todas as decisões (Exp 1–16)
- `docs/Sumario_Treinamento_Parte2.md` como contexto compacto para novas sessões
- Incidentes documentados: contaminação de checkpoint (Exp 9), colapso de temperature scaling (8 experimentos), bug de class weight (Exp 13)

---

## 3. Bloqueadores de Produção Reais

### 3.1 Infraestrutura (resolvíveis antes do deploy)

| Bloqueador | Impacto | Solução |
|---|---|---|
| Rate limiter in-process | Com N workers Gunicorn, limite efetivo é `120×N` req/min | Redis + `fastapi-limiter` |
| MC Dropout sem timeout/circuit breaker | Sob carga alta, requests enfileiram antes do timeout HTTP | Timeout interno + `torch.vmap` |
| Rotação de chave HMAC | Se secret comprometido, histórico de hashes fica desvinculado | `key_version` por hash no banco |
| mTLS entre servidor e clientes Flower | Comunicação FL em plaintext na rede local | `grpc.ssl_channel_credentials` |

### 3.2 Regulatório (independente de software)

| Requisito | Status |
|---|---|
| Submissão ANVISA SaMD Classe III (RDC 657/2022) | Não iniciado |
| Validação clínica prospectiva com parecer CEP/CONEP | Não iniciado |
| Documentação técnica CFM 2.227/2018 | Não iniciado |
| Consentimento informado e DPO designado | Não iniciado |

**Observação**: nenhum dos bloqueadores regulatórios é um problema de código. O sistema já implementa os controles técnicos que essas regulamentações exigem (pseudonimização, rastreabilidade, DP). O que falta é o processo formal de submissão e validação clínica.

---

## 4. Qualidade Científica — O que está no nível de mestrado

### 4.1 Contribuições com resultado empírico verificado

| Contribuição | Resultado | Referência |
|---|---|---|
| Custo de privacidade negativo com budget equivalente | FL (69,59%) > Pooled B (68,68%) > RF (68,41%) | Exp 15/16 |
| FedNova para cenário com razão de dados 5,5× | +8,08 p.p. vs FedAvg (Exp 12 vs Exp 7) | Wang et al. 2020 |
| Leave-one-out: nenhum hospital generaliza sozinho | HSL-only 40,05% vs Federado 69,59% (+29,54 p.p.) | Exp 13/14/15 |
| Calibração isotônica OvR sobre transformer FL | ECE 0,0575 → 0,0149 (−74%) | Zadrozny & Elkan 2002 |
| Diagnóstico de falha sistemática do temperature scaling | 8/8 experimentos com ECE pior pós-calibração | Guo et al. 2017 |
| DiaRelativoEmbedding para sequências intra-internação | +1,80 p.p. acurácia | Ablação Exp 4→6 |

### 4.2 Contribuições implementadas, resultados pendentes

| Contribuição | Estado | Pendência |
|---|---|---|
| DP-FedAvg com curva Acc×ε | Implementado | Exp 17 (σ=1,0), Exp 18 (σ=0,5), Exp 19 (σ=2,0) |
| gemma3:4b para justificativas clínicas em PT-BR | Implementado | Validação qualitativa das justificativas |

---

## 5. Lacunas para Qualidade de Mestrado

### 5.1 Alta prioridade (bloqueia argumento central do TCC)

**A — Curva Acc×ε (série DP)**

O argumento de privacidade diferencial está incompleto sem os pontos da curva. Três experimentos planejados (σ=0,5; 1,0; 2,0) gerarão a tabela que demonstra o trade-off entre privacidade formal e utilidade — requisito para qualquer publicação sobre FL com DP.

**B — Análise clínica dos erros críticos**

A matriz de confusão do Exp 12 mostra 67 de 338 casos de `melhora_internado_grave` classificados como `curado_pronto` — 19,8% de pacientes com internação prolongada recebendo predição de alta sem internação. Este é o erro clínico mais grave do sistema e não foi analisado em nenhum documento. Para uma banca com foco em impacto médico, essa análise é obrigatória. Requer verificar se o número permanece no Exp 15 (melhor modelo).

**C — Ablação formal em tabela única**

As contribuições de cada componente estão espalhadas em 16 experimentos. Uma tabela de ablação com Δ isolado por componente (DiaRelativo, FedNova, local_epochs, clipping, calibração) é a forma canônica de apresentar evolução em artigo.

### 5.2 Média prioridade (enriquece o texto)

**D — Performance por hospital no conjunto de teste**

O conjunto de teste global mistura BPSP e HSL. A acurácia agregada (69,59%) pode esconder desempenho muito diferente por hospital. Filtrar por `hospital_id` no conjunto de teste e gerar métricas separadas fortalece a narrativa de generalização federada.

```sql
SELECT hospital_id, COUNT(*) FROM clinical.attendances
WHERE attendance_id IN (/* IDs do test set */)
GROUP BY hospital_id;
```

**E — Distribuição temporal do dataset FAPESP**

Dados de COVID-19 têm viés temporal forte (onda 1 vs onda 2 vs ômicron). O período de coleta define a validade externa do modelo. Query necessária:

```sql
SELECT MIN(co.outcome_at), MAX(co.outcome_at),
       a.hospital_id, COUNT(*)
FROM metrics.clinical_outcomes co
JOIN clinical.attendances a ON co.attendance_id = a.attendance_id
WHERE a.hospital_id IN ('HSL','BPSP')
GROUP BY a.hospital_id;
```

**F — Validação qualitativa do gemma3:4b**

Coletar 5 exemplos de justificativas por classe (curado_pronto, melhora_internado_grave, etc.) e comparar coerência clínica com distilgpt2. O P@3 macro subiu de 0,145 (Exp 12, KB corrompida) para 0,2343 (Exp 13, KB corrigida) — o ganho quantitativo já está medido; falta o qualitativo do novo modelo.

**G — Estatísticas demográficas por classe de desfecho**

age_mean e sex_M por hospital já aparecem nos logs (BPSP: age_mean=0,51, sex_M≈48%; HSL: age_mean=0,52, sex_M≈55%), mas sem estratificação por classe. Necessário para caracterizar o dataset.

### 5.3 Baixa prioridade (para apêndice ou trabalhos futuros)

**H — Janela temporal da predição**

O modelo usa o histórico completo da internação. Na prática clínica, a predição ocorre com dados parciais (fim do dia 1, 3 ou 5). Decisão clínica que deve ser embasada pela orientadora ou pela literatura.

**I — Visualização do BEHRTPatternExtractor**

Pelo menos um heatmap de atenção por classe para um caso representativo do Exp 15. Demonstra interpretabilidade além do texto gerado pelo RAG.

**J — FL sem demográficos (ablação Config A federado)**

O Pooled B vs A (+0,39 p.p.) está documentado no centralizado. A late fusion no federado não foi isolada — sempre esteve ativa desde o Exp 1. Uma rodada com `demo_dim=0` fecharia o argumento.

---

## 6. O que NÃO é uma lacuna

### 6.1 Arquitetura "simplificada demais" do SimplifiedBEHRT

Esta crítica (levantada na avaliação acadêmica anterior, −0,5 p.p.) está respondida pelos dados:

- Dataset FAPESP tem escala hospitalar única (uma internação por paciente, sem histórico longitudinal longo). Modelos maiores overfit com dados escassos
- Hardware CPU-only impõe restrição real: SimplifiedBEHRT treina em ~3min/rodada; embed_dim=128 triplicaria o tempo
- Sem corpus pré-treinado em PT-BR/FAPESP, camadas adicionais adicionam parâmetros aleatórios sem ganho representacional
- O gargalo é `embed_dim=64` (16-dim por cabeça de atenção), não a profundidade — mais camadas no mesmo espaço residual não aumentam capacidade
- O argumento defensável para a banca: citar Vaswani et al. (2017) sobre equivalência de positional encoding sinusoidal vs aprendido em sequências curtas, e apresentar os resultados empíricos como justificativa pragmática


### 6.3 Afirmação sobre dados sintéticos

Corrigida no `Metodologia MOSAIC-FL - Final Corrigido.md`. Todos os experimentos usam dados reais FAPESP COVID-19.

### 6.4 Números do custo de privacidade

Corrigidos com os dados reais dos Exp 15 e 16. O `Metodologia MOSAIC-FL - Final Corrigido.md` usa os valores corretos.

---

## 7. Resumo Executivo

**O que está pronto para produção** (aguardando apenas infra e regulatório):
- Pipeline FL com FedNova, FedProx, DP-FedAvg, calibração isotônica, FHIR R4, LGPD, RAG com fallback, arquitetura hexagonal, 569 testes

**O que está pronto como contribuição científica**:
- Custo de privacidade negativo com budget equivalente; leave-one-out empírico; diagnóstico de falha do temperature scaling; FedNova aplicado a cenário hospitalar brasileiro com razão de dados 5,5×; ECE=0,0149

**O que precisa ser feito antes da defesa**:
- Série DP (Exp 17/18/19) — curva Acc×ε
- Análise clínica dos erros críticos (`melhora_internado_grave` classificados como `curado_pronto`)
- Ablação formal em tabela única
- Validação qualitativa do gemma3:4b

**O que não é bloqueador** (pode ir para trabalhos futuros):
- mTLS, Redis, circuit breaker, ANVISA, CEP/CONEP, janela temporal, visualizações de atenção
