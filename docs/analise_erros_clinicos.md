# Análise de Erros Clínicos — MOSAIC-FL

**Referência**: logs `run_complete_20260629_074506.log`, `experiments/data/behrt_pooled_20260629_172822.json`, `experiments/data/evaluation_round_120.json`
**Última atualização**: 2026-06-29 (pós-Exp 15/16)

---

## 1. Distribuição do Conjunto de Teste

O conjunto de teste global é fixo em todos os experimentos (split 70/10/10/10 determinístico, seed=42):

| Classe | N | % do teste |
|---|---|---|
| `curado_pronto` | 1.620 | 47,9% |
| `melhora_internado_breve` | 1.074 | 31,8% |
| `melhora_internado_grave` | 338 | 10,0% |
| `melhora_pronto` | 321 | 9,5% |
| `curado_internado` | 28 | 0,8% |
| **Total** | **3.381** | **100%** |

A distribuição é fortemente assimétrica. As duas classes com maior relevância clínica para erros graves (`melhora_internado_grave` e `curado_internado`) somam apenas 10,8% do teste — mas concentram os erros de maior impacto.

---

## 2. Métricas por Classe Disponíveis

### 2.1 BEHRT Federado Exp 15 — métricas globais (melhor modelo: round 79)

| Métrica | Valor |
|---|---|
| Accuracy | 69,59% |
| Macro F1 | 0,4946 |
| Macro AUC | 0,8181 |
| ECE (isotônica OvR) | 0,0149 |
| ECE pré-calibração | 0,0575 |

> A matriz de confusão por classe do round 79 (best checkpoint) não está gravada em arquivo — está serializada no checkpoint PostgreSQL. A extração requer um script de avaliação pós-treinamento. Isso é um **item de ação pendente** (ver Seção 5).

### 2.2 BEHRT Pooled A — per-class F1 (proxy para Exp 15)

O BEHRT Pooled A (120 épocas, sem demográficos, dados centralizados) tem Acc=68,29%, estrutura idêntica ao federado. Seus F1 por classe são o dado mais próximo disponível do breakdown do Exp 15:

| Classe | F1 | AUC | Interpretação |
|---|---|---|---|
| `curado_pronto` | **0,8487** | — | Bem aprendida — classe dominante (47,9%) |
| `melhora_pronto` | **0,8219** | — | Bem aprendida — alta frequência no HSL |
| `melhora_internado_breve` | 0,4634 | — | Aprendizagem parcial |
| `melhora_internado_grave` | 0,3501 | — | Aprendizagem fraca — alto risco de erro grave |
| `curado_internado` | **0,0714** | — | Praticamente não predita — 28 casos, quase nenhum correto |

### 2.3 Round 120 (não o melhor) — per-class detail

Do arquivo `evaluation_round_120.json` (avaliação na última rodada, acc=35,05% — modelo não convergido nesta rodada):

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `curado_pronto` | 0,643 | 0,101 | 0,175 | 1.620 |
| `curado_internado` | 0,000 | 0,000 | 0,000 | 28 |
| `melhora_pronto` | 0,323 | 0,461 | 0,380 | 321 |
| `melhora_internado_breve` | 0,352 | 0,739 | 0,476 | 1.074 |
| `melhora_internado_grave` | 0,193 | 0,234 | 0,211 | 338 |

> Estes valores são do round 120 (acc=35,05%), não do best checkpoint (round 79, acc=69,59%). Servem para ilustrar o padrão de erros mas **não representam o desempenho do modelo final**.

### 2.4 RAG Precision@3 — Exp 15 (proxy de dificuldade por classe)

A P@3 mede quais classes o sistema consegue recuperar casos similares na knowledge base — reflete indiretamente quais classes têm representação clínica coerente:

| Classe | P@3 |
|---|---|
| `melhora_pronto` | **0,6012** |
| `curado_internado` | 0,1667 |
| `curado_pronto` | 0,0821 |
| `melhora_internado_breve` | 0,0829 |
| `melhora_internado_grave` | **0,0424** |
| **Macro P@3** | **0,1284** |

O P@3=0,0424 de `melhora_internado_grave` indica que os perfis desta classe na knowledge base têm pouca similaridade entre si — padrão clínico heterogêneo, difícil de recuperar por embedding.

---

## 3. Tipologia dos Erros Clínicos

### Erro Tipo 1 — Paciente grave classificado como alta do pronto-socorro

**Classe real**: `melhora_internado_grave` (internação > 10 dias, evolução lenta)
**Classe predita**: `curado_pronto` (alta do pronto-socorro sem internação)

**Impacto clínico**: máximo. O paciente que necessita de internação prolongada recebe predição de alta sem admissão. Em um sistema de triagem assistida, essa predição pode influenciar a decisão de liberar o paciente antes da avaliação clínica completa.

**Evidência quantitativa**: a avaliação anterior do Exp 12 (acc=67,44%) mostrou 67 de 338 casos neste padrão (19,8%). No Exp 15 (acc=69,59%), o número exato requer extração do checkpoint, mas a F1=0,35 de `melhora_internado_grave` confirma que o modelo acerta menos de metade dos casos desta classe.

**Por que acontece**: `melhora_internado_grave` é a classe com menor P@3 (0,0424) — o padrão clínico é heterogêneo. Pacientes com evolução grave lenta podem ter exames iniciais semelhantes a pacientes de alta simples, com a diferença aparecendo nos dias seguintes — que o modelo não vê na triagem.

**Limitação estrutural do problema**: o modelo é treinado com o histórico completo da internação mas, em uso real, a predição ocorre com dados parciais do início do atendimento. A janela temporal da predição está indefinida (item pendente no TODO.md).

---

### Erro Tipo 2 — Classe ausente: curado_internado quase nunca predita

**Classe real**: `curado_internado` (28 casos no teste, 0,8%)
**Classe predita**: qualquer outra (F1≈0,07 no Pooled A; F1=0,0 no round 120)

**Impacto clínico**: moderado. `curado_internado` representa pacientes que foram internados e receberam alta curados — o desfecho positivo após hospitalização. Um erro nesta classe tende a sub-representar casos de internação breve com boa evolução, potencialmente classificando-os como `curado_pronto` (subestimando necessidade de internação) ou `melhora_internado_breve` (mais próximo clinicamente).

**Por que acontece**: 28 casos no teste (e proporcionalmente poucos no treino) tornam impossível ao modelo aprender o padrão desta classe com confiança. Com pesos de classe, o modelo tenta compensar — mas 28 amostras no teste implicam ainda menos no treino (~196 amostras, 0,8% do total). É uma classe estruturalmente sub-representada.

**Implicação metodológica**: o F1 macro é sensível a esta classe — um F1=0,07 em `curado_internado` puxa o macro para baixo mesmo com F1>0,8 nas classes frequentes. Isso é metodologicamente correto (a classe importa clinicamente) mas pode parecer que o modelo está "pior do que é" nas classes principais.

---

### Erro Tipo 3 — Confusão na fronteira melhora_internado_breve ↔ melhora_internado_grave

**Classes envolvidas**: `melhora_internado_breve` (internação ≤ 10 dias) vs `melhora_internado_grave` (internação > 10 dias)

**Impacto clínico**: moderado. Ambas as classes requerem internação — o erro não envia um paciente para casa indevidamente. O risco é de sub ou super-estimar a intensidade do suporte necessário.

**Por que acontece**: o limiar de 10 dias é um critério administrativo, não uma fronteira clínica nítida. Um paciente com 9 dias de internação é clinicamente indistinguível de um com 11 dias na admissão. O modelo aprende um padrão de exames que não tem equivalência direta com a duração da internação, especialmente porque a duração é determinada por fatores além dos exames iniciais (disponibilidade de leito, protocolo hospitalar, complicações secundárias).

**Dado disponível**: `melhora_internado_breve` F1=0,463 (Pooled A) e `melhora_internado_grave` F1=0,350 — ambas baixas, sugerindo confusão entre elas.

---

### Erro Tipo 4 — Classes bem aprendidas (referência positiva)

`curado_pronto` (F1=0,8487) e `melhora_pronto` (F1=0,8219) são bem aprendidas. Isso faz sentido:

- **curado_pronto** (47,9% do teste) tem volume suficiente para aprendizagem robusta e padrão clínico coerente: exames leves, sem progressão de gravidade
- **melhora_pronto** é dominante no HSL (61,5% dos casos) — o cliente HSL aprende bem este padrão e contribui para o modelo federado via FedNova

A P@3 da `melhora_pronto` (0,60) confirma: a knowledge base recupera casos similares com boa precisão para esta classe.

---

## 4. Implicações para o Desenvolvimento

### 4.1 O limiar de 10 dias é o principal ponto de pressão

A fronteira entre `melhora_internado_breve` e `melhora_internado_grave` é administrativa. Duas opções metodológicas a discutir com a orientadora:

**Opção A — Fusão das classes**: unir `melhora_internado_breve` e `melhora_internado_grave` em uma única classe `melhora_internado`. Reduz de 5 para 4 classes, elimina o erro Tipo 3, e provavelmente aumenta o F1 macro. Custo: perde a granularidade de gravidade.

**Opção B — Manter e documentar**: manter as 5 classes e documentar explicitamente que o Erro Tipo 3 é inerente ao critério de 10 dias, não uma falha do modelo. A banca pode aceitar esse argumento se a limitação estiver clara no texto.

### 4.2 curado_internado requer atenção especial na defesa

Com F1≈0,07, qualquer avaliador que olhar as métricas por classe questionará esta classe. A resposta está pronta: 28 casos no teste implica ~196 no treino (0,8% das amostras) — nenhum modelo aprende bem com essa representação. Opções:

- Documentar como limitação do dataset (não do modelo)
- Avaliar se a classe pode ser fundida com `curado_pronto` (ambas são alta; diferem apenas se houve internação)
- Propor como trabalho futuro: coleta de mais dados desta classe

### 4.3 melhora_internado_grave é o erro clinicamente mais grave

O Erro Tipo 1 é o que mais importa para validação clínica futura. Independentemente de qual ação for tomada nas Opções A ou B da seção 4.1, a análise do trade-off entre false negatives de `melhora_internado_grave` e o custo de false positives (pacientes internados desnecessariamente) deve constar do texto da defesa.

---

## 5. Ações Pendentes

### A — Extrair matriz de confusão do Exp 15 (best round 79)

O checkpoint do Exp 15 está no banco PostgreSQL (training_id=5, round=79, sha256=4b38dc9d5617). Para gerar a matriz de confusão real:

```python
# Script a ser executado no desktop com o banco ativo
from src.mosaicfl.core.evaluation import evaluate_model
from infrastructure.shared.checkpoint_store import get_checkpoint_store

store = get_checkpoint_store(FL_DB_URL)
model, meta = store.load(training_id=5)
# ... avaliar no test_loader e gerar confusion_matrix
```

Isso fornecerá os números exatos que a análise atual estimou por proxy.

### B — Definir janela temporal da predição com a orientadora

O modelo usa o histórico completo da internação. Em uso real, a predição ocorre na admissão ou ao final do 1º/3º dia. Essa decisão muda o frame clínico dos erros: erros de predição na admissão são mais graves do que erros no dia 5 (quando há mais informação).

### C — Avaliar fusão de classes (decisão clínica)

Decidir com a orientadora se:
1. `curado_internado` pode ser fundida com `curado_pronto`
2. `melhora_internado_breve` e `melhora_internado_grave` podem ser fundidas

Isso não é uma decisão técnica — é uma decisão clínica sobre o que o sistema precisa discriminar para ser útil.

---

## 6. Resumo para a Defesa

| Erro | Frequência estimada | Impacto clínico | Causa raiz | Solução proposta |
|---|---|---|---|---|
| `melhora_internado_grave` → `curado_pronto` | ~20% dos graves (Exp 12; Exp 15 pendente) | **Máximo** — alta indevida | Padrão clínico heterogêneo; janela temporal indefinida | Definir janela de predição; considerar fusão com `melhora_internado_breve` |
| `curado_internado` não predita | ~93% dos 28 casos | Moderado | Sub-representação estrutural (0,8% do dataset) | Fundir com `curado_pronto` OU documentar como limitação do dataset |
| `melhora_internado_breve` ↔ `melhora_internado_grave` | Alta (F1≈0,35–0,46) | Moderado — ambas requerem internação | Fronteira administrativa de 10 dias sem correlato clínico | Documentar como limitação; avaliar fusão |
| `curado_pronto` e `melhora_pronto` | Baixa (F1≈0,82–0,85) | Baixo | Classes bem representadas e com padrão coerente | — (referência positiva) |
