Explicar:
1 - Caracterização do Dataset FAPESP: Detalhando a coleta, o desbalanceamento de classes entre hospitais e o pré-processamento de tokens

2 - Arquitetura do Modelo SimplifiedBEHRT: Explicando a fusão demográfica (late fusion) e o embedding de tempo relativo

3 - Protocolo de Aprendizado Federado: Detalhando o ciclo de vida de uma rodada, a agregação via FedNova e os mecanismos de recuperação de falhas (SuperLink)

4 - Fundamentação Teórica Detalhada: Os documentos citam FedProx, BEHRT e RAG
. Para ganhar volume e profundidade, você precisará de seções explicando a matemática por trás do termo proximal (μ) do FedProx e a arquitetura interna de Self-Attention do Transformer usado no SimplifiedBEHRT

5 - Justificativa Clínica dos Hiperparâmetros: Por que escolher 128 como max_seq_len? Como os analitos do FAPESP foram selecionados clinicamente?

6 - Análise Qualitativa do RAG: Os documentos trazem métricas (Precision@3), mas para ocupar espaço, precisaríamos de exemplos reais de "justificativas" geradas pelo RAG para casos específicos (estudos de caso)

7 - Diagramas e Fluxogramas: O README menciona diagramas Mermaid
. Eu posso descrever esses fluxos textualmente para que você os converta em imagens, detalhando cada etapa do pipeline desde o SQL até a predição

8 - Discussão sobre LGPD e Segurança: Detalhar o processo de HMAC-SHA256 para pseudonimização e como o secret de ID do paciente é gerenciado localmente

9 - Profundidade Teórica: Detalhes sobre a matemática do FedProx (especialmente o papel do termo proximal μ) e do FedNova, além da mecânica de Self-Attention do Transformer no BEHRT

10 - Racional Clínico: Por que o max_seq_len foi fixado em 128 e qual a importância biológica dos analitos selecionados (como PCR, D-dímero e Ferritina) para a predição de COVID-19

11 - Estudos de Caso do RAG: Exemplos qualitativos de como o sistema recupera perfis similares para justificar uma predição de "melhora_internado_grave", por exemplo

12 - Diagramas de Engenharia: Detalhamento visual do pipeline que vai desde a query SQL no PostgreSQL até a geração do recurso RiskAssessment no padrão FHIR R4



---

# Respostas — baseadas exclusivamente no código-fonte (com citações arquivo:linha)

> **Critério adotado**: toda afirmação é suportada por uma linha de código ou docstring do repositório. Lacunas que requerem dados externos (métricas de runtime, literatura clínica) são sinalizadas explicitamente como **DADO NECESSÁRIO**.

---

## 1 · Caracterização do Dataset FAPESP

### Origem e escopo

O sistema consome a base **FAPESP COVID-19 Data Sharing/BR** via PostgreSQL. A query principal está em `src/mosaicfl/core/preprocessor.py` (`_SQL_ATENDIMENTOS`, linhas 347–380) e une quatro tabelas:

```
clinical.attendances         — admissões e tipo de atendimento
clinical.patients            — sexo e ano de nascimento
metrics.clinical_outcomes    — desfecho e data de saída
metrics.exam_records         — analito, valor, classificação e data do exame
```

### Hospitais utilizáveis

O docstring de `SequencePipeline` (linhas 482–487) documenta:

- **HSL** e **BPSP**: únicos hospitais com `attendance_id` vinculando exames a atendimentos.
- **HEI**: 0 % de vinculação.
- **HFL / HCSP**: sem exames vinculados a atendimentos.

A query filtra explicitamente: `a.hospital_id IN ('HSL', 'BPSP')` (linha 373).

### Critérios de exclusão

`WHERE co.outcome_class NOT IN (2, 3, 4)` (linha 369):

| Código | Significado | Motivo da exclusão |
|--------|-------------|-------------------|
| 2 | Alta administrativa | Saída burocrática sem relação com evolução clínica |
| 3 | Transferência | Desfecho clínico final desconhecido |
| 4 | Em atendimento | Dado censurado — desfecho ainda não ocorreu |

Adicionalmente: `(co.outcome_at - a.attended_at) >= 0` (linha 374) exclui registros com datas inconsistentes.

### Classes de desfecho (5 classes)

Definidas em `_map_outcome()` (linhas 285–310), cruzando `outcome_class × attendance_type × duration_days`:

| Classe | Label | Critério |
|--------|-------|---------|
| 0 | `curado_pronto` | outcome 0, não-internado (ambulatorial / pronto / externo) |
| 1 | `curado_internado` | outcome 0, internado (qualquer duração) |
| 2 | `melhora_pronto` | outcome 1, não-internado |
| 3 | `melhora_internado_breve` | outcome 1, internado, ≤ 10 dias |
| 4 | `melhora_internado_grave` | outcome 1, internado, > 10 dias |

### Tokenização

Cada linha do resultado SQL representa um exame. O token é composto por (`preprocessor.py`, linhas 330–344):

```
token = f"{analyte}_{classification}"   # ex: LEUCOCITOS_HIGH, PCR_NORMAL
```

Quando `classification == "NO_REF"` (sem intervalo de referência cadastrado), o token é apenas o nome do analito. Tokens especiais reservados (linhas 542–543):

```
PAD = 0  |  UNK = 1  |  CLS = 2
```

O token `<CLS>` é inserido pelo modelo, não pelo pipeline de dados.

### Construção do vocabulário

O vocabulário é construído globalmente com todos os hospitais para garantir consistência entre clientes (`build_per_hospital`, linhas 656–659):

```python
# top (max_vocab_size − 3) tokens por frequência absoluta
available = self.max_vocab_size - len(self._SPECIAL)  # 10.000 - 3 = 9.997
top_tokens = token_series.value_counts().index[:available].tolist()
```

(`preprocessor.py`, linhas 727–729). **Não há seleção clínica manual de analitos** — o critério é puramente frequência na base.

### Divisão dos dados

`dataloaders.py`, linhas 136–150: por hospital, com gerador determinístico (`RANDOM_SEED + cid`):

```
70 % treino  |  10 % validação  |  10 % calibração (temperature scaling)  |  10 % teste global
```

O conjunto de calibração (`cal_loader`) é usado **exclusivamente** para Temperature Scaling e Isotonic Calibration — nunca exposto ao loop federado (`dataloaders.py`, linha 89).

### Tratamento do desbalanceamento

`client.py`, linhas 69–91: pesos de classe calculados localmente por cliente:

```python
weight_i = total / (n_classes × count_i)    # inversamente proporcional à freq.
weights.clamp(max=15.0)                       # teto para evitar explosão de gradiente
```

Comentário na linha 86: "peso 47 no BPSP causava explosão de gradiente" — evidência de desbalanceamento severo nesse hospital.

### DADO NECESSÁRIO

- **Contagens absolutas** de pacientes e atendimentos por hospital (HSL, BPSP) — não estão no código, requerem execução de `SELECT COUNT(*) ...` contra a base de produção.
- **Distribuição real de classes** por hospital (necessária para quantificar o desbalanceamento inter-hospital no texto).
- **Período de coleta** dos dados (não está no código; o código usa `_FAPESP_REF_YEAR = 2021` como âncora para cálculo de idade, sugerindo dados de 2020–2021).

---

## 2 · Arquitetura do Modelo SimplifiedBEHRT

Fonte principal: `src/mosaicfl/core/model.py` e `src/mosaicfl/core/config.py`.

### Hiperparâmetros fixos (`ModelConfig`, `config.py` linhas 57–74)

```python
vocab_size  = 10.000     embed_dim  = 64
max_seq_len = 128        num_layers = 2
num_heads   = 4          ff_dim     = 128
num_classes = 5          dropout    = 0.1
```

### Componentes em ordem de forward pass

**1. Token Embedding** (`model.py`, linha 143)
```python
nn.Embedding(vocab_size=10000, embed_dim=64, padding_idx=0)
```
Mapeia índice do vocabulário → vetor de 64 dimensões. Índice 0 (PAD) recebe embedding zero.

**2. DiaRelativoEmbedding** (`model.py`, linhas 31–53)
```python
nn.Embedding(max_dia + 2, d_model, padding_idx=0)
# max_dia = 60 (model.py, linha 28)
```
Captura a posição temporal do exame dentro do episódio de internação. O dia relativo é o número de dias desde a admissão (`attended_at`). Deslocamento +1: índice 0 = padding (token CLS), índice 1 = dia 0 (admissão), índice 2 = dia 1, ..., índice 61 = dia ≥ 60. Valores acima de 60 são clampados (`line 53: clamp(0, max_dia + 1)`). O embedding de dia é **somado** ao embedding de token (linha 227):

```python
emb = emb + self.dia_embedding(dia_relativo)
```

**3. PositionalEncoding sinusoidal** (`model.py`, linhas 56–67)
```python
pe[:, 0::2] = sin(position × exp(−2i × log(10000) / d_model))
pe[:, 1::2] = cos(position × exp(−2i × log(10000) / d_model))
```
Encoding posicional não aprendível, adicionado ao embedding após o DiaRelativoEmbedding.

**4. Token CLS learnable** (`model.py`, linhas 141–142)
```python
self.cls_token = nn.Parameter(torch.empty(1, 1, MODEL_CFG.embed_dim))
nn.init.trunc_normal_(self.cls_token, std=0.02)
```
Prefixado à sequência (linha 231–234). Recebe `dia_relativo = 0` (padding_idx), portanto seu embedding temporal é zero.

**5. BEHRTEncoderLayer × 2** (`model.py`, linhas 70–124)

Substitui `nn.TransformerEncoderLayer` para expor pesos de atenção. Cada camada:
- Multi-Head Self-Attention (4 cabeças, `batch_first=True`)
- `need_weights=True, average_attn_weights=False` → shape `(batch, 4, seq, seq)` por camada
- Feed-Forward: `Linear(64, 128) → ReLU → Dropout → Linear(128, 64)`
- LayerNorm pós-atenção e pós-FF (Pre-LN não; é Post-LN)
- Residual connections em ambos os sub-blocos

**6. Pooling CLS** (`model.py`, linhas 249–253)
```python
pooled = out[:, 0]   # vetor da posição 0 (CLS) após o encoder
```
Alternativa não usada em produção: masked mean pooling sobre tokens não-PAD.

**7. Pre-classifier** (`model.py`, linhas 160–163)
```python
nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
```

**8. Late fusion demográfica** (`model.py`, linhas 257–261)
```python
if demographics is not None:
    pooled = torch.cat([pooled, demographics], dim=-1)
# demographics: (batch, 2) → [age_norm, sex_binary]
```
`age_norm = (2021 - birth_year) / 100.0`, clampado [0.0, 1.0]; `sex_binary = 1.0 se M, 0.0 se F`.

O Transformer aprende representação da sequência de exames sem interferência demográfica. O classifier pondera ambas as fontes de forma independente (comentário `model.py` linha 259).

**9. Classifier head** (`model.py`, linhas 167–173)
```python
nn.Linear(embed_dim + demo_dim, 64) → nn.ReLU() → nn.Dropout(0.1) → nn.Linear(64, 5)
```
`demo_dim = 0` quando dados demográficos não estão disponíveis (comportamento original); `demo_dim = 2` no modo com late fusion.

### DADO NECESSÁRIO

- Comparação arquitetural com o BEHRT original (Li et al. 2020) requer leitura do artigo; o código apenas nomeia a inspiração.
- A justificativa para embed_dim=64 (vs. 768 do BERT original) está na docstring de `config.py` (linhas 6–10): limitação de hardware (Dell Inspiron 5402, sem GPU). Confirmar se isso deve ser explicitado no texto de defesa.

---

## 3 · Protocolo de Aprendizado Federado — Ciclo de Vida de uma Rodada

Fonte: `experiments/training/fl_core.py` e `src/mosaicfl/core/client.py`.

### Inicialização

`run_federated_learning_manual()` (linha 127):
1. Seeds fixadas: `random.seed`, `np.random.seed`, `torch.manual_seed`, `cudnn.deterministic=True` (`fl_core.py`, linhas 138–141)
2. Modelo global inicializado: `SimplifiedBEHRT(use_cls_token=True)` com pesos aleatórios
3. Treinamento registrado no `CheckpointStore` via `register_training()` (linha 173)

### Por rodada (`for round_num in range(1, n_rounds + 1)`)

**Fase cliente (sequencial, sem Ray):**
Para cada hospital (cliente):

1. `FedProxClient(cid, train_loader, val_loader)` inicializado com pesos do modelo global
2. `client.set_parameters(global_params)`:
   - Carrega state_dict completo (treináveis + buffers) via `load_state_dict(strict=False)`
   - Armazena cópia dos pesos globais para o termo proximal
3. `client.fit(global_params, config={"current_round": round_num})`:
   - `local_epochs = 1` (linha 91 de `config.py`)
   - Loss por batch: `L = CrossEntropy(weighted) + (μ/2)·‖w_local − w_global‖²`, μ=0.1
   - Gradient clipping: `clip_grad_norm_(max_norm=1.0)` antes do `optimizer.step()`
   - τ_i: contador de batches processados (passos efetivos)
4. Retorna: `(pesos_atualizados, num_samples, {loss, tau, grad_norm})`

**Fase servidor (agregação):**

Se `use_fednova=True` (padrão, `config.py` linha 98):
```python
τ_eff = Σ p_i · τ_i                              # fl_core.py linha 78
w_{t+1}[k] = w_t[k] + τ_eff · Σ p_i · (w_i[k] − w_t[k]) / max(τ_i, 1)
```
(`fl_core.py`, linhas 78–87)

Se `use_fednova=False`:
```python
w_{t+1}[k] = Σ p_i · w_i[k]    # FedAvg ponderado por num_samples
```
(`fl_core.py`, linhas 38–52)

**Avaliação global:**
`evaluate_global_model(global_model, test_loader)` — acurácia e loss no conjunto de teste global (`fl_core.py`, linhas 92–112).

**Checkpoint (melhor rodada):**
Se `acc_global > best_accuracy`:
```python
checkpoint_store.save(round_num, state_dict, vocab, accuracy, loss, training_id)
```
(`fl_core.py`, linha 237). O checkpoint é salvo no PostgreSQL via `CheckpointStore`.

**Critério de convergência:**
- Warm-up: convergência só avaliada após rodada `MIN_ROUNDS = 20` (`config.py` linha 82)
- Δaccuracy < `CONVERGENCE_THRESHOLD = 0.005` por `CONVERGENCE_PATIENCE = 3` rodadas consecutivas
- Máximo de `NUM_ROUNDS = 120` rodadas

**Ao final do treinamento:**
- `checkpoint_store.complete_training(...)` atualiza o registro com `n_rounds_done`, `best_round`, `best_accuracy`, `converged`
- `checkpoint_store.load_best()` restaura o melhor checkpoint
- Calibração: Temperature Scaling + Isotonic Calibration (OvR) sobre `cal_loader`

### Mecanismo de recuperação de falhas (RoundDispatcher)

`infrastructure/mosaicfl_scheduler/round_training_fl_dispatcher.py`:
- O `RoundDispatcher.dispatch_round()` faz poll via HTTP GET em `/metrics/round/{n}` no servidor Flower
- Backoff exponencial: início em 5 s, dobra a cada tentativa, teto em 60 s, `max_wait = 600 s`
- HTTP 404: rodada ainda não concluída, aguarda. HTTP 200: métricas disponíveis.
- A convergência é verificada via `ConvergenceTracker` que replay o histórico completo a cada chamada

**Nota sobre "SuperLink"**: O termo SuperLink refere-se ao componente do Flower (framework) responsável pela coordenação central em arquiteturas de FL de larga escala. No código deste projeto, o modo de simulação usa `flwr.simulation.start_simulation()` (`fl_core.py`, linha 443). O mecanismo de recuperação concreto implementado é o `RoundDispatcher` com backoff exponencial. **Não há referência a SuperLink no código do repositório** — confirmar se o texto da defesa descreve o Flower SuperLink como mecanismo externo de infraestrutura ou se deve referenciar o `RoundDispatcher` implementado.

---

## 4 e 9 · Fundamentação Teórica: FedProx, FedNova e Self-Attention

### FedProx — termo proximal

Implementado em `client.py`, linhas 93–100:

```python
def _proximal_loss(self, loss, proximal_mu):
    proximal_term = 0.0
    for local_w, global_w in zip(self.model.parameters(), self.global_params):
        proximal_term += torch.norm(local_w - global_w, p=2) ** 2
    return loss + (proximal_mu / 2) * proximal_term
```

Formulação matemática (diretamente derivável do código):

```
L_FedProx(w) = L_CE(w) + (μ/2) · ‖w − w*‖²₂
```

onde `w` são os pesos locais após o update, `w*` são os pesos globais recebidos do servidor, e `μ = 0.1` (`config.py`, linha 84).

O comentário na linha 84 de `config.py` documenta a razão do valor:
> "aumentado de 0.01 → 0.1 (Exp 7): reduz drift não-IID (Li et al. 2020)"

O termo proximal penaliza a distância entre pesos locais e globais, impedindo que clientes com datasets muito diferentes (regime non-IID) divirjam excessivamente do modelo global durante o treinamento local.

### FedNova — normalização por passos efetivos

Implementado em `fl_core.py`, linhas 55–89, com referência explícita no docstring:

> "Wang et al. 2020 — 'Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization'"

Formulação extraída diretamente do código (linhas 78–87):

```
τ_eff = Σ_i  p_i · τ_i             # média ponderada dos passos efetivos

Δ_i = (w_i − w_global) / τ_i       # update normalizado do cliente i

w_{t+1} = w_global + τ_eff · Σ_i  p_i · Δ_i
```

onde `p_i = n_i / N_total` (fração de amostras do cliente i) e `τ_i` é o número de batches processados localmente (contador `tau` em `client.py`, linha 131: `tau += 1` por batch).

**Motivação (docstring, `fl_core.py`, linha 63):** "Clientes com mais dados (mais batches por rodada) têm seus updates normalizados por τ_i antes de agregar, equalizando a contribuição independente do volume local."

### Self-Attention no BEHRTEncoderLayer

Implementado em `model.py`, linhas 83–124:

```python
self.self_attn = nn.MultiheadAttention(
    embed_dim=64, num_heads=4, dropout=0.1, batch_first=True
)
```

A chamada (linhas 110–115):
```python
attn_out, attn_weights = self.self_attn(
    src, src, src,                    # Q = K = V = src (self-attention)
    key_padding_mask=src_key_padding_mask,
    need_weights=True,
    average_attn_weights=False,       # shape: (batch, 4_heads, seq, seq)
)
```

Formulação matemática padrão do Transformer (Vaswani et al. 2017), que o código implementa via `nn.MultiheadAttention`:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Q = X · W_Q ∈ ℝ^{L×d_k}
K = X · W_K ∈ ℝ^{L×d_k}
V = X · W_V ∈ ℝ^{L×d_v}

d_k = embed_dim / num_heads = 64 / 4 = 16

MultiHead(Q, K, V) = Concat(head_1, ..., head_4) · W_O
head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

O `average_attn_weights=False` (linha 114) preserva os pesos por cabeça individualmente para uso no `BEHRTPatternExtractor` (`interpretability.py`) — permite análise de quais analitos o modelo foca em cada cabeça.

---

## 5 e 10 · Justificativa de Hiperparâmetros

### max_seq_len = 128

**Justificativa técnica (código):**

A query SQL usa `ROW_NUMBER() OVER (PARTITION BY attendance_id ORDER BY dia_relativo, analyte)` e filtra `WHERE _rn <= :max_seq_len` (`preprocessor.py`, linhas 353–377). Ou seja, os primeiros 128 exames — em ordem cronológica por dia relativo, com desempate pelo nome do analito — são retidos por atendimento.

**Justificativa de hardware (código):**

`config.py`, linha 61 e tabela nas linhas 14–22: o valor é calibrado para rodar no Dell Inspiron 5402 (i7-1165G7, 16 GB RAM, sem GPU dedicada). Valores maiores aumentam consumo de memória na dimensão de atenção (O(L²)).

**DADO NECESSÁRIO — justificativa clínica:**

O código **não** documenta por que 128 é suficiente do ponto de vista clínico. Para o texto da defesa, é necessário citar literatura que estabeleça a densidade típica de exames laboratoriais em internações por COVID-19 (ex.: número médio de exames/dia, duração mediana de internação). Uma estimativa: para 10 dias de internação com ~8-12 exames/dia, 128 cobre a maioria dos casos — mas esse cálculo requer dados da própria base FAPESP para ser verificado.

### Seleção de analitos

**O que o código diz:** os analitos não são selecionados clinicamente. O vocabulário é construído por frequência absoluta na base (`preprocessor.py`, linha 728: `token_series.value_counts().index[:available]`). Os tokens mais frequentes entram automaticamente.

**DADO NECESSÁRIO:**

- Os analitos mais frequentes no vocabulário resultante (PCR, D-dímero, Ferritina etc.) precisam ser listados a partir do `standard_vocab.json` ou de uma query `SELECT analyte, COUNT(*) FROM metrics.exam_records GROUP BY analyte ORDER BY 2 DESC` — não disponível no código.
- A importância clínica desses analitos para COVID-19 **não está documentada no código** e requer fundamentação bibliográfica externa (ex.: Huang et al. 2020 para PCR; Tang et al. 2020 para D-dímero; Mehta et al. 2020 para ferritina e síndrome de hiperinflamação).

---

## 6 e 11 · Análise Qualitativa do RAG

### Arquitetura implementada

`src/mosaicfl/core/rag.py`:

- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (variável `RUNTIME_CFG.embedding_model`, `config.py` linha 107)
- **LLM para geração**: `distilgpt2` (variável `RUNTIME_CFG.llm_model`, `config.py` linha 108)
- **Backend de vetores**: `_PostgreSQLStore` (pgvector, tabela `knowledge.clinical_profiles`) quando `FL_DB_URL` configurado; `_InMemoryStore` (similaridade de cosseno via numpy) em experimentos sem banco
- **Recuperação**: distância de cosseno, top-k=3 por padrão (`config.py` linha 95)
- **Geração**: prompt estruturado com casos recuperados + predição do modelo + tokens do paciente; `max_new_tokens=64` (`config.py` linha 96); `temperature=0.7`, `do_sample=True`
- **Detecção de alucinação** (`rag.py`, linha 236): `probability < 0.6 AND "certeza" in justification.lower()` — heurística simples

### Métricas de precisão (Precision@k)

Calculada em `experiments/training/rag.py`, linhas 18–75:

```python
Precision@k = (casos recuperados com mesmo desfecho que ground_truth) / (k × n_queries)
```

**DADO NECESSÁRIO:** Os valores numéricos de Precision@3 por classe não foram localizados nos arquivos JSON disponíveis em `experiments/data/`. Os arquivos `rag_*.json` contêm o campo `precision_metrics` mas os mais recentes com dados reais precisam ser identificados na execução real.

### Estado atual dos experimentos de RAG (análise honesta)

O arquivo `experiments/data/rag_20260615_223046.json` foi inspecionado. **Não é adequado para uso como estudo de caso qualitativo** pelos seguintes motivos:

1. **Base de conhecimento corrompida**: os textos armazenados na knowledge base contêm artefatos de tokenização (a palavra "adulto" interpolada entre cada caractere do texto). Exemplo do arquivo:
   ```
   "adultoPadultoaadultocadultoiadultoeadultonadultotadulto..."
   ```
   Causa provável: o sentence-transformer foi aplicado sobre textos que já eram tokens individuais do BEHRT (ex.: `<MASK>`, `<UNK>`, `<CLS>`, `<PAD>`), fazendo a faixa etária "adulto" ser inserida pelo método `build_knowledge_base()` (`rag.py`, linha 169) sobre tokens especiais ao invés de texto clínico real.

2. **Geração incoerente**: a justificativa do distilgpt2 no arquivo (`rag_20260615_223046.json`) é:
   ```
   "o conseira de conseira de dão mândulo..."
   ```
   O distilgpt2 é um modelo de linguagem de propósito geral treinado em inglês; sem fine-tuning em português ou domínio clínico, a geração em português é incoerente.

3. **Conclusão**: os experimentos de RAG realizados até a data de análise (2026-06-29) **não produziram exemplos qualitativos válidos** que possam ser usados como estudos de caso no texto da defesa.

### DADO NECESSÁRIO

Para que este item seja coberto adequadamente no texto, é necessário um dos seguintes:
- **Opção A**: Executar o pipeline RAG com dados reais e base de conhecimento construída a partir de textos clínicos coerentes (não tokens especiais), e capturar os JSONs resultantes.
- **Opção B**: Construir exemplos hipotéticos baseados na estrutura do sistema (indicando explicitamente que são ilustrativos), usando os desfechos reais do sistema e os analitos do vocabulário.
- **Opção C**: Limitar a discussão qualitativa do RAG à descrição da arquitetura e das métricas quantitativas obtidas, sendo honesto sobre as limitações do modelo generativo atual (distilgpt2 sem fine-tuning).

---

## 7 e 12 · Pipeline de Engenharia: do SQL ao FHIR

### Pipeline completo (baseado no código)

```
[PostgreSQL]
     │
     │  _SQL_ATENDIMENTOS (preprocessor.py:347-380)
     │  JOIN: attendances × patients × clinical_outcomes × exam_records
     │  WHERE: hospital_id IN ('HSL','BPSP'), outcome_class NOT IN (2,3,4)
     │  ROW_NUMBER per attendance_id ORDER BY dia_relativo, analyte
     ▼
[SequencePipeline._load_dataframe()]  → DataFrame bruto
     │
     ▼
[_build_vocab()]
     │  token = f"{analyte}_{classification}" | analyte (se NO_REF)
     │  top 9.997 tokens por frequência global
     ▼
[_build_tensors()]
     │  sequences: LongTensor (N, 128) — token_ids com PAD=0
     │  labels:    LongTensor (N,)     — classes 0..4 via _map_outcome()
     │  demo:      FloatTensor (N, 2)  — [age_norm, sex_binary]
     │  dia_rels:  LongTensor (N, 128) — dia relativo deslocado +1
     ▼
[prepare_dataloaders_from_db()]
     │  Split por hospital: 70/10/10/10
     │  DataLoader com Generator(seed=RANDOM_SEED+cid) — determinístico
     ▼
[run_federated_learning_manual()]   ← loop de rodadas
     │
     ├─► [FedProxClient.fit()]  ← por cliente/hospital
     │       loss = CrossEntropy(weighted) + (μ/2)‖w_local − w_global‖²
     │       gradient clipping (max_norm=1.0)
     │       retorna: (pesos, n_samples, {loss, tau, grad_norm})
     │
     ├─► [aggregate_fednova() | aggregate_fedavg()]
     │       w_{t+1} = w_global + τ_eff · Σ p_i · (w_i − w_global) / τ_i
     │
     ├─► [evaluate_global_model()]  → accuracy, loss no test_loader
     │
     └─► [CheckpointStore.save()]   → PostgreSQL (se melhor rodada)
     │
     ▼
[Calibração pós-treinamento]
     │  TemperatureScaler.fit(model, cal_loader)
     │  IsotonicCalibrator.fit(model, cal_loader)
     ▼
[InferenceEngine]  (inference_engine.py)
     │  Carrega checkpoint + vocab do PostgreSQL
     │  records_to_tokens():
     │    1. normalize(exam_name) → knowledge.term_dictionary (canonical)
     │    2. value vs refs → knowledge.analyte_references (HIGH/NORMAL/LOW/NO_REF)
     │    3. _make_token(canonical, classification, token_mode)
     │    4. vocab.get(token, UNK=1)
     │  predict_proba(): MC Dropout (50 amostras) → mean/std por classe
     ▼
[POST /api/exams/ingest]  (routers/prediction.py)
     │  _pid_to_internal(patient_id) → HMAC-SHA256 (pseudonimização)
     │  Upsert paciente + exames no SQLite local
     │  proba = engine.predict_proba(history_records)
     │  Persiste risk_score + prediction
     │  PatientExport → ClinicalPath JSON
     │
     ▼
[FHIR RiskAssessment]
     fhir_output = InferenceOutput(predictions, model_round, temperature, ece, ...)
     fhir_ra = state._fhir_exporter.to_risk_assessment(fhir_output)
```

O recurso FHIR R4 `RiskAssessment` é gerado por `state._fhir_exporter.to_risk_assessment()` (linha 153 de `prediction.py`). **DADO NECESSÁRIO**: o arquivo `state.py` ou o módulo exporter de FHIR não foi inspecionado; a estrutura exata do RiskAssessment (campos, codings SNOMED/LOINC) requer leitura de `infrastructure/mosaicfl_api/state.py`.

---

## 8 · LGPD e Segurança

### Pseudonimização via HMAC-SHA256

`security.py`, linhas 38–44:

```python
def _pid_to_internal(raw_patient_id: str) -> str:
    if not _PID_SECRET:
        return raw_patient_id
    return hmac.new(
        _PID_SECRET.encode(),
        raw_patient_id.encode(),
        hashlib.sha256
    ).hexdigest()
```

O `_PID_SECRET` é lido da variável de ambiente `FL_PATIENT_ID_SECRET` (`security.py`, linha 21). **Não é compartilhado com o servidor central** — cada instância de hospital gerencia seu próprio secret localmente. Isso garante que o servidor central nunca receba o `patient_id` real, apenas o hash HMAC-SHA256, que é irreversível sem o secret local.

Âncora legal: o comentário `# (LGPD Art. 13 §4º)` na linha 38 documenta o artigo que fundamenta a pseudonimização.

### Autenticação

`security.py`, linhas 47–65:

- **JWT** via `FL_JWT_SECRET` (HMAC, HS256) ou `FL_JWT_PUBLIC_KEY_FILE` (RSA, RS256/RS512)
- **API Key** via header `X-API-Key`
- Modo desenvolvimento: `FL_AUTH_REQUIRED=false` desativa autenticação

### Rate Limiting (janela deslizante, sem dependências externas)

`security.py`, linhas 70–98:

```python
_api_limiter    = _SlidingWindowLimiter(max_calls=120, window_seconds=60.0)
_ingest_limiter = _SlidingWindowLimiter(max_calls=30,  window_seconds=60.0)
```

Aplicado por IP em todos os endpoints (`_rate_check`).

### Auditoria

`audit.log_access()` é chamado em todos os endpoints de predição (`prediction.py`, linha 166 e 203). Os campos logados incluem `patient_id_hash` (nunca o ID real), `exam_count`, `risk_score`.

### Dados nunca saem do hospital

No protocolo de FL implementado, **apenas os pesos do modelo** (`state_dict`) trafegam entre cliente e servidor. Os dados brutos (`train_loader`, exames, `patient_id`) permanecem no banco local do hospital. Isso é garantido pela interface `FedProxClient` (linha 147 de `client.py`):

```python
return self.get_parameters(config), total_samples, metrics
# get_parameters: apenas numpy arrays dos tensores do modelo
```

---

## Dados Necessários para Complementar o Texto

Os seguintes itens **não podem ser obtidos do código-fonte** e requerem coleta específica:

| # | Item | Como obter |
|---|------|-----------|
| A | Contagem de pacientes e atendimentos por hospital (HSL, BPSP) | `SELECT hospital_id, COUNT(DISTINCT attendance_id) FROM clinical.attendances GROUP BY 1` |
| B | Distribuição real das 5 classes por hospital | `SELECT hospital_id, outcome_class, COUNT(*) FROM metrics.clinical_outcomes JOIN clinical.attendances USING(attendance_id) GROUP BY 1,2` |
| C | Top-20 analitos mais frequentes no vocabulário | Query de frequência em `metrics.exam_records` ou inspeção do `standard_vocab.json` |
| D | Valores reais de Precision@3 por classe (RAG) | Reexecutar `run_rag_pipeline()` com dados reais e base de conhecimento válida |
| E | Exemplos qualitativos de justificativas RAG | Idem D; exige também avaliar viabilidade do distilgpt2 vs. modelo alternativo em PT-BR |
| F | Estrutura completa do FHIR RiskAssessment gerado | Ler `infrastructure/mosaicfl_api/state.py` (não inspecionado nesta análise) |
| G | Justificativa clínica formal para max_seq_len=128 | Requere dados da base: distribuição de `COUNT(exames) por attendance_id` e literatura sobre densidade de exames em COVID-19 |
| H | Período exato de coleta dos dados FAPESP | Documentação da base FAPESP ou query `SELECT MIN(date), MAX(date) FROM metrics.exam_records` |
| ~~I~~ | ~~Métricas finais do treinamento~~ | **Disponível** — ver Apêndice de Métricas abaixo |

---

## Apêndice · Métricas do Treinamento Federado (Rodada 120)

Fonte: `experiments/logs/evaluation_round_120.json` — gerado automaticamente ao final do treinamento.

**Conjunto de teste:** 3.381 amostras (global, ambos os hospitais).  
**Temperatura calibrada (Temperature Scaling):** T = 1.058.

### Métricas globais

| Métrica | Pré-calibração | Pós-Temperature Scaling |
|---------|---------------|------------------------|
| Acurácia | 0.6744 | 0.6744 (inalterada) |
| Macro F1 | 0.484 | 0.484 (inalterado) |
| Macro AUC | 0.8015 | 0.8013 |
| ECE | **0.0935** | 0.1086 (piorou) |
| MCE | 0.2545 | 0.2875 |

**Nota**: o Temperature Scaling (T=1.058) não melhorou a calibração neste caso — ECE aumentou de 0.0935 para 0.1086. Isso indica que o modelo já estava razoavelmente calibrado antes da temperatura, ou que o conjunto de calibração é insuficiente para ajustar T de forma eficaz.

### Métricas por classe (pré-calibração)

| Classe | Support (n) | % do teste | AUC-ROC | F1 | Precision | Recall |
|--------|------------|-----------|---------|-----|-----------|--------|
| curado_pronto | 1.620 | 47,9 % | 0.8762 | 0.8146 | 0.7695 | 0.8654 |
| curado_internado | 28 | 0,8 % | 0.5713 | 0.0323 | 0.0294 | 0.0357 |
| melhora_pronto | 321 | 9,5 % | 0.9553 | 0.6606 | 0.7854 | 0.5701 |
| melhora_internado_breve | 1.074 | 31,8 % | 0.8108 | 0.5819 | 0.6413 | 0.5326 |
| melhora_internado_grave | 338 | 10,0 % | 0.7936 | 0.3306 | 0.3050 | 0.3609 |

**Observações diretas dos dados:**

1. **Desbalanceamento severo confirmado**: `curado_internado` tem apenas 28 amostras no teste (0,8 % do conjunto), resultando em F1=0.03 — o modelo praticamente não aprende essa classe. A matrix de confusão confirma: apenas 1 de 28 classificado corretamente.

2. **Classes com bom desempenho**: `curado_pronto` (AUC=0.876, F1=0.815) e `melhora_pronto` (AUC=0.955, F1=0.661) têm desempenho razoável.

3. **Classes clinicamente críticas com desempenho moderado**: `melhora_internado_grave` (F1=0.33) é a classe de maior severidade clínica e tem o pior desempenho entre as classes com suporte adequado — 67 dos 338 casos são classificados como `curado_pronto` (erro clínico relevante).

### Matriz de confusão (pré-calibração)

```
                      Predito →
Real ↓          c_pronto  c_intern  m_pronto  m_int_breve  m_int_grave
curado_pronto    [1402       10       23         119          66]
curado_intern    [  12        1        2          10           3]
melhora_pronto   [  71       10      183          49           8]
m_int_breve      [ 270        9       22         572         201]
m_int_grave      [  67        4        3         142         122]
```
