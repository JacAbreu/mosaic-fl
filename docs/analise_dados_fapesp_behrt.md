# Análise dos Dados FAPESP e Impacto no Treinamento BEHRT

**Data:** 2026-06-08  
**Hospitais analisados:** HSL, BPSP, HEI, HCSP, HFL — Janeiro/Agosto 2021  
**Status:** todos os 5 hospitais carregados no PostgreSQL (migrations 001-005 aplicadas)

---

## 1. O que foi carregado (5 hospitais)

| hospital | pacientes | exames | outcomes |
|---|---|---|---|
| HSL — Hospital Sírio-Libanês | 8.971 | 1.346.802 | 42.598 |
| BPSP — BP — A Beneficência Portuguesa | 39.000 | 5.838.999 | 217.157 |
| HEI — Hospital Einstein | 79.863 | 3.029.830 | — |
| HCSP — Hospital das Clínicas | 3.751 | 2.320.739 | — |
| HFL — Grupo Fleury | 470.967 | 17.097.334 | — |
| **TOTAL** | **602.552** | **29.633.704** | **259.755** |

Apenas HSL e BPSP fornecem arquivos de desfechos na base FAPESP. HEI, HCSP e HFL não têm.

---

## 2. Distribuição de exames por paciente

```
min=1  p10=1  p25=1  p50=32  p75=94  p90=313  p95=621  max=15.599  média=150
```

| faixa de exames | pacientes | % |
|---|---|---|
| < 10 | 3.978 | 44,3% |
| 10–31 | 431 | 4,8% |
| 32–63 | 1.609 | 17,9% |
| 64–127 | 1.131 | 12,6% |
| 128–255 | 725 | 8,1% |
| ≥ 256 | 1.097 | 12,2% |

### Interpretação

A distribuição é **bimodal e altamente assimétrica**:

- **44,3% têm menos de 10 exames** (p10=1, p25=1). Isso não é ruído — reflete uma realidade
  do sistema de saúde: desigualdade de acesso. Pacientes com menos exames podem ser
  ambulatoriais, mas também podem ser pacientes que deterioraram rapidamente antes que
  mais exames fossem solicitados, ou pacientes de menor renda com menos acesso a
  investigação complementar. **Descartar esses pacientes seria enviesar o modelo para a
  população mais monitorada** — exatamente o perfil que menos precisa de predição clínica.
  A abordagem correta é incluí-los e sinalizar a densidade de dados para o modelo.

- **20,3% têm 128 ou mais exames**, chegando a 15.599 em um único paciente. Para esses,
  o `max_seq_len=128` atual trunca a sequência e pode perder informação crítica do final
  do internamento (quando o agravamento costuma ocorrer).

- **A mediana é 32 exames** — sequências com significado clínico real para a maioria
  dos pacientes.

---

## 3. Top 15 analitos (HSL)

| analito | ocorrências |
|---|---|
| Creatinina | 51.321 |
| Hemoglobina | 31.921 |
| Hematócrito | 30.120 |
| Plaquetas | 29.868 |
| Leucócitos | 29.543 |
| Eritrócitos | 29.541 |
| VCM | 29.541 |
| RDW | 29.497 |
| Neutrófilos | 29.458 |
| Basófilos | 29.458 |
| Eosinófilos (%) | 29.458 |
| Neutrófilos (%) | 29.458 |
| Monócitos (%) | 29.458 |
| Linfócitos | 29.458 |
| Linfócitos (%) | 29.458 |

### Interpretação

Hemoglobina, Hematócrito, Plaquetas, Leucócitos, Eritrócitos, VCM, RDW, Neutrófilos, Basófilos,
Eosinófilos, Monócitos, Linfócitos aparecem todos ~29.458 vezes — são o **hemograma completo**,
solicitado em bloco toda vez. No vocabulário do BEHRT eles ocupam 12 tokens distintos mas
carregam informação redundante quando tratados individualmente.

**Decisão pendente:** tokenizar por `analyte` (DE_ANALITO) ou por `exam_group` (DE_EXAME)?
- Por analito: vocabulário mais rico, mas hemograma vira 12 tokens por coleta
- Por exam_group: sequência mais compacta ("HEMOGRAMA" = 1 token), perde granularidade

---

## 4. Distribuição de outcomes (desfechos)

| classe | texto | contagem |
|---|---|---|
| 0 | Alta médica curado | 61 |
| 0 | Alta médica inalterado | 145 |
| 0 | Alta por abandono | 39 |
| 1 | Alta médica melhorado | 12.277 |
| 2 | Alta a pedido | 210 |
| 2 | **Alta Administrativa** | **29.613** |
| 3 | Transferência inter-hospitalar (ambulância) | 15 |
| 3 | Transferência inter-hospitalar (transporte próprio) | 2 |
| 4 | Desistência do atendimento | 229 |
| 4 | Assistência domiciliar | 7 |
| 5 | — ausente — | — |
| 6 | — ausente — | — |

### Distribuição completa (HSL + BPSP — todos os desfechos carregados)

| classe | descrição | texto original | total |
|---|---|---|---|
| 0 | recuperado | Alta curado/Inalterado/Pronto Atend./etc. | 85.496 |
| 1 | melhorado | Alta melhorado / Alta médica melhorado | 51.349 |
| 2 | administrativo | Alta Administrativa / Alta a pedido | 320.311 |
| 3 | transferido | Transferência inter-hospitalar | 646 |
| 4 | evasão | Evadiu-se / Desistência / Assist. Domiciliar | 1.175 |
| **5** | **UTI** | **— ausente —** | **0** |
| **6** | **óbito** | **— ausente —** | **0** |

### Interpretação — limitação crítica do dataset

**Ausência confirmada de óbitos e UTI em toda a base FAPESP (versão Jan/2021).**

A função `classify_outcome` em `integration/fapesp/transforms.py` reconhece corretamente
os padrões `obit|morte|falec` (classe 6) e `uti|terapia intensiva` (classe 5). A ausência
não é um bug de classificação — nenhum dos textos de desfecho nos arquivos de HSL e BPSP
contém essas palavras. Todos os 458.977 registros são variações de "Alta" ou "Transferência".

**Por que isso acontece:**  
O arquivo de desfechos do FAPESP registra o desfecho de *cada visita ambulatorial ou
internação*, não o desfecho final do paciente. Pacientes que foram a óbito podem ter
registros de "Alta" em atendimentos anteriores. O óbito em si ou não foi incluído nesta
versão do dataset, ou está registrado em uma dimensão que não foi compartilhada pelos
hospitais no acordo de data sharing da FAPESP.

**Impacto no TCC:**  
O objetivo original de predizer mortalidade por COVID-19 não pode ser alcançado com esses
dados. Isso é uma **limitação real e relevante a documentar** — reflete o estado atual do
compartilhamento de dados clínicos no Brasil, inclusive em contexto de pandemia.

**Alternativas de label para o modelo:**
- **Binário clínico vs administrativo:** 0 = Alta clínica (classes 0+1), 1 = Alta
  Administrativa (classe 2). Distingue desfecho clínico de gestão de leito.
- **Binário internação vs pronto atendimento:** diferencia pacientes que foram internados
  dos que foram dispensados no pronto atendimento — proxy de gravidade.
- **Regressão de gravidade:** usar o número de exames, tempo de internação ou analitos
  como target contínuo em vez de classificação binária.
- **Buscar dados complementares:** repositório FAPESP pode ter versões mais completas com
  óbitos. Vale contato com os autores do dataset.

**b) Classe 2 domina com 70% (Alta Administrativa)**  
"Alta Administrativa" é desfecho processual (gestão de leitos, convênio), não clínico.
Qualquer que seja o label escolhido, essa classe precisa ser tratada explicitamente
(removida, colapsada ou usada como classe separada com peso reduzido).

---

## 5. O modelo BEHRT atual vs. o que os dados permitem

### Como está implementado hoje

O `SimplifiedBEHRT` em `src/mosaicfl/core/model.py` recebe sequências de tokens de tamanho
`max_seq_len=128`. Mas o pipeline atual em `run_experiments_v2.py` constrói a sequência assim:

```python
seq = [sintoma_encoded, exame_encoded, diagnostico_encoded]
# → 3 tokens reais + 125 tokens de <PAD>
```

Ou seja: **não há sequência temporal real**. Cada paciente é representado por 3 colunas
de uma única linha, não por múltiplos eventos ao longo do tempo. O `max_seq_len=128` está
sendo desperdiçado.

### O que os dados da FAPESP permitem

Com os dados carregados no banco, cada paciente tem múltiplos registros em `metrics.exam_records`
com `collection_date`. É possível construir sequências temporais reais:

```
Paciente X:
  2020-03-01 → Hemoglobina   → token "hemoglobina"
  2020-03-01 → PCR           → token "pcr"
  2020-03-03 → Creatinina    → token "creatinina"
  2020-03-05 → D-Dímero      → token "d-dimero"
  ...até 128 eventos
```

O `positional encoding` passa a ter significado clínico real — a posição na sequência
representa ordem temporal.

---

## 6. Decisões pendentes antes de construir o pipeline de treino

### 6.1 Tratamento de pacientes com poucos exames (acesso desigual à saúde)

**Contexto:** Exames de saúde em volume ainda não são acessíveis a toda a população.
Pacientes com poucos registros não são outliers a descartar — representam uma parcela
real e relevante do sistema de saúde. Descartá-los enviesaria o modelo para atender bem
apenas quem já tem acesso privilegiado ao sistema.

**Decisão:** Incluir todos os pacientes, mas **classificar a densidade de dados** e
sinalizar isso explicitamente para o modelo.

**Grupos de densidade propostos** (baseados na distribuição do HSL):

| grupo | faixa de exames | % HSL | interpretação clínica |
|---|---|---|---|
| `DENSITY_SPARSE` | < 10 | 44,3% | ambulatorial ou caso agudo rápido |
| `DENSITY_LOW` | 10–31 | 4,8% | internação curta |
| `DENSITY_MEDIUM` | 32–127 | 30,5% | internação típica |
| `DENSITY_HIGH` | ≥ 128 | 20,3% | internação prolongada ou crônico |

**Como sinalizar para o modelo — 3 opções:**

- Opção A: **Token especial no início da sequência** (mais simples)
  ```
  [DENSITY_SPARSE] → Hemoglobina → <PAD> × 126
  [DENSITY_HIGH]   → Hemoglobina → PCR → Creatinina → ... × 128
  ```
  O modelo aprende que o token inicial define o contexto da sequência.
  Alinhado com como o BEHRT original usa segment embeddings.

- Opção B: **Embedding de densidade somado ao embedding de posição**
  O grupo de densidade vira um vetor de tamanho `embed_dim=64` somado a cada
  posição, similar ao `age embedding` do BEHRT original. O modelo vê a densidade
  em todo token, não só no início.

- Opção C: **Feature auxiliar no classificador final**
  O grupo de densidade é concatenado ao vetor poolado antes da camada linear.
  Mais simples de implementar, mas o modelo não usa a informação durante a atenção.

**Recomendação:** Opção A para começar — altera menos o modelo, é interpretável,
e pode ser evoluída para Opção B se a avaliação mostrar necessidade.

**Como avaliar o impacto:**
Reportar métricas (F1, AUC) separadas por grupo de densidade. Se o modelo tiver
desempenho muito inferior em `DENSITY_SPARSE`, significa que está descriminando
pacientes com menos acesso — informação crítica para o TCC sob a perspectiva de
equidade em saúde.

### 6.2 Estratégia para sequências longas (> 128 exames)

**Problema:** 20,3% têm ≥ 128 exames; máximo de 15.599.  
**Decisão:** Como truncar?  
- Opção A: **Janela ancorada no fim** — últimos 128 exames antes do desfecho. Captura o
  momento do agravamento. Mais alinhado com o BEHRT original.
- Opção B: **Janela deslizante** — múltiplas janelas por paciente. Aumenta o dataset, mas
  cria correlação entre janelas do mesmo paciente. O split treino/validação precisa ser
  obrigatoriamente por paciente, não por janela.
- Opção C: **Aumentar `max_seq_len`** para 256 ou 512. Cobre 95% dos pacientes sem
  truncamento (p95=621, então 512 cobre ~90%). Custo: memória quadrática no self-attention.

### 6.3 Granularidade do token

**Problema:** Hemograma completo = 12 analitos solicitados juntos toda coleta.  
**Decisão:** Token por analito ou por grupo de exame?  
- Opção A: Token = `analyte` (DE_ANALITO) — vocabulário ~500 tokens únicos no HSL,
  sequências mais longas por coleta
- Opção B: Token = `exam_group` (DE_EXAME) — "HEMOGRAMA" em vez de 12 analitos,
  sequências mais compactas, perde granularidade numérica
- Opção C: Token composto — `exam_group:analyte` ou incluir faixa do valor numérico
  (ex: "hemoglobina:baixo", "hemoglobina:normal", "hemoglobina:alto")

### 6.4 Tratamento dos outcomes

**Problema:** Classes 5 e 6 ausentes em toda a base. Classe 2 (Alta Administrativa) domina
com 70%. O label de treino precisa ser redefinido.

**Decisão:** Como definir o label de treino dado que não há óbitos?

- Opção A: **Binário clínico vs administrativo**
  0 = Alta clínica (classes 0+1) — o paciente saiu bem por decisão médica
  1 = Alta Administrativa (classe 2) — saiu por pressão de leito/convênio, desfecho clínico incerto
  Clinicamente relevante: o modelo aprende a distinguir recuperação real de alta forçada.
  Distribuição: 136.845 (0) vs 320.311 (1) — desbalanceado mas tratável.

- Opção B: **Binário internação longa vs curta** (proxy de gravidade)
  Usar tempo entre `attended_at` e `outcome_date` como threshold (ex: > 7 dias = grave).
  Não depende do texto de desfecho. Usa dados que já temos.

- Opção C: **Multiclasse clínica** — remover classe 2 (administrativa), manter 0, 1, 3, 4.
  Distribuição muito desbalanceada (classe 1 domina). Mais difícil de treinar.

- Opção D: **Contatar os autores do dataset FAPESP** para verificar se existe versão com
  óbitos. O repositório USP/FAPESP pode ter dados mais completos.

**Recomendação:** Opção A no curto prazo (permite começar o treino agora) + Opção D em
paralelo (para o TCC final ter o label mais relevante clinicamente).

### 6.5 Vocabulário: construir do zero ou pré-treinar

**Problema:** O vocabulário atual é sintético (`sintoma`, `exame`, `diagnostico`).  
**Decisão:**  
- Opção A: Construir vocabulário dos analitos reais da FAPESP (5 hospitais). Vantagem:
  tokens têm significado clínico direto. Desvantagem: vocabulário específico deste dataset.
- Opção B: Usar CID-10 ou LOINC como vocabulário pré-definido. Vantagem: transferível para
  outros datasets. Desvantagem: requer mapeamento dos analitos para LOINC (trabalhoso).

### 6.6 ~~Verificar óbitos no BPSP~~ — CONCLUÍDO

**Resultado:** BPSP carregado (217.157 outcomes). Classes 5 e 6 ausentes em toda a base
FAPESP. Confirmado que é limitação do dataset, não bug de classificação (ver seção 4).

---

## 7. Design do pipeline de sequência com densidade

O pipeline que constrói os tensores de treino a partir do banco precisará:

```
Para cada patient_id:
  1. Buscar todos os exames ordenados por collection_date
  2. Calcular exam_count → mapear para grupo de densidade
  3. Tokenizar analyte (ou exam_group) → lista de índices
  4. Truncar para max_seq_len (janela ancorada no fim)
  5. Pad com <PAD>=0 até max_seq_len
  6. Prepend token de densidade: [DENSITY_SPARSE/LOW/MEDIUM/HIGH]
  7. Buscar outcome_class do paciente (pior outcome registrado)
  8. Emitir: (tensor de tokens, label, grupo_densidade)
```

O vocabulário terá tokens reservados:

```
0  → <PAD>
1  → <UNK>
2  → <MASK>
3  → <CLS>
4  → <DENSITY_SPARSE>
5  → <DENSITY_LOW>
6  → <DENSITY_MEDIUM>
7  → <DENSITY_HIGH>
8+ → analitos / exam_groups reais
```

Durante a avaliação, o campo `grupo_densidade` permite quebrar as métricas por grupo
e reportar equidade do modelo.

---

## 8. Próximos passos sugeridos (em ordem)

1. ~~**Carregar os 5 hospitais**~~ — ✅ **CONCLUÍDO** (602.552 pacientes, 29.6M exames)
2. ~~**Verificar óbitos no BPSP**~~ — ✅ **CONCLUÍDO** (ausentes em toda a base, ver seção 4)
3. **Decidir 6.4 (label)** → define o que o modelo vai aprender. Recomendação: Opção A
   (clínico vs administrativo) para começar + contato com autores do dataset FAPESP.
4. **Decidir 6.1 (grupos de densidade)** → confirmar os 4 grupos propostos.
5. **Decidir 6.3 (granularidade do token)** → analyte vs exam_group.
6. **Construir o pipeline de sequência** com tokens de densidade (seção 7).
7. **Decidir 6.2 (sequências longas)** → implementar janela ancorada no fim.
8. **Ajustar `config.py`** — `max_seq_len`, `vocab_size`, `num_classes` para valores reais.
9. **Treinar e avaliar** com HSL+BPSP (únicos com labels), reportar métricas por grupo de
   densidade para demonstrar equidade do modelo.

---

## 9. Referência rápida — arquivos relevantes

| o que | onde |
|---|---|
| Modelo BEHRT | `src/mosaicfl/core/model.py` |
| Config (seq_len, vocab_size) | `src/mosaicfl/core/config.py` |
| Pipeline de dados atual (sintético) | `src/mosaicfl/core/data_loader.py`, `preprocessor.py` |
| Loader FAPESP | `integration/fapesp/loader.py` |
| Extractors | `integration/fapesp/patients_extract.py`, `exams_extract.py`, `outcomes_extract.py` |
| Tabelas no banco | `clinical.patients`, `clinical.attendances`, `metrics.exam_records`, `metrics.clinical_outcomes` |
| Migrations | `alembic/versions/001` a `005` |
