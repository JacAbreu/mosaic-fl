# Documentação — `POST /api/predict`

Endpoint principal para integração entre ClinicalPath e MOSAIC-FL.
Recebe exames de um paciente, processa via BEHRT + RAG e retorna
score de risco com justificativa clínica — sem persistir dados.

---

## Localização no código

`infrastructure/mosaicfl_api/service.py` — linha 326  
`infrastructure/mosaicfl_api/inference_engine.py` — lógica de inferência

---

## Estado atual

### Request

```http
POST /api/predict
Content-Type: application/json
Authorization: Bearer {token}
```

```json
{
  "patient_id": "PAC-001",
  "exams": [
    {
      "exam_name": "hemoglobina",
      "date": "2026-01-03",
      "value": 10.2,
      "phase": "IN",
      "ref_low": 12.0,
      "ref_high": 16.0,
      "origin": null,
      "exam_group": null,
      "value_text": null,
      "unit": "g/dL",
      "attendance_id": null
    }
  ]
}
```

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `patient_id` | string | sim | Identificador do paciente no sistema do hospital. Não é armazenado pelo MOSAIC-FL — usado apenas para devolver na resposta |
| `exams` | array | sim | Lista de exames laboratoriais. Mínimo 1 item |
| `exams[].exam_name` | string | sim | Nome do analito (ex: `hemoglobina`, `leucocitos`, `pcr`) |
| `exams[].date` | date (YYYY-MM-DD) | sim | Data de coleta do exame |
| `exams[].value` | float | sim | Valor numérico do resultado |
| `exams[].phase` | string | não | Fase clínica do atendimento: `AB` (ambulatorial), `EX` (externo), `IN` (internado), `OBITO` (óbito), `P_ALTA` (pós-alta). Padrão: `IN` |
| `exams[].ref_low` | float | não | Limite inferior da faixa de referência do analito. Padrão: `0.0` |
| `exams[].ref_high` | float | não | Limite superior da faixa de referência do analito. Padrão: `0.0` |
| `exams[].unit` | string | não | Unidade de medida (ex: `g/dL`, `mm3`) |
| `exams[].attendance_id` | string | não | ID do atendimento no EHR do hospital (não usado na inferência atual) |
| `exams[].origin` | string | não | Hospital de origem do exame (não usado na inferência atual) |
| `exams[].exam_group` | string | não | Grupo do exame (ex: `HEMOGRAMA`) (não usado na inferência atual) |
| `exams[].value_text` | string | não | Resultado textual quando não numérico (não usado na inferência atual) |

### Response (estado atual)

```json
{
  "patient_id": "PAC-001",
  "risk_score": 0.4200,
  "risk_date": "2026-06-11"
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `patient_id` | string | Devolvido sem modificação — MOSAIC-FL não armazena |
| `risk_score` | float [0.0–1.0] | **Atenção: valor com problema (ver seção abaixo)** |
| `risk_date` | date | Data da predição (data do servidor no momento da chamada) |

---

## Problemas identificados no estado atual

### Problema 1 — Tokenização incompatível com o treinamento

O `InferenceEngine` tokeniza os exames usando **MD5 do nome** do analito:

```python
def exam_name_to_token(name: str) -> int:
    digest = int(hashlib.md5(name.upper().encode()).hexdigest(), 16)
    return (digest % (_VOCAB_SIZE - 2)) + 1
```

O modelo BEHRT foi treinado pelo `SequencePipeline` com tokens no formato
`{analito}_{bucket}`, onde o bucket é calculado comparando o valor com `ref_low`
e `ref_high`:

```
hemoglobina_baixo   (value < ref_low)
hemoglobina_normal  (ref_low ≤ value ≤ ref_high)
hemoglobina_alto    (value > ref_high)
```

Os dois vocabulários são incompatíveis. O `InferenceEngine` envia tokens que
o modelo nunca viu durante o treino, produzindo predições sem significado clínico.

**Correção necessária:** usar o mesmo `SequencePipeline._make_token(analito, valor, ref_low, ref_high)`
na inferência, ou carregar o vocabulário salvo no checkpoint.

### Problema 2 — `risk_score` retorna a probabilidade da classe errada

O modelo tem **5 classes de duração de internação**:

| Classe | Faixa | Interpretação |
|---|---|---|
| 0 | 1–3 dias | internação curta |
| 1 | 4–7 dias | internação média |
| 2 | 8–14 dias | internação longa |
| 3 | 15–30 dias | internação muito longa |
| 4 | > 30 dias | internação prolongada |

O código atual retorna `probs[0, 1]` — a probabilidade da **classe 1 (média)**,
não um score de risco. Dias com maior risco clínico correspondem às classes 2, 3
e 4.

**Correção necessária:** definir o score de risco como probabilidade acumulada
das classes de maior duração:

```python
# risco = P(longa) + P(muito longa) + P(prolongada)
risk_score = float(probs[0, 2] + probs[0, 3] + probs[0, 4])
```

---

## Como os dados são recuperados e processados

```
ClinicalPath envia exames
        ↓
service.py: _to_record() converte ExamInput → ExamRecord
        ↓
InferenceEngine.predict(records)
  ├── records_to_tokens(): ordena por data, tokeniza, padding até seq_len=128
  ├── torch.tensor([tokens]) → forward pass no SimplifiedBEHRT
  ├── F.softmax(logits) → probabilidades das 5 classes
  └── retorna float (hoje: probs[0,1] — deve ser corrigido)
        ↓
PredictResponse devolvida ao ClinicalPath
```

O `patient_id` é incluído na resposta mas **não é armazenado**. Nenhum dado
de paciente persiste no servidor MOSAIC-FL neste endpoint.

---

## Evolução proposta — integração com RAG

### Motivação

Um número isolado (`risk_score: 0.42`) tem valor clínico limitado. O médico
precisa entender **por que** o modelo classificou o paciente naquela faixa.
O RAG já existe no MOSAIC-FL e tem exatamente esse papel: buscar padrões
similares no banco vetorial e gerar uma justificativa em linguagem natural.

### Response proposta (com RAG)

```json
{
  "patient_id": "PAC-001",
  "risk_score": 0.42,
  "risk_date": "2026-06-11",
  "predicted_class": 2,
  "predicted_label": "longa (8–14 dias)",
  "duration_probabilities": {
    "curta_1_3d":        0.08,
    "media_4_7d":        0.21,
    "longa_8_14d":       0.36,
    "muito_longa_15_30d":0.27,
    "prolongada_30d_mais":0.08
  },
  "justification": "Padrão compatível com internação longa: hemoglobina persistentemente baixa (10.2 g/dL) associada a leucocitose (14.800/mm³) nos primeiros dias de internação. Padrão similar identificado em pacientes com complicações inflamatórias.",
  "similar_profiles_found": 3
}
```

| Campo | Descrição |
|---|---|
| `risk_score` | P(classe 2 + 3 + 4) — probabilidade de internação longa ou mais |
| `predicted_class` | Classe de maior probabilidade (0–4) |
| `predicted_label` | Descrição legível da classe predita |
| `duration_probabilities` | Distribuição completa — permite ao ClinicalPath exibir gráfico de probabilidades |
| `justification` | Texto gerado pelo RAG (DistilGPT-2 + contexto do ChromaDB) explicando o padrão encontrado |
| `similar_profiles_found` | Quantos perfis similares foram recuperados do banco vetorial |

### Por que o RAG deve ser a camada central

O RAG não apenas gera texto — ele **afunila** o conhecimento:

1. O BEHRT extrai o padrão temporal dos exames e produz a distribuição de probabilidades
2. O `BEHRTPatternExtractor` identifica os tokens mais relevantes (via pesos de atenção)
3. O ChromaDB busca perfis clínicos similares no banco vetorial (`knowledge.clinical_profiles`)
4. O DistilGPT-2 sintetiza o contexto recuperado em linguagem natural

O resultado não é apenas "risco 0.42" — é "este padrão de exames se assemelha a
pacientes que ficaram 8–14 dias internados, com os seguintes marcadores relevantes".
Isso transforma o MOSAIC-FL de um classificador numérico em um **sistema de apoio
à decisão clínica interpretável**.

---

## Premissa de deploy (privacidade e FL)

O `POST /api/predict` é compatível com os princípios do aprendizado federado
**desde que** o `mosaicfl_api` rode dentro da infraestrutura do próprio hospital:

```
[Hospital A]
  ClinicalPath → POST /api/predict → mosaicfl_api (local) → BEHRT + RAG
                                                                   ↑
                                          modelo global (pesos chegaram via FL,
                                          mas a inferência é local — dados do
                                          paciente nunca saem do hospital)
```

Se o `mosaicfl_api` rodar em servidor centralizado externo ao hospital,
os exames do paciente sairiam do perímetro hospitalar — violando tanto
os princípios do FL quanto a LGPD. Essa premissa de deploy local precisa
ser explicitada na documentação de produção.

---

*MOSAIC-FL — MBA Big Data & IA, ICMC/USP, 2026-06-11*
