# Documentação de Etapas Legadas — MOSAIC-FL

> Este arquivo consolida documentos que registram decisões de design, análises e planejamentos
> de etapas anteriores do projeto. O conteúdo não reflete o estado atual do repositório,
> mas preserva o raciocínio e a trajetória de desenvolvimento para fins históricos e acadêmicos.
>
> **Não usar como referência operacional.** Para o estado atual, consulte `README.md`,
> `docs/FLUXO_APRENDIZADO_FEDERADO.md` e `docs/TODO.md`.

---

## Índice

1. [Análise dos Dados FAPESP e Impacto no Treinamento BEHRT](#1-análise-dos-dados-fapesp-e-impacto-no-treinamento-behrt-2026-06-08)
2. [Roadmap de Conformidade LGPD](#2-roadmap-de-conformidade-lgpd-2026-06-07)
3. [Guia de Configuração do SGBD](#3-guia-de-configuração-do-sgbd-versão-inicial)
4. [Documentação do Endpoint POST /api/predict](#4-documentação-do-endpoint-post-apipredict-2026-06-11)
5. [Adequações ClinicalPath ↔ MOSAIC-FL](#5-adequações-clinicalpath--mosaic-fl-2026-06-11)

---

## 1. Análise dos Dados FAPESP e Impacto no Treinamento BEHRT (2026-06-08)

> **Por que é legado:** escrito logo após o carregamento inicial dos 5 hospitais. Todas as
> decisões pendentes (seções 6.1–6.5) foram tomadas e implementadas nas sessões seguintes.
> O label scheme proposto aqui foi completamente redesenhado na sessão de 2026-06-24/25.
> Preservado como registro do processo de descoberta dos dados FAPESP.

**Data:** 2026-06-08
**Hospitais analisados:** HSL, BPSP, HEI, HCSP, HFL — Janeiro/Agosto 2021
**Status:** todos os 5 hospitais carregados no PostgreSQL (migrations 001-005 aplicadas)

---

### 1.1 O que foi carregado (5 hospitais)

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

### 1.2 Distribuição de exames por paciente

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

#### Interpretação

A distribuição é **bimodal e altamente assimétrica**:

- **44,3% têm menos de 10 exames** (p10=1, p25=1). Isso não é ruído — reflete uma realidade
  do sistema de saúde: desigualdade de acesso. Pacientes com menos exames podem ser
  ambulatoriais, mas também podem ser pacientes que deterioraram rapidamente antes que
  mais exames fossem solicitados, ou pacientes de menor renda com menos acesso a
  investigação complementar. **Descartar esses pacientes seria enviesar o modelo para a
  população mais monitorada** — exatamente o perfil que menos precisa de predição clínica.

- **20,3% têm 128 ou mais exames**, chegando a 15.599 em um único paciente. Para esses,
  o `max_seq_len=128` atual trunca a sequência.

- **A mediana é 32 exames** — sequências com significado clínico real para a maioria
  dos pacientes.

---

### 1.3 Top 15 analitos (HSL)

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

Hemograma completo = 12 tokens distintos por coleta. **Decisão tomada:** tokenizar por analito
individual (`DE_ANALITO`), aceitando a redundância do hemograma na sequência.

---

### 1.4 Distribuição de outcomes (desfechos)

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

#### Distribuição completa (HSL + BPSP)

| classe | descrição | total |
|---|---|---|
| 0 | recuperado | 85.496 |
| 1 | melhorado | 51.349 |
| 2 | administrativo (Alta Administrativa / a pedido) | 320.311 |
| 3 | transferido | 646 |
| 4 | evasão | 1.175 |
| **5** | **UTI** — ausente | **0** |
| **6** | **óbito** — ausente | **0** |

#### Limitação crítica identificada

**Ausência confirmada de óbitos e UTI em toda a base FAPESP (versão Jan/2021).**
O arquivo de desfechos registra o desfecho de cada visita, não o desfecho final do paciente.
Pacientes que foram a óbito têm registros de "Alta" em atendimentos anteriores.

**Decisão tomada (2026-06-24/25):** excluir classes 2, 3, 4 (administrativo, transferência,
evasão). Usar apenas classes 0 e 1. Esquema de 5 labels cruzando outcome × attendance_type ×
duration_days implementado em `preprocessor.py:_map_outcome()`.

---

### 1.5 Decisões pendentes (resolvidas nas sessões seguintes)

| # | Questão | Resolução |
|---|---|---|
| 6.1 | Pacientes com poucos exames — descartar ou incluir? | Incluídos (duration_days ≥ 0) |
| 6.2 | Sequências longas (> 128 exames) — janela? | ROW_NUMBER() OVER capping em max_seq_len |
| 6.3 | Token por analito ou por grupo de exame? | Por analito — 649 tokens no vocab final |
| 6.4 | Label de treino dado ausência de UTI/óbito? | 5 classes cruzando outcome × tipo × duração |
| 6.5 | Vocabulário do zero ou pré-treinado? | Do zero com dados FAPESP (build_standard_vocab.py) |

---

## 2. Roadmap de Conformidade LGPD (2026-06-07)

> **Por que é legado:** escrito na sessão 1 do projeto, quando todos os itens LGPD estavam
> pendentes. Itens 1, 2, 4, 7 e parcialmente 3 foram implementados nas sessões 4 e 5.
> Preservado como registro do planejamento de conformidade.
> Para o estado atual de cada item, consulte `AVALIACAO_PROJETO.md`.

**Data:** 2026-06-07
**Status na data:** todos os itens pendentes (dados sintéticos, sem PII real)

### Resumo executivo (estado em 2026-06-07)

| Item | Artigo LGPD | Complexidade | Status na data |
|---|---|---|---|
| Trilha de auditoria | Art. 37 | Média | **Pendente → Implementado** (`audit.py`) |
| Differential Privacy nos pesos | Art. 46 | Alta | Pendente — no roadmap |
| Minimização de dados | Art. 6, III | Baixa | **Pendente → Parcialmente** (pipeline filtra por schema) |
| Pseudonimização de identificadores | Art. 13 §2 | Baixa | **Pendente → Implementado** (HMAC-SHA256 em `FL_PATIENT_ID_SECRET`) |
| Controle de consentimento | Art. 7, I | Alta | Pendente — Art. 7 IX (pesquisa com CEP) |
| Retenção e exclusão de dados | Art. 15/16 | Alta | Pendente — roadmap |
| Controle de acesso / autenticação | Art. 46 | Alta | **Pendente → Implementado** (JWT, TLS obrigatório) |
| Notificação de incidentes | Art. 48 | Média | Pendente — roadmap |

### 2.1 Trilha de Auditoria (Art. 37)

O logging operacional atual serve para observabilidade. Para LGPD, cada acesso a dado pessoal
precisa de um registro imutável e separado com: quem acessou, qual dado, com qual finalidade, quando.

**Implementado em:** `infrastructure/mosaicfl_api/audit.py` — `log_access()` com JSON
estruturado, `token_fingerprint` SHA-256[:12], `patient_id_hash` SHA-256[:16].

### 2.2 Differential Privacy nos Pesos (Art. 46)

Sem DP, pesos do modelo podem vazar informação via Model Inversion e Membership Inference attacks.

Mecanismo Gaussiano proposto para `FedProxClient.get_parameters()`:
```python
FL_DP_NOISE_MULTIPLIER=0.0   # 0 = desabilitado; 0.5–1.0 = proteção razoável
FL_DP_MAX_GRAD_NORM=1.0      # norma de clipping
```

**Status atual:** não implementado. Documentado em `TODO.md` como bloqueador para uso com dados reais.

### 2.3 Pseudonimização (Art. 13 §2)

**Implementado em:** HMAC-SHA256 do `patient_id` com `FL_PATIENT_ID_SECRET`.
Em `FL_ENV=production`, o processo não sobe sem o secret configurado.

### 2.4 Controle de Acesso (Art. 46)

**Implementado em:** JWT HS256/RS256 na API; TLS obrigatório via `EnvironmentError`
se `FL_TLS_CERT_DIR` ausente.

### 2.5 Consentimento (Art. 7)

Para pesquisa com aprovação CEP (Art. 7, IX), uso sem consentimento individual é permitido
desde que: dados pseudonimizados, finalidade limitada à pesquisa, ROPA documentado.

### 2.6 Retenção e Exclusão (Art. 15/16)

Pesos do modelo são função não-invertível de todos os dados. Pedido de exclusão individual
requer re-treino completo excluindo o paciente — computacionalmente caro. Documentar no ROPA.

### 2.7 Ordem de implementação recomendada (2026-06-07)

1. Pseudonimização — **feito**
2. Minimização de dados — **parcialmente feito**
3. Trilha de auditoria — **feito**
4. Differential Privacy — pendente
5. Consentimento — pendente (Art. 7 IX como base legal para pesquisa)
6. Retenção e exclusão — pendente
7. mTLS / RBAC — **parcialmente feito** (TLS + JWT; mTLS pendente)
8. Notificação de incidentes — pendente

---

## 3. Guia de Configuração do SGBD (versão inicial)

> **Por que é legado:** escrito antes da definição do schema FAPESP. Referencia variáveis de
> ambiente (`MOSAICFL_SOURCE_TYPE`, `MOSAICFL_DB_QUERY`) e tabelas (`prontuarios_covid`) que
> não existem no schema atual. O sistema atual usa exclusivamente PostgreSQL com o schema
> definido nas migrations 001-010. Para configuração atual, consulte `README.md` seção
> "Configuração de Banco de Dados" e `ambiente_simulacao.md`.

**Variáveis de ambiente ANTIGAS (não usar):**
```bash
# Obsoleto — estas variáveis não existem mais
export MOSAICFL_SOURCE_TYPE=postgresql
export MOSAICFL_DB_URL="postgresql://usuario:senha@localhost:5432/nome_do_banco"
export MOSAICFL_DB_QUERY="SELECT * FROM prontuarios_covid WHERE ano >= 2022"
```

**Variáveis de ambiente ATUAIS:**
```bash
# Correto — schema atual
export FL_DB_URL="postgresql://mosaicfl:senha@localhost:5432/mosaicfl"
export FL_HOSPITAL_ID=BPSP
export FL_ENV=production
```

A arquitetura de Strategy Pattern (`FileDataSource` / `DatabaseDataSource`) foi mantida em
`data_loader.py`, mas o schema de tabelas passou a ser fixo: `clinical.patients`,
`clinical.attendances`, `metrics.exam_records`, `metrics.clinical_outcomes`, conforme
definido pelas migrations em `alembic/versions/`.

---

## 4. Documentação do Endpoint POST /api/predict (2026-06-11)

> **Por que é legado:** escrito em 2026-06-11 com o esquema de 5 classes de **duração de
> internação** (1–3 dias, 4–7 dias, 8–14 dias, 15–30 dias, >30 dias). Esse esquema foi
> completamente substituído em 2026-06-24/25 pelo esquema de 5 classes cruzando outcome ×
> attendance_type × duration (curado_pronto, curado_internado, melhora_pronto,
> melhora_internado_breve, melhora_internado_grave).
> Os "Problemas identificados" desta seção foram tratados nas sessões seguintes.

**Data:** 2026-06-11
**Schema de labels na data (obsoleto):**

| Classe | Faixa | Interpretação |
|---|---|---|
| 0 | 1–3 dias | internação curta |
| 1 | 4–7 dias | internação média |
| 2 | 8–14 dias | internação longa |
| 3 | 15–30 dias | internação muito longa |
| 4 | > 30 dias | internação prolongada |

**Schema de labels atual (2026-06-25):**

| Classe | Critério |
|---|---|
| 0 — curado_pronto | outcome=0, não internado |
| 1 — curado_internado | outcome=0, internado |
| 2 — melhora_pronto | outcome=1, não internado |
| 3 — melhora_internado_breve | outcome=1, internado ≤10 dias |
| 4 — melhora_internado_grave | outcome=1, internado >10 dias |

### Request (ainda válido)

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
      "ref_high": 16.0
    }
  ]
}
```

### Problemas identificados (2026-06-11) e resoluções

**Problema 1 — Tokenização incompatível com o treinamento (RESOLVIDO)**

O `InferenceEngine` usava MD5 do nome do analito. O modelo era treinado com tokens
`{analito}_{classificação}` (HIGH/NORMAL/LOW/NO_REF). Os vocabulários eram incompatíveis.

**Resolução:** vocabulário canônico via `build_standard_vocab.py` carregado no checkpoint.
`InferenceEngine` carrega o vocab do checkpoint e reutiliza o mesmo `SequencePipeline._make_token()`.

**Problema 2 — `risk_score` retornava probabilidade da classe errada (RESOLVIDO)**

O código retornava `probs[0, 1]` (classe 1 = internação média). O score deveria ser
probabilidade acumulada das classes de maior risco.

**Resolução:** `risk_score` calculado como `sum(prob × linspace(0,1,n_classes))` em
`inference_engine.py`, compatível com o schema atual de 5 classes.

### Response atual (referência)

```json
{
  "patient_id": "PAC-001",
  "risk_score": 0.42,
  "risk_date": "2026-06-25",
  "predicted_class": 3,
  "predicted_label": "melhora_internado_breve",
  "class_probabilities": {
    "curado_pronto": 0.08,
    "curado_internado": 0.12,
    "melhora_pronto": 0.21,
    "melhora_internado_breve": 0.42,
    "melhora_internado_grave": 0.17
  },
  "uncertainty": { ... },
  "model_metadata": { "trained": true, "calibrated": true, "temperature": 1.098 }
}
```

---

## 5. Adequações ClinicalPath ↔ MOSAIC-FL (2026-06-11)

> **Por que é legado:** análise pontual realizada em 2026-06-11 comparando os arquivos
> gerados pelo `ClinicalPathExporter` com os arquivos reais do ClinicalPath v2.0.
> Os itens de alta prioridade foram tratados; os pendentes estão no `docs/TODO.md`
> (seção "ClinicalPath"). Preservado como registro da análise de compatibilidade.

**Data:** 2026-06-11

### Status dos arquivos por arquivo (na data)

| Arquivo | Gerado pelo MOSAIC-FL | Formato correto? |
|---|---|---|
| `exam-id.txt` | sim | ✅ idêntico |
| `timestamp_to_date.txt` | sim | ✅ idêntico |
| `node-inline-time.txt` | sim | ❌ coluna 3 errada |
| `node-inline-time-complete.txt` | sim | ❌ coluna 3 errada + colunas a mais |
| `time-metadata.txt` | sim | ⚠️ esquema de IDs incompatível |
| `network.txt` | **não** | ❌ arquivo ausente |
| `patient-metadata.txt` (raiz) | **não** | ❌ arquivo ausente |
| `list_exams.txt` (raiz) | **não** | ❌ arquivo ausente |

### 5.1 Coluna 3 nos arquivos node-inline-time — crítico

O ClinicalPath espera categoria do resultado em relação à referência:
`-2` (muito baixo) / `-1` (baixo) / `0` (normal) / `1` (alto) / `2` (muito alto).

O MOSAIC-FL colocava a fase clínica do atendimento (`OUTPATIENT=-2`, `HOSPITALIZED=0`).

**Correção implementada:**
```python
def _result_category(value: float, ref_low: float, ref_high: float) -> int:
    if ref_low == 0.0 and ref_high == 0.0:
        return 0
    if value < ref_low * 0.7:    return -2
    elif value < ref_low:         return -1
    elif value <= ref_high:       return  0
    elif value <= ref_high * 1.3: return  1
    else:                         return  2
```

### 5.2 `network.txt` ausente

Um arquivo por paciente listando todos os `exam_ids` presentes. Sem ele, o ClinicalPath
não carrega o paciente.

**Status:** pendente — ver `docs/TODO.md`, seção "ClinicalPath: implementar geração do `network.txt`".

```python
def _write_network(self, patient_dir, patient_id, exam_ids):
    lines = [f"{patient_id} {idx} 0" for idx in sorted(exam_ids.values())]
    (patient_dir / "network.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

### 5.3 `node-inline-time-complete.txt` — colunas a mais

O ClinicalPath espera 6 colunas; o MOSAIC-FL gerava 8 (adicionava `sex_ref_low sex_ref_high`).
**Corrigido** — exporter agora usa apenas as 6 colunas canônicas.

### 5.4 `time-metadata.txt` — esquema de IDs incompatível

O ClinicalPath usa IDs internos do grafo JGraphX (`exam_id × num_timestamps + timestamp_id`).
Não é possível determinar a fórmula exata sem o código-fonte Java (apenas `.jar` disponível).

**Status:** pendente — bloqueado por acesso ao código-fonte. Ver `docs/TODO.md`.

### 5.5 Arquivos raiz ausentes — menor prioridade

- `patient-metadata.txt`: arquivo único com todos os pacientes (`{patient_id} {sexo} {idade}`)
- `list_exams.txt`: catálogo global de analitos — requer autorização do Prof. Claudio Linhares
  para incluir `FL_PROB_*` (email enviado, pendente resposta)

### Priorização original (2026-06-11)

| Prioridade | Item | Status atual |
|---|---|---|
| 🔴 Alta | Coluna 3 (categoria de resultado) | ✅ Corrigido |
| 🔴 Alta | Gerar `network.txt` | Pendente no TODO |
| 🟡 Média | Colunas de `node-inline-time-complete` (6 em vez de 8) | ✅ Corrigido |
| 🟡 Média | `patient-metadata.txt` na raiz | Pendente |
| 🟠 Baixa | `time-metadata.txt` (IDs JGraphX) | Bloqueado (sem código-fonte .jar) |
| 🟠 Baixa | `list_exams.txt` com `FL_PROB_*` | Bloqueado (aguarda Prof. Claudio) |

---

*Arquivo gerado em 2026-06-25 consolidando documentos de etapas anteriores do MOSAIC-FL.*
*Autora: Jacqueline Abreu — MBA Big Data & IA, ICMC/USP*
