# Adequações necessárias — Integração MOSAIC-FL → ClinicalPath

Análise realizada comparando os arquivos gerados pelo `ClinicalPathExporter`
(`integration/clinical-path/exporter.py`) com os arquivos reais do ClinicalPath v2.0
(repositório `github.com/claudiogl/ClinicalPath`, ZIPs inspecionados em 2026-06-11).

---

## Contexto

O `ClinicalPathExporter` foi implementado com base numa interpretação do formato
esperado pelo ClinicalPath — não a partir da inspeção direta dos arquivos do repositório
real. A inspeção revelou incompatibilidades que impediriam o ClinicalPath de exibir
corretamente os dados exportados pelo MOSAIC-FL.

---

## Status atual por arquivo

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

---

## Detalhamento das inadequações

### 1. Coluna 3 em `node-inline-time.txt` e `node-inline-time-complete.txt` — crítico

**O que o ClinicalPath espera:**
A terceira coluna representa a **categoria do resultado do exame** em relação aos
valores de referência:

| Valor | Significado |
|---|---|
| `-2` | Muito baixo (very low) |
| `-1` | Baixo (low) |
| `0` | Normal |
| `1` | Alto (high) |
| `2` | Muito alto (very high) |

Exemplo real: `0 51 -2 8.4 13.5 17.5` → exame 0 (Hb), timestamp 51, **resultado muito baixo**, valor=8.4, ref_low=13.5, ref_high=17.5.

**O que o MOSAIC-FL coloca:**
O código atual usa `r.phase.status_code`, que é a **fase clínica do atendimento**
(`OUTPATIENT=-2`, `PRE_HOSPITAL=-1`, `HOSPITALIZED=0`, `POST_DISCHARGE=1`).

**Impacto:**
Todos os exames de um paciente internado (`HOSPITALIZED`) apareceriam como categoria
`0` (normal) no ClinicalPath, independentemente do valor real. A visualização seria
completamente enganosa.

**Correção necessária:**
Calcular a categoria a partir de `value`, `ref_low` e `ref_high`. O ClinicalPath original
usa a mediana da distribuição dos valores baixos/altos para definir os limites de
"muito baixo" e "muito alto" (conforme Figura 2 do artigo). Uma aproximação prática:

```python
def _result_category(value: float, ref_low: float, ref_high: float) -> int:
    if ref_low == 0.0 and ref_high == 0.0:
        return 0  # sem referência disponível — assume normal
    if value < ref_low * 0.7:    return -2
    elif value < ref_low:         return -1
    elif value <= ref_high:       return  0
    elif value <= ref_high * 1.3: return  1
    else:                         return  2
```

Para o `FL_RISK_SCORE`: acima de `ref_high=0.3` → categoria `1` ou `2`; abaixo → `0` ou `-1`.

---

### 2. `network.txt` ausente — necessário para carregar o paciente

**O que o ClinicalPath espera:**
Um arquivo por paciente listando todos os exam_ids presentes, no formato:
```
{patient_id} {exam_id} 0
```

Exemplo real (`Patients/103007/network.txt`):
```
103007 10 0
103007 1 0
103007 0 0
...
```

**O que o MOSAIC-FL faz:**
Não gera esse arquivo. Sem ele, o ClinicalPath provavelmente não carrega o paciente.

**Correção necessária:**
Adicionar `_write_network()` ao exporter, iterando sobre `exam_ids.values()`:
```python
def _write_network(self, patient_dir, patient_id, exam_ids):
    lines = [f"{patient_id} {idx} 0" for idx in sorted(exam_ids.values())]
    (patient_dir / "network.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

---

### 3. `node-inline-time-complete.txt` — colunas a mais

**O que o ClinicalPath espera:**
6 colunas: `exam_id timestamp categoria valor ref_low ref_high`

**O que o MOSAIC-FL gera:**
8 colunas: adiciona `sex_ref_low sex_ref_high` ao final.

**Impacto:**
Pode causar erro de parsing no ClinicalPath (depende de como o leitor Java trata
colunas extras — pode ignorar ou falhar silenciosamente).

**Correção necessária:**
Remover as duas últimas colunas, ou usar apenas o par de referência principal.

---

### 4. `time-metadata.txt` — esquema de IDs incompatível

**O que o ClinicalPath usa:**
IDs internos do grafo JGraphX, misturando dois tipos:
- IDs de timestamp puro (ex: `13 AB`, `14 AB`) para timestamps próximos ao início
- IDs de nó calculados como `exam_id × num_timestamps + timestamp_id` para outros casos

Exemplo (paciente 1553025, 26 exames, 71 timestamps):
- `16 AB` → timestamp_id = 16 (direto)
- `158 AB` → exam_id=2, timestamp=16: `2 × 71 + 16 = 158`

**O que o MOSAIC-FL gera:**
Índices de timestamp sequenciais simples (`0 AB`, `1 AB`, `2 AB`...).

**Impacto:**
As cores de fase clínica (internado=vermelho, ambulatorial=azul etc.) podem não
aparecer corretamente na linha do tempo, pois os IDs referenciariam nós errados no
grafo JGraphX.

**Correção necessária:**
Requer teste com o ClinicalPath.jar real para confirmar o comportamento. A fórmula
exata de mapeamento de nós não pode ser determinada sem o código-fonte Java
(apenas o `.jar` compilado está disponível no repositório).

---

### 5. Arquivos raiz ausentes — menor prioridade

**`patient-metadata.txt`** (diretório raiz do ClinicalPath):
```
{patient_id} {sexo} {idade}
```
Não é por paciente — é um arquivo único com todos os pacientes. O MOSAIC-FL não
o gera. Pode ser gerado pela API ao exportar.

**`list_exams.txt`** (diretório raiz):
Catálogo global de exames com categorias e nomes em PT/EN. É um arquivo fixo que
acompanha o ClinicalPath — provavelmente não precisa ser gerado pelo MOSAIC-FL,
apenas garantir que o vocabulário de analitos do MOSAIC-FL esteja mapeado nele.

---

## Priorização sugerida

| Prioridade | Item | Esforço |
|---|---|---|
| 🔴 Alta | Corrigir coluna 3 (categoria de resultado) | Médio — nova função no exporter |
| 🔴 Alta | Gerar `network.txt` | Baixo — 5 linhas no exporter |
| 🟡 Média | Corrigir colunas de `node-inline-time-complete` (6 em vez de 8) | Baixo |
| 🟡 Média | Gerar `patient-metadata.txt` na raiz via API | Baixo |
| 🟠 Baixa | Corrigir `time-metadata.txt` (IDs JGraphX) | Alto — requer teste com .jar real |
| 🟠 Baixa | Mapear analitos do MOSAIC-FL no `list_exams.txt` | Médio — trabalho manual de mapeamento |

---

## Observação sobre a integração bidirecional

O módulo de exportação é **unidirecional** — terminal de saída do MOSAIC-FL.
Nenhuma parte do projeto lê os arquivos gerados de volta. O ClinicalPath leria
esses arquivos de forma independente, a partir do diretório `FL_CLINICALPATH_OUTPUT`.

A integração no sentido MOSAIC-FL → ClinicalPath existe em código, mas **não foi
validada com o ClinicalPath real**. As inadequações acima foram identificadas por
inspeção estática dos formatos, não por execução da ferramenta.

---

*Análise: Jacqueline Abreu — MOSAIC-FL, MBA Big Data & IA, ICMC/USP, 2026-06-11*
