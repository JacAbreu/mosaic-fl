# Racional das Classes de Prognóstico do MOSAIC-FL

Este documento reconstrói, a partir do histórico real do repositório (commits + docs
existentes), como as 5 classes atuais (`curado_pronto`, `curado_internado`,
`melhora_pronto`, `melhora_internado_breve`, `melhora_internado_grave`) foram definidas.

**Aviso importante, pra não confundir expectativa com o que aconteceu de fato:** as
classes atuais **não vieram de uma exploração de dados (clustering, análise estatística
etc.)** — foram uma decisão de desenho clínico/teórico. O script de clustering que você
está construindo agora (`avaliacao-dados-scripts/exploracao-dados-fapesp-dependencias-novas.py`)
é a primeira vez que se tenta uma descoberta de classes orientada a dados neste projeto.
Isso não invalida as classes atuais, só significa que não há um "mesmo processo" pra
reproduzir — o que existe é a REGRA determinística que define as classes, e os dados
que ela produz.

## Linha do tempo (rastreada via `git log`)

### Esquema 1 — 2026-06-11 (obsoleto)

5 classes de **duração de internação pura**: 1–3, 4–7, 8–14, 15–30, >30 dias.
Documentado em `docs/documentacao_etapas_legadas.md` (seção "Endpoint POST /api/predict").

**Problema:** só cobria pacientes internados — atendimentos ambulatoriais/pronto-socorro
ficavam de fora do esquema de classes inteiramente.

### Esquema 2 — 2026-06-24/25 (atual)

Commit `6193d45` ("Redefini as labels para ter abrangência também para atendimentos
feitos no pronto socorro e nas internações, classificando em curado/melhora e se foi
breve ou grave"), autoria da própria autora do projeto.

Implementado em `src/mosaicfl/core/preprocessor/outcomes.py`, função `_map_outcome()`,
cruzando 3 dimensões:

| Dimensão | Fonte | Valores |
|---|---|---|
| `outcome_class` | `metrics.clinical_outcomes.outcome_class` | 0=curado, 1=melhora (2-6 excluídos: censurado/UTI/óbito, raros/ausentes nos dados FAPESP) |
| `attendance_type` | `clinical.attendances.attendance_type` | internado ou não |
| `duration_days` | `clinical_outcomes.outcome_at − attendances.attended_at` | só relevante quando internado |

Resultado:

| Índice | Classe | Critério |
|---|---|---|
| 0 | `curado_pronto` | outcome=0, não internado |
| 1 | `curado_internado` | outcome=0, internado (qualquer duração) |
| 2 | `melhora_pronto` | outcome=1, não internado |
| 3 | `melhora_internado_breve` | outcome=1, internado, ≤10 dias |
| 4 | `melhora_internado_grave` | outcome=1, internado, >10 dias |

**Motivação documentada (commit + docstring da função):** ampliar a cobertura pra incluir
atendimentos não-internados, e distinguir gravidade (curado vs. melhora) e tempo de
recuperação (breve vs. grave) quando há internação.

**Ponto em aberto, sem justificativa registrada em lugar nenhum do repositório:** o
limiar de 10 dias entre `melhora_internado_breve`/`melhora_internado_grave` está
**fixo no código** (`return 3 if duration_days <= 10 else 4`,
`src/mosaicfl/core/preprocessor/outcomes.py:34`) — não há evidência de que tenha sido
derivado de uma análise estatística da distribuição real de `duration_days` (ex.:
mediana, percentil, ponto de corte ótimo). É um valor de bom senso clínico, não
validado empiricamente no próprio projeto. **Isso é uma oportunidade concreta pra sua
exploração atual** — ver se 10 dias é de fato um corte que separa bem os dados, ou se
outro valor discriminaria melhor.

## Achado empírico posterior (não usado para desenhar as classes, só descoberto ao treinar)

`docs/Metodologia MOSAIC-FL - Final Corrigido.md`, seção 2.4, documenta um desbalanceamento
severo e heterogêneo entre hospitais nessas 5 classes:

| Classe | BPSP (N=28.599) | HSL (N=5.174) |
|---|---|---|
| curado_pronto | 55,6% | 1,3% |
| curado_internado | 1,1% | 0,9% |
| melhora_pronto | **0,4%** | **61,5%** |
| melhora_internado_breve | 33,0% | 24,7% |
| melhora_internado_grave | 9,9% | 11,6% |

`melhora_pronto` é praticamente inexistente no BPSP mas domina o HSL — a principal fonte
de heterogeneidade (non-IID) do projeto, motivo direto de usar FedProx/FedNova em vez de
FedAvg puro. Esse padrão foi descoberto DEPOIS de definir as classes (analisando
resultados de treino), não foi usado para escolhê-las.

## Por que isso importa pra sua exploração de hoje

Sua pergunta original era sobre privacidade — classes menos desbalanceadas/heterogêneas
entre hospitais tendem a exigir menos rodadas de federação pra convergir e a sofrer menos
com ruído de DP (uma classe rara é a primeira a "sumir" sob ruído — ver
[[project_dp_ruido_seletivo]] na memória). Se o clustering k-prototypes encontrar uma
divisão que:
- distribua os pacientes de forma mais equilibrada entre os hospitais, e/ou
- separe bem `outcome_class` (a variável que de fato importa clinicamente),

isso seria uma alternativa concreta e testável às 5 classes atuais — com a vantagem de
ter sido descoberta a partir dos dados, não só de uma regra teórica.
