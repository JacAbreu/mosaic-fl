# Comparativo: Metodologia Planejada (`Metodologia3_JacquelineAbreu.pdf`) vs. Execução Real

**Propósito:** o PDF em `Metodologia3_JacquelineAbreu.pdf` é o plano de metodologia submetido antes da execução do projeto. Este documento mapeia, item a item, onde a execução real (documentada em `docs/Linha_do_Tempo_MOSAIC-FL.md`, `docs/Sumario_Treinamento.md` e `docs/Sumario_Treinamento_Parte2.md`) confirma, diverge ou substitui o que foi planejado. Todo parâmetro técnico citado aqui como "confirmado" foi verificado diretamente no código nesta sessão (`src/mosaicfl/core/config.py`, `model.py`, `client.py`, `rag/`), não de memória. O objetivo é te dar uma lista de decisões a tomar sobre o texto da metodologia — eu não reescrevo o texto, só aponto onde ele não bate mais com o que foi feito.

---

## 1. Resumo executivo — o que precisa de decisão editorial

| # | Item | Situação |
|---|---|---|
| 1 | **5 "Experimentos" planejados** (padronização, equalizador, non-IID, RAG-Likert, eficiência) | **Nenhum foi executado na forma exata especificada.** O projeto seguiu por um caminho experimental diferente (T1–T16+, Bloco 1/2). Precisa de decisão: reescrever a seção 3.7 para descrever o que foi *de fato* feito, ou ainda executar (parte de) os 5 experimentos como planejados. |
| 2 | **Algoritmo de agregação**: FedProx μ=0,01 fixo → **FedNova** | Mudança de algoritmo bem documentada e justificada (non-IID severo, 5,5× de razão de volume BPSP/HSL). Precisa entrar na metodologia como decisão motivada, não como nota de rodapé. |
| 3 | **Escopo de hospitais**: base "particionada em subconjuntos" (implicitamente todos os 5) → **apenas 2 hospitais (BPSP, HSL)** | Só HSL e BPSP fornecem dados de desfecho no FAPESP COVID-19 Data Sharing/BR — os outros 3 (HEI, HCSP, HFL) têm exames mas não outcomes. Isso restringe o "efeito equalizador do FL" a uma rede de 2 nós, não N nós — precisa estar explícito no texto. |
| 4 | **Esquema de desfechos**: não especificado no plano (genérico, "conforme os desfechos da base") → **5 classes, redesenhadas em 2026-06-24/25 após identificar um label censurado** | Esse redesign foi o maior risco acadêmico identificado no projeto (penalidade −1,5 em avaliação formal). O plano não antecipa essa complexidade — precisa de uma seção própria. |
| 5 | **RAG**: ChromaDB + DistilGPT-2 → **PostgreSQL/pgvector + Ollama/gemma3:4b** (DistilGPT-2 relegado a fallback de teste) | Mudança de infraestrutura de armazenamento vetorial E de modelo de geração. Resolve uma limitação que o próprio plano já havia previsto (seção 3.9: "recomenda-se um LLM maior"). |
| 6 | **Avaliação do RAG**: Likert 1–5 por avaliador humano em 50 amostras → **Precision@3 (métrica de recuperação automática)** | São métricas de coisas diferentes: Likert mede qualidade da *justificativa gerada*; P@3 mede se os casos *recuperados* pertencem à classe certa. Nenhuma avaliação Likert foi feita até 2026-07-01 — isso deixa a hipótese de "redução de incerteza diagnóstica" sem validação qualitativa. |
| 7 | **RETAIN (Choi et al., 2016)** citado no plano como mecanismo de interpretabilidade → **nunca implementado** | A interpretabilidade real vem da atenção do próprio BEHRT (`BEHRTPatternExtractor`), não de um modelo RETAIN separado. Citação deveria ser removida ou o texto deveria esclarecer que a abordagem final é diferente. |
| 8 | **Calibração (Temperature Scaling → Isotônica)**: ausente do plano | Um dos achados mais ricos do projeto (isotônica supera temperature em 100% dos ~14 experimentos medidos) não tem nenhuma âncora na metodologia planejada — precisa de uma seção nova. |
| 9 | **Escopo de engenharia**: "protótipo" de pesquisa → **sistema com topologia de produção** (SuperLink, TLS obrigatório, LGPD audit trail, Docker/K8s, API REST, DP-FedAvg implementado) | O plano (seção 3.8) fala em disponibilizar "o código-fonte do protótipo". A execução foi muito além disso. Bom para a nota, mas o texto metodológico deveria justificar esse escopo ampliado (há uma instrução da própria autora registrada no TODO.md sobre isso). |
| 10 | **Métrica principal**: AUC-ROC (plano) → **F1 macro como critério de checkpoint, accuracy/AUC/ECE como métricas reportadas** | Mudança bem justificada (accuracy favorece a classe majoritária com 5 classes desbalanceadas), mas muda o que a seção 3.7 do plano define como "métrica principal". |

---

## 1.5 Categorização das divergências — nem toda diferença pesa igual

Antes de entrar item a item, vale separar as ~10 divergências acima em 3 categorias, porque elas têm implicação muito diferente para a defesa:

### Categoria A — Evolução justificada pelos dados (não é problema, é ponto forte se bem narrado)

FedNova no lugar de FedProx fixo, calibração isotônica, redesenho do esquema de labels, arquitetura de produção completa, `DiaRelativoEmbedding`. Em todos esses casos, o plano razoavelmente não podia antecipar a decisão sem já ter tocado nos dados reais — a mudança foi motivada por evidência coletada durante a própria execução. Isso é normal em pesquisa aplicada e, contado como "planejei X, os dados mostraram Y, mudei para Z, aqui está a evidência", costuma ser visto como maturidade metodológica, não como desvio do plano.

### Categoria B — Ajuste técnico sem consequência conceitual

`batch_size` (32→16), épocas locais (3→1), número de rodadas (50→120), frequência de avaliação (a cada 5 → a cada rodada). São mudanças de tunagem que não alteram a pergunta de pesquisa nem a arquitetura conceitual — precisam estar documentadas por rigor, mas não pedem justificativa extensa no texto.

### Categoria C — Lacuna real (isso sim precisa de decisão consciente)

Os **Experimentos 3 e 4 do plano nunca foram executados em nenhuma forma** (ver Seção 10). O Experimento 4 é o mais grave: ele é a validação empírica direta da hipótese central do trabalho ("RAG reduz a incerteza diagnóstica ao fundamentar predições em evidências recuperadas") — sem ele, essa parte da hipótese fica sem teste qualitativo, só com Precision@3 (uma métrica de recuperação, que mede se os casos recuperados pertencem à classe certa, não se a justificativa gerada é clinicamente sã e não-alucinada). Isso não é "o plano mudou" — é "uma parte do plano ficou sem ser feita", e precisa de uma decisão explícita: executar antes da defesa, ou declarar como limitação assumida conscientemente (diferente de uma lacuna não percebida).

---

## 2. Dataset — escopo real vs. planejado

**Plano (seção 3.2):** base FAPESP COVID-19 Data Sharing/BR, 5 instituições (HF1–HF5, identificadas no artigo original como Hospital Albert Einstein, laboratórios Fleury, Hospital Sírio-Libanês, Beneficência Portuguesa/SP e Hospital das Clínicas da FMUSP), ~595.000 pacientes, ~32,1 milhões de resultados laboratoriais brutos (reduzidos a ~10 milhões após a normalização do próprio ClinicalPath). O plano fala em "cada hospital (cliente) mantém seus prontuários localmente" e particiona a base "em subconjuntos... conforme as distribuições naturais presentes nos dados" — não restringe explicitamente a 2 instituições.

**Execução:** apenas **2 clientes — BPSP (Beneficência Portuguesa/SP) e HSL (Hospital Sírio-Libanês)** — de 5 instituições disponíveis na base. Confirmado em `docs/documentacao_etapas_legadas.md` (Linha do Tempo, Parte 2): dos 5 hospitais carregados (HSL, BPSP, HEI, HCSP, HFL), **só HSL e BPSP fornecem arquivos de desfecho** — os outros 3 têm ~22,4 milhões de exames combinados sem outcome associado, inutilizáveis para o problema de predição definido.

| | Plano | Execução |
|---|---|---|
| Instituições participantes do FL | Implícito: até 5 | **2** (BPSP, HSL) |
| Pacientes utilizados | ~595.000 (base completa) | BPSP: 39.000 \| HSL: 8.971 \| **total ~48.000** |
| Razão de volume entre clientes | Não quantificada no plano | **5,5×** (BPSP/HSL) — o dado central que motivou a troca de FedProx para FedNova |

**O que precisa entrar no texto:** uma justificativa explícita de que apenas 2 das 5 instituições do FAPESP possuem desfecho registrado, e que a arquitetura federada foi validada com N=2 clientes, não N=5. Isso não invalida o "efeito equalizador do FL" (Objetivo/Experimento 2 do plano), mas restringe seu escopo empírico — vale nomear isso como limitação adicional na seção 3.9, complementando a já existente "Escopo da base de dados".

---

## 3. Esquema de desfechos (labels) — ausente do plano, redesenhado 3× na execução

**Plano:** não define o esquema de classes. O glossário trata "Desfechos (outcomes)" genericamente como "resultados finais dos tratamentos (variável-alvo)". A arquitetura do BEHRT no plano (seção 3.5) até prevê flexibilidade: "MLP de duas camadas (64 → 2 neurônios para classificação binária, **ou número de classes conforme os desfechos da base**)" — ou seja, o plano deixou o número de classes em aberto, mas não discute a dificuldade de definir o que é um "desfecho" válido.

**Execução:** o esquema passou por pelo menos 3 formulações distintas antes de estabilizar (ver `docs/Linha_do_Tempo_MOSAIC-FL.md`, Partes 2–4):
1. Inicialmente, 5 classes de **duração de internação** (1–3, 4–7, 8–14, 15–30, >30 dias) — documentado ainda em uso em 2026-06-11.
2. Depois, 4 classes de **prognóstico** (alta, internação prolongada, UTI, óbito) — mas **UTI e óbito nunca existiram nos dados** (limitação estrutural do FAPESP confirmada em 2026-06-08), e "internação prolongada" era um **estado de censura** (paciente ainda internado no momento do snapshot, não um desfecho real). Essa formulação recebeu a maior penalidade acadêmica do projeto (−1,5, "maior risco acadêmico identificado") em avaliação formal de 2026-06-24.
3. Esquema final (2026-06-24/25, vigente até hoje): **5 classes cruzando outcome (curado/melhora) × tipo de atendimento (internado/pronto-socorro) × duração (limiar de 10 dias)** — `curado_pronto`, `curado_internado`, `melhora_pronto`, `melhora_internado_breve`, `melhora_internado_grave`.

**O que precisa entrar no texto:** esta é provavelmente a decisão metodológica mais substantiva de todo o projeto e está ausente do plano. Vale uma subseção própria explicando por que o desfecho não pôde ser definido a priori (limitação do próprio dataset — ausência de óbito/UTI) e como o esquema final foi derivado empiricamente. O limiar de 10 dias entre "breve" e "grave" continua **sem referência clínica formal citada** — sinalizado como questão aberta para a orientadora desde 2026-06-25, ainda sem resposta registrada.

---

## 4. Arquitetura BEHRT — confirmado no código nesta sessão

| Parâmetro | Plano | Execução (verificado em `config.py`/`model.py`) | Status |
|---|---|---|---|
| `embed_dim` | 64 | **64** | ✓ confirmado |
| Camadas do encoder | 2 | **2** | ✓ confirmado |
| Cabeças de atenção | 4 | **4** | ✓ confirmado |
| `dim_feedforward` | 128 | **128** | ✓ confirmado |
| Embedding posicional | Seno/cosseno (posição/ordem do evento) | **Preservado** — `PositionalEncoding` sinusoidal continua ativo | ✓ confirmado, **sem divergência** |
| Camada adicional de tempo | Não prevista no plano | **`DiaRelativoEmbedding`** (`nn.Embedding`, aprendido, indexado pelo dia relativo real desde a admissão) — somado ao embedding de token, **antes** do encoder | ⚠️ **adição não planejada** |
| Otimizador | Adam, lr=0,001 | **Adam, lr=0,001** | ✓ confirmado |
| Função de perda | Entropia cruzada | Entropia cruzada **com pesos por classe** (`class_weights`, clamp em 15,0) | ⚠️ pesos por classe não estavam no plano |
| Classificação final | MLP 64→2 (ou N classes conforme a base) | MLP para 5 classes | ✓ consistente com a flexibilidade que o plano já previa |

**Nota importante de correção:** ao investigar este item, cheguei a supor inicialmente que o `DiaRelativoEmbedding` teria *substituído* o encoding sin/cos do plano — não é o caso. Os dois coexistem: o `PositionalEncoding` sinusoidal permanece exatamente como especificado, e o `DiaRelativoEmbedding` é uma **camada adicional**, somada antes do encoder Transformer, carregando a informação de *quantos dias desde a admissão* aquele exame ocorreu (diferente de "ordem sequencial", que é o que o encoding sin/cos já cobria). Essa adição foi responsável pelo maior ganho de acurácia de uma única alteração arquitetural em todo o projeto (+3,08 p.p., ver Linha do Tempo Parte 5, "Experimento 6"). **O texto da metodologia deveria descrever essa camada explicitamente — é uma contribuição real que o plano não previa.**

**Class weights com clamp em 15,0** também não está no plano — foi adicionado depois que um peso bruto de ~47× (para a classe rara `melhora_pronto` no BPSP) causou explosão de gradiente. Vale uma frase na metodologia.

---

## 5. Framework de agregação federada — mudança de algoritmo

**Plano (seção 3.4.1):** "O servidor central coordena o treinamento seguindo o algoritmo **FedProx**... O hiperparâmetro `proximal_mu` será fixado em **0,01** após testes preliminares."

**Execução (confirmado em `config.py`):**
```python
proximal_mu: float = 0.1   # aumentado de 0.01 → 0.1 (Exp 7)
use_fednova: bool  = True  # Exp 9: substitui FedAvg por normalização por τ_i (Wang et al. 2020)
```

O que de fato aconteceu foi uma combinação, não uma simples troca:
- O **termo proximal do FedProx continua ativo no cliente** (`_proximal_loss()` em `client.py`) — μ subiu de 0,01 para **0,10**, não ficou fixo como planejado.
- A **agregação no servidor** deixou de ser a média ponderada por amostras (estilo FedAvg/FedProx) e passou a ser **FedNova** (Wang et al. 2020) — normaliza cada update pelo número de passos efetivos τᵢ de cada cliente, especificamente para corrigir o viés de agregação quando BPSP e HSL têm volumes de batch muito diferentes (razão 5,5×).
- **SCAFFOLD foi avaliado e descartado** como alternativa (risco de viés com apenas 2 clientes) — essa decisão nunca aparece no plano, porque o plano não previa a necessidade de trocar de algoritmo.

**Referência ausente da bibliografia do plano:** Wang et al. (2020), FedNova, NeurIPS, arXiv:2007.07481 — precisa ser adicionada às referências junto com a justificativa da troca.

---

## 6. Hiperparâmetros de treinamento federado

| Parâmetro | Plano (seção 3.4.2) | Execução (confirmado em `config.py`) | Status |
|---|---|---|---|
| Épocas locais por rodada | 3 | **1** (reduzido de 2 → 1 no Exp13, para reduzir client drift) | ⚠️ diverge — direção oposta à do plano |
| `batch_size` | 32 | **16** | ⚠️ diverge |
| Rodadas máximas | 50 | **120** | ⚠️ diverge — mais que o dobro |
| Frequência de avaliação | A cada 5 rodadas | **A cada rodada** (necessário para o checkpoint guloso) | ⚠️ diverge |
| Critério de parada antecipada | Δ acurácia < 0,5% por 3 rodadas | `convergence_threshold=0.005`, `convergence_patience=3` | ✓ **confirmado, bate exatamente** (embora hoje o critério seja sobre F1 macro em vez de accuracy — ver seção 9) |

**Contexto da divergência de épocas/batch/rodadas:** o plano assumiu 50 rodadas como teto suficiente. Na prática, a heterogeneidade non-IID real do BPSP/HSL exigiu 120 rodadas para sequer se aproximar de convergência (e a fase federada do Bloco 2 nem convergiu em 120). Épocas locais foram *reduzidas* (não aumentadas) porque mais épocas locais por rodada aumentavam o *client drift* — o oposto do que "3 épocas" do plano assumia implicitamente (mais treino local = melhor). O `batch_size=16` (vs. 32 planejado) não tem uma justificativa registrada nos documentos — vale a autora confirmar se foi decisão deliberada ou herdada de uma configuração anterior sem revisão.

---

## 7. RAG — infraestrutura e modelo de geração trocados

| Componente | Plano | Execução | Status |
|---|---|---|---|
| Armazenamento vetorial | ChromaDB | **PostgreSQL + pgvector** (`_PostgreSQLStore`, tabela `knowledge.clinical_profiles`) — ChromaDB só sobrevive como dependência de um `ChromaDBConfigLoader`, que não tem nada a ver com o RAG | ⚠️ substituído |
| Modelo de embeddings | all-MiniLM-L6-v2, 384 dim | **all-MiniLM-L6-v2** — confirmado em `config.py` (`FL_EMBEDDING_MODEL`, padrão exatamente esse modelo) | ✓ confirmado, sem divergência |
| Modelo de geração (LLM) | DistilGPT-2 | **Ollama + gemma3:4b** (produção); DistilGPT-2 rebaixado a fallback (Ollama indisponível) e padrão só em ambiente de teste | ⚠️ substituído — mas resolve uma limitação que o próprio plano (seção 3.9) já havia identificado ("recomenda-se um LLM maior... em cenário real") |
| `top_k` (documentos recuperados) | 3 | **3** (`FED_CFG.top_k = 3`) | ✓ confirmado |
| Perfis protótipos por desfecho | "até 50 perfis" | **`top_n=50`** em `extract_top_patterns()` | ✓ confirmado |
| Anonimização estrutural (idade exata → faixa etária) | Sim, especificado no plano | **Confirmado** — `rag/__init__.py` substitui `idade_exacta` por `faixa_etaria` no texto da KB antes da vetorização (mesma lógica que gerou o bug do `replace("", "adulto")` corrigido em 2026-06-29, ver Linha do Tempo Parte 7) | ✓ confirmado |

**Consequência para o texto:** a seção 3.6 do plano ("Construção da base de conhecimento") descreve ChromaDB e DistilGPT-2 em detalhe — precisa ser reescrita para refletir PostgreSQL/pgvector + Ollama/gemma3:4b, mantendo a lógica de "50 perfis protótipos por desfecho" e `top_k=3`, que **permaneceram exatamente como planejado**.

---

## 8. RETAIN — citado na revisão bibliográfica, nunca implementado

O plano (seção 2, Revisão Bibliográfica) apresenta o RETAIN (Choi et al., 2016) como o modelo responsável por "garantir a interpretabilidade das predições", com a lógica de atenção reversa no tempo. **Não há nenhuma implementação de RETAIN em nenhum ponto do código do projeto.** A interpretabilidade efetivamente implementada vem da própria atenção do BEHRT, extraída via `BEHRTPatternExtractor.extract_top_patterns()` (`interpretability.py`) — uma abordagem diferente (atenção nativa do Transformer, não um modelo dedicado de atenção reversa).

**O que precisa de decisão:** ou (a) remover a citação de RETAIN da revisão bibliográfica e substituir por uma citação do próprio BEHRT como fonte de interpretabilidade, ou (b) manter RETAIN como "trabalho relacionado considerado mas não adotado", com uma frase explicando por quê (provavelmente: a atenção nativa do BEHRT já fornece o sinal necessário para o RAG, tornando um segundo modelo de interpretabilidade redundante para o escopo deste TCC).

---

## 9. Métricas e critério de seleção de modelo

**Plano:** métrica principal declarada é **AUC-ROC** (explicitamente nomeada como "métrica principal" no Experimento 2). Critério de parada é baseado em "estabilização da acurácia global".

**Execução:** evolução em 3 fases, nenhuma delas usando AUC-ROC como critério de seleção:
1. Accuracy como critério de checkpoint (Bloco 1, T1–T12) — Macro AUC é *reportada*, mas nunca foi o critério de seleção do melhor checkpoint.
2. F1 macro como critério (a partir do Bloco 2, 2026-06-30) — decisão explícita, motivada por um gap de 20 p.p. entre accuracy (70,19%) e F1 macro (0,4994) no melhor checkpoint do Bloco 1, causado por accuracy favorecer `curado_pronto` (48% do dataset).
3. **ECE (Expected Calibration Error)** com **calibração isotônica One-vs-Rest** (Zadrozny & Elkan, 2002) tornou-se um eixo de análise tão relevante quanto accuracy/F1 — isotônica supera temperature scaling em **100% das ~14 medições feitas ao longo do projeto** (ver Linha do Tempo, "Decisões técnicas transversais"). Esse é um achado empírico consistente e forte que **não tem nenhuma âncora na metodologia planejada** — a palavra "calibração" não aparece no PDF.

**O que precisa entrar no texto:** a seção 3.7 do plano precisa de uma reformulação para declarar F1 macro como critério de checkpoint (com justificativa do trade-off accuracy vs. equilíbrio entre classes) e adicionar calibração (temperature scaling → isotônica) como um eixo de investigação — mesmo não tendo sido planejado, é provavelmente um dos resultados mais citáveis do trabalho (referências a adicionar: Zadrozny & Elkan 2002; Guo et al. 2017, "On Calibration of Modern Neural Networks").

---

## 10. Os "5 Experimentos" planejados vs. o que foi de fato executado

Esta é a maior divergência estrutural do documento. O plano define 5 experimentos com desenho experimental específico (seção 3.7). A execução seguiu uma sequência de **16+ "Treinamentos"** (T1–T16, depois Bloco 2, GPU, DP pendente) com objetivos diferentes dos 5 originais. Mapeamento aproximado:

| Experimento planejado | Desenho especificado no plano | O que foi (ou não foi) executado |
|---|---|---|
| **1 — Desafios de padronização** | Pré-processamento sistemático, log de transformações, % de amostras rejeitadas, tempo de normalização | **Parcialmente equivalente.** Não houve um "experimento" isolado, mas a infraestrutura de padronização foi construída como *produção* permanente: `integration/term_manager` (validação de analitos, resolução canônica, `knowledge.term_dictionary`), `scan_analytes()`/`validate_analytes_before_load()` antes de cada carga. O esforço de padronização existe e é medível (nº de analitos canônicos, termos pendentes), mas nunca foi empacotado como o "Experimento 1" com as métricas qualitativas exatas do plano (nº de transformações, dificuldade de automação). |
| **2 — Efeito equalizador do FL** | Local (silo) vs. Federado, métrica AUC-ROC por subconjunto, foco no ganho do hospital menor | **Reformulado, não idêntico.** Os treinamentos leave-one-out (T13 BPSP-only, T14 HSL-only) e o treinamento federado (T15) medem algo próximo — mas com foco em accuracy/F1/AUC macro, não especificamente "ganho relativo do subconjunto com menor volume" isolado como o plano pede. O achado mais forte nessa linha (BPSP-only nunca aprende `melhora_pronto`, F1=0,000, e a federação resolve isso) é conceitualmente **exatamente** o que o Experimento 2 queria demonstrar — mas nunca foi reportado nesse formato específico (AUC-ROC por subconjunto, local vs. federado, lado a lado). |
| **3 — Impacto da heterogeneidade (non-IID) na convergência** | Comparar distribuição non-IID natural vs. versão artificialmente balanceada (IID simulado), acurácia por subgrupo demográfico ao longo das rodadas | **Nunca executado.** Não existe, em nenhum documento revisado, uma versão "balanceada artificialmente" dos dados para contraste IID vs. non-IID. Toda a discussão de non-IID no projeto foi feita observando o comportamento natural dos dados (ex.: razão de volume BPSP/HSL, dominância de classe por hospital) — rica, mas não é o desenho de contraste controlado que o plano especifica. |
| **4 — Contribuição do RAG na redução de incertezas** | Avaliação humana (autora e/ou colega), 50 amostras, escala Likert 1–5, % com nota ≥4, frequência de "alucinações" | **Nunca executado.** A métrica usada no lugar (Precision@3) mede recuperação, não qualidade da justificativa textual gerada. Nenhuma avaliação Likert foi feita até 2026-07-01. Esta é a lacuna mais crítica para a hipótese central do trabalho ("RAG reduz a incerteza diagnóstica") — sem avaliação qualitativa humana, a hipótese permanece não testada da forma como foi formulada. |
| **5 — Eficiência operacional da rede** | Acurácia de validação por rodada, custo de comunicação acumulado (MB), rodada de convergência ótima (Topt) | **Parcialmente equivalente, formato diferente.** O projeto rastreia extensivamente duração por rodada, RAM, CPU (via `psutil`, `fl_round_history.round_duration_s`) e velocidade CPU vs. GPU (~10,9× de ganho medido) — mas não fecha o cálculo de "MB transmitidos vs. acurácia" nem determina um `Topt` formal como o plano pede. O foco real acabou sendo velocidade de treino (CPU vs. GPU) mais do que custo de comunicação de rede. |

**Decisão necessária:** a seção 3.7 da metodologia precisa ser reescrita para descrever a sequência real de treinamentos (T1–T16, Bloco 1/2), ou os 3 experimentos nunca executados (3, 4, e uma versão fechada do 5) precisam ser rodados antes da defesa para que o texto planejado continue válido. Dado o estágio atual do projeto (pós-modularização, GPU disponível — ciclo de treino caiu de ~7h para minutos), os Experimentos 3 e 5 ficaram mais baratos de rodar agora do que estariam em CPU. O Experimento 4 (avaliação Likert) não depende de poder computacional — só de tempo da autora — e é o mais fácil de fechar rapidamente.

---

## 11. Ambiente experimental e ferramentas

**Plano (seção 3.8):** "Todos os experimentos serão executados em hardware a ser definido." Bibliotecas: Python 3.10, `flwr`, `torch`, `transformers`, `pandas`, `numpy`, `datasets`, `langchain`, `chromadb`, `sentence-transformers`, `scikit-learn`, Git+GitHub. "O código-fonte do **protótipo** será disponibilizado em um repositório público."

**Execução:** hardware definido (notebook Dell Inspiron i7-1165G7, depois GPU RTX 4070 Ti adicionada). Bibliotecas confirmadas em `pyproject.toml`: `torch`, `transformers`, `flwr>=1.8.0`, `sentence-transformers`, `pandas`, `numpy`, `scikit-learn`, mais **não previstas no plano**: `sqlalchemy`, `psycopg2-binary`, `pgvector`, `fastapi`, `uvicorn`, `apscheduler`, `prometheus_client`, `python-json-logger`, `alembic`. Ausentes do que foi de fato usado: `langchain` (não aparece em nenhuma dependência do projeto), `datasets` (biblioteca HuggingFace Datasets não é usada — os dados vêm de SQL direto via SQLAlchemy).

**A palavra "protótipo" do plano não descreve mais o que existe.** O sistema atual tem: topologia de produção Flower (SuperLink + ServerApp + SuperNode), TLS obrigatório, audit trail LGPD, API REST completa (FastAPI, JWT, rate limiting), scheduler com APScheduler, observabilidade Prometheus, Dockerfiles e Helm chart, e Differential Privacy (DP-FedAvg) implementado. Isso é consistente com uma instrução explícita da autora registrada em `docs/TODO.md` (não datada, mas anterior à adoção do FedNova): *"não foque em um trabalho simples para TCC, foque em um sistema que tem que estar prestes a ser um MVP sólido em produção"*. Essa instrução — que efetivamente redefiniu o escopo de engenharia do projeto — não está refletida em lugar nenhum da metodologia planejada e merece uma frase explícita no texto, porque explica por que o projeto foi muito além de "protótipo de pesquisa".

---

## 12. Limitações do plano (seção 3.9) — o que mudou de status

| Limitação declarada no plano | Status em 2026-07-01 |
|---|---|
| "RAG com LLM pequeno... recomenda-se um LLM maior... (BioGPT, LLaMA-med)" | **Resolvida** — Ollama/gemma3:4b (4B parâmetros) substituiu DistilGPT-2 como padrão de produção. |
| "Este trabalho não implementa privacidade diferencial, deixando-a como trabalho futuro" | **Parcialmente resolvida** — DP-FedAvg (McMahan et al. 2018) está implementado (clipping + ruído gaussiano), mas os experimentos que mediriam seu efeito (Exp 17/18/19, σ=1,0/0,5/2,0) **nunca foram executados**. A limitação evolui de "não implementado" para "implementado, não validado empiricamente" — ainda é uma lacuna para a defesa, mas de natureza diferente. |
| "Simulação de médicos: avaliação qualitativa do RAG será feita pelo próprio autor... futuros trabalhos devem incluir validação com médicos reais" | **Pior do que o planejado** — nem a avaliação do próprio autor (a mínima prometida no plano) foi feita. Ver seção 10, Experimento 4. |
| "Escopo da base de dados: embora real, a base do ClinicalPath pode não representar toda a diversidade de prontuários eletrônicos brasileiros" | **Inalterada**, ainda válida — e agora agravada pelo fato de que só 2 das 5 instituições da base têm desfecho utilizável (ver Seção 2 acima). Vale reforçar essa limitação com o dado concreto. |
| "Modelo BEHRT simplificado... pode sacrificar desempenho" | **Inalterada como trade-off consciente** — mantido embed_dim=64/2 camadas, mas com a adição do `DiaRelativoEmbedding` (não previsto) que compensou parte da simplificação. Avaliação formal de 2026-06-25 concluiu que "mais camadas não ajudam" porque o gargalo é a dimensão de embedding, não a profundidade — argumento a favor de manter a arquitetura simplificada, vale citar. |
| "Privacidade e segurança... não elimina completamente riscos residuais (ex.: inferência de atributos)" | **Inalterada, ainda válida.** Nenhum trabalho adicional sobre ataques de inferência de atributos foi feito. |

---

## 13. Bibliografia — referências usadas na execução mas ausentes do plano

| Referência | Onde entra |
|---|---|
| Wang et al. (2020), FedNova — "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization", NeurIPS, arXiv:2007.07481 | Algoritmo de agregação real do projeto — ver Seção 5 |
| McMahan et al. (2018), DP-FedAvg — "Learning Differentially Private Recurrent Language Models", ICLR, arXiv:1710.06963 | Differential Privacy implementado — ver Seção 12 |
| Zadrozny & Elkan (2002) — calibração isotônica | Calibrador adotado como padrão desde o Exp13 — ver Seção 9 |
| Guo et al. (2017) — "On Calibration of Modern Neural Networks" | Framework teórico para o diagnóstico de subconfiança do modelo — ver Seção 9 |
| Karimireddy et al. (2020), SCAFFOLD | Avaliado e descartado como alternativa ao FedNova — vale citar mesmo tendo sido descartado, porque documenta uma decisão de design deliberada |

**Referência do plano cuja aplicação deve ser revisada:** Choi et al. (2016), RETAIN — ver Seção 8.

---

## 14. Itens do plano que se confirmaram exatamente como especificado

Para não passar a impressão de que tudo mudou — o núcleo conceitual do plano se manteve firme:

- Arquitetura de 3 camadas (dados distribuídos → FL → RAG) — preservada integralmente na estrutura do sistema.
- `embed_dim=64`, 2 camadas, 4 cabeças de atenção, `dim_feedforward=128` — todos confirmados no código.
- Otimizador Adam, `lr=0,001` — confirmado.
- Critério de convergência "Δ < 0,5% por 3 rodadas" — confirmado quase literalmente (`convergence_threshold=0.005`, `convergence_patience=3`), só migrou de "sobre accuracy" para "sobre F1 macro".
- `top_k=3` para recuperação do RAG — confirmado.
- "Até 50 perfis protótipos" por desfecho — confirmado (`top_n=50`).
- Modelo de embeddings `all-MiniLM-L6-v2`, 384 dimensões — confirmado.
- Anonimização estrutural (idade exata → faixa etária) na base de conhecimento do RAG — confirmado.
- Uso do framework Flower (Beutel et al., 2020) — confirmado, inclusive expandido para topologia de produção (SuperLink).
- FedProx como base do treino local (termo proximal) — confirmado, ainda ativo, só com μ diferente e complementado por FedNova na agregação.
- Base de dados FAPESP COVID-19 Data Sharing/BR, uso condicionado à LGPD, dados anonimizados — confirmado.

---

## 15. Onde a execução é melhor do que a proposta original

| Item | Por que a execução ganha |
|---|---|
| **FedNova em vez de FedProx fixo (μ=0,01)** | O plano fixava μ sem mecanismo para lidar com a razão de volume de 5,5× entre BPSP e HSL. FedNova ataca exatamente esse tipo de viés de agregação (normaliza por passos efetivos τᵢ), o que o FedProx sozinho não resolve — FedProx mitiga *client drift*, não desigualdade de volume entre clientes. Ganho empírico medido: +8,08 p.p. (T8→T12). |
| **Calibração isotônica formal** | O plano não menciona calibração em nenhum ponto. A execução descobriu (e documentou, de forma reprodutível, em ~14 medições) que o modelo é sistematicamente subconfiante e que temperature scaling piora a calibração em 100% dos casos — achado que não existiria se o plano tivesse sido seguido à risca. |
| **`DiaRelativoEmbedding`** | Camada adicional não prevista, responsável pelo maior ganho de acurácia de uma única alteração arquitetural do projeto (+3,08 p.p.). Sem ela, o modelo só teria a ordem sequencial dos eventos (via encoding sin/cos), não a distância real em dias desde a admissão — informação clinicamente relevante que o plano não cogitou. |
| **Redesenho do esquema de labels** | O esquema que o plano deixaria implícito (herdado da base tal como veio) continha um label censurado (`internacao_prolongada` = "ainda internado no momento do snapshot", não um desfecho real). A execução identificou e corrigiu isso — um erro que, se não corrigido, invalidaria silenciosamente os resultados do modelo preditivo inteiro. |
| **`training_id` scoping + checkpoint guloso** | O plano não previa nenhum mecanismo de isolamento entre execuções. Sem ele, a execução real teria (e de fato teve, no Exp9) um bug de cross-contamination — o checkpoint de um treinamento sendo erroneamente avaliado como se fosse de outro. O plano ficaria vulnerável ao mesmo erro sem sequer ter esse tipo de proteção descrita. |
| **Gemma3:4b via Ollama em vez de DistilGPT-2** | O próprio plano (seção 3.9) já reconhecia essa limitação e recomendava corrigi-la ("recomenda-se um LLM maior... em cenário real"). A execução resolveu exatamente a lacuna que o plano tinha antecipado. |
| **PostgreSQL/pgvector em vez de ChromaDB para a base de conhecimento do RAG** | Consolida o armazenamento vetorial na mesma base relacional que já guarda checkpoints, métricas e vocabulário — uma tecnologia de banco a menos para operar, fazer backup e auditar (relevante em contexto de LGPD, onde auditoria e controle de acesso unificado importam). |
| **Rigor de rastreabilidade experimental** | Cada treinamento tem `training_id`, critério de checkpoint registrado (`checkpoint_criterion`), histórico por rodada (`fl_round_history` com τ_eff, F1 por classe, duração), consultável via SQL. O plano não especifica nenhum mecanismo de rastreabilidade além de "logs" genéricos — a execução foi muito além do necessário para o TCC, mas isso facilita diretamente a escrita do capítulo de resultados (todos os números são reconstituíveis a partir do banco, não de anotações manuais). |

---

## 16. Onde o plano original era melhor ou mais completo do que a execução

| Item | Por que o plano tinha um desenho melhor |
|---|---|
| **Experimento 4 (avaliação Likert humana do RAG)** | Desenho simples, barato (50 amostras, escala 1–5, sem depender de poder computacional) e diretamente alinhado com a hipótese central do trabalho. Foi abandonado no meio do caminho sem substituto equivalente — Precision@3 mede uma coisa relacionada, mas não a mesma coisa (recuperação ≠ qualidade da geração). Esta é a maior perda do processo: um experimento barato e metodologicamente correto que simplesmente não aconteceu. |
| **Experimento 3 (contraste IID vs. non-IID artificial)** | O plano propunha criar uma versão artificialmente balanceada dos dados para comparar contra a distribuição não-IID natural — um desenho de contraste controlado, que isola causalmente o efeito da heterogeneidade. A execução só observou o comportamento non-IID natural, sem contrafactual — o que permite descrever padrões, mas não afirmar causalidade ("a heterogeneidade causa X de queda") com o mesmo rigor que o desenho planejado permitiria. |
| **AUC-ROC como métrica única e estável ao longo de todo o trabalho** | O plano escolhia uma métrica principal e a mantinha do início ao fim. A execução migrou de accuracy (Bloco 1) para F1 macro (Bloco 2) como critério de seleção — mudança bem justificada tecnicamente (Seção 9), mas com um custo real: os resultados do Bloco 1 e do Bloco 2 **não são diretamente comparáveis** entre si, porque o "melhor checkpoint" de cada bloco foi escolhido por critérios diferentes. Isso complica a narrativa de "evolução de resultados" que normalmente se espera num capítulo de resultados de TCC. |
| **50 rodadas / batch=32 / 3 épocas locais (footprint mais leve)** | Essa configuração, mesmo tendo se mostrado insuficiente para a heterogeneidade real (daí a mudança), teria dado ciclos de experimento muito mais rápidos de iterar e depurar. A configuração real (120 rodadas, batch=16, 1 época) só se tornou operacionalmente viável depois da GPU — em CPU, cada ciclo completo levava ~7h, o que atrasou a velocidade de iteração do projeto por semanas. |
| **Prompt do RAG especificado literalmente no texto** | O plano cita o prompt exato usado para gerar a justificativa (`"Com base nos seguintes casos clínicos semelhantes: [...]. Justifique brevemente:"`) — bom nível de especificidade para reprodutibilidade. Vale conferir se a documentação atual do RAG mantém esse mesmo nível de detalhe explícito no texto da metodologia (o prompt real está no código, mas nem sempre reproduzido literalmente na documentação). |
| **Stack mais simples de reproduzir (`chromadb`, `langchain`, sem servidor de banco dedicado)** | Para alguém tentando reproduzir a pesquisa a partir do zero, a stack do plano (SQLite/ChromaDB embutidos, sem servidor externo) tem barreira de entrada bem menor do que a stack atual (PostgreSQL com extensão pgvector, migrations Alembic, certificados TLS). Isso é uma tensão real entre "rigor de produção" e "reprodutibilidade acadêmica simples" — vale nomear essa tensão no texto, já que a defesa pode ser cobrada sobre reprodutibilidade. |

---

## 17. Alinhamento com estado da arte e prática de indústria

> **Aviso de fonte:** esta seção usa conhecimento geral sobre a literatura de FL/RAG/MLOps e práticas de indústria — **não vem dos dados nem do código do projeto**, e não foi verificado contra as referências específicas que você vai citar no TCC. Trate como ponto de partida para pesquisa bibliográfica própria, não como afirmação pronta para citar. Cite as fontes originais (não este documento) no texto final.

| Decisão | Mais alinhada com estado da arte/indústria | Comentário |
|---|---|---|
| Algoritmo de agregação: FedNova vs. FedProx | **Execução (FedNova)**, com ressalva | FedNova (Wang et al., NeurIPS 2020) ataca especificamente o problema de inconsistência de objetivo em agregação heterogênea, mais adequado ao cenário de 2 clientes com razão de volume 5,5× do que FedProx (que resolve client drift, um problema relacionado mas diferente). Ressalva: a literatura mais recente (2023–2025) sobre FL com poucos clientes e heterogeneidade severa também discute métodos como FedDyn e variantes com redução de variância — vale checar se algum deles é citado como "estado da arte atual" na literatura que você já revisou, para não ficar apenas no FedNova de 2020 como o mais recente citável. |
| Armazenamento vetorial: pgvector vs. ChromaDB | **Execução (pgvector)** | Consolidar embeddings na mesma base relacional que já guarda dados clínicos, checkpoints e trilha de auditoria é um padrão que ganhou tração justamente em 2023–2025, motivado por sistemas RAG em produção que precisam de controle de acesso e consistência transacional unificados — relevante em contexto de dado sensível/LGPD. ChromaDB (e ferramentas similares) permanece mais comum em protótipos e pesquisa exploratória, exatamente o contexto que o plano original endereçava. |
| LLM para geração de justificativa: gemma3:4b/Ollama vs. DistilGPT-2 | **Execução**, claramente | DistilGPT-2 (2019, 82M parâmetros, sem ajuste de instrução) está bastante atrás do estado da arte para geração de texto condicionado a contexto recuperado. Servir um modelo pequeno-porém-moderno localmente via Ollama é um padrão comum atualmente para cenários que exigem que os dados não saiam do ambiente local (privacidade/LGPD) — evita depender de uma API externa de LLM, o que seria inaceitável dado o requisito de privacidade do próprio projeto. |
| Calibração pós-hoc (isotônica) | **Execução, e deveria ter estado no plano desde o início** | Probabilidades calibradas (não só acurácia discriminativa) vêm se tornando expectativa mínima para qualquer sistema de ML clínico com pretensão de uso real — reguladores e literatura de ML em saúde cada vez mais cobram isso. A ausência de qualquer menção a calibração no plano original é, sob essa ótica, uma lacuna do desenho inicial, não um refinamento opcional. |
| Topologia federada de produção (SuperLink separado do ServerApp) | **Execução** | Separar o plano de controle/infraestrutura (persistente, "raramente cai") da lógica de treino (que pode ser reiniciada sem perder o histórico da federação) é o padrão que o próprio Flower recomenda para produção, e reflete a arquitetura usada por outras plataformas de FL em saúde. Para o escopo de um TCC (protótipo de pesquisa), esse nível de robustez é mais do que estritamente necessário — mas não é incorreto, é além do exigido. |
| Contraste IID vs. non-IID controlado (Experimento 3 do plano) | **Plano** | Esse tipo de ablação controlada (partição sintética balanceada vs. distribuição real) é como a própria literatura citada no trabalho (inclusive o paper do FedNova e o do FedProx) demonstra robustez a heterogeneidade — é o desenho padrão esperado por revisores de artigos de FL. A execução, ao não fazer esse contraste, fica com uma afirmação de "efeito do non-IID" mais fraca do ponto de vista de rigor experimental do que o que a área normalmente exige. |
| Avaliação de qualidade de geração (Experimento 4 do plano) | **Plano, com uma ressalva de atualização** | Avaliação humana por escala Likert continua sendo prática padrão para julgar texto gerado por LLM em aplicações médicas. Uma atualização razoável ao desenho do plano — não uma substituição — seria complementar o Likert humano com métricas automáticas de "groundedness"/fidelidade ao contexto recuperado (a área de avaliação de RAG evoluiu bastante desde 2023, com frameworks desse tipo ganhando popularidade) — mas isso seria um adicional, não um substituto do julgamento humano que o próprio plano já previa corretamente. |
