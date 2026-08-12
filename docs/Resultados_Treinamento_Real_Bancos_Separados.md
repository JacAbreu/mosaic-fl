# Resultados — Treinamentos Reais (Caminho B) com 3 Bancos Totalmente Separados

**Propósito deste documento:** registrar os resultados oficiais dos "6 treinamentos formais" descritos em `docs/Tutorial_Treinamento_Real_Bancos_Separados.md` (Parte 7) — a primeira execução formal do Caminho B na topologia final de privacidade (servidor, BPSP e HSL em 3 bancos Postgres totalmente isolados, sem nenhum compartilhamento de container/porta/volume). Preenchido conforme cada treino é concluído — não antes, pra não registrar número provisório como se fosse final.

**Para o histórico de incidentes de infraestrutura desta fase** (crash do desktop, bugs de descoberta de vocabulário, backfill de classification faltando no HSL, bug do `evaluation_json` etc.) **ver `docs/Linha_do_Tempo_MOSAIC-FL.md`, seção "Fase '3 bancos separados'"** — este documento cobre só os números, não a investigação de como se chegou a eles.

**Critério de inclusão (objetivo, verificável direto no banco):** um `training_id` só entra na tabela abaixo se `metrics.fl_trainings.status='completed'` **e** `run_classification='treinamento_real'`. Qualquer outro status (`running` órfão, `interrupted`, `invalid`) não conta como treino formal, independente do motivo — não precisa de julgamento narrativo caso a caso, é a própria coluna `status` que decide. Tentativas com `status≠'completed'` ficam de fora da tabela principal, registradas só na tabela de descartadas abaixo (motivo) e na linha do tempo (investigação completa).

---

## Conceitos: como funciona o treinamento, o que é convergência, como os dados são escolhidos

Seção de referência rápida — pra entender os números das tabelas abaixo sem precisar abrir código. Não repete o que já está detalhado no rascunho do TCC (`docs/rascunho_tcc/`); é a versão curta, prática.

### Como funciona uma rodada de treinamento federado

O modelo nunca vê o dado de paciente diretamente — só os pesos que cada hospital calcula localmente. Cada rodada segue sempre a mesma sequência:

1. **Distribuição**: o servidor (`SuperLink`) envia os pesos do modelo global atual pra cada hospital conectado (BPSP e HSL, hoje).
2. **Treino local**: cada hospital treina esses pesos por 1 época sobre seu próprio dado (nunca sai da máquina do hospital), usando FedProx — a função de perda tem um termo extra que penaliza o modelo local se afastar demais do modelo global recebido, pra reduzir o quanto cada hospital "puxa" o modelo pra sua própria distribuição.
3. **Envio de volta**: cada hospital manda de volta só os pesos atualizados (não o dado) e quantos passos de otimização (`τ`) ele de fato executou naquela rodada.
4. **Agregação**: o servidor combina os pesos dos dois hospitais em um único modelo global. Usa FedNova, que pondera a contribuição de cada hospital pelo número de passos efetivos (`τ`) que ele executou — necessário porque o BPSP tem ~5,5× mais dado que o HSL, e uma média simples deixaria o modelo dominado pelo BPSP.
5. **Avaliação**: o modelo agregado é testado contra um conjunto de teste que nenhum dos dois hospitais usou pra treinar — dá a acurácia/F1/AUC daquela rodada.
6. **Checkpoint guloso**: se essa rodada bateu o melhor resultado já visto no treino (por `f1_macro`, o critério padrão), os pesos são salvos como "melhor rodada" — o modelo não melhora sempre, então guardar só a última rodada perderia o pico real.

Isso se repete até bater o teto de rodadas configurado (110, nos 6 treinos formais) ou até a convergência ser detectada (se `FL_EARLY_STOP` estiver ligado).

### O que é "convergência" aqui (e o que ela NÃO é)

"Convergência", neste sistema, significa só uma coisa: **a métrica de avaliação (`f1_macro`) parou de mudar de forma significativa por várias rodadas seguidas** (o `ConvergenceTracker` olha uma janela de rodadas recentes e compara a variação contra um limiar pequeno). Não avalia se o resultado é bom — só se ele está estável.

Isso importa porque estabilidade e qualidade são coisas diferentes. Um exemplo real, já documentado: sob ruído de privacidade diferencial (DP), o modelo pode ficar preso repetindo os mesmos poucos resultados ruins rodada após rodada — isso também conta como "convergência" pelo critério de estabilidade, mesmo sendo uma solução degenerada, não uma boa solução (ver seção de "estados atratores" abaixo). Por isso a rodada de convergência e a melhor rodada quase nunca são a mesma — a melhor rodada é sobre qualidade (métrica mais alta já vista), a de convergência é só sobre quando a métrica parou de balançar.

Há também um período de "aquecimento" (*warm-up*) no início do treino em que a convergência nunca é avaliada — nas primeiras rodadas a métrica sempre varia bastante (o modelo ainda está longe de qualquer solução), então checar estabilidade cedo demais daria falso positivo.

### Como os dados de treino são escolhidos

Cada hospital participante mantém seu próprio banco de dados local — nenhum dado de paciente é compartilhado entre BPSP, HSL e o servidor, nunca. Dentro do banco de cada hospital, o conjunto de pacientes elegíveis (com desfecho clínico utilizável, ver critérios de inclusão) é dividido em 4 partes, sempre na mesma proporção 70/10/10/10:

- **70% treino**: usado no loop federado, rodada a rodada.
- **10% validação**: usado localmente por cada hospital durante o treino.
- **10% calibração**: reservado exclusivamente para ajustar o calibrador de probabilidade (temperatura ou isotônica) — nunca visto durante o treino, pra não inflar artificialmente a qualidade da calibração.
- **10% teste**: combinado entre os hospitais no fim, usado só pra avaliação final — é sobre esse conjunto que accuracy/F1/AUC da tabela principal são calculados.

A divisão é feita com uma semente aleatória fixa por hospital — a mesma configuração sempre produz exatamente a mesma divisão, condição necessária pra qualquer comparação entre treinos ser válida (senão, uma diferença de resultado poderia vir só de um *split* de dado diferente, não de uma mudança real de configuração). O particionamento respeita a distribuição real de cada hospital (`partition_mode=natural`) — não há balanceamento artificial entre classes ou entre hospitais; é exatamente essa distribuição desbalanceada real (o *label skew* documentado nas seções de achados) que motiva o uso de FedProx/FedNova em vez de FedAvg simples.

### Por que rodar o mesmo cenário mais de uma vez pode dar resultados diferentes

O *split* de dado é fixo (mesma semente, mesma divisão sempre) — a variação entre execuções **não vem dos dados**, vem de outras fontes de aleatoriedade que continuam ativas mesmo com o *split* fixo:

- **Inicialização dos pesos do modelo**: cada execução começa de um ponto diferente e aleatório no espaço de parâmetros, salvo se uma semente específica de inicialização também for fixada (o que hoje só é feito pro *split* de dado, não pros pesos iniciais).
- **Embaralhamento dos lotes de treino (*shuffling*)**: a ordem em que os exemplos de treino são apresentados ao modelo, a cada época, é aleatória.
- **Dropout**: ativo durante o treino, apaga aleatoriamente uma fração dos neurônios a cada passo — cada execução treina, na prática, uma sub-rede ligeiramente diferente.
- **Ruído de privacidade diferencial** (só nos cenários com DP): o ruído gaussiano adicionado à agregação é sorteado de novo em cada rodada — duas execuções do mesmo cenário com DP nunca recebem exatamente o mesmo ruído.

Nenhuma dessas fontes é um defeito — são parte normal do treinamento de rede neural, e por isso a prática padrão (adotada no desenho dos 6 treinos formais) é sempre rodar réplicas, nunca confiar num resultado de execução única.

**Por que isso afeta desproporcionalmente as classes raras.** Uma classe com milhares de exemplos (`curado_pronto`) tem sinal de gradiente forte e consistente — pequena variação de inicialização ou embaralhamento não muda se o modelo aprende aquele padrão, só ajusta o quanto. Já uma classe com pouquíssimos exemplos (`curado_internado`, ~82 no teste) tem um sinal de gradiente fraco e ruidoso — o suficiente para que uma diferença de inicialização determine se o modelo "encontra" um caminho no espaço de parâmetros que preserva algum sinal residual para essa classe, ou cai num mínimo local que a ignora completamente. É exatamente o padrão observado entre `training_id=6` e `training_id=11` (seção acima).

### A relação entre o treinamento e o RAG — e o que mais réplicas mudam nele

A base de conhecimento do RAG (Seção do rascunho do TCC sobre RAG) **não usa prontuário de paciente real** — ela é construída extraindo os padrões de atenção do próprio modelo BEHRT já treinado: para cada classe de desfecho, o sistema identifica quais combinações de exame/idade/sintoma o modelo mais "olhou" ao prever aquela classe, e sintetiza até 50 perfis clínicos representativos a partir disso. Ou seja: **o RAG é construído a partir do resultado do treinamento, não é independente dele.**

Isso tem uma consequência direta: se o modelo nunca aprendeu um padrão de atenção significativo para uma classe (porque ela colapsou — F1≈0, como `curado_internado` em `training_id=11`), simplesmente não existe padrão nenhum de qualidade para extrair para essa classe. A base de conhecimento do RAG fica sem bons perfis para recuperar quando um paciente novo apresentar um quadro parecido com esse desfecho — e é exatamente esse efeito que já aparece medido na métrica *Precision@3* do RAG (rascunho do TCC, seção de avaliação do RAG): classes onde o modelo tem pouco sinal tendem a ter recuperação pior.

**O que isso implica sobre rodar mais treinamentos**: cada réplica que colapsa uma classe diferente (ou nenhuma) vai gerar uma base de conhecimento RAG diferente também — não é só a métrica de classificação que varia entre execuções, a qualidade da explicação que o RAG consegue dar para os casos mais raros e clinicamente sensíveis varia junto. Ponto adicional já registrado como achado deste projeto (ver `docs/rascunho_tcc/`, seção de limitações): hoje o RAG é construído a partir do modelo da **última rodada** do treino, não da **melhor rodada** (`best_round`) — então mesmo que o melhor *checkpoint* do treino tenha capturado uma classe rara razoavelmente bem em algum ponto, se a classe colapsou justamente na última rodada (o padrão real observado em `training_id=11` — ver comparação de rodadas na tela `/fl-training-results`), a base de conhecimento do RAG construída a partir do modelo efetivamente ativo em produção pode estar pior do que o melhor modelo já treinado sugeriria. Mais réplicas ajudam a entender se esse descasamento é a regra ou a exceção — e reforçam o caso para revisar essa decisão de qual *checkpoint* alimenta o RAG.

### É viável usar o `best_round` no RAG? O que o código permite hoje, e o que a literatura diz

**Pergunta levantada (2026-08-09):** dá pra trocar o RAG pra usar os pesos da melhor rodada em vez da última? Investigação no código (não é só opinião) e busca de embasamento na literatura, nessa ordem.

**O que o código faz hoje, exatamente.** A extração de padrões de atenção pro RAG acontece do lado do **cliente** (`FedProxClient._extract_rag_patterns`, `src/mosaicfl/core/client.py`), sobre o modelo que o Flower acabou de carregar pra rodada de avaliação — e isso só é pedido na **última rodada configurada** (`is_final_round`, `infrastructure/mosaicfl_server/strategy/core.py`), nunca na rodada de melhor critério. O servidor já guarda os pesos da melhor rodada (`self._best_state_dict`), e até existe uma função pronta pra recarregar o melhor *checkpoint* de um treinamento (`CheckpointStore.load_best(training_id)`, já usada exatamente pra esse fim no Caminho A — `experiments/training/core/fl_core/manual_loop.py` recarrega o melhor modelo antes de rodar o RAG) — mas essa recarga **nunca acontece no Caminho B** (produção, rede real) antes da extração de padrão. A calibração (temperatura/isotônica) já resolveu um problema parecido: os pesos *persistidos* junto com o calibrador são os da melhor rodada (correção de 2026-07-28, documentada no próprio código), mas mesmo lá o calibrador em si é ajustado sobre os *logits* da última rodada — então nem a calibração está 100% resolvida nesse sentido, só parcialmente.

**Conclusão técnica**: é viável, sim — a peça que falta (`load_best`) já existe e já é usada em outro lugar do próprio projeto pra resolver exatamente esse tipo de descasamento. O trabalho não é trivial (recarregar o melhor modelo no servidor não move os pesos pro cliente sozinho — extração de padrão roda no cliente, então seria necessário um novo ciclo de comunicação Flower enviando os pesos da melhor rodada de volta pros clientes especificamente pra esse fim, não é só trocar uma variável local), mas não exige nenhuma peça nova, só recombinar peças que já existem.

**Busca de embasamento na literatura**: não encontrada nenhuma referência que trate especificamente de "qual rodada usar para extrair padrão de interpretabilidade que alimenta um sistema de recuperação (RAG) a partir de um modelo federado" — essa combinação exata (FL + extração de atenção + RAG) parece não ter precedente direto publicado. Existe, porém, uma base indireta bem estabelecida e diretamente aplicável por analogia: a literatura clássica de *early stopping* argumenta que o critério de seleção de modelo deveria ser sempre a melhor métrica de validação vista durante o treino, nunca a última época \citep{prechelt1998} — e um trabalho recente reforça esse ponto indo além: comparando critérios de seleção pós-treino contra seleção durante o treino, mostra que examinar todas as épocas depois do fato e escolher a melhor supera qualquer critério de parada aplicado durante o treino \citep{apicella2026validation}. Achado também relevante, na direção oposta: a prática comum relatada na literatura de FL é justamente salvar só o modelo da rodada final (não o de melhor critério) — ou seja, o comportamento atual do RAG deste projeto está alinhado com o que muita implementação de FL faz por padrão, não é uma escolha peculiar deste projeto — só não é o que a literatura geral de seleção de modelo recomenda como melhor prática.

**Correção (2026-08-09, mesmo dia): a alegação empírica inicial estava errada.** A primeira versão desta seção dizia que "a classe que colapsou na última rodada tinha sinal melhor em rodadas anteriores" — isso não foi verificado antes de ser escrito. Comparando `best_round` × última rodada, **classe a classe, em todas as 5 execuções com dado utilizável** (as 3 originalmente descartadas por outro motivo, o bug do `evaluation_json`, entram aqui porque `per_class_f1` vem de `fl_round_history`, fonte não afetada por aquele bug):

| `training_id` | Cenário | `best_round` | F1 macro @ melhor | `curado_internado` @ melhor | `melhora_pronto` @ melhor | Última rodada | F1 macro @ última | `curado_internado` @ última | `melhora_pronto` @ última |
|---|---|---|---|---|---|---|---|---|---|
| 6 | Sem DP, sem *early stop* | 108 | 0,4179 | 0,149 | 0,0 | 110 | 0,4120 | 0,149 | 0,0 |
| 7 | Sem DP, *early stop* | 19 | 0,3710 | 0,0 | 0,0 | 27 | 0,3594 | 0,0 | 0,0 |
| 8 | DP uniforme, *early stop* | 6 | 0,2805 | 0,015 | 0,0 | 48 | **0,0241** | 0,0 | 0,0 |
| 10 | Sem DP, *early stop* | 33 | 0,3772 | 0,0 | 0,0 | 33 (= melhor) | — | — | — |
| 11 | Sem DP, sem *early stop* | 75 | 0,3921 | 0,0 | 0,0 | 110 | 0,3861 | 0,0 | 0,0 |

**Leitura honesta, agora com dado real:**

- **Nos 4 cenários sem DP** (6, 7, 10, 11) — com e sem *early stop* — `curado_internado` e `melhora_pronto` têm o **mesmo valor** (ou seguem em 0,0) tanto na melhor rodada quanto na última. Não há, nestes dados, nenhuma "janela de recuperação perdida" — a classe que colapsa, colapsa cedo e permanece colapsada até o fim; a classe com sinal fraco (`curado_internado` no treino 6) mantém esse mesmo sinal fraco até a última rodada. **Para o cenário sem DP, os dados disponíveis não sustentam a alegação de que o RAG estaria perdendo qualidade por usar a última rodada em vez da melhor.**
- **No único cenário DP disponível** (treino 8), o quadro é dramaticamente diferente: F1 macro cai de 0,2805 (melhor rodada, 6) para **0,0241** (última rodada, 48) — uma queda de mais de 10×, e não só nas classes raras: até `curado_pronto`, a classe majoritária, cai de 0,770 para 0,0 na última rodada. Isso **não é o mesmo fenômeno** da hipótese original (classe rara perdendo sinal) — é o padrão de "estados atratores" já documentado nesta mesma página (ruído DP fazendo o modelo oscilar e cair repetidamente em soluções degeneradas). Mas é evidência real, direta, de que sob DP a diferença entre melhor rodada e última rodada pode ser enorme — e é justamente sob DP que o RAG mais sofreria ao usar a última rodada.

**Como reportar isso, então (revisado, sem inventar base que não existe e sem manter uma alegação empírica que não se sustentou):** não há literatura específica validando "use o `best_round` pro RAG". A evidência empírica deste projeto **não sustenta** o ganho no cenário sem DP (4 execuções checadas, nenhuma mostra diferença relevante entre melhor rodada e última); **sustenta fortemente** no cenário com DP (1 execução, mas com diferença de mais de 10× em F1 macro) — pelo mecanismo já conhecido de estados atratores, não por um mecanismo novo. A frase correta pro rascunho, revisada: *"não foi encontrada base direta na literatura para esta combinação específica; a evidência empírica deste projeto é mista — não sustenta ganho no cenário sem privacidade diferencial (4 execuções verificadas), mas sustenta fortemente no cenário com privacidade diferencial, onde a última rodada pode ser drasticamente pior que a melhor por efeito já documentado de estados atratores sob ruído DP. Fica como trabalho futuro escopado: revisar o *checkpoint* usado pelo RAG especificamente para cenários com DP, não como recomendação geral."*

---

## Os treinamentos planejados

**Desenho correto (esclarecido 2026-08-09 — versões anteriores desta tabela rotulavam errado como "1ª/2ª réplica"):** os 3 cenários (`make superlink-dp-off` / `make superlink-dp-uniform` / `make superlink-dp-layer-group`) têm, cada um, **duas variações**, não duas réplicas da mesma configuração — uma **com** `FL_EARLY_STOP` e uma **sem**. É assim que dá pra comparar de forma empírica, dentro do mesmo cenário de privacidade, a última rodada × rodada de convergência × melhor rodada: com *early stop* ligado, a última rodada e a de convergência coincidem por definição; sem *early stop*, elas se separam e sobra rodada extra até o teto de 110 pra ver se a métrica melhora, piora ou entra em platô depois da convergência detectada.

**Nova dimensão adicionada (2026-08-09): peso de classe.** Achado no mesmo dia: `class_weight_overrides_json` está vazio no banco do servidor e do BPSP desta fase ("3 bancos separados") — os treinos 10-14 rodaram todos em `completo_sem_peso` (só `ClassBalancedStrategy` padrão, nenhum peso explícito de classe rara). A autora decidiu configurar pesos (Seção acima, "peso de classe") e refazer os treinos — cada cenário/variação agora tem, em potencial, duas linhas: `completo_sem_peso` (já rodada) e `completo_com_peso` (a rodar). Isso não substitui os 6 slots originais — os adiciona como uma segunda dimensão, pra permitir comparar diretamente o efeito do peso de classe sobre o mesmo cenário de privacidade.

| Cenário | Early Stop | Peso de classe | `training_id` | Status | Última rodada | Melhor rodada | Rodada convergência | Accuracy | Macro F1 | Macro AUC | ECE (pré→pós) | ε (RDP) | Duração |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sem DP | com | completo_sem_peso | 10 | 🟢 concluído (2026-08-08) | 33 | 33 | 32 | 0,7548 | 0,3772 | 0,8299 | 0,1126→0,0081 | — (DP off) | 1h22min |
| Sem DP | com | completo_com_peso | 17 | 🟢 concluído (2026-08-09) | 28 | 22 | 27 | 0,7205 | 0,3722 | 0,8006 | 0,0513→0,0177 | — (DP off) | 2h58min |
| Sem DP | sem | completo_sem_peso | 11 | 🟢 concluído (2026-08-09) | 110 | 75 | 92 | 0,7515 | 0,3921 | 0,8569 | 0,0727→0,0082 | — (DP off) | 4h24min |
| Sem DP | sem | completo_com_peso | 20 | 🟢 concluído (2026-08-10) | 110 | 96 | 47 | 0,7420 | 0,3913 | 0,8489 | 0,0847→0,0145 | — (DP off) | 4h09min |
| DP uniforme | com | completo_sem_peso | 12 | 🟢 concluído (2026-08-09) | 33 | 16 | 32 | 0,5622 | 0,1758 | 0,4549 | 0,5171→0,1655 | RDP 119,1 / simples 319,8 | 1h19min |
| DP uniforme | com | completo_com_peso | 18 | 🟢 concluído (2026-08-09) | 35 | 3 | 34 | 0,4043 | 0,1913 | 0,5154 | 0,3814→0,0573 | RDP 124,7 / simples 339,1 | 1h24min |
| DP uniforme | sem | completo_sem_peso | 13 | 🟢 concluído (2026-08-09) | 110 | 42 | 35 | 0,6661 | 0,2705 | 0,5943 | 0,0609→0,0 | RDP 318,9 / simples 1065,9 | 3h43min |
| DP uniforme | sem | completo_com_peso | 22 | 🟢 concluído (2026-08-10) | 110 | 65 | 44 | 0,5721 | 0,2005 | 0,5568 | 0,3827→0,0 | RDP 318,9 / simples 1065,9 | 3h37min |
| DP layer_group | com | completo_sem_peso | 25 (após `14`, `23` interrompidos) | 🟢 concluído (2026-08-11) | 64 | 19 | 63 | 0,6453 | 0,2906 | 0,6101 | 0,3543→0,0909 | RDP 956,6 / simples 1240,3 | 2h18min |
| DP layer_group | com | completo_com_peso | 19 | 🟢 concluído (2026-08-09) | 89 | 63 | 88 | 0,6191 | 0,2026 | 0,5006 | 0,6356→0,0950 | RDP 1286,6 / simples 1724,8 | 3h09min |
| DP layer_group | sem | completo_sem_peso | 24 | 🟢 concluído (2026-08-11) | 110 | 10 | 72 | 0,4001 | 0,2273 | 0,6888 | 0,6486→0,1176 | RDP 1563,8 / simples 2131,7 | 3h46min |
| DP layer_group | sem | completo_com_peso | 21 | 🟢 concluído (2026-08-10) | 110 | 14 | 47 | 0,6214 | 0,2374 | 0,5399 | 0,3543→0,1082 | RDP 1563,8 / simples 2131,7 | 3h44min |

**Legenda de status:** 🟡 em andamento · 🟢 concluído e validado · 🔴 interrompido/descartado · ⬜ não iniciado.

**Legenda de peso de classe:** `completo_sem_peso` = só `ClassBalancedStrategy` padrão (peso por frequência local, sem override); `completo_com_peso` = com `class_weight_overrides_json` configurado (`cost_sensitive` nas classes com override, `class_balanced` nas demais).

**Nota — colunas "Última rodada" e "Melhor rodada" lado a lado (2026-08-08):** quando o *early stop* está ligado, a última rodada é (por definição) a rodada de convergência — a tabela deixa essa relação visível de cara, sem precisar abrir cada treino pra entender se o resultado reportado é do pico real ou de onde o treino simplesmente parou.

**Nota — as 3 primeiras tentativas nesses cenários (`training_id` 6, 7, 8) foram invalidadas (2026-08-08) por um bug real, corrigido no mesmo dia** — ver seção abaixo. As métricas agregadas (accuracy/F1/AUC) delas continuam corretas (confirmado via `fl_round_history`, fonte independente do bug), mas não contam como resultado oficial (critério simples: `status='completed'` decide, sem exceção) — precisaram ser reexecutadas com o código corrigido, o que gerou `training_id` 10/11/12.

**Progresso (atualizado 2026-08-11): MATRIZ COMPLETA — 12 de 12 slots preenchidos.** `training_id=25` fechou o último slot (`DP layer_group, com early stop, sem peso`) na 3ª tentativa (`14` e `23` tinham falhado antes). A tabela acima já reflete os 12 `training_id` finais, um por linha — write-up individual de cada um nas seções abaixo.

### Qual comando `make` gerou cada treino concluído (2026-08-09)

| `training_id` | Comando `make` | `FL_EARLY_STOP` | Peso de classe |
|---|---|---|---|
| 10 | `make superlink-dp-off` | `true` | `completo_sem_peso` |
| 17 | `make superlink-dp-off` | `true` | `completo_com_peso` |
| 11 | `make superlink-dp-off` | `false` | `completo_sem_peso` |
| 12 | `make superlink-dp-uniform` | `true` | `completo_sem_peso` |
| 13 | `make superlink-dp-uniform` | `false` | `completo_sem_peso` |

**Pares com comando idêntico** (mesma configuração de privacidade e *early stop* — a única variável de configuração real entre eles é o peso de classe, ou a semente/execução quando não há par):

- **`training_id=10` × `training_id=17`** — comando **igual** (`superlink-dp-off`, `FL_EARLY_STOP=true`). Única diferença de configuração real: peso de classe (10 = sem, 17 = com) — ver comparação completa na seção do `training_id=17` acima.
- `training_id=11`, `12`, `13` — cada um é o único representante do seu comando entre os concluídos até agora, sem "gêmeo com peso" ainda.
- `superlink-dp-layer-group` — nenhum treino concluído usou esse comando ainda; é a única família sem resultado formal fechado (`training_id=14` tentou `FL_EARLY_STOP=true` mas foi interrompido).

**Agrupado por `FL_EARLY_STOP`** (a mesma tabela, cortada pela outra dimensão — útil pra comparar convergência × última rodada entre cenários de privacidade diferentes):

| `FL_EARLY_STOP=true` | Comando | Peso | | `FL_EARLY_STOP=false` | Comando | Peso |
|---|---|---|---|---|---|---|
| `training_id=10` | `superlink-dp-off` | sem peso | | `training_id=11` | `superlink-dp-off` | sem peso |
| `training_id=17` | `superlink-dp-off` | com peso | | `training_id=13` | `superlink-dp-uniform` | sem peso |
| `training_id=12` | `superlink-dp-uniform` | sem peso | | | | |

`training_id=14` (DP layer_group, `FL_EARLY_STOP=true`) ficaria nessa coluna também, mas foi interrompido antes de completar — não entra na tabela de concluídos. Não há, até agora, nenhum treino concluído de `DP layer_group` em nenhuma das duas colunas, e nenhum `FL_EARLY_STOP=false` com peso de classe.

## Peso de classe — configurado em 2026-08-09, a partir daqui os novos treinos usam

**Origem da decisão**: comentário do marido da autora (leigo na área) de que classes raras deveriam ter peso diferente pra não sumirem do treinamento — o que levou a checar se esse mecanismo já existia (existia, implementado antes desta sessão — ver `Makefile`, alvos `server-set-class-weights`/`client-set-class-weights`, e `scripts/set_class_weight_overrides.py`) e se estava ativo nos treinos formais em andamento (não estava).

**Achado que motivou a mudança**: `class_weight_overrides_json` estava vazio tanto no banco do servidor quanto no do BPSP desta fase ("3 bancos separados") — `updated_at` de ambos batendo com a data de criação do banco (2026-08-06), não alterado desde então. Confirma que os 5 treinos formais já concluídos ou em andamento até este ponto (`training_id` 10, 11, 12, 13, 14) rodaram todos só com `ClassBalancedStrategy` padrão (peso por frequência local, sem nenhum ajuste explícito de classe rara) — classificados retroativamente como `completo_sem_peso` na tabela acima.

**Literatura que sustenta o mecanismo** (validada em textos de alta confiabilidade, a pedido da autora):
- **He & Garcia (2009)**, *IEEE Transactions on Knowledge and Data Engineering* — revisão canônica sobre aprendizado com dado desbalanceado (milhares de citações).
- **King & Zeng (2001)**, *Political Analysis* — origem direta da fórmula já implementada como `ClassBalancedStrategy` (`peso = total / (num_classes × contagem)`); é a mesma fórmula usada pelo `class_weight='balanced'` do scikit-learn, que atribui a origem a este artigo.
- **Cui et al. (2019)**, CVPR — refinamento mais recente do mesmo princípio, mesmo nome de estratégia (`class-balanced loss`).
- **Elkan (2001)** — já citado no rascunho do TCC, fundamenta a estratégia `cost_sensitive` (peso explícito por julgamento, não só por frequência).
- Todas as 4 referências completas em `docs/referencias_mosaic_fl.bib`.

**Mecanismo — onde gravar importa, achado ao esclarecer dúvida da autora**: existem dois lugares possíveis, com comportamento diferente. Gravar no banco do **servidor** força o mesmo peso nos dois hospitais (autoritativo, empurrado toda rodada — ver `src/mosaicfl/core/client.py`, prioridade 1 na função `_compute_class_weights`). Gravar no banco **local de cada hospital** (BPSP e HSL separadamente) permite pesos diferentes entre eles (prioridade 2, usada quando a config do servidor está vazia). Como `melhora_pronto` precisa de peso só no BPSP (é maioria no HSL, pesar lá seria contraproducente), a gravação foi feita nos bancos locais, não no do servidor — apesar do nome do alvo (`server-set-class-weights`) sugerir o contrário; o nome se refere a "rodar a partir do desktop", não a "gravar no banco do servidor".

**Valores configurados, com justificativa** (baseados nos 4 treinos formais já concluídos: `training_id` 10, 11, 12, 13):

| Classe | BPSP | HSL | Por quê |
|---|---|---|---|
| `curado_internado` | 20 | 20 | Colapsou (F1=0) nos 4 treinos, rara nos dois hospitais (1,1-1,3% BPSP, 0,4-0,9% HSL) |
| `melhora_pronto` | 20 | sem override | Colapsou nos 4 treinos, mas só é rara no BPSP (0,4-1,1%) — maioria no HSL (61,5-80,6%) |
| `melhora_internado_grave` | 6 | 6 | Só colapsa nos cenários DP (12, 13), não nos sem-DP (10, 11) — peso mais leve, desbalanceamento moderado |

`FL_CLASS_WEIGHT_CLAMP` já estava em 25 no `.env` (ajuste de uma decisão anterior, 2026-07-27) — peso 20 não é cortado.

**Precedente e ressalva honesta**: o valor 20 pra `curado_internado` já tinha sido testado antes, numa topologia diferente (pré-"3 bancos separados") — moveu o F1 de 0,000 pra 0,171 no melhor round, mas de forma instável (voltou a 0,000 na rodada final). Não é garantia de que vai resolver o colapso desta vez, é um ponto de partida com algum precedente empírico, não um valor derivado de otimização formal.

**Execução, passo a passo (2026-08-09)**:
1. BPSP — gravado e **verificado diretamente no banco** (porta 5438): `{"curado_internado": 20, "melhora_pronto": 20, "melhora_internado_grave": 6}`, `updated_at` 17:19:47.
2. HSL — comando `make client-set-class-weights OVERRIDES='{"curado_internado": 20, "melhora_internado_grave": 6}'` executado pela autora na máquina do HSL, e confirmado via `make client-show-class-weights` (resultado igual ao esperado). Não verificado diretamente por mim (máquina não alcançável deste ambiente) — confirmação vem da própria autora, rodando a consulta na máquina do HSL.
3. **Confirmado dos dois lados** (BPSP verificado direto no banco por mim; HSL verificado pela autora via `make client-show-class-weights`) — peso de classe ativo nos dois hospitais a partir deste ponto.
4. `training_id=15` (Sem DP, com *early stop*) já estava em andamento quando o peso do BPSP foi configurado — como o peso é lido do banco a cada rodada (não só no início), esse treino ficaria com rodadas iniciais sem peso e rodadas finais com peso, misturado. Decisão: não injetar peso no meio do treino — `training_id=15` segue classificado como `completo_sem_peso`; o primeiro treino oficialmente `completo_com_peso` será o próximo, iniciado agora que os dois hospitais estão confirmados.

### Trabalho futuro: como os pesos de classe seriam sugeridos numa rede federada real (sem acesso privilegiado aos dados)

Os valores da tabela acima só puderam ser propostos porque, neste trabalho, há acesso direto — via pesquisa — às duas bases (`mosaicfl-db-bpsp-final` e o banco do HSL) para consultar a contagem real de cada classe por hospital. Numa implantação federada real de produção, esse acesso não existiria por definição: é exatamente a premissa que a arquitetura federada protege (dado clínico nunca sai do hospital, nem para fins de configuração). Vale registrar como direção de trabalho futuro, não implementado nesta pesquisa, com o que a literatura de alta confiabilidade já propõe:

- **Wang et al. (2021)**, *AAAI* — "Addressing Class Imbalance in Federated Learning": o servidor infere a distribuição de classes por rodada **sem** os clientes enviarem contagem alguma, monitorando indiretamente os gradientes da última camada (que carregam sinal correlacionado à composição de rótulos, usando um pequeno conjunto auxiliar balanceado do lado do servidor). A partir dessa distribuição inferida, propõem uma função de perda (*Ratio Loss*) que corrige o desbalanceamento sem nunca ver um rótulo bruto. É a opção mais forte em privacidade, mas mais complexa de implementar (exige o conjunto auxiliar e a lógica de monitoramento por gradiente).
- **Duan et al. (2019)**, *IEEE ICDCS* — "Astraea": os clientes enviam ao servidor (um "mediador") apenas a **contagem agregada por classe** (não os dados), e o mediador computa a distribuição global e reagenda o treinamento dos clientes por divergência KL. É mais simples de implementar — e mais alinhado à intuição de "fazer no servidor" — mas expõe mais informação que a abordagem de Wang et al. (a contagem por classe de cada cliente, ainda que sem os dados em si).
- **Agregação segura** (técnica geral de FL, não específica de classe): se a preocupação é não revelar nem a contagem individual de cada hospital, a soma das contagens por classe pode ser calculada via protocolo de agregação segura (soma multipartidária) — o servidor aprenderia só a distribuição **global agregada**, nunca a de um hospital isolado. Combinar essa técnica com a ideia do Astraea resolveria o ponto fraco de privacidade dele.

**Não há consenso único** na literatura sobre qual abordagem é superior — a escolha é um trade-off entre privacidade (Wang et al. não expõe nenhuma contagem; Astraea expõe contagem agregada por cliente; agregação segura fica no meio) e complexidade de implementação (Wang et al. exige infraestrutura de monitoramento por gradiente + dado auxiliar; Astraea + agregação seguem o mesmo padrão de "cliente computa localmente, servidor agrega e decide" já usado neste projeto para o próprio peso de classe, `_compute_class_weights`).

**Direção prática para o MOSAIC-FL, se essa pesquisa continuar**: dado que o mecanismo de override já é *server-authoritative* (prioridade 1 na função `_compute_class_weights`, ver acima), a extensão mais natural é fazer o **servidor** computar automaticamente os pesos a cada rodada a partir de contagens por classe que os clientes já reportam de qualquer forma (ou passam a reportar, via agregação segura) — sem exigir que a pesquisadora consulte manualmente o banco de cada hospital como foi feito aqui. Isso confirma a intuição inicial de que fazer no servidor tende a ser mais simples: o servidor já é o ponto natural de agregação no FedProx/FedAvg, e já envia `class_weight_overrides_json` a cada rodada — só faltaria automatizar o cálculo desse valor, hoje feito manualmente.

**Verificação de viabilidade no código atual (2026-08-09)**: conferido diretamente em `src/mosaicfl/core/client.py` — a contagem por classe **já é calculada** localmente em `_compute_class_weights` (aparece até no log, `counts={0: 23487, ...}`), mas hoje nunca sai do cliente: o retorno de `fit()` só inclui `{loss, tau, grad_norm, dp_update_norm, resource_metrics}`. Ou seja, a versão "proposta pelo cliente" (estilo Astraea) está a uma distância pequena de ser implementável: (1) cliente passa a incluir as contagens no retorno do `fit()` (ex.: `class_counts_json`, mesmo padrão já usado em `vocab_candidates_json` — nunca expõe registro bruto, só o agregado por classe); (2) servidor ganha uma nova estratégia em `class_weighting/` (coexistindo com a manual, sem substituí-la — *Strategy pattern*) que agrega as contagens recebidas no `aggregate_fit`, aplica a mesma fórmula do `ClassBalancedStrategy` já implementado, e empurra o resultado via `class_weight_overrides_json` — o canal de push para a próxima rodada já existe, é o mesmo usado pelo override manual.

A versão "proposta pelos gradientes" (Wang et al. 2021) tem uma barreira mais séria que complexidade de engenharia: ela pressupõe que o **servidor** tenha um conjunto auxiliar balanceado para inferir a distribuição a partir do gradiente da última camada — e a arquitetura do MOSAIC-FL, por decisão já tomada e documentada (ver `[[project_decisao_privacidade_sem_dp_fedavg]]`), depende explicitamente do servidor nunca ter dado de paciente algum, nem auxiliar. Dar ao servidor esse conjunto, mesmo pequeno e sintético, tensiona diretamente essa premissa.

**Decisão de escopo (2026-08-09)**: a versão "proposta pelo cliente" será implementada **depois** de fechar a matriz de treinos formais em andamento (não entra como nova dimensão experimental agora, para não atrasar o fechamento da coleta de dados) — e só se houver tempo hábil, em paralelo à escrita do rascunho. A versão "proposta pelos gradientes" fica só como trabalho futuro, com o enquadramento mais preciso: não é "implementar depois", é uma pergunta de pesquisa em aberto — **quanto essa abordagem preserva ou fere o princípio federado de privacidade** que o MOSAIC-FL adota (dado nunca sai do hospital, nem em forma agregada auxiliar), dado que ela exige que o servidor deixe de ser "cego" a qualquer sinal de distribuição de dado.

---

## `training_id=10` e `training_id=11` — cenário "Sem DP", com e sem early stop

Os dois slots oficiais do cenário "Sem DP", ambos concluídos com o código já corrigido (2026-08-08) — as métricas agregadas nunca foram afetadas pelo bug antigo, mas agora também o `per_class_f1` do `evaluation_json` está correto.

| | `training_id=10` (sem DP, com early stop) | `training_id=11` (sem DP, sem early stop) |
|---|---|---|
| Última rodada | 33 | 110 |
| Melhor rodada | 33 | 75 |
| Rodada convergência | 32 | 92 |
| Accuracy | 0,7548 | 0,7515 |
| Macro F1 | 0,3772 | 0,3921 |
| Macro AUC | 0,8299 | 0,8569 |
| ECE (pré→pós) | 0,1126→0,0081 | 0,0727→0,0082 |
| Classes colapsadas (F1=0) | 2 (`curado_internado`, `melhora_pronto`) | 2 (`curado_internado`, `melhora_pronto`) — mesmas duas |
| Duração | 1h22min | 4h24min |

**Dado registrado, sem conclusão fechada neste momento:**

- Accuracy: com *early stop* (10) fica 0,33 p.p. acima de sem *early stop* (11).
- Macro F1: sem *early stop* (11) fica 1,49 p.p. acima de com *early stop* (10).
- Macro AUC: sem *early stop* (11) fica 2,70 p.p. acima — a maior diferença entre os dois.
- Classes colapsadas: empate, 2 em cada.
- Custo de tempo: rodar as 110 rodadas em vez de parar em 33 custa 3h02min a mais (3,3× o tempo).

Uma comparação anterior, com dado hoje descartado (`training_id` 6×7, invalidados pelo bug do `evaluation_json`, mas com `per_class_f1` vindo de `fl_round_history` — fonte independente do bug, ver seção de "achados preservados" abaixo), tinha mostrado uma diferença maior em todas as métricas a favor de "sem *early stop*". Com os dois slots oficiais agora preenchidos, a diferença observada é menor em accuracy e F1, e as classes colapsadas empatam. Ainda faltam a 2ª variação de cada um dos outros dois cenários (DP uniforme, DP layer_group) para qualquer leitura mais ampla.

### Erro clinicamente relevante confirmado em `training_id=11`

A matriz de confusão da melhor rodada (75) mostra um padrão mais grave do que "F1=0" sozinho comunica: dos 1.166 casos reais de `melhora_pronto` no teste, **1.156 (99,1\%) foram classificados como `curado_pronto`** — não é apenas ausência de sinal para a classe, é um viés sistemático maciço em direção à classe majoritária do BPSP. O mesmo padrão, em menor escala, aparece em `training_id=10` (1.159 de 1.166, 99,4\%). O F1 por cliente (`fl_round_history`) confirma que o colapso é estrutural nos dois hospitais individualmente antes da agregação, não um artefato de diluição federada — ambos os clientes têm F1=0,0 local para `melhora_pronto` e `curado_internado` na rodada 75.

### `training_id=6` × `training_id=11` — mesmo cenário, contagem de colapso diferente (1 vs. 2 classes)

Pergunta levantada (2026-08-09): por que os treinos de ajuste anteriores (fase Bloco 1) e o `training_id=6` (mesmo cenário "sem DP, sem early stop" do `training_id=11`, mas invalidado por outro motivo) mostravam só **1** classe colapsada, e agora `training_id=11` mostra **2**?

Comparação direta, re-verificada no banco (não de memória) — ambos na melhor rodada de cada um:

| | `training_id=6` (melhor rodada 108) | `training_id=11` (melhor rodada 75) |
|---|---|---|
| `curado_pronto` | 0,799 | 0,809 |
| `curado_internado` | **0,149** (sinal fraco, mas real) | **0,0** (colapsado) |
| `melhora_pronto` | 0,0 (colapsado) | 0,0 (colapsado) |
| `melhora_internado_breve` | 0,643 | 0,697 |
| `melhora_internado_grave` | 0,499 | 0,454 |

`melhora_pronto` colapsa nos dois — isso é estável, consistente com o *label skew* extremo já documentado (praticamente ausente do BPSP). A diferença está em `curado_internado`: a classe mais rara de todas (82 casos no teste, ~0,8% do total, escassa nos dois hospitais). Num treino ela sobrevive com sinal fraco (0,149); no outro, colapsa por completo (0,0).

**Dado registrado:** `training_id=6` foi uma tentativa anterior para o mesmo slot que `training_id=11` preenche oficialmente hoje ("Sem DP, sem *early stop*") — descartada por outro motivo (o bug do `evaluation_json`), não por este achado. As duas são execuções distintas do mesmo cenário nominal (sementes/inicialização diferentes), e a classe mais rara é onde a diferença aparece. `training_id=6` não conta como resultado oficial, mas o valor de `curado_internado` registrado nela (0,149, vindo de `fl_round_history`, fonte não afetada pelo bug) fica aqui como dado de referência.

---

## Bug achado e corrigido: `evaluation_json.best_per_class_f1` vinha da rodada errada

**O que estava errado:** `ProductionFedProxStrategy._persist_federated_calibration()` (`infrastructure/mosaicfl_server/strategy/core.py`) salva o checkpoint da **melhor rodada** — pesos (`state_dict`), `accuracy`, `f1_macro`, `macro_auc` todos vinham corretos de `self._best_*`, atributos capturados no exato momento em que aquela rodada foi a melhor. Mas o campo `per_class_f1` dentro do `evaluation_json` lia `aggregated_metrics.get("per_class_f1_json")` — as métricas da rodada em que a **calibração** roda (normalmente a última rodada configurada, ou onde o early stop dispara), quase nunca a mesma da melhor rodada.

**Por que passou despercebido antes:** é a mesma classe de bug já corrigida em 2026-07-28 (checkpoint salvando pesos da última rodada em vez da melhor) — só que aquela correção cobriu `state_dict`/`accuracy`, e `per_class_f1` ficou de fora por não ter sido notado na hora.

**Como foi achado:** comparando manualmente o `per_class_f1` da melhor rodada (via `fl_round_history`, fonte independente) contra o `evaluation_json.best_per_class_f1` do checkpoint, pra 3 treinos reais (`training_id` 6, 7 e 8) — os valores nunca batiam com a melhor rodada, e sempre batiam **exatamente** (float idêntico) com a última rodada de cada treino.

| `training_id` | `best_round` real | `per_class_f1` real da melhor rodada | `per_class_f1` gravado no `evaluation_json` | De qual rodada esse valor realmente é |
|---|---|---|---|---|
| 6 | 108 | `[0.799, 0.149, 0.0, 0.643, 0.499]` | `[0.802, 0.149, 0.0, 0.611, 0.499]` | rodada 110 (última) |
| 7 | 19 | `[0.802, 0.0, 0.0, 0.581, 0.471]` | `[0.802, 0.0, 0.0, 0.536, 0.458]` | rodada 27 (última) |
| 8 | 6 | `[0.770, 0.015, 0.0, 0.527, 0.090]` | `[0.0, 0.0, 0.0, 0.0, 0.121]` | rodada 47 (colapsada) |

**Correção aplicada (2026-08-08):** novo atributo `self._best_per_class_f1`, capturado no mesmo bloco/instante que `self._best_state_dict` (junto com o resto dos `self._best_*`), consumido por `_persist_federated_calibration()` no lugar de `aggregated_metrics`. Fallback preservado pra recovery de treinos antigos que reiniciaram no meio sem passar pelo bloco de melhora nesta run. 2 testes de regressão novos em `tests/unit/test_persist_federated_calibration.py` (um confirma que usa `_best_per_class_f1` mesmo com `aggregated_metrics` diferente, outro confirma o fallback) — sem esses testes, a próxima pessoa mexendo nessa função podia reintroduzir o mesmo bug sem perceber. Regressão completa: 1053 passando (4 falhas pré-existentes, sem relação — ambiente Docker específico, `/app` não existe neste sandbox).

**Escopo do bug:** só o campo `per_class_f1` dentro de `evaluation_json` (usado por quem lê o checkpoint pra saber o desempenho por classe da melhor rodada). `accuracy`/`f1_macro`/`macro_auc`/os pesos do modelo salvo nunca foram afetados — sempre estavam certos. Meça a análise por classe feita nesta fase (comparação treino 6×7, achado das classes colapsadas) usou `fl_round_history` diretamente, não esse campo — não foi afetada pelo bug, mas os 3 treinos foram invalidados mesmo assim por decisão da autora (regra simples: `status='completed'` é o único critério de "conta como oficial", sem exceção caso a caso).

---

## Achados preservados dos treinos invalidados (não contam como resultado oficial, mas o método/achado continua válido)

Os números abaixo vêm de `fl_round_history` (não do `evaluation_json` com bug) — tecnicamente corretos, mas pertencem a treinos com `status='invalid'`. Mantidos aqui como referência/direção de pesquisa, não como resultado citável do treinamento formal. **Terão que ser replicados nos treinos oficiais que ainda vão rodar.**

### Comparação `training_id=6` (sem DP) × `training_id=8` (DP uniforme) × `training_id=7` (sem DP + early stop)

| | Treino 6 (sem DP, sem early stop) | Treino 7 (sem DP, early stop) | Treino 8 (DP uniforme, early stop) |
|---|---|---|---|
| Última rodada | 110 | 27 | 48 |
| Melhor rodada | 108 | 19 | 6 |
| Rodada convergência | 46 | 26 | 47 |
| Accuracy | 0,7306 | 0,7157 | 0,6727 |
| Macro F1 | 0,4179 | 0,3710 | 0,2805 |
| Macro AUC | 0,8422 | 0,8289 | 0,5834 |
| Classes colapsadas (F1=0) na melhor rodada | 1 (`melhora_pronto`) | 2 (`curado_internado`, `melhora_pronto`) | 4 de 5 (só `melhora_internado_grave` com sinal) — mas isso é da rodada 47, não da 6 (ver estados atratores abaixo) |
| ε (RDP) | — (DP off) | — (DP off) | 160,84 |

### `FL_EARLY_STOP=true` no cenário sem DP — perde qualidade real, mas não por ruído

**Pergunta:** o cenário sem DP também perde qualidade se parar no momento em que a convergência é detectada (mesma classe de risco já documentada pra cenários com DP)?

**Resultado:** sim — treino 7 (early stop) fica 4,7 p.p. de F1 macro e 1,5 p.p. de accuracy atrás do treino 6 (sem early stop), e tem 2 classes colapsadas contra 1. Comparei tanto na "melhor rodada de cada treino" quanto nas mesmas rodadas absolutas (19, 26, 27) nos dois — o padrão se repete nos dois recortes, não é coincidência de um ponto isolado.

**Conclusão:** o cenário sem DP também perde com early stop, mas por mecanismo diferente do DP — não é ruído afogando sinal, é parar durante um platô que só se resolve com mais rodadas (ver análise de platô abaixo). Sem custo de privacidade acumulando, não há nada compensando essa perda — confirma empiricamente a recomendação já existente no tutorial (`FL_EARLY_STOP` só recomendado com DP).

### Quantas rodadas a mais são necessárias pra sair do platô? (análise sobre `training_id=6`)

**Contexto:** treinos 6 e 7 têm F1 macro parecido (~0,35–0,37) entre as rodadas 19 e 46 — platô visível a olho nu. O treino 6, que continuou depois, só atinge seu melhor resultado (F1=0,4179) na rodada 108.

**Método, com justificativa de cada escolha:**
1. **Janela do platô = rodadas 19–46**, não arbitrária: 19 é o `best_round` do treino 7 e 46 é a `convergence_round` do treino 6 — os dois extremos vêm de critérios já usados pelo sistema, não escolhidos a dedo.
2. **Limiar de rompimento = média do platô + N desvios-padrão**, testado com N∈{1,2,3} e 2 valores fixos (0,39/0,40) **de propósito** — escolher um limiar só depois de ver o resultado seria viés de confirmação; testar vários e ver se convergem é o controle contra isso.
3. **Média móvel de 5 rodadas**, não o valor bruto — o F1 por rodada tem ruído real (desvio-padrão de 0,0156 já dentro do próprio platô).
4. **"Rompimento sustentado"** = a média móvel ultrapassa o limiar e não recai por mais de 2 rodadas seguidas depois — descarta pico passageiro.

**Resultado (sensibilidade ao limiar):**

| Limiar | Rodada de rompimento | Rodadas além da convergência (46) | Rodadas além de onde o treino 7 parou (27) |
|---|---|---|---|
| μ+1σ (fraco, 68% cobertura) | 69 | +23 | +42 |
| μ+2σ | 104 | +58 | +77 |
| μ+3σ (rígido) | 108 | +62 | +81 |
| Valor fixo 0,39 | 97 | +51 | +70 |
| Valor fixo 0,40 | 107 | +61 | +80 |

Critérios de 2σ+ convergem na faixa **rodada 97–108** (~50–62 rodadas além da convergência detectada). **Limitação explícita:** trajetória única (`training_id=6`, hoje descartado), descritiva, não generalizável. `training_id=11` (mesmo slot oficial, "Sem DP, sem *early stop*") é o dado equivalente disponível hoje para conferir se esse rompimento de platô se repete — comparação ainda não feita nesta seção.

### Estados atratores sob DP — reprodução do padrão do `training_id=85` (achado anterior)

No treino 8 (DP uniforme), o modelo nunca mais chegou perto da qualidade da rodada 6 (pico real) depois dela — oscila violentamente (F1 de 0,003 a 0,19) e **repete valores idênticos** (floats exatos, não aproximados) entre rodadas diferentes:

| `f1_macro` (valor exato) | Nº de rodadas | Rodadas |
|---|---|---|
| 0,024112567 | 11 | 20, 26, 28, 31, 36, 37, 42, 45, 46, 47, 48 |
| 0,07507446 | 4 | 16, 18, 21, 22 |
| 0,13848834 | 4 | 14, 17, 30, 35 |
| 0,0031969242 | 2 | 33, 34 |

**43,8% das 48 rodadas** presas em 4 estados repetidos — quase idêntico ao "41% das rodadas" já documentado pro `training_id=85` (memória do projeto, mesmo fenômeno, treino diferente). **Justificativa do método:** agrupamento por igualdade EXATA de ponto flutuante — coincidência de vários dígitos decimais entre rodadas é improvável por acaso, só acontece se o modelo agregado for genuinamente idêntico entre elas (o treino ciclando entre os mesmos pontos do espaço de parâmetros). O `ConvergenceTracker` interpretou a sequência final (45–48, 4 rodadas idênticas) como "estabilidade" — é estabilidade num estado degenerado, não uma boa solução.

**Boa notícia, independente do bug do `evaluation_json`:** o `checkpoint_criterion` funcionou certo — salvou a rodada 6 (pico real) como `best_round`, não a 47/48 (estado colapsado). O bug corrigido acima estava só no campo auxiliar `per_class_f1` do relatório, nunca nos pesos do modelo de fato salvo/servido.

### `training_id=12` (2026-08-09) — mesmo padrão, agora em treino OFICIAL

`training_id=12` (DP uniforme, com *early stop* — preenche a tabela principal) reproduz o mesmo fenômeno, com números muito próximos: F1 macro oscila de 0,003 a 0,176 ao longo de 33 rodadas, com 3 estados repetidos por igualdade exata de ponto flutuante:

| `f1_macro` (valor exato) | Nº de rodadas | Rodadas |
|---|---|---|
| 0,07507446 | 5 | 23, 25, 30, 31, 32 |
| 0,027232518 | 4 | 14, 15, 21, 26 |
| 0,13848834 | 2 | 13, 18 |

**33,3% das 33 rodadas** presas em 3 estados repetidos — mesmo padrão já registrado em `training_id=85` (~41%) e `training_id=8` (~43,8%, descartado), agora observado em `training_id=12`, que conta oficialmente para os 6 formais. Dado consistente com a decisão já registrada no rascunho do TCC de não apoiar a garantia de privacidade do MOSAIC-FL neste mecanismo (Seção `sec:dp-decisao`).

**Efeito direto sobre a pergunta "vale a pena usar `best_round` no RAG?"**: aqui sim, com dado 100% oficial. F1 macro cai de 0,1758 (rodada 16, melhor) para 0,0571 (rodada 33, última) — queda de 67,5%. `curado_pronto`, a classe mais comum, vai de F1=0,674 na melhor rodada para **0,0** na última — não é só classe rara sendo perdida, é o modelo inteiro caindo num estado degenerado. Some-se a `training_id=8` (informal) e a evidência de que, sob DP, `best_round` importa muito mais que a última rodada agora tem **duas execuções independentes**, uma delas oficial.

#### Dado obtido de log, recomputado — indicado registrar para a evolução do trabalho: ε por rodada em `training_id=12`

**Origem do dado**: `metrics.fl_trainings.dp_epsilon_rdp` só é gravado no banco **uma vez, no final do treino** (rodada de melhor critério ou conclusão) — não existe uma coluna com o ε por rodada. O log do `ServerApp` também só imprime o ε *simples* por rodada, não o RDP. O valor abaixo **não veio direto de nenhum log nem do banco** — foi recalculado localmente, replicando exatamente a mesma chamada do `RDPAccountant` (biblioteca `opacus`) usada no código de produção (`infrastructure/mosaicfl_server/strategy/core.py`, `sample_rate=1,0`, `noise_multiplier=0,5`, 1 passo por rodada), rodada a rodada de 1 a 33. Conferido: o valor recalculado na rodada 33 (119,088) bate com o `dp_epsilon_rdp=119,08837` gravado no banco — confirma que a reconstrução está correta.

| Rodada | ε RDP (recalculado) | Rodada | ε RDP (recalculado) |
|---|---|---|---|
| 1 | 10,73 | 18 | 75,02 |
| 2 | 16,51 | 19 | 78,12 |
| 3 | 21,45 | 20 | 81,12 |
| 4 | 25,93 | 21 | 84,12 |
| 5 | 30,13 | 22 | 87,12 |
| 6 | 34,13 | 23 | 90,12 |
| 7 | 37,93 | 24 | 93,12 |
| 8 | 41,65 | 25 | 96,12 |
| 9 | 45,25 | 26 | 99,12 |
| 10 | 48,80 | 27 | 102,12 |
| 11 | 52,20 | 28 | 105,09 |
| 12 | 55,60 | 29 | 107,89 |
| 13 | 59,00 | 30 | 110,69 |
| 14 | 62,22 | 31 | 113,49 |
| 15 | 65,42 | 32 | 116,29 |
| 16 | 68,62 (melhor rodada) | 33 | 119,09 (última rodada) |
| 17 | 71,82 | | |

**Dado registrado, sem conclusão fechada**: sob o critério de interpretação já usado neste projeto (ε≤10 = proteção formal significativa), nenhuma rodada deste treino específico ficou dentro do limiar — a rodada 1 já está em 10,73. Esse mesmo cálculo pode ser reaplicado a qualquer outro treino com DP (histórico de `noise_multiplier` por rodada precisa estar disponível — está, via o log do `ServerApp` ou a estratégia de ruído configurada) sem precisar re-treinar nada.

#### `training_id=13` (DP uniforme, sem *early stop*) — concluído, preenche o 4º dos 6 slots oficiais

Mesma configuração de ruído de `training_id=12` (`UniformNoiseStrategy`, σ=0,5, sem partição por grupo de camada). Rodou as 110 rodadas completas (sem *early stop*, ao contrário do `training_id=12`, que parou na 33).

| | Valor |
|---|---|
| Melhor rodada | 42 |
| Rodada de convergência | 35 |
| Última rodada | 110 |
| Accuracy (melhor rodada) | 0,6661 |
| Macro F1 (melhor rodada) | 0,2705 |
| Macro AUC (melhor rodada) | 0,5943 |
| ECE pré→pós | 0,0609→0,0 |
| ε RDP / simples (final, 110 rodadas) | 318,86 / 1065,86 |
| Duração | 3h43min |

**Per-class F1 na melhor rodada (42)**: `[curado_pronto=0,740, curado_internado=0,0, melhora_pronto=0,0, melhora_internado_breve=0,613, melhora_internado_grave=0,0]` — **3 das 5 classes colapsadas**, mesmo na melhor rodada do treino. A matriz de confusão confirma: dos 1.166 casos reais de `melhora_pronto`, 1.023 foram previstos como `curado_pronto`; dos 653 casos de `melhora_internado_grave`, nenhum foi acertado (15 foram para `curado_pronto`, 638 para `melhora_internado_breve`); dos 82 casos de `curado_internado`, nenhum foi acertado.

**Comparação com `training_id=12`** (mesmo cenário nominal, `com` em vez de `sem` *early stop*): a melhor rodada de `training_id=13` (42) é **melhor** que a de `training_id=12` (16) em todas as métricas — accuracy 0,666 vs.\ 0,562, F1 macro 0,271 vs.\ 0,176 — ou seja, sob DP uniforme, continuar treinando além do ponto em que o *early stop* teria parado **encontrou um pico melhor** nesta réplica. Ao mesmo tempo, `training_id=12` também tinha 3 classes colapsadas na sua melhor rodada (`curado_internado`, `melhora_pronto`, `melhora_internado_grave`, idêntico padrão) — o colapso de múltiplas classes sob DP uniforme agora está confirmado em **duas execuções oficiais independentes**, não é um resultado isolado.

**Última rodada (110) comparada à melhor (42)**: F1 macro cai de 0,2705 para 0,0751 — queda de 72,3\%, maior ainda que a de `training_id=12` (67,5\%). Na última rodada, até `curado_pronto` (a classe mais comum) vai a zero — só `melhora_internado_breve` sobrevive com sinal (0,375). **Terceira execução oficial/informal seguida em que a última rodada é dramaticamente pior que a melhor sob DP** (`training_id=8` informal, `training_id=12` oficial, `training_id=13` oficial) — o padrão de que `best_round` importa mais que a última rodada sob DP está, agora, replicado três vezes.

**Sobre "qual rodada ainda tinha proteção real"**: recalculando o RDP até a rodada 42 (mesma métodologia de antes, replicando o `RDPAccountant`) dá ε≈144,3 — já muito acima do limiar de 10. Como a rodada 1 já cruza esse limiar (ε≈10,73, achado anterior), **não há nenhuma rodada com proteção formal significativa neste treino também** — o mesmo resultado do `training_id=12`, agora confirmado numa segunda execução completa.

---

## `training_id=17` — primeiro treino oficial `completo_com_peso` (Sem DP, com early stop)

Mesmo slot nominal de `training_id=10` (Sem DP, com *early stop*), primeira execução com os pesos de classe configurados (`curado_internado=20`, `melhora_pronto=20` só no BPSP, `melhora_internado_grave=6` — ver seção "Peso de classe" acima). Convergiu e parou por *early stop* na rodada 27, `best_round=22`, 28 rodadas rodadas ao todo (de um máximo de 110), duração 2h58min.

| | `training_id=10` (sem peso) | `training_id=17` (com peso) |
|---|---|---|
| Melhor rodada | 33 | 22 |
| Rodada de convergência | — | 27 |
| Accuracy (melhor rodada) | 0,7548 | 0,7205 |
| Macro F1 (melhor rodada) | 0,3772 | 0,3722 |
| Macro AUC (melhor rodada) | 0,8299 | 0,8006 |
| ECE pré-calibração | 0,1126 | 0,0513 |
| ECE pós-calibração | 0,0081 | 0,0177 |

**Per-class F1, melhor rodada** — `[curado_pronto, curado_internado, melhora_pronto, melhora_internado_breve, melhora_internado_grave]`:
- `training_id=10` (sem peso): `[0,809, 0,0, 0,0, 0,715, 0,362]`
- `training_id=17` (com peso): `[0,802, 0,0, 0,0, 0,609, 0,450]`

**As duas classes com peso mais alto (20) não saíram do colapso.** A matriz de confusão de `training_id=17` mostra 0 acertos para `curado_internado` (82 casos reais: 16→`curado_pronto`, 51→`melhora_internado_breve`, 15→`melhora_internado_grave`) e 0 acertos para `melhora_pronto` (1.166 casos reais: 1.147→`curado_pronto`). Idêntico ao padrão sem peso — o peso=20 não moveu nenhuma predição, em nenhum dos dois hospitais (`per_client_f1_json` confirma F1=0,0 para as duas classes nos dois clientes, nos dois treinos).

**A classe com peso moderado (`melhora_internado_grave`, peso=6, aplicado nos dois hospitais) teve efeito real e bilateral.** Comparando o F1 por cliente entre os dois treinos, na melhor rodada de cada um:

| Cliente | `training_id=10` (sem peso) | `training_id=17` (com peso) |
|---|---|---|
| Cliente com `curado_pronto` baixo (≈0,03 — provável HSL, minoria de `curado_pronto` nesse hospital) | F1 `melhora_internado_grave` = 0,074 | F1 `melhora_internado_grave` = 0,294 |
| Cliente com `curado_pronto` alto (≈0,94-0,95 — provável BPSP) | F1 `melhora_internado_grave` = 0,414 | F1 `melhora_internado_grave` = 0,478 |

Subiu nos dois hospitais de forma consistente — não é ruído de um cliente só. Em troca, `melhora_internado_breve` caiu um pouco nos dois clientes no mesmo comparativo (0,691→0,641 e 0,720→0,603) — sugere troca de sinal entre as duas classes de gravidade adjacentes (*breve* ↔ *grave*), não um ganho livre de custo.

**Leitura honesta, sem generalizar de uma única execução**: peso alto (20) nas duas classes mais raras não resolveu o colapso estrutural — efeito nulo, não parcial. Peso moderado (6) numa classe menos extrema moveu sinal real e replicado nos dois hospitais, mas à custa da classe vizinha mais comum. Accuracy, AUC e ECE pós-calibração pioraram; ECE pré-calibração melhorou bastante. Uma única execução por braço — não há réplica ainda para saber se esse padrão se repete.

**Observação adicional (2026-08-09): convergência em menor quantidade de rodadas.** `training_id=17` (com peso) convergiu na rodada 27; `training_id=10` (mesmo comando `make`, mesmo `FL_EARLY_STOP=true`, sem peso) convergiu na rodada 32 — 5 rodadas a menos (~15%). Dado registrado como está, sem hipótese de mecanismo causal (não pesquisado ainda) e sem generalizar — é **um único par de execuções**, e já vimos neste mesmo documento pares do mesmo cenário nominal convergirem em pontos bem diferentes só por variação de semente/inicialização (`training_id=6` × `training_id=11`). O treino `DP uniforme, com peso, com early stop` em andamento no momento deste registro, comparado a `training_id=12` (mesmo cenário, sem peso), vai dar um segundo ponto — se repetir o padrão de menos rodadas até convergir, deixa de ser um dado isolado.

---

## `training_id=18` — DP uniforme, com peso, com early stop

Mesmo slot nominal de `training_id=12` (DP uniforme, com *early stop*), primeira execução com peso de classe. Convergiu (por detecção de estado atrator, ver abaixo) na rodada 34, parou na 35. `best_round=3` — pico muito precoce, atípico frente aos outros treinos deste documento.

| | `training_id=12` (sem peso) | `training_id=18` (com peso) |
|---|---|---|
| Melhor rodada | 16 | **3** |
| Rodada de convergência | 32 | 34 |
| Accuracy (melhor rodada) | 0,5622 | 0,4043 |
| Macro F1 (melhor rodada) | 0,1758 | 0,1913 |
| Macro AUC (melhor rodada) | 0,4549 | 0,5154 |
| ECE pré→pós | 0,5171→0,1655 | 0,3814→0,0573 |
| ε RDP / simples | 119,1 / 319,8 | 124,7 / 339,1 |
| Duração | 1h19min | 1h24min |

**Trajetória muito mais instável que o par sem peso.** `f1_macro` por rodada de `training_id=18` oscila entre 0,015 e 0,191 de forma não-monotônica ao longo de todo o treino — não é um platô que se aproxima de um pico, é ruído mesmo depois da rodada 3: `[r1=0,057, r2=0,017, r3=0,191, r4=0,170, r5=0,024, r6=0,097, r7=0,024, r8=0,152, r9=0,116, r10=0,075, ...]`. `accuracy` acompanha a mesma volatilidade (0,014 a 0,598 rodada a rodada). Isso é bem mais caótico do que `training_id=12` (que tinha uma trajetória mais suave até o pico na rodada 16).

**Novo caso de estado atrator, agora sob "DP uniforme + peso de classe".** As rodadas 32, 33 e 34 têm **valores idênticos** de `f1_macro` (0,024112567) e `accuracy` (0,0641769) — três casas decimais batendo exatamente, o padrão já documentado em `training_id=85`, `training_id=8` e `training_id=12` (achado "Estados atratores sob DP", seção acima). A convergência detectada na rodada 34 é esse padrão cíclico se repetindo, não uma melhora real estabilizando — quinta ocorrência confirmada do mesmo fenômeno, agora numa configuração ainda não testada antes (com peso de classe).

**Per-class F1, melhor rodada** — `[curado_pronto, curado_internado, melhora_pronto, melhora_internado_breve, melhora_internado_grave]`:
- `training_id=12` (sem peso, rodada 16): `[0,674, 0,0, 0,0, 0,204, 0,0]`
- `training_id=18` (com peso, rodada 3): `[0,475, 0,0133, 0,0, 0,367, 0,102]`

`curado_internado` sai de F1=0,0 pra F1=0,0133 — primeira vez, em qualquer treino deste documento, que essa classe tem F1 não-nulo. Mas o número em si é minúsculo: *recall* de 1,22% (1 acerto em 82 casos reais, confirmado na matriz de confusão — `[27, **1**, 0, 49, 5]` na linha de `curado_internado`). Não dá pra chamar de "resolvido" nem "melhorou de forma relevante" — é sinal, não é solução.

**Observação (2026-08-09): menor número de classes zeradas de todos os 6 treinos formais concluídos até agora.**

| `training_id` | Cenário | Classes com F1=0 na melhor rodada |
|---|---|---|
| 10 | Sem DP, sem peso | 2 (`curado_internado`, `melhora_pronto`) |
| 11 | Sem DP, sem peso | 2 (mesmas) |
| 17 | Sem DP, com peso | 2 (mesmas) |
| 12 | DP uniforme, sem peso | 3 (+ `melhora_internado_grave`) |
| 13 | DP uniforme, sem peso | 3 (mesmas) |
| 18 | DP uniforme, com peso | **1** (só `melhora_pronto`) |

`training_id=18` tem menos classes colapsadas que qualquer outro treino formal, incluindo os cenários "Sem DP" (que, em toda métrica agregada, são muito melhores). Duas leituras corretas e simultâneas, sem uma cancelar a outra: (1) é a rodada com menos colapso estrutural de toda a tabela; (2) é também a rodada com a pior accuracy geral (0,4043) — porque veio de um pico isolado numa trajetória caótica (a `best_round=3`, antes da instabilidade descrita acima dominar o resto do treino), não de um treino que estabilizou bem. Não há como separar, com um único dado, se foi o peso de classe que espalhou o sinal entre mais classes, ou coincidência de onde o ruído do DP bateu naquela rodada específica — mas é um padrão que vale a pena olhar de novo se mais réplicas de `DP uniforme com peso` forem executadas.

**Leitura honesta**: este treino não é comparável ponto a ponto com `training_id=17` (Sem DP) — o cenário DP uniforme já é, por si, muito mais instável (achado de "estados atratores" preexistente, sem relação com peso de classe). A `best_round=3` provavelmente captura um pico de ruído favorável, não um ponto de aprendizado real consolidado — a trajetória posterior nunca mais chega perto desses valores. Accuracy piorou (0,562→0,404), macro F1 e AUC subiram um pouco, mas dentro da faixa de ruído já observada na trajetória. Não dá pra afirmar que o peso ajudou ou atrapalhou aqui — o cenário DP uniforme parece dominado por instabilidade estrutural própria, que o peso de classe não muda visivelmente.

---

## `training_id=19` — primeiro treino `DP layer_group` concluído (com peso, com early stop)

Primeiro treino da família `DP layer_group` a chegar até o fim — `training_id=14` (mesmo slot, sem peso) foi interrompido na rodada 11 e nunca completou, então ainda não existe par direto `sem_peso`×`com_peso` neste cenário. 89 rodadas, `best_round=63`, convergência detectada na rodada 88 — 25 rodadas depois do melhor, o maior desses gaps já visto no documento (maior até que o de `training_id=11`, 17 rodadas, citado na tela `/fl-training-results`).

| Métrica | Valor |
|---|---|
| Accuracy (melhor rodada) | 0,6191 |
| Macro F1 | 0,2026 |
| **Macro AUC** | **0,5006** |
| ECE pré-calibração | 0,6356 |
| ECE pós-calibração | 0,0950 |
| ε RDP / simples | 1286,6 / 1724,8 |
| Duração | 3h09min |

**Resultado ruim na maioria das métricas, sem suavizar.** `macro_auc=0,5006` é o pior de todo o documento — nível de chance, o modelo não separa as classes melhor que sorteio aleatório na melhor rodada dele. `ε_RDP=1286,6` é de longe o maior epsilon já registrado neste projeto (mais de 3× o pior valor anterior, `training_id=13` com 318,9) — sem proteção prática de privacidade nenhuma, mesmo padrão já documentado (`[[project_decisao_privacidade_sem_dp_fedavg]]`), agora num extremo bem mais severo. `ece_pre=0,6356` é a pior calibração pré-ajuste de todo o documento.

**Per-class F1, melhor rodada**: `[curado_pronto=0,716, curado_internado=0,0, melhora_pronto=0,0, melhora_internado_breve=0,256, melhora_internado_grave=0,041]` — as mesmas duas classes de sempre colapsadas (2, não 3 ou mais — na melhor rodada especificamente, apesar de rodadas vizinhas na trajetória chegarem a colapsar 4 das 5, ver observação de trajetória abaixo).

**Trajetória oscilou de forma severa até a rodada 63.** Registro em tempo real, rodada a rodada: rodada 8 foi a única com as 5 classes simultaneamente não-zeradas (sinal espalhado fino, `f1_macro=0,066`, não um bom resultado — ver observação anterior); rodadas 52-53 tiveram 3-4 classes zeradas; rodadas 60-61 alternaram entre "só a classe majoritária sobrevive" (`curado_pronto=0,692`, resto zerado) e "só uma classe minoritária fraca sobrevive" (`melhora_internado_grave=0,117`, resto zerado) — nunca as duas coisas boas ao mesmo tempo. É a trajetória mais volátil de todas registradas neste documento até agora.

**Por cliente, na melhor rodada**: um hospital com `curado_pronto` F1=0,840, o outro com **0,027** — diferença de ordem de grandeza, mais consistente com ruído do que com aprendizado real nesse cliente específico para essa classe.

**Leitura honesta**: este treino não teve nenhum aspecto claramente positivo. É o pior resultado de privacidade (ε), o pior AUC e a pior calibração pré-ajuste do documento inteiro — preenche uma lacuna real (primeiro `DP layer_group` concluído), mas o dado em si é negativo em quase toda dimensão avaliada. Sem par `completo_sem_peso` concluído no mesmo cenário ainda, não dá pra saber se o peso piorou algo aqui ou se `DP layer_group` é assim mesmo, sem peso nenhum.

---

## `training_id=20` — segundo par direto Sem DP (com peso), replica o achado do primeiro par

Mesmo slot nominal de `training_id=11` (Sem DP, sem *early stop*), com peso de classe. 110/110 rodadas completas, `best_round=96`, convergência detectada na rodada 47.

| | `training_id=11` (sem peso) | `training_id=20` (com peso) |
|---|---|---|
| Melhor rodada | 75 | 96 |
| Rodada de convergência | 92 | 47 |
| Accuracy | 0,7515 | 0,7420 |
| Macro F1 | 0,3921 | 0,3913 |
| Macro AUC | 0,8569 | 0,8489 |
| ECE pré→pós | 0,0727→0,0082 | 0,0847→0,0145 |
| Duração | 4h24min | 4h09min |

Praticamente empatados no agregado — todas as métricas ligeiramente piores com peso, diferenças pequenas.

**Per-class F1, melhor rodada**: `training_id=11`: `[0,809, 0,0, 0,0, 0,697, 0,454]` — `training_id=20`: `[0,810, 0,0, 0,0, 0,643, 0,503]`.

**Achado replicado (2026-08-10): `curado_internado` e `melhora_pronto` seguem em F1=0,0, agora confirmado em DOIS pares independentes** (`10`×`17` e `11`×`20`) — quatro execuções completas, peso=20 nunca moveu uma única predição pra essas duas classes. O mesmo trade-off entre classes vizinhas também se repete: `melhora_internado_grave` sobe (0,454→0,503), `melhora_internado_breve` cai (0,697→0,643) — o padrão bilateral já visto no primeiro par volta a aparecer aqui, com números bem próximos inclusive (o primeiro par tinha ido de 0,362→0,450 e 0,715→0,609). Por cliente, a mesma assimetria estrutural continua: um hospital com `curado_pronto` F1=0,951, o outro 0,031 — igual ao padrão de todos os outros treinos "Sem DP" deste documento.

**Convergência × melhor rodada, direção oposta à de `training_id=11`.** Em `training_id=11`, a convergência (92) veio *depois* do melhor ponto (75) — 17 rodadas de atraso. Em `training_id=20`, a convergência (47) veio bem *antes* do melhor ponto (96) — 49 rodadas de antecedência, na direção contrária. Como `FL_EARLY_STOP=false` nos dois, isso não interrompeu nenhum dos dois treinos, só mostra que o `ConvergenceTracker` (baseado em estabilidade do `f1_macro`) pode disparar cedo demais ou tarde demais em relação ao pico real — mais um dado a favor de tratar convergência e melhor rodada como sinais distintos (ver seção "É viável usar o `best_round` no RAG?", no início deste documento).

**Degradação da última rodada é bem mais branda que sob DP.** `f1_macro` cai de 0,3913 (rodada 96) pra 0,3835 (rodada 110) — queda de só ~2%, nada parecido com os 60-70%+ de queda vistos nos cenários DP (`training_id=8`, `12`, `13`). Confirma, com mais um dado, que o descolamento entre última rodada e melhor rodada é essencialmente um fenômeno de DP, não algo geral do treinamento federado.

---

## `training_id=21` — DP layer_group, com peso, sem early stop

110/110 rodadas completas, `best_round=14` (pico bem precoce, como em `training_id=18`), convergência detectada na rodada 47 — 33 rodadas *depois* do melhor ponto.

| Métrica | Valor |
|---|---|
| Accuracy (melhor rodada) | 0,6214 |
| Macro F1 | 0,2374 |
| Macro AUC | 0,5399 |
| ECE pré→pós | 0,3543→0,1082 |
| ε RDP / simples | 1563,8 / 2131,7 |
| Duração | 3h44min |

Per-class F1 na melhor rodada: `[0,731, 0,0, 0,0, 0,168, 0,289]` — as duas classes de sempre colapsadas. Sem par `completo_sem_peso` no mesmo cenário ainda (nenhum "DP layer_group sem early stop" rodou sem peso).

**Degradação severa até a última rodada**: `f1_macro` cai de 0,2374 (rodada 14) pra 0,0751 (rodada 110) — queda de 68%, e a última rodada tem só `melhora_internado_breve` com sinal (0,375), as outras 4 zeradas.

## Achado maior: estados atratores são idênticos ENTRE treinos diferentes, não só dentro de cada um

Até agora, "estados atratores sob DP" significava: dentro de um mesmo treino, o `f1_macro` repete o mesmo valor em rodadas não-consecutivas. Comparando os **7 treinos sob DP** já concluídos (`training_id` 18, 19, 21, 22, 23 — interrompido, mas com 79 rodadas de dado —, 24 e 25; DP uniforme e DP layer_group, com e sem peso, com e sem *early stop*, sementes/execuções distintas), apareceu algo mais forte: **o mesmo vetor de F1 por classe, idêntico até a 15ª casa decimal, se repete entre treinos diferentes.** Consulta final, definitiva, contra os 7 treinos de uma vez:

| Estado (per-class F1) | 18 | 19 | 21 | 22 | 23\* | 24 | 25 | **Total** |
|---|---|---|---|---|---|---|---|---|
| `[0, 0, 0, 0, 0,12056283799036835]` (só `melhora_internado_grave`) | 4 | 16 | 24 | 47 | 9 | 25 | 4 | **129** |
| `[0, 0, 0, 0,37537229765957636, 0]` (só `melhora_internado_breve`) | — | 7 | 23 | 17 | 11 | 22 | 3 | **83** |
| `[0,6924416983388537, 0, 0, 0, 0]` (só `curado_pronto`) | 1 | 7 | 11 | 7 | 9 | 12 | 5 | **52** |

\* `training_id=23` foi interrompido (processo zumbi), mas as 79 rodadas que rodaram antes de travar contam normalmente.

**Três estados degenerados distintos, somando 264 rodadas, entre 7 execuções independentes** — cobrindo as duas estratégias de ruído (uniforme e `layer_group`), com e sem peso de classe, com e sem parada antecipada. O primeiro estado aparece em **todos os 7 treinos sem exceção**; o terceiro também. Não é coincidência numérica — três pontos fixos específicos, sempre os mesmos até a 15ª casa decimal, sendo redescobertos repetidamente por treinamentos com sementes e configurações diferentes. A leitura mais provável — que o espaço de soluções acessível sob DP-FedAvg, para este modelo e este dado, tenha um número pequeno de pontos fixos degenerados — é uma **hipótese**, não uma conclusão com base teórica citável (ver a ressalva sobre `bu2023`/Teorema 1 já registrada no rascunho do TCC, Seção 5.4.2): o teorema explica *por que* o treino sob DP oscila, mas não prevê nem prova a recorrência numérica exata entre execuções independentes. Fica registrada como direção de trabalho futuro concreta (Seção 7.4 do rascunho) — inspecionar os pesos do modelo nos *checkpoints* onde o estado se repete, não só as métricas de saída, seria o próximo passo real de investigação.

---

## `training_id=22` — DP uniforme, com peso, sem early stop — fecha a matriz `completo_com_peso` de DP uniforme

Mesmo slot nominal de `training_id=13`, com peso de classe. 110/110 rodadas completas, `best_round=65`, convergência na 44 (antes do melhor ponto, como em `training_id=20`).

| | `training_id=13` (sem peso) | `training_id=22` (com peso) |
|---|---|---|
| Melhor rodada | 42 | 65 |
| Rodada de convergência | 35 | 44 |
| Accuracy | 0,6661 | 0,5721 |
| Macro F1 | 0,2705 | 0,2005 |
| Macro AUC | 0,5943 | 0,5568 |
| ECE pré→pós | 0,0609→0,0 | 0,3827→0,0 |
| ε RDP / simples | 318,9 / 1065,9 | 318,9 / 1065,9 |
| Duração | 3h43min | 3h37min |

O ε RDP/simples ficou **praticamente idêntico** entre os dois (318,9 e 1065,9 nos dois) — confirma que o acúmulo de epsilon depende só do número de rodadas e da configuração de ruído, não do peso de classe, como esperado.

**Per-class F1, melhor rodada**: `training_id=13`: `[0,740, 0,0, 0,0, 0,613, 0,0]` — `training_id=22`: `[0,731, 0,0, 0,0, 0,0, 0,272]`.

**Inversão completa entre `melhora_internado_breve` e `melhora_internado_grave`** — não é mais só uma troca parcial de sinal como nos outros dois pares (`10`×`17`, `11`×`20`): aqui `melhora_internado_breve` vai de F1=0,613 pra **F1=0,0** (colapso total), e `melhora_internado_grave` vai de F1=0,0 pra F1=0,272 (sinal parcial). O peso=6 nessa classe não só ajuda — nesse par específico, ele parece ter *trocado de lugar* qual das duas classes colapsa, não apenas dividido o sinal entre as duas. **Terceiro par consecutivo** com essa dinâmica breve↔grave, agora mais extrema.

**`curado_internado` e `melhora_pronto` seguem em F1=0,0 — confirmado agora em TRÊS pares independentes, dois cenários de privacidade diferentes** (`10`×`17` e `11`×`20` em Sem DP; `13`×`22` em DP uniforme). Em nenhuma das 6 execuções completas com peso=20 nessas duas classes houve um único acerto.

**Degradação até a última rodada, a mais severa registrada**: `f1_macro` cai de 0,2005 (rodada 65) pra 0,0241 (rodada 110) — queda de 88%. A última rodada cai exatamente no estado atrator `[0,0, 0,0, 0,0, 0,0, 0,1206]` já documentado acima (repetido 47 vezes só dentro deste treino).

---

## Verificação "Passo 0" — como a agregação funciona e se o colapso de classe é diluição ou estrutural (2026-08-11)

Motivada por um documento de avaliação externa (`Validacao_Hipotese_MOSAIC-FL.pdf`, gerado a pedido da autora) que levantou a suspeita de que o F1=0 agregado de `curado_internado`/`melhora_pronto` pudesse ser artefato da agregação ponderada, não colapso real. Duas verificações, ambas de custo zero (sem treino novo — só leitura de código e consulta ao banco já existente).

**0.1 — A agregação é mesmo ponderada por volume, favorecendo o BPSP?** Confirmado direto no código, não presumido. `infrastructure/mosaicfl_server/runner/superlink.py` (o runner real usado em todos os `make superlink-dp-*` rodados nesta fase) registra `evaluate_metrics_aggregation_fn=weighted_average_evaluate_metrics`. Essa função (`src/mosaicfl/core/federated.py`) calcula `accuracy`, `f1_macro` e `per_class_f1` agregados como **média ponderada pelo número de amostras de cada cliente** (`sum(n × métrica) / total`). Com o BPSP tendo ≈5,5× o volume do HSL, o F1 macro "global" reportado nas Tabelas 16–21 do rascunho **é**, de fato, estruturalmente dominado pelo desempenho do BPSP. A suspeita do documento está correta nesse ponto.

**0.2 — O F1=0 agregado é diluição, ou colapso real nos dois hospitais?** Extraído `per_client_f1_json` (F1 por classe, por hospital, calculado ANTES da agregação — existe desde a migration 027, ver `[[project_per_client_f1_capturado]]`) na melhor rodada dos 10 treinos formais concluídos (`training_id` 10, 11, 12, 13, 17, 18, 19, 20, 21, 22). Resultado, olhando só `curado_internado` e `melhora_pronto`, por hospital, não agregado:

| | `curado_internado`, cliente A | `curado_internado`, cliente B | `melhora_pronto`, cliente A | `melhora_pronto`, cliente B |
|---|---|---|---|---|
| 10, 11, 12, 13, 17, 19, 20, 21, 22 | 0,0 | 0,0 | 0,0 | 0,0 |
| 18 (exceção) | 0,0 | **0,0870** | 0,0 | 0,0 |

De 10 treinos × 2 hospitais × 2 classes = **40 combinações, 39 são exatamente zero nos dois hospitais, independentemente um do outro**. A única exceção é `training_id=18` (a execução mais ruidosa e atípica documentada, pico na rodada 3) — um cliente com `curado_internado`=0,087, sinal minúsculo, não estrutural.

**Conclusão, sem inflar além do que os dados sustentam**: a suspeita de que "talvez o zero seja só diluição da agregação" **não se confirma** nos dados que já temos — o colapso é estrutural e ocorre nos dois hospitais quase sempre ao mesmo tempo, não é um artefato de como a métrica global é calculada. Ao mesmo tempo, o mecanismo de agregação ponderada por volume (0.1) é real e confirmado — só não é ele que explica o colapso dessas duas classes específicas.

---

## `training_id=24` — DP layer_group, sem peso, sem early stop — fecha a matriz `completo_sem_peso` (6/6)

Mesmo slot nominal de `training_id=21` (com peso). 110/110 rodadas completas, `best_round=10` (pico bem precoce), convergência na 72 — 62 rodadas depois do melhor ponto, o maior gap entre convergência e melhor rodada já registrado neste documento.

| | `training_id=21` (com peso) | `training_id=24` (sem peso) |
|---|---|---|
| Melhor rodada | 14 | 10 |
| Rodada de convergência | 47 | 72 |
| Última rodada | 110 | 110 |
| Acurácia | 0,6214 | 0,4001 |
| Macro F1 | 0,2374 | 0,2273 |
| Macro AUC | 0,5399 | **0,6888** |
| ECE (pré→pós) | 0,3543→0,1082 | 0,6486→0,1176 |
| ε RDP / simples | 1563,8 / 2131,7 | 1563,8 / 2131,7 |
| Duração | 3h44min | 3h46min |

ε idêntico entre os dois, de novo confirmando que o acúmulo de privacidade não depende do peso de classe.

**Resultado misto, sem direção limpa desta vez.** Acurácia caiu bastante sem peso (0,621→0,400), mas o AUC macro **subiu** (0,540→0,689) — o melhor AUC de qualquer treino `DP layer_group` registrado, e um dos melhores sob DP em geral. F1 macro ficou praticamente igual (0,237 vs 0,227). Per-class na melhor rodada: `training_id=21`: `[0,731, 0,0, 0,0, 0,168, 0,289]` — `training_id=24`: `[0,452, 0,0093, 0,0, 0,380, 0,295]`. `melhora_internado_breve` melhora bastante sem peso (0,168→0,380); `curado_internado` tem um sinal minúsculo sem peso (0,0093, 1 acerto em 82, num só dos dois hospitais) que não aparece com peso — o oposto do padrão usual, mas o número é pequeno demais pra significar solução real.

**Confirmação mais forte do achado dos estados atratores — agora sem peso de classe.** A última rodada (110) cai de novo em `[0,0, 0,0, 0,0, 0,0, 0,12056283799036835]`, e esse mesmo estado se repete **25 vezes só neste treino** (mais 22 vezes de um segundo estado, `[0,0,0,0,0,375,0,0]`, e 12 de um terceiro, `[0,692,0,0,0,0,0]`). Somando às ocorrências já documentadas em `training_id` 18, 19, 21, 22 (110 rodadas) e as 3 do `training_id=23` (interrompido, mas com dados até a rodada 79), o primeiro estado sozinho já soma **138 rodadas entre 6 treinos independentes** — e, pela primeira vez, dois desses treinos (`23` e `24`) são `completo_sem_peso`. Isso resolve uma dúvida em aberto do achado anterior: **o fenômeno não depende do peso de classe** — aparece com e sem *override* ativo, sob as duas estratégias de ruído DP já testadas (uniforme e `layer_group`), com e sem parada antecipada.

**Matriz `completo_sem_peso` fica em 5 de 6** — só falta `DP layer_group, com early stop, sem peso`, o mesmo slot que `training_id=14` e `23` já tentaram e falharam; `training_id=25`, em andamento agora, é a 3ª tentativa desse slot específico (ainda sem peso — os pesos de classe seguem limpos no BPSP). O slot `DP layer_group, com early stop, com peso` (do lado `completo_com_peso`) segue como o único da matriz de 12 ainda sem nenhuma tentativa.

---

## `training_id=25` — DP layer_group, sem peso, com early stop — fecha a matriz inteira (12/12)

3ª tentativa deste slot (`14` parou manualmente na rodada 11; `23` travou com processo zumbi na rodada 79). Desta vez completou 64/110 rodadas, parada antecipada disparando corretamente na rodada 63 (convergência detectada na 63, `best_round=19`).

| | `training_id=19` (com peso) | `training_id=25` (sem peso) |
|---|---|---|
| Melhor rodada | 63 | 19 |
| Rodada de convergência | 88 | 63 |
| Última rodada | 89 | 64 |
| Acurácia | 0,6191 | 0,6453 |
| Macro F1 | 0,2026 | 0,2906 |
| Macro AUC | 0,5006 | 0,6101 |
| ECE (pré→pós) | 0,6356→0,0950 | 0,3543→0,0909 |
| ε RDP / simples | 1286,6 / 1724,8 | 956,6 / 1240,3 |
| Duração | 3h09min | 2h18min |

**Resultado limpo, diferente dos outros pares — `sem peso` ganha em todas as métricas desta vez**: acurácia, F1 macro, AUC, ECE pré-calibração, epsilon (menor, mais barato em privacidade) e até duração (menos rodadas até parar). Per-class na melhor rodada: `training_id=19`: `[0,716, 0,0, 0,0, 0,256, 0,041]` — `training_id=25`: `[0,783, 0,0, 0,0, 0,392, 0,278]` — sem peso vence em todas as classes com sinal. `curado_internado`/`melhora_pronto` seguem exatamente em zero nos dois, mais uma confirmação (agora 7 pares/execuções isoladas).

**Estados atratores, mais uma vez.** Três estados já documentados se repetem neste treino: `[0,692,0,0,0,0]` 5×, `[0,0,0,0,0,1206]` 4×, `[0,0,0,0,375,0]` 3×. A última rodada (64, onde a parada antecipada disparou) cai exatamente no estado `[0,692,0,0,0,0]` — a convergência declarada é, mais uma vez, esse ciclo degenerado, não um mínimo genuíno.

**Leitura honesta**: uma única execução por lado, sem réplica — não dá pra afirmar que "sem peso é melhor sob DP layer_group com early stop" como regra geral, só que nesta comparação específica foi. É o par mais unilateral de todos os 6 registrados neste documento; os outros 5 mostraram resultado misto ou trade-off. Vale registrar como dado, sem generalizar.

**Matriz de 12 configurações formais, oficialmente completa.** Todos os 6 pares `sem_peso`×`com_peso`, nos 3 cenários de privacidade × 2 estados de parada antecipada, têm dado real agora.

---

## Leave-one-out no Caminho B — validação direta da hipótese central (iniciado 2026-08-11)

Motivado pela avaliação externa `Validacao_Hipotese_MOSAIC-FL.pdf` (Seção "Verificação Passo 0" acima): o único teste direto do "efeito equalizador do FL" até agora vem do Bloco 1 do Caminho A (T13/T14/T15), dado que o próprio rascunho do TCC desqualifica como "não resultado final citável de defesa". Este é o teste equivalente, real, no Caminho B.

**Desenho**: célula mais limpa disponível — sem privacidade diferencial, sem parada antecipada, sem peso de classe (pesos confirmados limpos nos dois hospitais desde 2026-08-10 23:05). Dois treinos `local_only_hospital` (capacidade portada ao Caminho B desde a migration 030, nunca usada nesta fase até agora), comparados contra o federado já existente no mesmo cenário (`training_id=11`).

| `training_id` | Configuração | Status |
|---|---|---|
| 26 | BPSP isolado (`min-clients=1`) | 🟢 concluído (2026-08-11, 12:18→16:34 UTC, ~4h17min) — acc=0,8439, F1 macro=0,4537, AUC macro=0,8571, best\_round=51, convergência na 65, 110/110 rodadas — **válido, HSL de fato não conectou** |
| 27 | HSL "isolado" (`min-clients=1`), 1ª tentativa | 🔴 **INVÁLIDO como leave-one-out** — ver achado abaixo. Treino em si concluiu normalmente (110/110 rodadas), mas não isolou o HSL |
| 28 | HSL isolado (`min-clients=1`), 2ª tentativa | 🟢 concluído (2026-08-11 23:07→2026-08-12 00:47 UTC, ~1h40min) — acc=0,8591, F1 macro=0,4013, AUC macro=0,8593, best\_round=99, convergência na 72, 110/110 rodadas — **válido, confirmado 1 único cliente em todas as 110 rodadas** |
| 11 | Federado (referência, já concluído) | 🟢 concluído — acc=0,7515, F1 macro=0,3921 |

**Comandos executados para `training_id=26`** (BPSP isolado): `make superlink FL_EARLY_STOP=false` → `make supernode FL_CLIENT_ID=BPSP FL_DATA_SOURCE=sgbd` (sozinho, HSL não conectado) → `make server-app-local-only LOCAL_ONLY_HOSPITAL=BPSP`. Registrado no banco com `local_only_hospital='BPSP'`, `early_stop_enabled=false`.

**Comandos executados para `training_id=27`** (pretendia ser HSL isolado): mesma sequência, com `LOCAL_ONLY_HOSPITAL=HSL` — mas o SuperNode do BPSP (subido para o `training_id=26`, às 09:17) **nunca foi encerrado** e continuava rodando (reconexão automática do cliente, `legacy_client.py`). Quando o novo `SuperLink` subiu para este treino, o SuperNode do BPSP reconectou sozinho.

**Achado 2026-08-11 — `training_id=27` não isolou o HSL, apesar do nome**: `local_only_hospital='HSL'` no banco é só um rótulo — a flag `LOCAL_ONLY_HOSPITAL` no `server-app-local-only` **relaxa** `min_clients` para 1 (permite o treino começar com um hospital só), mas **não impede** outros clientes de se conectarem se já estiverem de pé. Evidência direta no log (`experiments/logs/serverapp_20260811_151936.log`, linha 1112): `ece_pre_computed round=110 [...] n_samples=10175 n_clients=2` — **dois** clientes, não um. Confirmado no log do SuperLink (`superlink_20260811_151836.log`): os dois `node_id` (BPSP e HSL) aparecem 4811 vezes combinadas, do início ao fim do treino, não só numa reconexão pontual. `evaluation_json.confusion_matrix_stats` de `training_id=27` tem `n_total=10175` e `curado_pronto` respondendo por 58% do suporte — perfil de classe consistente com o BPSP dominando o teste (`curado_pronto` é só ~1,5% da distribuição real do HSL), não com uma avaliação isolada do HSL. Confirmado também via `ps aux`: o processo `flower-supernode --node-config client-id="BPSP"` iniciado às 09:17 (para o `training_id=26`) **ainda está rodando agora**, nunca foi encerrado.

**Na prática, `training_id=27` é um segundo treino federado válido no cenário "sem DP, sem parada antecipada, sem peso" — não um silo do HSL.** Os números (acc=0,7398, F1 macro=0,3939) são reais e utilizáveis como uma réplica adicional de `training_id=11` (mesmo cenário, seed/timing diferente — útil para checar estabilidade do resultado federado, não para o teste de *leave-one-out*), mas **não substituem** o silo do HSL que o experimento original pedia. `local_only_hospital='HSL'` no banco está factualmente incorreto para este `training_id` e precisa de correção (não fiz ainda — aguardando decisão de como marcar).

**`training_id=28` — segunda tentativa, corrigida**: antes de rodar de novo, o processo do SuperNode do BPSP ainda ativo foi encerrado (`kill`) e um novo `SuperLink` foi iniciado. Confirmado, com três evidências independentes, que desta vez o isolamento foi real do início ao fim: (1) o log do SuperLink mostra `[Fleet.DeactivateNode]` do `node_id` do BPSP às 20:07:28, sem nenhuma reaparição dele até o fim do log; (2) `per_client_f1_json` de **todas as 110 rodadas** de `training_id=28` tem exatamente 1 cliente (`jsonb_array_length=1`, verificado rodada a rodada, sem exceção); (3) o perfil de classe da rodada 1 já bate com a distribuição conhecida do HSL (`melhora_pronto` alto, `curado_pronto` zero) — o oposto do observado no `training_id=27` inválido.

**Nota sobre classificação**: `training_id=26` e `training_id=28` foram reclassificados para `run_classification='treinamento_real'` (2026-08-11) — os dois silos válidos são resultado formal citável. `training_id=27` permanece `treinamento_real` também, mas como réplica federada (não como *leave-one-out* — seu `local_only_hospital='HSL'` no banco está factualmente incorreto, mantido sem correção por ora, documentado aqui para não confundir).

### Resultado — comparação silo × federado, avaliados no mesmo teste local

A comparação correta usa sempre o **mesmo conjunto de teste** dos dois lados: o silo é avaliado no teste do seu próprio hospital (é a única opção que ele tem); o federado precisa ser avaliado **no teste desse mesmo hospital**, não no conjunto global agregado — daí `per_client_f1_json` de `training_id=11` (rodada 75, melhor rodada), que guarda o F1 por classe que cada cliente calculou **antes** da agregação, exatamente o que falta. Identificação dos dois `client_id` numéricos confirmada por duas evidências independentes: o footprint de recursos (`resource_per_client_json` — o primeiro cliente reporta `gpu_power_w`/`gpu_energy_wh`, consistente com o desktop/BPSP, que tem RTX 4070 Ti mesmo sem usá-la no treino; o segundo não reporta nenhum campo de GPU, consistente com o notebook/HSL) e o próprio perfil de classe (`curado_pronto` alto no primeiro cliente, `melhora_pronto` baixíssimo — compatível com BPSP; o inverso no segundo — compatível com HSL).

| | **BPSP** — Federado (`training_id=11`, avaliado no teste do BPSP) | **BPSP** — Silo (`training_id=26`) | **HSL** — Federado (`training_id=11`, avaliado no teste do HSL) | **HSL** — Silo (`training_id=28`) |
|---|---|---|---|---|
| F1 por classe | `[0,950; 0,0; 0,0; 0,707; 0,474]` | `[0,948; 0,176; 0,0; 0,648; 0,497]` | `[0,031; 0,0; 0,0; 0,640; 0,341]` | `[0,0; 0,0; 0,967; 0,641; 0,398]` |
| **F1 macro** | **0,4263** | **0,4537** | **0,2024** | **0,4013** |
| Δ (silo − federado) | | **+0,0275** | | **+0,1989** |

(ordem das classes: `curado_pronto`, `curado_internado`, `melhora_pronto`, `melhora_internado_breve`, `melhora_internado_grave`)

**O silo vence nos dois hospitais, evidência direta contra a hipótese do "efeito equalizador do FL" nesta configuração.** No BPSP a diferença é pequena (+2,75 p.p., dentro do que uma única execução sem réplica já não permite generalizar com confiança). No HSL a diferença é grande (+19,89 p.p.) e tem uma explicação visível nos próprios números: o modelo federado tem F1 **zero** em `melhora_pronto` mesmo avaliado no teste do próprio HSL — a classe que domina 72,5% do teste local do HSL (support=1126/1554). O modelo federado erra sistematicamente a classe majoritária do hospital menor; o silo, sem a competição do gradiente do BPSP (5,5× mais volume), acerta essa mesma classe com F1=0,967.

**Isso não é um achado isolado — é consistente com o motivo original documentado para adotar o FedNova** (rascunho do TCC, seção "Framework e estratégia de agregação"): a média ponderada por amostras do FedAvg/FedProx dilui sistematicamente o sinal do cliente menor. E é consistente com o achado independente já registrado nesta mesma investigação (Seção "FedProx/FedNova" acima): `training_id=11`, como todos os 12 treinos formais do Caminho B, agregou por **FedProx puro** (equivalente a FedAvg), nunca por FedNova. **Hipótese, não conclusão**: é plausível que a diluição do sinal do HSL observada aqui seja exatamente o efeito que o FedNova deveria corrigir e que, por não estar portado ao Caminho B, nunca foi corrigido nos resultados formais. Essa hipótese só vira achado citável depois que a Prioridade 1 (portar FedNova, rerodar os 12 treinos) estiver pronta e a mesma comparação for refeita com a agregação corrigida.

**Limitações explícitas desta comparação** (honestidade sobre o que não dá pra afirmar): (1) execução única de cada lado, sem réplica — a diferença de +2,75 p.p. no BPSP não é distinguível de ruído de uma única semente; a de +19,89 p.p. no HSL é grande o bastante pra ser notável mesmo sem réplica, mas ainda é uma medição; (2) não há teste de significância pareado (McNemar) porque o sistema, por desenho de privacidade, nunca centraliza predição por paciente — só `per_class_f1`/matriz de confusão agregada por rodada estão disponíveis, não há como parear exemplo a exemplo; (3) a comparação usa F1 macro, não acurácia, porque `per_client_f1_json` só captura F1 por classe — acurácia por cliente do treino federado não é persistida separadamente; (4) os dois lados (silo e federado) usam o critério de melhor rodada (`f1_macro`) de cada treino individualmente, não a mesma rodada calendário.

---

## Achado operacional não planejado: recuperação automática após queda de um SuperNode (`training_id=17`, 2026-08-09)

Durante o `training_id=17` (cenário "Sem DP", com pesos de classe, com *early stop*), o SuperNode do HSL foi derrubado sem aviso — a usuária estava copiando dados na máquina do HSL, e essa operação interrompeu o processo do supernode. Isso não foi um teste planejado de resiliência; foi um incidente real, registrado aqui porque o comportamento do sistema durante e depois dele é um dado relevante.

**Linha do tempo, reconstruída a partir dos logs (`serverapp_20260809_143404.log`, `superlink_20260809_143334.log`)**:

| Horário | Evento |
|---|---|
| 15:12:55 | Round 17 inicia normalmente (`configure_fit`, 2 clientes amostrados) |
| 15:15:04 | `aggregate_fit: received 1 results and 1 failures` — só o BPSP completou o fit; o HSL falhou |
| 15:15:04–16:38 | Nenhuma linha nova de progresso no log do servidor (~83 minutos parado). O superlink ficou fazendo `PullMessages` do node BPSP a cada ~3s, sem nada para enviar |
| 16:38:35 | Erro no superlink: `Failed to store Message reply: ... does not exist or has expired, or was deleted because the target SuperNode was removed from the federation` — a resposta tardia do HSL chegou depois do TTL da mensagem e foi rejeitada |
| 16:38:40 | `configure_evaluate` do round 17 finalmente executa |
| 16:39:01 | `aggregate_evaluate: received 2 results and 0 failures` — os dois clientes responderam a essa etapa; checkpoint do round 17 salvo; round 18 inicia normalmente |

Um detalhe técnico: o `node_id` do HSL depois da reconexão é diferente do anterior (`16765870621546141878` no lugar do id original) — comportamento esperado do Flower, que trata um SuperNode reconectando como uma nova identidade de nó, não como retomada da mesma sessão.

**O que isso mostra, com cautela**: o treinamento sobreviveu à queda completa de um dos dois clientes por ~83 minutos sem intervenção manual e sem precisar reiniciar o processo — quando o HSL reconectou, a rodada travada foi concluída e o treino seguiu adiante normalmente a partir do round 18. Isso é uma evidência pontual (uma única ocorrência, não planejada, sem grupo de controle) de que a arquitetura federada tolera a queda temporária de um participante sem perda do estado do treinamento em andamento — mas não permite generalizar limites (não se sabe, por exemplo, qual seria o tempo máximo de queda tolerável, nem o que aconteceria se o `checkpoint_store` também estivesse indisponível nesse intervalo).

---

## Tentativas descartadas (não contam como treino formal)

| `training_id` | O que aconteceu | Detalhe na linha do tempo |
|---|---|---|
| 1 | Tentativa de ~1min, crash do desktop antes de completar rodada 1 | "Crash do desktop interrompe o primeiro treino real" |
| 2 | Crash do desktop na rodada 82/110 (~74% concluído) | idem |
| 3 | Descoberta de vocabulário pulou (HSL conectou depois do timeout de 120s) | "Bug: descoberta de vocabulário nunca tem segunda chance" |
| 4 | Nasceu e morreu junto com o `training_id=3` (parada manual) | idem |
| 5 | Vocabulário funcionou, mas HSL falhava toda rodada (`classification` NULL) | "Bug: backfill de classification nunca rodou no HSL recarregado" |
| 6 | Sem DP, sem early stop (tentativa anterior ao slot que `training_id=11` preenche hoje) — completou 110/110 rodadas limpo, mas `evaluation_json.best_per_class_f1` estava errado (bug corrigido) | "Bug: evaluation_json.best_per_class_f1 vinha da rodada errada" |
| 7 | Sem DP, com early stop (tentativa anterior ao slot que `training_id=10` preenche hoje) — mesmo bug do 6 | idem |
| 8 | DP uniforme, com early stop (tentativa anterior ao slot que `training_id=12` preenche hoje) — mesmo bug, e também mostrou estados atratores sob DP (achado preservado) | idem |
| 14 | DP layer_group, com early stop, `completo_sem_peso` — chegou à rodada 11, parada manual (`Task stopped by user`) durante a mesma sessão de configuração dos pesos de classe (2026-08-09, tarde). Slot segue em aberto. | seção "Peso de classe — configurado em 2026-08-09" |
| 15 | Sem DP, com early stop, `completo_sem_peso` (registrado antes do peso do BPSP ser confirmado) — chegou à rodada 3 (`round_timeout` no log), parada manual | idem |
| 16 | Sem DP, com early stop, provável `completo_com_peso` (registrado depois do peso do BPSP confirmado) — parou logo na rodada 1, parada manual quase imediata; superado por `training_id=17`, que preencheu o mesmo slot com sucesso | idem |
| 23 | DP layer_group, com early stop, `completo_sem_peso` — segunda tentativa desse mesmo slot (`training_id=14` foi a primeira). Chegou à rodada 79/110 progredindo normalmente, sem nenhum erro de desconexão de rede desta vez (os dois nós seguiam conectados, fazendo `PullMessages`). O subprocesso `flwr-clientapp` responsável pela avaliação da rodada travou e morreu como processo zumbi (`<defunct>`) nos dois hospitais simultaneamente, sem log de erro explícito — servidor ficou 17 minutos esperando resposta de `configure_evaluate` sem nenhuma linha nova. Marcado `interrupted` manualmente. Slot segue em aberto, precisa de 3ª tentativa. | — |

---

## Como atualizar este documento

Ao concluir (ou interromper definitivamente) um dos 6 treinos:

1. Consultar `metrics.fl_trainings` pelo `training_id` (`status`, `best_round`, `best_accuracy`, `macro_f1`, `macro_auc`, `ece_pre`, `ece`, `dp_epsilon_rdp`, `convergence_round`, `total_duration_s`) — ou a tela `/fl-training-results`, que já mostra os mesmos campos. `n_rounds_done` dá a "última rodada"; comparar com `convergence_round` e checar se `FL_EARLY_STOP` estava ligado (log do ServerApp, `early_stop_enabled`) diz se a última rodada é a de convergência ou o teto configurado (110).
2. Atualizar a linha correspondente na tabela "Os 6 treinamentos planejados" com os valores finais e trocar o status pra 🟢.
3. Se o treino foi descartado (bug, resultado inválido), mover pra tabela de "Tentativas descartadas" com o motivo, e voltar o cenário/variação (com ou sem *early stop*) pra ⬜ na tabela principal — o próximo `training_id` que tentar esse mesmo slot substitui a linha.
4. Antes de citar `evaluation_json.best_per_class_f1` de qualquer checkpoint gravado **antes** de 2026-08-08, comparar contra `fl_round_history` na rodada certa — pode estar com o bug (corrigido nesse dia, mas checkpoints antigos continuam com o valor errado gravado).
5. Depois dos 6 completos, adicionar uma seção de análise comparativa (sem DP vs. DP uniforme vs. DP layer_group — accuracy, F1 macro de classe rara, custo de ε) — mesma régua já usada em `docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md` pra decisão anterior sobre DP-FedAvg (seção 18).
