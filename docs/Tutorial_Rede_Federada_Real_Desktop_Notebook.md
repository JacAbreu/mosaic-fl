# Tutorial — Rede Federada Real (Desktop + Notebook), Caminho B com banco separado

**Cenário:** desktop = servidor (SuperLink) + dados BPSP; notebook = cliente (SuperNode) + dados HSL.
**Variante:** banco do servidor é um Postgres **novo e separado** (porta 5433), sem mexer no
`mosaicfl-db` já existente no desktop — que continua intacto com o dataset combinado (BPSP+HSL)
usado nos Treinamentos Reais 1/2/3 e na curva Acc×ε (ver `docs/Sumario_Treinamento_Parte3.md`).

Contexto geral do Caminho B (o que é SuperLink/ServerApp/SuperNode, TLS, `~/.flwr/config.toml`)
está documentado no `README.md`, seção "Rede Federada Real via SuperLink (Desktop + Notebook) —
Caminho B". Este tutorial é o passo a passo prático da variante com banco separado.

## Caminho A vs. Caminho B — nomes corretos e por que escolhemos o B

O projeto tem **dois caminhos** pra rodar aprendizado federado, e coexistem no mesmo
repositório (portas diferentes, não competem entre si):

| | **Caminho A** — `fl-server`/`fl-client` | **Caminho B** — `superlink`/`supernode` |
|---|---|---|
| API do Flower | Legada — sockets diretos (`fl.server.start_server`/`fl.client.start_client`) | Produção — `flower-superlink` + `ServerApp`/`ClientApp`, via `flwr run` |
| Portas | 8080 (server) / 8081 (health) | 9091 (Fleet API) / 9092 (ServerAppIo, interno) / 9093 (Control API) |
| TLS | Suportado, mas os passos originais nem sempre mencionavam | Obrigatório desde o início — scripts falham cedo e com erro claro sem certificado |
| Onde roda historicamente | Testes/depuração, Treinamentos Reais 1-3 (single-machine, `training-full[-cuda]`) | Validação real entre 2 máquinas físicas (desktop + notebook), a partir de 2026-07-04 |
| Mais próximo de um deploy real? | Não — arquitetura legada, sem o modelo de deployment do Flower moderno | **Sim** — é a arquitetura que o Flower recomenda pra cross-silo real, com FAB (empacotamento automático de código) e coordenação via SuperLink |

**Por que optamos pelo Caminho B pra validar a rede real (desktop+notebook):** o Caminho A
nunca foi desenhado pra rodar em 2 máquinas fisicamente separadas de verdade — era usado
em simulação (um processo simula vários hospitais na mesma máquina) ou, no máximo, 2
processos na mesma rede local sem a infraestrutura de produção do Flower por trás
(distribuição de código via FAB, gerenciamento de sessão via SuperLink, etc.). O Caminho B
é o que o próprio Flower recomenda pra esse cenário — e por isso, embora tenha exigido uma
sessão inteira de depuração pra ficar funcional (ver `docs/Linha_do_Tempo_MOSAIC-FL.md`,
2026-07-05 a 07 — 4 bugs reais encontrados só ao rodar de verdade com 2 máquinas), é o
caminho certo pra validar que o sistema funciona como um deploy real funcionaria.

---

## Parte 1 — Desktop (Servidor, BPSP)

### 1.1 Subir o Postgres do servidor numa porta separada (5433)

```bash
docker run -d \
  --name mosaicfl-db-bpsp \
  -e POSTGRES_DB=mosaicfl -e POSTGRES_USER=mosaicfl \
  -e POSTGRES_PASSWORD=senhaForte \
  -p 5433:5432 \
  -v mosaicfl_db_bpsp_data:/home/postgres/pgdata/data \
  timescale/timescaledb-ha:pg16

# Confirmar que subiu:
docker exec mosaicfl-db-bpsp pg_isready -U mosaicfl -d mosaicfl
```

### 1.2 Rodar as migrations neste banco novo

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
bash scripts/db/migrate.sh upgrade head
```

### 1.3 Gerar os dois seeds (BPSP e HSL) a partir dos dados FAPESP

Os dois são gerados aqui no desktop, onde estão os arquivos brutos (`FL_DATA_DIR`):

```bash
make server-generate-seed FL_DB_URL="$FL_DB_URL"   # gera scripts/db/seeds/bpsp_seed.sql.gz
make client-generate-seed FL_DB_URL="$FL_DB_URL"   # gera scripts/db/seeds/hsl_seed.sql.gz (para o notebook)
```

### 1.4 Carregar só o BPSP neste banco

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
make server-load-bpsp FL_DB_CONTAINER=mosaicfl-db-bpsp
```

> `server-load-bpsp` já inclui o backfill de `classification` automaticamente (ver
> seção 2.4b do Caminho A/HSL) — não precisa de passo manual separado.

### 1.4b Gerar o vocabulário padrão compartilhado (obrigatório, só no desktop)

Sem isso, cada cliente FL construía seu **próprio** vocabulário local a partir dos
dados que vê. Como BPSP e HSL têm conjuntos de analitos diferentes, os vocabulários
saíam com tamanhos diferentes — a camada de embedding sempre tem o mesmo formato
(`MODEL_CFG.vocab_size` é fixo), então isso não crasha, mas o mesmo índice de
embedding passa a representar tokens diferentes em cada cliente: a agregação
federada (média dos pesos) fica sem sentido semântico, em silêncio — as rodadas
rodam sem erro nos dois lados, mas o resultado agregado é lixo.

> **Achado 2026-07-26 — o catálogo curado (`knowledge.term_dictionary`) cobre só
> 15 analitos de um painel COVID escolhido à mão, mas os dados reais têm centenas
> de analitos distintos.** Qualquer exame fora desses 15 vira `<UNK>` na
> tokenização — não some, mas perde toda a informação clínica. Antes de gerar o
> vocab, rode a descoberta de lacunas do catálogo (idempotente, seguro de repetir):
> ```bash
> export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
> python3 scripts/discover_analyte_catalog_gaps.py --dry-run   # revisa a lista antes
> python3 scripts/discover_analyte_catalog_gaps.py             # grava
> ```
> Precisa rodar **depois** de `server-load-bpsp`/`client-load-hsl` (que já fazem o
> backfill de `classification` automaticamente) — sem isso, boa parte dos
> candidatos de alto volume aparece sem `classification` válida e fica de fora.

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
mkdir -p checkpoints
python3 scripts/build_standard_vocab.py --output checkpoints/standard_vocab.json
```

**Só precisa existir no desktop.** O servidor (`ServerApp`) lê esse arquivo e
distribui o vocab automaticamente a cada cliente, embutido na config de cada
rodada (`vocab_json`) — não precisa copiar pro notebook. Se o servidor não tiver
um vocab pra mandar (arquivo ausente e nenhum checkpoint anterior), o `ServerApp`
agora falha ao subir, em vez de deixar os clientes com vocabulários incompatíveis.

> **Cuidado com checkpoint velho sobrepondo o vocab novo.** O `ServerApp` carrega
> o vocab embutido no último checkpoint (via `logs/training_state.json` →
> `last_checkpoint`) **sem checar se é de uma sessão de treino diferente** — se
> esse arquivo existir e apontar pra um `.pt` válido, ele silenciosamente ganha do
> `standard_vocab.json` que você acabou de reconstruir. Depois de rodar
> `discover_analyte_catalog_gaps.py`/`build_standard_vocab.py`, confira
> `logs/training_state.json`: se `last_checkpoint` existir em disco e for de um
> treino anterior à correção, mova ou apague esse arquivo antes do próximo
> `flwr run` pra garantir que o vocab novo seja usado.

### 1.5 Gerar certificados TLS com o IP real do desktop

```bash
hostname -I   # anote o IP mostrado
bash scripts/gerar_certs_tls.sh certs <SEU_IP>
export FL_TLS_CERT_DIR="$(pwd)/certs"
```

> O IP precisa ser passado na geração para entrar como `IP:` no certificado (SAN) —
> sem isso, a validação TLS falha ao conectar via IP real (só funcionaria via `localhost`).

### 1.6 Liberar a porta no firewall (Fleet API)

```bash
sudo ufw allow 9091/tcp && sudo ufw reload
```

### 1.7 Subir o SuperLink, usando o banco novo

```bash
FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl" make superlink
```

Isso imprime o IP local e o comando pronto para colar no notebook. **Deixe rodando** —
é o processo que fica de pé durante todo o treinamento.

> **Regra geral (achado 2026-07-29):** qualquer variável de ambiente que o `ServerApp` precisa
> ler (`FL_DP_NOISE*`, `FL_CALIBRATION_METHOD`, `FL_LLM_BACKEND`/`FL_LLM_MODEL`/`FL_LLM_HF_MODEL`)
> tem que ser passada **aqui**, na subida do SuperLink — nunca em `make server-app`, que só
> submete o treino e não tem efeito sobre o ambiente do `ServerApp`. Ver comentário completo no
> alvo `superlink` do `Makefile`.
>
> **Rodando um cenário com DP-FedAvg?** Troque `make superlink` por
> `make superlink-dp-uniform` ou `make superlink-dp-layer-group` — ver seção 3.4.

> **Log em arquivo:** além de aparecer no terminal, o log é gravado em
> `experiments/logs/superlink_<timestamp>.log` (caminho exato impresso ao subir).
> Sobrescreva com `FL_LOG_FILE=/caminho/seu.log` se preferir.

### 1.8 Subir o SuperNode do próprio desktop (cliente BPSP) — obrigatório

**Etapa que faltava numa versão anterior deste tutorial** (achado em 2026-07-12, ao notar que
só o cliente do notebook estava documentado). `pyproject.toml` exige `min-clients = 2` — com
só o notebook (HSL) conectado, o quórum nunca fecha e o treinamento **nunca começa**, sem erro
explícito, só fica esperando o segundo cliente. É a mesma estrutura já validada na primeira vez
que o Caminho B rodou de ponta a ponta (`docs/Linha_do_Tempo_MOSAIC-FL.md`, 2026-07-06/07):
desktop roda SuperLink **e** um SuperNode local (BPSP); notebook roda o outro SuperNode (HSL).

Em outro terminal do desktop, ainda apontando pro banco BPSP (porta 5433):

```bash
export FL_TLS_CERT_DIR=/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
make supernode FL_CLIENT_ID=BPSP FL_SUPERLINK_ADDRESS=localhost:9091 FL_DATA_SOURCE=sgbd
```

`localhost:9091`, não o IP externo — é o mesmo desktop se conectando ao SuperLink que já está
rodando nele. Só use `FL_DEVICE=cuda` antes desse comando se o desktop tiver GPU disponível
(ver seção "Usando GPU no Caminho B" mais abaixo). **Deixe rodando**, junto com o SuperLink.

> **Por que o servidor não usa esse banco BPSP para calibrar/testar centralizado:** já
> investigado e documentado (`docs/Linha_do_Tempo_MOSAIC-FL.md`, seção sobre F1 federado,
> 2026-07-07) — dar ao `ServerApp` acesso a um conjunto de teste centralizado (mesmo que
> hospedado na mesma máquina, neste teste específico) violaria o princípio de privacidade que
> o Caminho B deveria generalizar para um deploy real, onde o coordenador não fica colado a
> nenhum hospital. Por isso a calibração pós-treinamento (`FL_CALIBRATION_METHOD`, seção 3.1)
> **não roda** no Caminho B hoje — fica pulada (`calibration_skipped`) mesmo com o cliente BPSP
> local conectado. Corrigir isso exige um mecanismo de calibração federada (calibrar no lado do
> cliente, agregar com preservação de privacidade) — pesquisa já em andamento, não uma correção
> rápida neste tutorial.

---

## Parte 2 — Notebook (Cliente, HSL)

Pré-requisitos: repositório clonado (`git clone`), Docker e Python 3.10+ instalados.

### 2.1 Instalar dependências

```bash
cd mosaic-fl
make setup
```

### 2.2 Subir o banco local do notebook

```bash
make client-db-up FL_DB_PASSWORD=senhaForte
```

### 2.3 Rodar as migrations

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"
make client-migrate
```

### 2.4 Trazer o seed HSL do desktop e carregar

Transferir via `scp`, pendrive ou git (o arquivo é `scripts/db/seeds/hsl_seed.sql.gz`, gerado no
passo 1.3):

```bash
scp usuario@IP_DESKTOP:~/mosaic-fl/scripts/db/seeds/hsl_seed.sql.gz scripts/db/seeds/
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"
make client-load-hsl
```

> **`client-load-hsl` já inclui o backfill de `classification`** (o seed deixa essa coluna
> propositalmente `NULL` — ver comentário em `scripts/db/generate_hsl_seed.py` — e a consulta
> que carrega os dados no treino exige `classification IS NOT NULL`). O alvo roda
> `scripts/compute_analyte_references.py` automaticamente logo após a carga, usando o
> `FL_DB_URL` exportado acima — não precisa de passo manual separado. Desde a correção de
> 2026-07-26, esse script também garante `NO_REF` (em vez de `NULL` permanente) pra
> qualquer analito sem referência institucional válida em hospital nenhum.

### 2.5 Copiar o certificado CA do desktop

**Só o `ca.crt`** — nunca `ca.key`/`server.key`, que ficam só no desktop:

```bash
scp usuario@IP_DESKTOP:~/mosaic-fl/certs/ca.crt certs/ca.crt
export FL_TLS_CERT_DIR="$(pwd)/certs"
```

### 2.6 Conectar o SuperNode ao SuperLink do desktop

> **Terminal novo? Reexporte tudo.** As variáveis dos passos 2.3–2.5
> (`FL_DB_URL`, `FL_TLS_CERT_DIR`) só valem para o terminal onde foram
> exportadas. Se você fechou o terminal ou abriu um novo para rodar o
> `make supernode`, exporte as duas variáveis de novo antes do comando —
> **sem `FL_DB_URL` o cliente sobe, conecta ao SuperLink, mas falha ao
> tentar ler os dados** com `RuntimeError: Fonte de dados inválida:
> Connection string não configurada. Defina FL_DB_URL.`

```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"
export FL_TLS_CERT_DIR="$(pwd)/certs"
make supernode FL_CLIENT_ID=HSL FL_SUPERLINK_ADDRESS=<IP_DO_DESKTOP>:9091 FL_DATA_SOURCE=sgbd
```

O log deve mostrar o SuperNode conectado e aguardando rodadas — deixe rodando.

> **Log em arquivo:** gravado em `experiments/logs/supernode_<client-id>_<timestamp>.log`
> (caminho exato impresso ao subir). Sobrescreva com `FL_LOG_FILE=/caminho/seu.log`.

---

## Parte 3 — Disparar o treinamento

Depois que o SuperNode do notebook já estiver conectado (Parte 2, passo 2.6), volte para o
**desktop** e submeta o treinamento:

```bash
make server-app
```

Na primeira execução, o flwr migra a configuração de `pyproject.toml` para
`~/.flwr/config.toml` (arquivo local desta máquina, não versionado — ver README para detalhes).
Se quiser conferir depois:

```bash
cat ~/.flwr/config.toml
```

### 3.1 Calibração pós-treinamento (`FL_CALIBRATION_METHOD`) — agora federada (client-side)

**Atualização de 2026-07-12 (mesma sessão, corrigido no mesmo dia):** o achado original desta
seção (calibração sempre pulada, porque `superlink.py` nunca constrói um `test_loader`
centralizado — decisão de privacidade deliberada, ver `docs/Linha_do_Tempo_MOSAIC-FL.md`,
seção sobre F1 federado, 2026-07-07) continua correto — mas a resposta não é abandonar a
calibração, é federá-la: cada cliente ajusta o calibrador **localmente**, na última rodada
configurada (mesmo timing de `extract_rag_patterns`), e devolve só o resultado agregado/
comprimido (escalar T, ou breakpoints pós-PAV — nunca dado bruto por amostra). O servidor
combina o que recebeu de cada cliente e persiste o calibrador federado no checkpoint. Mesma
arquitetura de Cormode & Markov (VLDB 2023) e do FedTemp do Maddock et al. (preprint), já
pesquisada em `docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md`, seção 9.

Para esta rodada de validação, o plano volta a ser usar `auto` — cada cliente ajusta os dois
calibradores localmente, o servidor persiste o que tiver menor ECE agregado.

> **Correção 2026-07-29 — o comando abaixo estava documentado no lugar errado.**
> `FL_CALIBRATION_METHOD` é lido pelo `ServerApp` (`infrastructure/mosaicfl_server/runner/superlink.py`)
> e enviado a cada cliente via `fit_config` — mas o `ServerApp` é subido como subprocesso do
> **SuperLink** (`subprocess.Popen` sem `env=`, herda o ambiente do processo do SuperLink), não
> do comando `flwr run` usado por `make server-app`, que só *submete* o treino. Ou seja:
> `make server-app FL_CALIBRATION_METHOD=auto` (como este tutorial recomendava até agora)
> **não tinha efeito nenhum** — o valor efetivamente usado era o que já estivesse no ambiente do
> processo que rodou `make superlink` (o default do `Makefile` já é `auto`, então se você nunca
> exportou nada manualmente no terminal do SuperLink, provavelmente já rodou com `auto` por
> coincidência de default — mas se alguma vez tentou trocar para `temperature`/`isotonic` só no
> `server-app`, isso nunca surtiu efeito). Confirmado lendo o código-fonte do `flwr`
> (`flwr/supercore/superexec/executor/subprocess_executor.py`), mesma causa raiz do achado de
> DP-FedAvg documentado na seção 3.4. **Se algum resultado já citado no TCC depende do método de
> calibração ter sido "auto" especificamente (e não o default de código `temperature`), vale
> conferir a coluna `calibration_method`/`per_client_f1_json` do treino em questão em
> `metrics.fl_trainings` antes de citar.**

O comando certo é passar a variável na subida do SuperLink (passo 1.7), não no `server-app`:

```bash
FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl" \
  make superlink FL_CALIBRATION_METHOD=auto
```

(Como `auto` já é o default do `Makefile`, um `make superlink` puro também já usa `auto` —
só precisa passar a variável explicitamente se quiser um valor diferente, ex. `temperature`.)

`make server-app` continua sendo o comando que dispara o treino, sem passar `FL_CALIBRATION_METHOD`
nele (não tem efeito, ver aviso acima).

Fique de olho nos logs `local_calibration_fit` (por cliente) e `federated_calibration_persisted`
(no servidor, ao final). **Esta é a primeira vez que esse mecanismo roda com Flower de verdade**
— só foi validado com mocks/testes unitários até agora (629 testes, `tests/unit/test_fedprox_client.py`,
`test_aggregate_calibration.py`, `test_persist_federated_calibration.py`). Se algo falhar aqui,
é esperado precisar de ajuste — reporte o log de erro completo.

### 3.2 RAG (`FL_LLM_BACKEND`/`FL_LLM_MODEL`) — default correto, mas a variável vai no `superlink`

`server-app` constrói a base de conhecimento do RAG (`ClinicalRAG`) após a convergência, e a
API (`make api`) gera as justificativas clínicas usando o mesmo backend. O `Makefile` define
`FL_LLM_BACKEND=ollama` e `FL_LLM_MODEL=gemma3:4b` como default — **não precisa exportar nada
manualmente** para o caso comum. (Achado de 2026-07-07: sem essas variáveis em lugar nenhum, o
backend cai silenciosamente para `huggingface`/`distilgpt2`, mesmo com Ollama disponível.)

> **Correção 2026-07-29:** mesma causa raiz da seção 3.1 — quem realmente usa
> `FL_LLM_BACKEND`/`FL_LLM_MODEL`/`FL_LLM_HF_MODEL` é o `ServerApp`
> (`infrastructure/mosaicfl_server/strategy/core.py:_build_rag_knowledge_base`), que herda o
> ambiente do **SuperLink**, não do `server-app`. O `Makefile` já foi corrigido para passar
> essas variáveis no alvo `superlink` — o default (`ollama`/`gemma3:4b`) continua se aplicando
> sem ação manual, mas se você quiser **trocar** o backend, troque no `superlink`, não no
> `server-app`:
>
> ```bash
> make superlink FL_LLM_BACKEND=huggingface FL_LLM_MODEL=algum-outro-modelo
> ```

### 3.3 Duração esperada com `num-rounds=50`

Esta é a primeira vez que o Caminho B roda de ponta a ponta com `num-rounds=50`
(`pyproject.toml`, alterado em 2026-07-07 — antes era 10, nunca testado nesse valor até agora).
Com base no tempo por rodada observado em testes anteriores (~1-2 min por rodada do BPSP), um
treino completo pode levar de **50 minutos a quase 2 horas** por fase, se não convergir antes.
Não é motivo de alarme se o processo parecer "parado" por vários minutos entre rodadas.

### 3.4 DP-FedAvg (`FL_DP_NOISE*`) — variáveis vão no `superlink`, não no `server-app`

**Achado 2026-07-29:** quem aplica o ruído DP-FedAvg é o `ServerApp`
(`ProductionFedProxStrategy.aggregate_fit()`), e o `ServerApp` é subido pelo próprio SuperLink
como subprocesso (`flwr/supercore/superexec/executor/subprocess_executor.py`,
`subprocess.Popen(args)` **sem** `env=`) — ou seja, o `ServerApp` herda o ambiente do processo
do **SuperLink**, não do terminal onde você roda `flwr run`/`make server-app` (que só
*submete* o treino pro SuperLink já em execução). Exportar `FL_DP_NOISE` só na hora do
`make server-app` **não tem efeito nenhum** — é a mesma classe do achado de 2026-07-07 com
`FL_LLM_BACKEND` (seção 3.2), só que dessa vez confirmado lendo o código-fonte do `flwr`, não
só observado no comportamento.

Por isso o passo 1.7 (subir o SuperLink) tem 3 variantes prontas — escolha uma no lugar do
`make superlink` genérico, sem precisar copiar/colar valores soltos:

```bash
# Cenário baseline — sem ruído (equivalente ao make superlink puro)
FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl" make superlink-dp-off

# Cenário "uniforme" — ruído igual em todo o modelo, sigma=0.5 (comportamento histórico)
FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl" make superlink-dp-uniform

# Cenário "layer_group" — metade do ruído só na cabeça de classificação (sigma=0.5,
# head_scale=0.5), resto do modelo com ruído cheio
FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl" make superlink-dp-layer-group
```

Os três alvos sobem o mesmo `scripts/iniciar_servidor_fl.sh` de sempre — só variam os valores
de `FL_DP_NOISE`/`FL_DP_CLIP`/`FL_DP_NOISE_STRATEGY`/`FL_DP_NOISE_HEAD_SCALE` já embutidos no
`Makefile` (ver comentário no topo do arquivo). Depois de subir com um desses, o resto do
tutorial (1.8 em diante) não muda — `make server-app` continua sendo o comando que dispara o
treino.

**Como confirmar que o cenário certo realmente ativou:** depois da 1ª rodada, as colunas
`dp_epsilon_simple`, `dp_epsilon_rdp` e `dp_noise_strategy` de `metrics.fl_trainings` não devem
ficar `NULL` quando `FL_DP_NOISE > 0`. Se ficarem `NULL` com um cenário DP ativo, o SuperLink
provavelmente foi subido sem passar por um desses alvos (ex.: reaproveitando um terminal antigo
com `make superlink` puro ainda de pé) — pare o processo e suba de novo com o alvo certo.

---

## Usando GPU no Caminho B (desktop, achado em 2026-07-07)

Por padrão, o Caminho B roda inteiramente em **CPU** — `RUNTIME_CFG.device` (`src/mosaicfl/core/config.py`)
só usa GPU se `FL_DEVICE=cuda` estiver **explicitamente** definido (não há detecção
automática, mesmo com GPU disponível). Nenhum alvo do Caminho B (`superlink`, `supernode`,
`server-app`) define isso por conta própria — só os alvos do Caminho A (`training-full-cuda`
e afins) fazem.

**Quem se beneficia de GPU:** o treino/avaliação de verdade acontece do lado do **cliente**
(`FedProxClient.fit()`/`evaluate()`, dentro do `SuperNode`) — é lá que importa. O `SuperLink`
(coordenador) só agrega arrays, não roda forward/backward — não se beneficia de GPU.

**Comando, se você estiver rodando um cliente (SuperNode) local aqui no desktop** (ex.: o
BPSP local, usado nos testes de validação da mecânica do Caminho B):
```bash
export FL_DEVICE=cuda
export FL_TLS_CERT_DIR="/caminho/para/certs"
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:PORTA/BANCO"
make supernode FL_CLIENT_ID=BPSP FL_SUPERLINK_ADDRESS=localhost:9091 FL_DATA_SOURCE=sgbd
```

Confirme que a GPU está disponível antes:
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -c "import torch; print('cuda disponivel:', torch.cuda.is_available())"
```

> O mesmo vale pro notebook, se ele tiver GPU — exportar `FL_DEVICE=cuda` antes do
> `make supernode` de lá também. Sem GPU, deixe `FL_DEVICE` sem definir (cai no padrão CPU).

---

## Recarregar um seed já carregado (ex.: depois de regenerá-lo por uma correção)

Se você regenerar `bpsp_seed.sql.gz`/`hsl_seed.sql.gz` (por exemplo, após uma correção no
gerador) e o banco de destino **já tiver dados de uma carga anterior**, recarregar direto
falha assim:

```
ERROR:  duplicate key value violates unique constraint "patients_pkey"
DETAIL:  Key (patient_id)=(...) already exists.
```

É preciso truncar as tabelas antes de recarregar. `metrics.exam_records` e
`metrics.clinical_outcomes` **não** são apagadas automaticamente por `TRUNCATE
clinical.patients CASCADE` — não têm chave estrangeira para `patients`, então precisam
ser truncadas explicitamente.

**Atenção — nome do container:** `make server-db-reset`/`server-load-bpsp`/`client-db-reset`/
`client-load-hsl` sempre operam no container `mosaicfl-db` (o nome padrão do
`docker-compose.db.yml`) **a menos que você sobrescreva `FL_DB_CONTAINER` explicitamente**.
Este tutorial (Parte 1, passo 1.1) usa um container com nome customizado, `mosaicfl-db-bpsp`,
para não afetar o `mosaicfl-db` original. **Rodar `make server-db-reset` sem `FL_DB_CONTAINER`
trunca o container errado** (o `mosaicfl-db` principal, com os dados de outros treinos) —
já aconteceu uma vez nesta sessão. Sempre inclua a variável neste cenário:

**No desktop** (servidor — container customizado `mosaicfl-db-bpsp`):

```bash
make server-db-reset FL_DB_CONTAINER=mosaicfl-db-bpsp
gunzip -c scripts/db/seeds/bpsp_seed.sql.gz | \
  docker exec -i mosaicfl-db-bpsp psql -U mosaicfl -d mosaicfl -v ON_ERROR_STOP=1
```

**No notebook** (cliente — usa o container padrão `mosaicfl-db`, não precisa de override):

```bash
make client-db-reset
make client-load-hsl
```

---

## Checklist rápido

- [ ] Desktop: `mosaicfl-db-bpsp` rodando na porta 5433, migrations aplicadas, seed BPSP carregado
- [ ] Desktop: certificados gerados com o IP real, `FL_TLS_CERT_DIR` exportado
- [ ] Desktop: porta 9091 liberada no firewall
- [ ] Desktop: `make superlink` (ou `make superlink-dp-uniform`/`superlink-dp-layer-group` se o
      cenário usar DP; adicione `FL_CALIBRATION_METHOD=`/`FL_LLM_BACKEND=` aqui, não em
      `server-app`, se quiser um valor diferente do default — ver seção 3.4) rodando (deixar
      aberto num terminal)
- [ ] Desktop: `make supernode FL_CLIENT_ID=BPSP FL_SUPERLINK_ADDRESS=localhost:9091 FL_DATA_SOURCE=sgbd` rodando (segundo terminal, ver seção 1.8 — **obrigatório**, sem ele o quórum `min-clients=2` nunca fecha)
- [ ] Notebook: dependências instaladas (`make setup`)
- [ ] Notebook: banco local subido, migrations aplicadas, seed HSL carregado
- [ ] Notebook: `ca.crt` copiado do desktop
- [ ] Notebook: `make supernode` conectado ao IP do desktop (deixar aberto num terminal)
- [ ] Desktop: `make server-app` disparado (sem variáveis extras — `FL_CALIBRATION_METHOD`/`FL_LLM_BACKEND`/`FL_DP_NOISE*` já ficaram configuradas no `superlink`, passá-las aqui não tem efeito, ver seção 3.4)
- [ ] Desktop: Ollama rodando com `gemma3:4b` puxado (`ollama serve` + `ollama list`) — FL_LLM_BACKEND/FL_LLM_MODEL já vêm com default correto, só precisa o serviço estar de pé

## Se algo der errado

Qualquer erro nos logs de qualquer uma das duas máquinas, copie a mensagem e traga para a
próxima conversa — cada passo deste tutorial já foi validado individualmente (ver
`docs/Linha_do_Tempo_MOSAIC-FL.md`, seção "Rede Federada Real via SuperLink (Caminho B)"),
mas a combinação real entre duas máquinas físicas ainda não tinha sido testada até agora.


Logs cliente durante a validação:

make supernode FL_CLIENT_ID=HSL FL_SUPERLINK_ADDRESS=192.168.68.116:9091 FL_DATA_SOURCE=sgbd
FL_TLS_CERT_DIR=/home/jacabreu/studies/usp/tcc/simulacao-cliente-federado/certs \
FL_CLIENT_ID=HSL \
FL_DATA_SOURCE=sgbd \
FL_SUPERLINK_ADDRESS=192.168.68.116:9091 \
bash scripts/iniciar_cliente_fl.sh
INFO :      Starting Flower SuperNode
INFO :      Flower Deployment Runtime: Starting ClientAppIo API on 0.0.0.0:9094
INFO :      SuperNode ID: 908162020209674566
INFO :      Starting Flower SuperExec
INFO :      
INFO :      [RUN 7945113081761369056]
INFO :      Receiving: get_parameters message (ID: 7530dd3e64090b99d2fd3b3cd700fd278f3a9ae2a9b8ee14344a09eef1552ebd)
INFO :      Received successfully
INFO :      Start `flwr-clientapp` process
Successfully installed mosaicfl to /home/jacabreu/.flwr/apps/mosaic-fl.mosaicfl.0.2.0.366a23ab.
[SGBD] standard_vocab.json não encontrado — vocab será construído localmente. Execute scripts/build_standard_vocab.py antes do treinamento federado em produção.
[pipeline] conectando ao banco: postgresql://mosaicfl:senhaForte@localhost:5432/mo...
[pipeline] conexão OK
[pipeline] executando query (max_seq_len=128 — pode levar alguns minutos)...
[pipeline] query concluída em 0.0s — 0 linhas
ERROR :     ClientApp raised an exception
Traceback (most recent call last):
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/.venv/lib/python3.12/site-packages/flwr/supernode/runtime/run_clientapp.py", line 173, in run_clientapp
    reply_message = client_app(message=message, context=context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/.venv/lib/python3.12/site-packages/flwr/clientapp/client_app.py", line 144, in __call__
    return self._call(message, context)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/.venv/lib/python3.12/site-packages/flwr/clientapp/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/.venv/lib/python3.12/site-packages/flwr/client/message_handler/message_handler.py", line 97, in handle_legacy_message_from_msgtype
    client = client_fn(context)
             ^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/.flwr/apps/mosaic-fl.mosaicfl.0.2.0.366a23ab/infrastructure/mosaicfl_client/runner/supernode.py", line 45, in _client_fn
    _loader_cache[cache_key] = _split_loader(source.load())
                                             ^^^^^^^^^^^^^
  File "/home/jacabreu/.flwr/apps/mosaic-fl.mosaicfl.0.2.0.366a23ab/infrastructure/mosaicfl_client/datasource/sgbd.py", line 106, in load
    sequences, labels, vocab = pipeline.build(vocab=standard_vocab)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/src/mosaicfl/core/preprocessor/sequence_pipeline.py", line 183, in build
    df = self._load_dataframe()
         ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jacabreu/studies/usp/tcc/mosaic-fl/src/mosaicfl/core/preprocessor/sequence_pipeline.py", line 291, in _load_dataframe
    raise RuntimeError(
RuntimeError: Nenhum registro retornado. Verifique connection_string e o schema do banco.
INFO :      
INFO :      [RUN 7945113081761369056]
INFO :      Sending: get_parameters message (ID: e022156e2097c6f3cddd10dbca7930715f73d686996a8ebd9469738f664f8ee1)
INFO :      
INFO :      Sent successfully



