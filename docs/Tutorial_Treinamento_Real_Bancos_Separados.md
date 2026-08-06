# Tutorial — Treinamentos Reais (Caminho B) com 3 Bancos Totalmente Separados

**Contexto:** até aqui (`docs/Tutorial_Rede_Federada_Real_Desktop_Notebook.md`), o SuperLink (servidor) e o SuperNode BPSP (cliente, ambos no desktop) apontavam pro **mesmo** banco Postgres (porta 5433) — conveniente, mas significa que o processo do servidor tecnicamente tinha acesso de leitura ao dado bruto de paciente do BPSP, mesmo nunca usando isso no código. Como a privacidade do MOSAIC-FL passou a se apoiar explicitamente na arquitetura do Federated Learning (localidade de dado — ver seção 18 do doc de pesquisa, `docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md`), essa premissa só é defensável de verdade se o servidor **não puder** acessar dado de paciente de nenhum hospital, nem em princípio. Este tutorial corrige isso: **3 bancos Postgres totalmente separados, sem nenhum compartilhamento de container/porta/volume entre eles.**

**Investigação prévia (2026-08-02) que confirma que isso é viável sem perder funcionalidade:**
- O servidor só precisa, do próprio banco: schema completo (migrations), `knowledge.term_dictionary` (catálogo estático de analitos, **seedado pela própria migration 008** — não depende de dado de paciente), `clinical.fl_orchestration_config` (config, sem PHI), e as tabelas de `metrics.fl_trainings`/`fl_checkpoints`/`fl_round_history` (resultado do treino, sem PHI) — confirmado lendo `scripts/build_standard_vocab.py` (só consulta `knowledge.term_dictionary`/`knowledge.analyte_references`) e `infrastructure/shared/checkpoint_store/postgres_store.py` (só `metrics.*`).
- `knowledge.analyte_references` (faixas de referência por analito) **não precisa existir no banco do servidor**: sem ela, o vocabulário de boot fica mais grosseiro (só tokens `NO_REF`, sem `HIGH`/`NORMAL`/`LOW`) — mas isso não quebra nada, é um caminho já previsto no código (`build_standard_vocab.py`, sem limite mínimo de analitos-com-referência). O vocabulário fino (com referências reais) chega depois, **antes da Rodada 1**, pelo mecanismo de descoberta bidirecional de vocabulário já implementado (`ProductionFedProxStrategy._discover_and_curate_vocab`, ver memória do projeto `project_vocabulario_bidirecional_proposto`) — cada cliente consulta **seu próprio banco local** (`SGBDDataSource.find_vocab_candidates`) e manda só candidatos agregados por analito ao servidor, nunca dado bruto. **Esse mecanismo está implementado e testado (30 testes) desde 2026-07-26, mas nunca foi validado com 2 máquinas físicas reais** — este tutorial é a primeira vez que isso acontece de verdade (seção 6 abaixo).

**Make novo criado hoje** (`db-instance-up`/`db-instance-down`, Makefile) — até agora, subir um Postgres adicional nomeado (fora do `docker-compose.db.yml` padrão) exigia copiar um `docker run` na mão (era assim que `mosaicfl-db-bpsp` era criado no tutorial antigo) — sem `make`, risco real de esquecer um parâmetro. Agora:
```bash
make db-instance-up FL_DB_INSTANCE_CONTAINER=<nome> FL_DB_INSTANCE_PORT=<porta>
make db-instance-down FL_DB_INSTANCE_CONTAINER=<nome>   # pra desmontar depois, se precisar
```

---

## Topologia final

| Máquina | Papel | Container | Porta | Conteúdo |
|---|---|---|---|---|
| Desktop | **Servidor** (SuperLink/ServerApp) | `mosaicfl-db-server` | 5434 | Schema completo, SEM dado de paciente. Term dictionary, config, resultados de treino. |
| Desktop | **Cliente BPSP** (SuperNode) | `mosaicfl-db-bpsp` | 5433 | Dado real do BPSP (como já era). |
| Notebook | **Cliente HSL** (SuperNode) | `mosaicfl-db-hsl` (novo, não o `mosaicfl-db` padrão) | 5435 | Dado real do HSL, container próprio — não reaproveita o banco usado nos treinos de ajuste. |

Nenhum dos três containers compartilha volume, porta ou container entre si. O processo do servidor (`FL_DB_URL` do SuperLink) só aponta pra porta 5434 — nunca pra 5433 ou 5435 (nem pro `mosaicfl-db` antigo do notebook, que fica intocado).

Certificados TLS: reaproveitar os que já existem em `/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs` (gerados no tutorial anterior, seção 1.5) — não precisa gerar de novo, a rede (desktop+notebook) continua a mesma.

---

## Parte 1 — Desktop: banco do SERVIDOR (novo, sem dado de paciente)

```bash
make db-instance-up FL_DB_INSTANCE_CONTAINER=mosaicfl-db-server FL_DB_INSTANCE_PORT=5434
```

Só migrations — **nunca** rode `server-load-bpsp`/`client-load-hsl` contra este banco:
```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5434/mosaicfl"
make full-db-migrate FL_DB_URL="$FL_DB_URL"
```

Confirme que `term_dictionary` já veio populado pela migration (sem precisar carregar nenhum dado de paciente) e que não há nenhuma tabela clínica com linhas:
```bash
docker exec mosaicfl-db-server psql -U mosaicfl -d mosaicfl -c \
  "SELECT count(*) AS termos FROM knowledge.term_dictionary WHERE term_type='analyte';"
docker exec mosaicfl-db-server psql -U mosaicfl -d mosaicfl -c \
  "SELECT count(*) AS pacientes FROM clinical.patients;"
```
O primeiro deve retornar um número > 0 (catálogo estático). **O segundo TEM que retornar 0** — se não retornar, pare e não prossiga (algo rodou `server-load-bpsp` nesse banco por engano).

---

## Parte 2 — Desktop: banco do BPSP (cliente)

Mesmo processo de sempre, só que agora via `make` em vez de `docker run` manual:
```bash
make db-instance-up FL_DB_INSTANCE_CONTAINER=mosaicfl-db-bpsp FL_DB_INSTANCE_PORT=5433

export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
make full-db-migrate FL_DB_URL="$FL_DB_URL"
```

Carregar o BPSP (gera o seed se ainda não existir, ver tutorial anterior seção 1.3, e carrega):
```bash
make server-load-bpsp FL_DB_CONTAINER=mosaicfl-db-bpsp FL_DB_URL="$FL_DB_URL"
```

---

## Parte 3 — Notebook: banco do HSL (cliente)

**Correção 2026-08-06 em relação ao tutorial anterior:** NÃO reaproveitar o container padrão do compose (`mosaicfl-db`, porta 5432) — esse container já acumulou uso das trainings de ajuste anteriores. Pra manter a mesma garantia de "banco novo, sem estado herdado" que já vale pro servidor (Parte 1) e pro BPSP (Parte 2), o HSL também sobe com `db-instance-up`, container e porta próprios. O `mosaicfl-db` antigo fica intacto, sem ser tocado — não precisa apagar nada.

```bash
git pull   # scripts/db/seeds/hsl_seed.sql.gz já vem versionado no git, não precisa scp

make db-instance-up FL_DB_INSTANCE_CONTAINER=mosaicfl-db-hsl FL_DB_INSTANCE_PORT=5435

export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5435/mosaicfl"
make client-migrate

make client-load-hsl FL_DB_CONTAINER=mosaicfl-db-hsl FL_DB_URL="$FL_DB_URL"
```
`client-load-hsl` já inclui o backfill de `classification` e o cálculo de `analyte_references` local (`compute_analyte_references.py`) — é exatamente esse dado local que alimenta `find_vocab_candidates()` na Parte 6, então nada fica faltando pra validar a descoberta de vocabulário mesmo com o container novo.

Copiar o certificado CA do desktop (só `ca.crt`):
```bash
scp usuario@IP_DESKTOP:/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs/ca.crt certs/ca.crt
```

Subir o SuperNode HSL, apontando pro banco novo (5435):
```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5435/mosaicfl"
export FL_TLS_CERT_DIR="$(pwd)/certs"
make supernode FL_CLIENT_ID=HSL FL_SUPERLINK_ADDRESS=<IP_DO_DESKTOP>:9091 FL_DATA_SOURCE=sgbd
```

---

## Parte 4 — Firewall e certificados

Igual ao tutorial anterior, seções 1.5/1.6 — certificados já existem (`/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs`), só exportar `FL_TLS_CERT_DIR`; porta 9091 já deve estar liberada se você já rodou o tutorial anterior nesta máquina.

---

## Parte 5 — Subir os processos (ordem importa)

**Desktop — SuperLink, agora apontando pro banco do SERVIDOR (5434), não mais 5433:**
```bash
export FL_TLS_CERT_DIR=/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5434/mosaicfl"
make superlink-dp-off      # ou superlink-dp-uniform / superlink-dp-layer-group — ver Parte 7
```

**Desktop — SuperNode BPSP, apontando pro banco do BPSP (5433), outro terminal:**
```bash
export FL_TLS_CERT_DIR=/home/jacabreu/studies/usp/mba-bigdata-art-int/tcc/certs
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5433/mosaicfl"
make supernode FL_CLIENT_ID=BPSP FL_SUPERLINK_ADDRESS=localhost:9091 FL_DATA_SOURCE=sgbd
```

**Notebook — SuperNode HSL, banco NOVO (5435, não mais 5432 — ver Parte 3):**
```bash
export FL_DB_URL="postgresql://mosaicfl:senhaForte@localhost:5435/mosaicfl"
export FL_TLS_CERT_DIR="$(pwd)/certs"
make supernode FL_CLIENT_ID=HSL FL_SUPERLINK_ADDRESS=<IP_DO_DESKTOP>:9091 FL_DATA_SOURCE=sgbd
```

---

## Parte 6 — Descoberta bidirecional de vocabulário: o que esperar de verdade (achado 2026-08-06)

**Correção da expectativa original deste tutorial.** Investigando o pedido de garantir vocabulário assimétrico real entre HSL/servidor/BPSP, confirmei com queries diretas contra o BPSP já carregado (não suposição):

- Existem **145 analitos** com volume real (≥100 registros) no BPSP fora de `knowledge.term_dictionary` — a assimetria que motivaria a descoberta EXISTE de verdade nos dados.
- Mas `find_vocab_candidates()` usa `min_records=100` **fixo no código** (nunca sobrescrito por nenhum chamador), e `select_insertable()` só aceita candidato com referência institucional real (`has_real_ref=True`). **Dos 145 analitos com volume ≥100, nenhum tem referência real** — os que têm (72, achados baixando o piso pra ≥10) ficam abaixo do piso de 100 e nunca chegam a ser considerados.
- Ou seja: com o dado real de hoje, o caminho de **crescimento** do vocabulário (candidato aceito e inserido) não deve disparar com o piso padrão — confirma o mesmo padrão já visto nos treinos de ajuste desta semana (`vocab_discovery_candidates_rejected`).
- **Ponto de virada confirmado:** o candidato de maior volume com referência real (`PROTEINA_S_ATIVIDADE`) tem exatamente 94 registros — `min_records ≤ 94` já é suficiente pra pelo menos 1 candidato passar.

**Correção 2026-08-06 (mesma sessão):** `FL_VOCAB_MIN_RECORDS` criado — piso configurável por treino, sem mudar o default de produção (100, preservado quando a env var não é setada). No SuperNode de qualquer um dos dois hospitais (BPSP no desktop e/ou HSL no notebook — não precisa nos dois, um já basta pra provar o caminho):
```bash
make supernode FL_CLIENT_ID=BPSP FL_DATA_SOURCE=sgbd FL_VOCAB_MIN_RECORDS=90 ...
```

**O que validar nesta rodada:**
```bash
tail -f "$(ls -t experiments/logs/serverapp_*.log | head -1)" | grep -e vocab_discovery -e vocab_boot -e vocab_construído
```
Com `FL_VOCAB_MIN_RECORDS=90` em pelo menos um cliente, espera-se `vocab_discovery_candidates_rejected` sumir e aparecer algo como candidatos aceitos/inseridos, com o vocab final maior que o de boot (novos tokens `NO_REF`, já que o servidor continua sem `analyte_references` próprias — a granularidade `HIGH`/`NORMAL`/`LOW` desses novos analitos nunca viria do servidor mesmo, só cresce o catálogo, não a classificação). **Só reporte como problema real se aparecer erro/exceção, timeout de `client_manager.wait_for`, ou o treino travando antes da Rodada 1.**

---

## Parte 7 — Os 6 treinamentos formais

`FL_RUN_CLASSIFICATION=treinamento_real` em todos — sem essa flag, o treino grava como `ajuste` e não conta como resultado formal (`make server-app` sozinho grava `ajuste` por padrão, ver Makefile, achado 2026-08-02).

**2× sem DP:**
```bash
# subir o SuperLink (Parte 5) com: make superlink-dp-off
make server-app FL_RUN_CLASSIFICATION=treinamento_real
# repetir mais uma vez, do zero (novo SuperLink + 2 SuperNodes), pra 2ª réplica
```

**2× DP uniforme:**
```bash
# subir o SuperLink com: make superlink-dp-uniform FL_EARLY_STOP=true
make server-app FL_RUN_CLASSIFICATION=treinamento_real
# repetir mais uma vez pra 2ª réplica
```

**2× DP layer_group:**
```bash
# subir o SuperLink com: make superlink-dp-layer-group FL_EARLY_STOP=true
make server-app FL_RUN_CLASSIFICATION=treinamento_real
# repetir mais uma vez pra 2ª réplica
```

`FL_EARLY_STOP=true` recomendado nos cenários com DP (correção validada nesta semana, ver seção 16 do doc de pesquisa) — opcional no cenário sem DP (convergência não tem o mesmo problema de custo de privacidade acumulado, mas não atrapalha usar também).

Cada treino registra um `training_id` novo — confira em `/fl-training-results` que `run_classification=treinamento_real` e o `dp_noise_strategy` batem com o que você pretendia rodar antes de considerar o treino válido.

---

## Checklist rápido

- [ ] `mosaicfl-db-server` (5434): migrations aplicadas, `term_dictionary` populado, **zero** linhas em `clinical.patients`
- [ ] `mosaicfl-db-bpsp` (5433): migrations + `server-load-bpsp` aplicados
- [ ] Notebook `mosaicfl-db` (5432): migrations + `client-load-hsl` aplicados
- [ ] SuperLink sobe apontando pro banco **5434** (não 5433)
- [ ] SuperNode BPSP sobe apontando pro banco **5433**
- [ ] SuperNode HSL sobe apontando pro banco **5432** do notebook
- [ ] Descoberta de vocabulário confirmada no log (Parte 6) — vocab final maior que o de boot, com tokens `HIGH`/`NORMAL`/`LOW`
- [ ] 6 treinos rodados (2× cada cenário), todos com `run_classification=treinamento_real` confirmado em `/fl-training-results`
