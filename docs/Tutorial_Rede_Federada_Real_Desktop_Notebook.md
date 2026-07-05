# Tutorial — Rede Federada Real (Desktop + Notebook), Caminho B com banco separado

**Cenário:** desktop = servidor (SuperLink) + dados BPSP; notebook = cliente (SuperNode) + dados HSL.
**Variante:** banco do servidor é um Postgres **novo e separado** (porta 5433), sem mexer no
`mosaicfl-db` já existente no desktop — que continua intacto com o dataset combinado (BPSP+HSL)
usado nos Treinamentos Reais 1/2/3 e na curva Acc×ε (ver `docs/Sumario_Treinamento_Parte3.md`).

Contexto geral do Caminho B (o que é SuperLink/ServerApp/SuperNode, TLS, `~/.flwr/config.toml`)
está documentado no `README.md`, seção "Rede Federada Real via SuperLink (Desktop + Notebook) —
Caminho B". Este tutorial é o passo a passo prático da variante com banco separado.

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
gunzip -c scripts/db/seeds/bpsp_seed.sql.gz | \
  docker exec -i mosaicfl-db-bpsp psql -U mosaicfl -d mosaicfl -v ON_ERROR_STOP=1
```

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

> **Log em arquivo:** além de aparecer no terminal, o log é gravado em
> `experiments/logs/superlink_<timestamp>.log` (caminho exato impresso ao subir).
> Sobrescreva com `FL_LOG_FILE=/caminho/seu.log` se preferir.

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
> `FL_DB_URL` exportado acima — não precisa de passo manual separado.

### 2.5 Copiar o certificado CA do desktop

**Só o `ca.crt`** — nunca `ca.key`/`server.key`, que ficam só no desktop:

```bash
scp usuario@IP_DESKTOP:~/mosaic-fl/certs/ca.crt certs/ca.crt
export FL_TLS_CERT_DIR="$(pwd)/certs"
```

### 2.6 Conectar o SuperNode ao SuperLink do desktop

```bash
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
- [ ] Desktop: `make superlink` rodando (deixar aberto num terminal)
- [ ] Notebook: dependências instaladas (`make setup`)
- [ ] Notebook: banco local subido, migrations aplicadas, seed HSL carregado
- [ ] Notebook: `ca.crt` copiado do desktop
- [ ] Notebook: `make supernode` conectado ao IP do desktop (deixar aberto num terminal)
- [ ] Desktop: `make server-app` disparado

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



