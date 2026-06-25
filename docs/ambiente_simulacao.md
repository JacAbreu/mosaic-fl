# Ambiente de Simulação Federada — Desktop (Servidor) + Notebook (Cliente)

Este documento descreve o ambiente físico utilizado para validar a federação do MOSAIC-FL
em duas máquinas reais comunicando-se via rede local, simulando a topologia de um servidor
central e um hospital cliente.

> **Contexto:** em um cenário real de implantação hospitalar, o MOSAIC-FL seria instalado
> nos servidores de cada instituição. Para fins de validação acadêmica do TCC, a federação
> é demonstrada entre dois equipamentos físicos distintos operados pela pesquisadora.

---

## Topologia da Simulação

```
Desktop (Servidor FL + Hospital BPSP)      Notebook (Cliente FL + Hospital HSL)
┌──────────────────────────────────┐       ┌──────────────────────────────────┐
│  flower-superlink  :9091         │       │  flower-supernode                │
│  ServerApp (FedProx)             │◄─────►│  FedProxClient                   │
│  Dados BPSP (local)              │  LAN  │  Dados HSL (local)               │
│  PostgreSQL :5432                │       │  PostgreSQL :5432                │
│  API REST   :8000                │       │                                  │
└──────────────────────────────────┘       └──────────────────────────────────┘
         pesos globais (~2,8 MB) ──────────────► round local
         ◄──────────────── pesos atualizados ───
```

Apenas os **pesos do modelo** trafegam pela rede. Os dados clínicos de cada hospital
permanecem exclusivamente na máquina local — nunca são transmitidos.

---

## Equipamentos

### Servidor — Desktop

| Componente       | Especificação                                         |
|------------------|-------------------------------------------------------|
| **CPU**          | Intel Core i9-13900K — 24 núcleos físicos, 32 threads, até 6,0 GHz |
| **RAM**          | 32 GB                                                 |
| **Cache L2**     | 32 MB (12 instâncias)                                 |
| **Cache L3**     | 36 MB                                                 |
| **GPU**          | NVIDIA GeForce RTX 4070 Ti (12 GB VRAM) — driver não instalado* |
| **Armazenamento**| NVMe 937 GB                                           |
| **SO**           | Linux 6.17.0                                          |
| **Papel no FL**  | Servidor (SuperLink + ServerApp) + Hospital BPSP      |

> \* A RTX 4070 Ti está presente no barramento PCI mas o driver NVIDIA não está instalado.
> O PyTorch opera em modo CPU. Com o driver instalado, o ganho de desempenho seria
> substancial — o treinamento BEHRT em GPU reduziria as 20 rodadas federadas de ~15 min
> para ordem de minutos.

### Cliente — Notebook

| Componente       | Especificação                                         |
|------------------|-------------------------------------------------------|
| **Modelo**       | Dell Inspiron 5402                                    |
| **CPU**          | Intel Core i7-1165G7 — 4 núcleos, 8 threads, até 4,7 GHz |
| **RAM**          | 16 GB                                                 |
| **GPU**          | Intel Iris Xe (integrada) — sem suporte CUDA          |
| **SO**           | Linux                                                 |
| **Papel no FL**  | Cliente (SuperNode) + Hospital HSL                    |

---

## Configuração de Software Calibrada para o Notebook

Os parâmetros em `src/mosaicfl/core/config.py` foram ajustados para rodar de forma
estável no notebook (hardware mais restrito dos dois):

| Parâmetro             | Valor | Motivo do ajuste                              |
|-----------------------|-------|-----------------------------------------------|
| `DEVICE`              | `cpu` | Intel Iris Xe sem suporte CUDA                |
| `OMP_NUM_THREADS`     | 4     | Libera 4 threads para o SO, evita travamento  |
| `MKL_NUM_THREADS`     | 4     | Idem                                          |
| `TOKENIZERS_PARALLELISM` | false | Elimina conflito de threads do HuggingFace |
| `BATCH_SIZE`          | 16    | Reduz uso de RAM por round                    |
| `LOCAL_EPOCHS`        | 2     | Menos iterações por rodada federada           |
| `NUM_ROUNDS`          | 20    | Balanceia qualidade × tempo de execução       |
| `MAX_NEW_TOKENS`      | 64    | Geração de texto mais rápida no RAG           |
| `max_seq_len`         | 128   | Sequência máxima do BEHRT                     |
| `vocab_size`          | 10.000| Tamanho do vocabulário de tokens clínicos     |

O desktop opera com os mesmos parâmetros para garantir reprodutibilidade dos resultados.
Em execuções exclusivas no desktop, os parâmetros podem ser aumentados via variáveis de
ambiente sem alterar o código (ex.: `FL_BATCH_SIZE=32`, `FL_NUM_ROUNDS=50`).

---

## Dados por Máquina

| Máquina  | Hospital | Tabela principal | Pacientes | Exames     | Desfechos |
|----------|----------|------------------|-----------|------------|-----------|
| Desktop  | BPSP     | `metrics.exam_records` | ~39.000 | ~5,3 M | ~218.000 |
| Notebook | HSL      | `metrics.exam_records` | ~8.971  | ~1,5 M | ~42.691  |

Fonte: repositório USP-FAPESP Data Sharing COVID-19
([uspdigital.usp.br](https://repositoriodatasharingfapesp.uspdigital.usp.br/handle/item/1)).
Ambos os hospitais possuem tabela `Desfechos`, requisito para geração do label de prognóstico
clínico (4 classes: alta, internação prolongada, UTI, óbito).

---

## Como Preparar o Banco do Cliente (Notebook)

Os ZIPs do FAPESP ficam no desktop. Para carregar os dados do HSL no notebook sem
precisar transferir os arquivos originais (~19 MB comprimido, ~230 MB descomprimido),
o desktop gera um arquivo SQL comprimido que é então carregado no banco do notebook.

### Passo 1 — Gerar o seed no Desktop

```bash
# No desktop (onde estão os ZIPs FAPESP):
make client-generate-seed
# Gera: scripts/db/seeds/hsl_seed.sql.gz (~20–40 MB)
#
# Se os ZIPs estiverem em outro diretório:
make client-generate-seed FL_DATA_DIR=/outro/caminho/Covid-19
#
# Com resolução canônica de analitos (recomendado se o banco local estiver populado):
make client-generate-seed FL_DB_URL=postgresql://mosaicfl:senha@localhost:5432/mosaicfl
```

O arquivo contém blocos `COPY FROM stdin` para as 4 tabelas:
`clinical.patients` · `clinical.attendances` · `metrics.clinical_outcomes` · `metrics.exam_records`

### Passo 2 — Transferir o seed para o Notebook

```bash
# Opção A — via git (recomendado se o arquivo couber no repositório):
git add scripts/db/seeds/hsl_seed.sql.gz
git commit -m "seed: adiciona dados HSL para simulação cliente"
git push
# No notebook: git pull

# Opção B — via scp (na mesma rede local):
scp scripts/db/seeds/hsl_seed.sql.gz usuario@<IP_NOTEBOOK>:~/mosaic-fl/scripts/db/seeds/

# Opção C — pendrive / qualquer mídia removível
```

### Passo 3 — Preparar o banco no Notebook

```bash
# No notebook — um único comando faz tudo:
make client-setup
# Equivalente a:
#   make client-db-up      # sobe PostgreSQL via Docker
#   make client-migrate    # aplica migrations 001→010 (inclui registro do nó como cliente HSL)
#   make client-load-hsl   # carrega hsl_seed.sql.gz no banco

# Se o seed estiver em outro caminho:
make client-setup HSL_SEED=/outro/caminho/hsl_seed.sql.gz
```

A migration 010 registra automaticamente este banco como nó cliente do HSL —
ao inspecionar o banco, o papel do nó fica explícito:

```sql
SELECT node_role, hospital_id, description FROM clinical.simulation_node_config;
-- client | HSL | Simulação federada TCC — Notebook Dell Inspiron 5402 ...
```

---

## Como Preparar o Banco do Servidor (Desktop)

O servidor usa os dados do **BPSP** — o hospital com maior volume dentre os que possuem
desfechos, requisito obrigatório para geração dos labels de prognóstico no `SequencePipeline`.

> **Por que não Einstein?**
> O Einstein possui 3,4M de exames (mais que o BPSP antes do corte), mas **não tem arquivo
> de desfechos**. Sem desfechos não é possível gerar os labels de prognóstico clínico
> (alta / internação prolongada / UTI / óbito). O BPSP tem 6,3M de exames e 218K desfechos.

| Hospital | Exames  | Desfechos | Viável para treino FL? |
|----------|---------|-----------|------------------------|
| Fleury   | 19,3 M  | —         | Não                    |
| **BPSP** | **6,3 M** | **218K** | **Sim — servidor**    |
| Einstein | 3,4 M   | —         | Não                    |
| HC-SP    | 2,5 M   | —         | Não                    |
| HSL      | 1,5 M   | 42K       | Sim — cliente          |

### Passo 1 — Gerar o seed no Desktop

```bash
# No desktop (onde estão os ZIPs FAPESP):
make server-generate-seed
# Gera: scripts/db/seeds/bpsp_seed.sql.gz

# Com resolução canônica de analitos (recomendado):
make server-generate-seed FL_DB_URL=postgresql://mosaicfl:senha@localhost:5432/mosaicfl
```

### Passo 2 — Carregar os dados no Desktop

```bash
# Sequência completa (sobe banco + migrations + reset + carrega BPSP):
make server-setup

# Ou passo a passo:
make db-up                # sobe PostgreSQL via Docker
make client-migrate       # aplica migrations 001→010
make server-db-reset      # apaga dados anteriores (preserva schema)
make server-load-bpsp     # carrega bpsp_seed.sql.gz no banco
```

O seed BPSP registra automaticamente este banco como nó servidor:

```sql
SELECT node_role, hospital_id, description FROM clinical.simulation_node_config;
-- server | BPSP | Simulacao federada TCC - Desktop i9-13900K / 32 GB ...
```

---

## Como Executar o Treinamento Federado

Após preparar ambos os bancos (`make server-setup` no desktop e `make client-setup` no
notebook), consulte a seção **Rede Federada Real** no [`README.md`](../README.md) para
configuração de rede e abertura de portas.

```bash
# No desktop (servidor + BPSP):
make fl-server FL_DB_URL=postgresql://mosaicfl:senha@localhost:5432/mosaicfl \
               FL_HOSPITAL_ID=BPSP
# O servidor imprime o IP local ao subir — copie para o próximo comando.

# No notebook (cliente + HSL):
make fl-client FL_SERVER=<IP_DO_DESKTOP>:8080 \
               FL_DB_URL=postgresql://mosaicfl:senha@localhost:5432/mosaicfl \
               FL_HOSPITAL_ID=HSL
```

---

## Relevância para o TCC

Esta configuração de dois nós físicos demonstra os seguintes aspectos do aprendizado federado:

1. **Isolamento real de dados** — cada banco PostgreSQL reside exclusivamente em sua máquina;
   nenhum dado clínico cruza a rede.
2. **Heterogeneidade de hardware** — o cliente (notebook) é significativamente menos
   potente que o servidor; o FedProx acomoda essa assimetria via termo proximal.
3. **Heterogeneidade de dados (não-IID)** — HSL e BPSP possuem populações distintas
   (8.971 vs. 39.000 pacientes) e distribuições de desfecho diferentes, cenário
   realista de fragmentação hospitalar.
4. **Overhead de comunicação mensurável** — ~2,8 MB por round × 2 direções × 20 rounds
   ≈ 112 MB de tráfego total, verificável com `benchmark.py`.
