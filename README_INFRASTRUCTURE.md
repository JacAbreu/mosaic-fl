# MOSAIC-FL — Infraestrutura de Produção

## Visão Geral

A pasta `infrastructure/` contém os **pacotes plug-and-play** para deploy em produção:

| Pacote | O que faz | Onde instalar |
|---|---|---|
| `mosaicfl-server` | Servidor Flower 24/7, agregação de pesos, checkpoints | AWS, nuvem, USP |
| `mosaicfl-client` | Cliente no hospital, treina com EHR local, heartbeat | Cada hospital (HF1-HF5) |

Ambos dependem do pacote core `mosaicfl` (pasta `src/`).

## Estrutura do Projeto

```
mosaic-fl/                          ← repositório GitHub único
│
├── src/                            ← pacote core: mosaicfl
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── preprocess.py
│   ├── data_loader.py
│   ├── client.py
│   ├── server.py
│   └── ...
│
├── infrastructure/
│   ├── server/
│   │   ├── pyproject.toml          ← define pacote: mosaicfl-server
│   │   └── mosaicfl_server/
│   │       ├── __init__.py
│   │       ├── __main__.py         ← entrypoint CLI
│   │       ├── strategy.py         ← FedProx + checkpoint + convergência
│   │       └── runner.py           ← orquestrador do servidor
│   │
│   └── client/
│       ├── pyproject.toml          ← define pacote: mosaicfl-client
│       └── mosaicfl_client/
│           ├── __init__.py
│           ├── __main__.py         ← entrypoint CLI
│           ├── heartbeat.py        ← escreve client_registry.json
│           └── runner.py           ← orquestrador do cliente
│
├── publish-server.sh               ← build + publish mosaicfl-server
├── publish-client.sh               ← build + publish mosaicfl-client
├── install.sh                      ← instala tudo localmente (dev)
├── pyproject.toml                  ← pacote core: mosaicfl
├── run_v2_unified.py               ← experimento TCC
└── ...
```

## Instalação

### 1. Desenvolvimento (tudo local)

```bash
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl
./install.sh
```

Isso instala:
- `mosaicfl` (pacote core da pasta `src/`)
- `mosaicfl-server` (da pasta `infrastructure/server/`)
- `mosaicfl-client` (da pasta `infrastructure/client/`)

### 2. Servidor na AWS (apenas server)

```bash
# Na instância AWS
pip install mosaicfl-server

# Ou com o pacote wheel
pip install mosaicfl_server-0.2.0-py3-none-any.whl

# Roda
mosaicfl-server --port 8080 --min-clients 3 --rounds 20
```

### 3. Cliente no Hospital (apenas client)

```bash
# Na máquina do hospital
pip install mosaicfl-client

# Ou com o pacote wheel
pip install mosaicfl_client-0.2.0-py3-none-any.whl

# Roda (conecta ao servidor AWS)
mosaicfl-client --server 52.67.123.45:8080 --client-id hospital_a
```

## Publicação (para distribuição)

### Publicar server

```bash
./publish-server.sh
```

O script:
1. Entra em `infrastructure/server/`
2. Limpa builds antigos
3. Roda `python -m build`
4. Publica no PyPI (ou instala localmente para teste)

### Publicar client

```bash
./publish-client.sh
```

Mesmo processo para o pacote cliente.

## Comandos CLI

### Servidor

```bash
# Ajuda
mosaicfl-server --help

# Opções comuns
mosaicfl-server --port 8080 --min-clients 3 --rounds 20 --mu 0.01
mosaicfl-server --address 0.0.0.0:9090 --checkpoint-dir /data/checkpoints
```

### Cliente

```bash
# Ajuda
mosaicfl-client --help

# Opções comuns
mosaicfl-client --server 192.168.1.100:8080 --client-id hospital_a
mosaicfl-client --server aws.mosaicfl.org:8080 --client-id hf2 --device cpu
```

## Variáveis de Ambiente

| Variável | Default | Aplica-se a | Descrição |
|---|---|---|---|
| `FL_SERVER_ADDRESS` | `0.0.0.0:8080` | server | Endereço de escuta |
| `FL_MIN_AVAILABLE_CLIENTS` | `3` | server | Mínimo para iniciar round |
| `FL_NUM_ROUNDS` | `20` | server | Máximo de rounds |
| `FL_PROXIMAL_MU` | `0.01` | server | FedProx mu |
| `FL_CHECKPOINT_DIR` | `checkpoints/` | server | Onde salvar modelos |
| `FL_LOG_DIR` | `logs/` | ambos | Logs e métricas |
| `FL_CLIENT_ID` | `client_0` | client | ID do hospital |
| `FL_HEARTBEAT_INTERVAL` | `60` | client | Segundos entre heartbeats |
| `FL_RECONNECT_DELAY` | `30` | client | Segundos para reconectar |
| `FL_DEVICE` | `cpu` | ambos | PyTorch device |

## Dependências

```
mosaicfl-client  ──►  mosaicfl (core)
       │
       ▼
    flwr, torch, pandas, numpy

mosaicfl-server  ──►  mosaicfl (core)
       │
       ▼
    flwr, torch, apscheduler
```

Ambos dependem do `mosaicfl` core, que contém `model.py`, `config.py`, `preprocess.py`, etc.

## Diferença: TCC vs. Produção

| Aspecto | TCC (`run_v2_unified.py`) | Produção (`mosaicfl-server` + `mosaicfl-client`) |
|---|---|---|
| **Instalação** | `pip install -e .` (repo inteiro) | `pip install mosaicfl-server` ou `pip install mosaicfl-client` |
| **Servidor** | Loop local | `mosaicfl-server` daemon na porta 8080 |
| **Cliente** | DataLoader na mesma máquina | `mosaicfl-client` em máquina separada |
| **Dados** | CSV em memória | EHR local no hospital |
| **Rede** | Loopback | Internet/VPN com gRPC |
| **Comando** | `python run_v2_unified.py` | `mosaicfl-server` + `mosaicfl-client` |

## Próximos Passos

1. **TLS/SSL**: Configure certificados no gRPC do Flower
2. **Docker**: Crie `Dockerfile` para server e client
3. **Kubernetes**: Use Helm charts para deploy na AWS
4. **PyPI**: Publique os pacotes para instalação via `pip install`
5. **CI/CD**: GitHub Actions para build e publish automático
