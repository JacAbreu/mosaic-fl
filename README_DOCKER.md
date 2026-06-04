# MOSAIC-FL — Docker e Docker Compose

## Visão Geral

Imagens Docker plug-and-play para deploy do MOSAIC-FL em produção.

| Imagem | O que faz | Tamanho estimado |
|---|---|---|
| `mosaicfl-server` | Servidor Flower 24/7 | ~1.2GB (inclui PyTorch) |
| `mosaicfl-client` | Cliente no hospital | ~1.2GB (inclui PyTorch) |

## Arquivos

| Arquivo | Descrição |
|---|---|
| `Dockerfile.server` | Imagem do servidor |
| `Dockerfile.client` | Imagem do cliente |
| `docker-compose.yml` | Orquestra servidor + 3 clientes para teste |
| `.dockerignore` | Exclui arquivos desnecessários do build |
| `build-and-push.sh` | Build e push para registry |

## Uso Rápido

### 1. Teste local com Docker Compose

```bash
# Build e inicia tudo (servidor + 3 clientes)
docker-compose up --build

# Em outro terminal: ver logs
docker-compose logs -f mosaicfl-server

# Parar
docker-compose down

# Parar e remover volumes
docker-compose down -v
```

### 2. Deploy do servidor na AWS

```bash
# Build local
docker build -f Dockerfile.server -t mosaicfl-server:latest .

# Tag para ECR (AWS)
docker tag mosaicfl-server:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/mosaicfl-server:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/mosaicfl-server:latest

# Na instância AWS
docker run -d \
  -p 8080:8080 \
  -e FL_MIN_AVAILABLE_CLIENTS=3 \
  -e FL_NUM_ROUNDS=20 \
  -v /data/checkpoints:/app/checkpoints \
  -v /data/logs:/app/logs \
  --name mosaicfl-server \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/mosaicfl-server:latest
```

### 3. Deploy do cliente no hospital

```bash
# Build local
docker build -f Dockerfile.client -t mosaicfl-client:latest .

# Na máquina do hospital
docker run -d \
  -e FL_SERVER_ADDRESS=52.67.123.45:8080 \
  -e FL_CLIENT_ID=hospital_a \
  -v /hospital/data:/app/data:ro \
  -v /hospital/logs:/app/logs \
  --name mosaicfl-client \
  mosaicfl-client:latest
```

## Variáveis de Ambiente (Docker)

### Servidor

| Variável | Default | Descrição |
|---|---|---|
| `FL_SERVER_ADDRESS` | `0.0.0.0:8080` | Endereço de escuta |
| `FL_MIN_AVAILABLE_CLIENTS` | `3` | Mínimo para iniciar round |
| `FL_NUM_ROUNDS` | `20` | Máximo de rounds |
| `FL_PROXIMAL_MU` | `0.01` | FedProx mu |
| `FL_CHECKPOINT_DIR` | `/app/checkpoints` | Checkpoints |
| `FL_LOG_DIR` | `/app/logs` | Logs |

### Cliente

| Variável | Default | Descrição |
|---|---|---|
| `FL_SERVER_ADDRESS` | `localhost:8080` | Endereço do servidor |
| `FL_CLIENT_ID` | `client_0` | ID do hospital |
| `FL_HEARTBEAT_INTERVAL` | `60` | Heartbeat (segundos) |
| `FL_LOG_DIR` | `/app/logs` | Logs |

## Volumes

| Volume | Serviço | Descrição |
|---|---|---|
| `server-checkpoints` | server | Modelos salvos a cada rodada |
| `server-logs` | server | Logs e métricas |
| `client-*-logs` | client | Logs do cliente |
| `./data/hospital_*` | client | Dados EHR locais (bind mount) |

## Docker Compose — Serviços

```yaml
services:
  mosaicfl-server:    # 1 servidor
  mosaicfl-client-a:  # Hospital A
  mosaicfl-client-b:  # Hospital B
  mosaicfl-client-c:  # Hospital C
  mosaicfl-scheduler: # Scheduler (opcional)
```

## Rede

Todos os serviços compartilham a rede `mosaicfl-network` (bridge). O servidor é acessível como `mosaicfl-server:8080` pelos clientes.

## Health Checks

- **Servidor**: Tenta conectar na porta 8080 a cada 30s
- **Cliente**: Escreve heartbeat a cada 60s

## Build e Push (script)

```bash
# Configura registry
export DOCKER_REGISTRY=jacabreu
export VERSION=0.2.0

# Build e push
./build-and-push.sh
```

## Otimizações Futuras

1. **Multi-stage build**: Reduzir tamanho da imagem (PyTorch é pesado)
2. **GPU support**: Adicionar `nvidia-docker` para treinamento em GPU
3. **Kubernetes**: Helm charts para deploy em K8s
4. **TLS**: Certificados em volumes secretos
5. **Registry privado**: Harbor ou AWS ECR
