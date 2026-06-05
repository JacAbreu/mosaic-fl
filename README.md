# MOSAIC-FL

**Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas**

Extensão preditiva do ClinicalPath (Linhares et al., 2023) combinando:
- **Aprendizado Federado (FedProx)** para dados hospitalares fragmentados
- **BEHRT simplificado** para sequências clínicas temporais
- **RAG (ChromaDB + DistilGPT-2)** para justificativa diagnóstica interpretável

> **Nota sobre escopo:** Esta implementação é uma **simulação local** do FL, projetada para validação acadêmica. Em implantação real, cada `FedProxClient` rodaria em um hospital distinto com comunicação criptografada via TLS, garantindo que os prontuários **nunca saiam da instituição**.

---

## Índice

1. [Arquitetura](#arquitetura)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Instalação](#instalação)
4. [Execução Experimentos para Desenvolvimento do Mosaic-FL](#execução-experimentos-para-desenvolvimento-do-mosaic-fl)
5. [Testes](#testes)
6. [Rodando Localmente](#rodando-localmente-scheduler--servidor--cliente)
7. [Infraestrutura de Produção](#infraestrutura-de-produção)
8. [Docker](#docker)
9. [Kubernetes (Helm)](#kubernetes-helm)
10. [Experimentos](#experimentos)
11. [Solução de Problemas](#solução-de-problemas)
12. [Referências](#referências)

---

## Arquitetura

### Simulação Local (este repositório)

```
┌─────────────────────────────────────────────┐
│      MÁQUINA LOCAL (intel i7/16 GB RAM)     │
│                                             │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │   Servidor   │◄──►│  Hospital A (cid=0)│ │
│  │  (server.py) │◄──►│  Hospital B (cid=1)│ │
│  │              │◄──►│  Hospital C (cid=2)│ │
│  │ • Agrega FL  │◄──►│  Hospital D (cid=3)│ │
│  │ • Avalia     │◄──►│  Hospital E (cid=4)│ │
│  │ • RAG        │    └────────────────────┘ │
│  └──────────────┘                           │
│                                             │
│  Dados: split_by_institution() divide o     │
│  dataset FAPESP em 5 partições locais       │
└─────────────────────────────────────────────┘
```

### Arquitetura de Produção

```
                    Internet / VPN
┌───────────────┐  gRPC sobre TLS (8080)  ┌───────────────┐
│    SERVIDOR   │◄───────────────────────►│   HOSPITAL A  │
│ (server_      │                         │ (client_      │
│  daemon.py)   │  • Fica escutando       │  daemon.py)   │
│               │  • Agenda rounds        │               │
│ • Agrega pesos│◄───────────────────────►│ • Lê EHR local│
│ • Checkpoints │                         │ • Treina local│
│ • Exporta     │◄───────────────────────►│ • Devolve     │
│   métricas    │                         │   só pesos    │
└───────┬───────┘                         └───────────────┘
        │
        ▼
┌───────────────┐
│   SCHEDULER   │ ← Opcional: só inicia rodadas
│ (APScheduler) │   em janelas de manutenção
│ • 2h-4h manhã │   (ex: madrugada)
└───────────────┘

⚠️  PRONTUÁRIOS NUNCA SAEM DOS HOSPITAIS — apenas os pesos do modelo trafegam.
```

### Como funciona o Federated Learning

1. **Servidor inicia** — envia o modelo global (pesos iniciais) para cada hospital
2. **Hospital treina localmente** com seus próprios prontuários (dados nunca saem)
3. **Hospital devolve apenas os pesos** — nunca os dados brutos
4. **Servidor agrega** via FedProx (média ponderada) e envia novo modelo global
5. **Repete por N rodadas** até convergência (Δacurácia < threshold por patience rodadas)

---

## Estrutura do Projeto

```
mosaic-fl/
│
├── src/                                ← pacote core instalável (mosaicfl)
│   └── mosaicfl/
│       ├── __init__.py
│       ├── v1/                         ← experimentos sintéticos desenvolvimento mosaic-fl
│       │   ├── config.py
│       │   ├── model.py                # BEHRT v1 (mean pooling)
│       │   ├── client.py               # FedProxClient v1
│       │   ├── server.py               # Servidor v1
│       │   ├── preprocess.py           # Preprocessador v1
│       │   ├── rag_system.py           # RAG v1 (ChromaDB)
│       │   ├── extract_patterns.py     # Perfis prototípicos BEHRT
│       │   └── experiments/
│       │       ├── runner.py           # Orquestrador dos 5 experimentos
│       │       └── run_experiments.py  # Legado → redireciona para runner.py
│       └── v2/                         ← integração com dados reais
│           ├── config.py               # Hiperparâmetros (hardware-aware)
│           ├── model_v2.py             # BEHRT v2 (CLS token pooling)
│           ├── client_v2.py            # FedProxClient v2
│           ├── server_v2.py            # Servidor v2 + ConvergenceTracker
│           ├── preprocess_v2.py        # Preprocessador v2 (unidades médicas)
│           ├── rag_system_v2.py        # RAG v2 (type-safe, truncagem GPT-2)
│           └── data_loader.py          # Strategy: SGBD → CSV → sintético
│
├── infrastructure/                     ← daemons de produção
│   ├── mosaicfl_server/
│   │   ├── server_daemon.py            # Servidor Flower 24/7
│   │   ├── strategy.py                 # FedProx + checkpoint + métricas + configure_fit
│   │   ├── config_loader.py            # Config de runtime: ChromaDB | arquivo (FL_CONFIG_BACKEND)
│   │   ├── runner.py                   # Orquestrador do servidor
│   │   ├── __init__.py
│   │   └── __main__.py                 # python -m mosaicfl_server
│   ├── mosaicfl_client/
│   │   ├── client_daemon.py            # Cliente Flower 24/7 (hospital)
│   │   ├── client_daemon_v2.py         # Cliente v2 com datasource flexível
│   │   ├── heartbeat.py               # Registry de status (único ponto)
│   │   ├── runner.py                   # Orquestrador do cliente
│   │   ├── __init__.py
│   │   └── __main__.py                 # python -m mosaicfl_client
│   └── mosaicfl_scheduler/
│       ├── scheduler_daemon.py         # APScheduler — agenda rounds
│       ├── scheduler_cli.py            # Entrypoint (cron/systemd)
│       ├── schedule_state.py           # Estado persistente entre reinicializações
│       ├── round_training_fl_dispatcher.py  # Monitora e registra rounds
│       └── client_availability_checker.py  # Verifica quais hospitais estão online
│
├── tests/
│   ├── test_mosaicfl.py            # Testes unitários core (modelo, RAG, preprocessamento)
│   ├── test_v2_core.py             # Testes de integração v2 (pipeline EHR → modelo)
│   ├── test_infrastructure.py      # Testes da infra de produção (scheduler, servidor, cliente)
│   ├── test_config_loader.py       # Testes do config_loader (ChromaDB, arquivo, strategy)
│   └── test_fl_cycle_explained.py  # Documentação executável do ciclo FL (ver seção Testes)
│
├── ci_cd/
│   ├── ci-cd-github-actions.yml        # GitHub Actions CI/CD
│   └── helm/                           # Kubernetes Helm Chart
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── server-deployment.yaml
│       ├── client-deployment.yaml
│       ├── scheduler-cronjob.yaml
│       └── pvcs.yaml
│
├── run_experiments.py                  # Experimentos v1 (sintético)
├── run_experiments_v2.py               # Experimentos v2 (dados reais)
├── benchmark.py                        # Benchmark de performance
├── datasource.py                       # Strategy Pattern standalone
│
├── Dockerfile.server                   # Imagem Docker do servidor
├── Dockerfile.client                   # Imagem Docker do cliente
├── docker-compose.yml                  # Stack local completo
│
├── pyproject.toml                      # Pacote core mosaicfl
├── setup.sh                            # Setup Linux/macOS
├── setup.bat                           # Setup Windows
├── install.sh                          # Instala todos os pacotes
├── makefile                            # Atalhos de desenvolvimento
└── README.md                           # Este arquivo
```

---

## Instalação

### Desenvolvimento (tudo local)

```bash
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl
bash setup.sh
source .venv/bin/activate
```

O `setup.sh` cria `.venv` e instala `mosaicfl` em modo editável (`pip install -e .`). Qualquer edição em `src/` tem efeito imediato sem reinstalar.

### Windows

```bat
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl
setup.bat
.venv\Scripts\activate
```

---

## Execução Experimentos para desenvolvimento do Mosaic-FL

### Experimentos v1 — sintéticos (Utilizado para desenvolver o mosaic-fl)

```bash
source .venv/bin/activate
python run_experiments.py
```

### Experimentos v2 — dados reais / fallback sintético

```bash
source .venv/bin/activate
python run_experiments_v2.py
```

O v2 tenta carregar dados nesta ordem: **SGBD → CSV explícito → CSV padrão → sintético**.
Se nenhuma fonte real estiver disponível, usa dados sintéticos com aviso explícito.

Para conectar ao PostgreSQL:
```bash
export MOSAICFL_DB_URL="postgresql://user:pass@localhost:5432/mosaicfl"
python run_experiments_v2.py
```

Para forçar um CSV específico:
```bash
python -c "
from mosaicfl.v2.data_loader import load_with_fallback
df = load_with_fallback(csv_path='data/minha_base.csv', allow_synthetic=False)
print(df.shape)
"
```

### Benchmark de performance

O `benchmark.py` mede o custo computacional de uma simulação FL completa com dados sintéticos — útil para estimar viabilidade em hardware diferente antes de rodar os experimentos reais.

**O que é medido por rodada:**
- Tempo de treino + agregação
- Uso de RAM (antes, depois e pico)
- Uso de CPU (%)
- Tráfego de rede estimado (tamanho real do state_dict × número de clientes)
- Throughput (amostras/segundo)
- Acurácia global (quando `evaluate_fn` está ativa)

```bash
source .venv/bin/activate

# Configuração padrão: 1000 amostras, 10 rodadas, 5 clientes
python benchmark.py

# Configuração customizada
python benchmark.py --samples 2000 --rounds 5 --clients 3 --output meus_resultados
```

**Artefatos gerados em `benchmark_results/`:**
- `benchmark_<timestamp>.json` — métricas por rodada e resumo estatístico
- `benchmark_<timestamp>.png` — 6 gráficos: tempo, RAM, CPU, tráfego, acurácia, throughput

### Makefile

```bash
make setup   # cria venv e instala dependências
make run     # executa experimentos v1
make test    # roda testes unitários
make clean   # remove venv e caches
```

---

## Testes

```bash
make test          # roda todos os testes
make test-cov      # com relatório de cobertura
```

A suite tem **291 testes** distribuídos em 5 arquivos:

| Arquivo | Foco | Testes |
|---|---|---|
| `test_mosaicfl.py` | Unidades core: modelo, RAG, preprocessamento, data loader | ~147 |
| `test_v2_core.py` | Integração v2: pipeline EHR → FedProxClient → modelo | ~36 |
| `test_infrastructure.py` | Daemons de produção: scheduler, servidor, cliente (com mocks) | 61 |
| `test_config_loader.py` | Config de runtime: `_cast`, ChromaDB, arquivo, `configure_fit` | 55 |
| `test_fl_cycle_explained.py` | Documentação executável do ciclo FL completo | 29 |

### `test_fl_cycle_explained.py` — Documentação executável

Este arquivo é o ponto de entrada para entender **como o MOSAIC-FL funciona na prática**. Cada classe de teste cobre uma fase do ciclo federado e imprime logs detalhados descrevendo quem envia o quê e como os dados fluem.

```bash
# Ver todos os prints do ciclo (recomendado para entender o projeto)
pytest tests/test_fl_cycle_explained.py -v -s

# Ver só uma fase específica
pytest tests/test_fl_cycle_explained.py -v -s -k "TestServerAggregates"
```

**Fases cobertas:**

| Classe | Fase do ciclo | O que demonstra |
|---|---|---|
| `TestSchedulerDispatchesFLRound` | 1 — Scheduler | Quórum mínimo, convergência, max_rounds — quando o scheduler dispara ou para |
| `TestServerSendsModelToClient` | 2 — Servidor → Cliente | `set_parameters()` carrega pesos no modelo local; armazena cópia para termo proximal |
| `TestClientLocalTraining` | 3 — Treino local | `fit()` retorna `(params, n_samples, {"loss": float})`; FedProx adiciona regularização |
| `TestClientReturnsWeightsToServer` | 4 — Cliente → Servidor | `get_parameters()` exporta state_dict completo (32 params + 2 buffers = 34 tensores) |
| `TestServerAggregatesWeights` | 5 — Agregação | `weighted_average()` agrega **métricas** (accuracy); `_fedavg_params()` agrega **pesos** |
| `TestServerConvergenceTracking` | 6 — Convergência | `ConvergenceTracker` usa `stable_count` incremental: converge quando Δ < threshold por `patience` rounds consecutivos |
| `TestFullFLCycle` | 7 — End-to-end | 1 cliente, 3 clientes, 5 rounds com rastreamento de convergência |

**APIs documentadas pelos testes:**

```python
# Cliente
FedProxClient(client_id: int, train_loader, val_loader)
client.fit(global_params, {})       # → (List[np.ndarray], n_samples, {"loss": float})
client.evaluate(params, {})         # → (loss, n_samples, {"accuracy": float, "client_id": int})
client.get_parameters({})           # → List[np.ndarray]  (34 tensores: state_dict completo)
client.set_parameters(params)       # carrega List[np.ndarray] no state_dict

# Servidor
weighted_average([(n, {"accuracy": v}), ...])   # agrega MÉTRICAS, não pesos
ConvergenceTracker(threshold, patience).check(accuracy)  # → bool
```

---

## Rodando Localmente (scheduler + servidor + cliente)

Para observar o ciclo completo de comunicação em uma única máquina, abra **três terminais**:

### Terminal 1 — Servidor Flower

```bash
source .venv/bin/activate
python infrastructure/mosaicfl_server/server_daemon.py \
  --address 0.0.0.0:8080 \
  --min-clients 1 \
  --rounds 3
```

O servidor fica aguardando conexões de clientes. Quando o quórum (`--min-clients`) é atingido, inicia o round automaticamente.

### Terminal 2 — Cliente (Hospital A)

```bash
source .venv/bin/activate
python infrastructure/mosaicfl_client/client_daemon.py \
  --server localhost:8080 \
  --client-id hospital_a
```

O cliente conecta ao servidor, recebe o modelo global, treina localmente com seus dados e devolve apenas os pesos. Os dados nunca saem da máquina.

### Terminal 3 — Scheduler (opcional)

```bash
source .venv/bin/activate
python infrastructure/mosaicfl_scheduler/scheduler_daemon.py \
  --interval 1 \
  --min-clients 1 \
  --max-rounds 3
```

O scheduler monitora a disponibilidade de clientes e o estado de convergência. Com `--interval 1` ele verifica a cada 1 hora; use `--once` para executar um único ciclo de verificação imediatamente.

### Verificando o estado

```bash
# Status do servidor
cat logs/round_1_metrics.json

# Heartbeat dos clientes
cat logs/client_registry.json

# Estado do scheduler (rounds e convergência)
cat scheduler_state.json
```

### Variáveis de ambiente úteis

```bash
export FL_SERVER_ADDRESS=0.0.0.0:8080     # endereço do servidor
export FL_CLIENT_ID=hospital_a            # identificador do cliente
export FL_SCHEDULER_MIN_CLIENTS=1         # quórum mínimo
export FL_SCHEDULER_MAX_ROUNDS=3          # limite de rounds
export FL_SCHEDULER_CONV_THRESHOLD=0.005  # Δacurácia para convergência
export FL_SCHEDULER_CONV_PATIENCE=3       # rounds estáveis para convergir
export FL_CONFIG_BACKEND=file             # backend de config de runtime (file | chroma)
```

### Alterando configuração em tempo de execução

O servidor lê `FL_CONFIG_BACKEND` para decidir como buscar config antes de cada round — sem necessidade de reiniciar.

**Backend `file` (desenvolvimento):**
```bash
# Cria ou atualiza logs/runtime_config.json — aplicado no próximo round
cat > logs/runtime_config.json <<EOF
{"proximal_mu": 0.005, "pause_seconds": 0, "stop": false}
EOF
```

**Backend `chroma` (padrão em produção):**
```python
from infrastructure.mosaicfl_server.config_loader import ChromaDBConfigLoader
loader = ChromaDBConfigLoader()

# Atualiza proximal_mu no próximo round
loader.write({"proximal_mu": 0.005, "stop": False})

# Para o servidor graciosamente após o round atual
loader.write({"stop": True})

# Remove config (volta aos defaults do servidor)
loader.clear()
```

Chaves suportadas: `proximal_mu` (float), `pause_seconds` (float), `stop` (bool).

---

## Infraestrutura de Produção

### Servidor (nuvem)

```bash
# Inicia servidor Flower que fica escutando indefinidamente
python infrastructure/mosaicfl_server/server_daemon.py \
  --address 0.0.0.0:8080 \
  --min-clients 3 \
  --rounds 20

# Variáveis de ambiente equivalentes
export FL_SERVER_ADDRESS=0.0.0.0:8080
export FL_MIN_AVAILABLE_CLIENTS=3
export FL_NUM_ROUNDS=20
python infrastructure/mosaicfl_server/server_daemon.py
```

### Cliente (hospital)

```bash
# Inicia cliente que reconecta automaticamente ao servidor
python infrastructure/mosaicfl_client/client_daemon.py \
  --server 52.67.123.45:8080 \
  --client-id hospital_a

# Com dados reais (PostgreSQL local do hospital)
export FL_SERVER_ADDRESS=52.67.123.45:8080
export FL_CLIENT_ID=hospital_a
export MOSAICFL_DB_URL="postgresql://ehr_user:pass@localhost:5432/prontuarios"
python infrastructure/mosaicfl_client/client_daemon.py
```

### Scheduler de rounds (APScheduler)

O scheduler verifica periodicamente quais hospitais estão online e, quando o quórum mínimo (`MIN_FIT_CLIENTS`) é atingido, aguarda a conclusão de um round e registra as métricas.

```bash
# Modo daemon — roda indefinidamente, verifica a cada 6h
python infrastructure/mosaicfl_scheduler/scheduler_daemon.py \
  --interval 6 \
  --min-clients 3 \
  --max-rounds 20

# Modo one-shot — ideal para cron (executa 1 ciclo e termina)
python infrastructure/mosaicfl_scheduler/scheduler_cli.py --once

# Via crontab (executa às 2h da manhã todos os dias)
# crontab -e
0 2 * * * /path/to/.venv/bin/python /path/to/infrastructure/mosaicfl_scheduler/scheduler_cli.py --once
```

Estado do scheduler persiste em `scheduler_state.json` — reinicializações não perdem histórico de convergência.

### Fluxo de comunicação scheduler ↔ servidor ↔ cliente

```
scheduler               server_daemon            client_daemon
    │                        │                        │
    │── verifica registry ──►│                        │
    │                        │◄── heartbeat (60s) ────│
    │◄── clientes ativos ────│                        │
    │                        │                        │
    │── dispatch_round() ───►│                        │
    │                        │── solicita treino ────►│
    │                        │                        │── treina local
    │                        │◄── devolve pesos ──────│
    │                        │── agrega FedProx        │
    │                        │── salva checkpoint      │
    │                        │── escreve métricas JSON │
    │◄── poll métricas ──────│                        │
    │── atualiza state ───────────────────────────────►│
```

### ⚠️ Limitações do Scheduler (Arquitetura Atual)

> **Importante:** O scheduler atual **NÃO dispara rounds ativamente** no servidor Flower. Ele atua como um **supervisor/monitor** que:
> 
> 1. Verifica quais clientes estão online (via heartbeat registry)
> 2. Aguarda o servidor Flower completar rounds naturalmente (quando clientes conectam)
> 3. Faz polling das métricas em `logs/round_{N}_metrics.json`
> 4. Detecta convergência e persiste estado

**Pré-requisitos para o funcionamento correto:**
```bash
# 1. Servidor Flower DEVE estar rodando
python infrastructure/mosaicfl_server/server_daemon.py --port 8080

# 2. Clientes DEVEM estar conectados ao servidor
python infrastructure/mosaicfl_client/client_daemon.py --server localhost:8080 --client-id hospital_a

# 3. SÓ ENTÃO o scheduler pode monitorar
python infrastructure/mosaicfl_scheduler/scheduler_daemon.py --interval 6
```

**Para produção:** A arquitetura atual é suficiente para a simulação local. Para o funcionamento em ambientes reais, ou seja hospitais, veja [`TODO.md`](TODO.md).

---

## Docker

### Stack completo (desenvolvimento)

```bash
docker-compose up --build
```

### Servidor na nuvem

```bash
docker build -f Dockerfile.server -t mosaicfl-server:latest .
docker run -d \
  -p 8080:8080 \
  -e FL_MIN_AVAILABLE_CLIENTS=3 \
  -e FL_NUM_ROUNDS=20 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  --name mosaicfl-server \
  mosaicfl-server:latest
```

### Cliente no hospital

```bash
docker build -f Dockerfile.client -t mosaicfl-client:latest .
docker run -d \
  -e FL_SERVER_ADDRESS=52.67.123.45:8080 \
  -e FL_CLIENT_ID=hospital_a \
  -e MOSAICFL_DB_URL="postgresql://user:pass@db:5432/ehr" \
  -v /hospital/logs:/app/logs \
  --name mosaicfl-client \
  mosaicfl-client:latest
```

---

## Kubernetes (Helm)

```bash
# Instalação padrão
helm install mosaicfl ./ci_cd/helm

# Com valores de produção
helm install mosaicfl ./ci_cd/helm -f values-production.yaml

# Verificar pods
kubectl get pods -l app.kubernetes.io/name=mosaicfl

# Logs do servidor
kubectl logs -f deployment/mosaicfl-server
```

O CronJob do scheduler (`scheduler-cronjob.yaml`) executa por padrão às **2h da manhã** (`0 2 * * *`), configurável em `values.yaml`.

---

## Experimentos

| # | Experimento | Componente | Status |
|---|---|---|---|
| 1 | Padronização e pré-processamento | `EHRPreprocessor` | ✅ Real |
| 2 | Efeito equalizador do FL | FedProx + AUC por cliente | ⚠️ Seed fixo |
| 3 | Impacto heterogeneidade não-IID | Curvas por subgrupo demográfico | ⚠️ Curva aproximada |
| 4 | RAG e detecção de alucinação | ChromaDB + DistilGPT-2 | ✅ Real |
| 5 | Eficiência operacional | Convergência vs. comunicação | ✅ Real |

Resultados salvos em `experiment_results.json` após cada execução.

Para documentação detalhada de cada experimento (hipótese, limitações, caminho para dados reais), veja `EXPERIMENTOS.md`.

---

## Solução de Problemas

**`externally-managed-environment` ao rodar `pip install`**
Use `bash setup.sh` em vez de `pip install` direto — cria um venv isolado automaticamente.

**`ModuleNotFoundError: No module named 'mosaicfl'`**
```bash
source .venv/bin/activate
pip install -e . --force-reinstall
```

**`ImportError: No module named 'round_dispatcher'`**
O nome correto é `round_training_fl_dispatcher`. Verifique se está usando a versão mais recente do projeto.

**Cliente não conecta ao servidor**
```bash
nc -zv localhost 8080        # verifica se porta está aberta
echo $FL_SERVER_ADDRESS      # verifica variável de ambiente
cat logs/server_health.json  # verifica status do servidor
```

**Scheduler não detecta clientes**
```bash
cat logs/client_registry.json   # verifica heartbeats dos clientes
# Clientes precisam ter reportado heartbeat nos últimos 10 minutos
```

---

## Referências


### Frameworks e Bibliotecas

- **Flower** — Beutel et al., 2020. *Flower: A Friendly Federated Learning Research Framework*. arXiv:2007.14390.  
  [https://arxiv.org/abs/2007.14390](https://arxiv.org/abs/2007.14390)

### Algoritmos

- **FedAvg** — McMahan et al., 2017. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- **FedProx** — Li et al., 2020. *Federated Optimization in Heterogeneous Networks*. MLSys.

### Modelos

- **Med-BERT/BEHRT** — Rasmy et al., 2021. *Med-BERT: Pretrained Contextualized Embeddings on Large-scale Structured Electronic Health Records for Disease Prediction*. npj Digital Medicine.

### RAG

- **RAG** — Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.

### Base do Projeto

- **ClinicalPath** — Linhares et al., 2023. *ClinicalPath: Um Sistema de Apoio à Decisão Clínica Baseado em Evidências*.

---

## Licença

Apache 2.0 — veja `pyproject.toml` para detalhes.

---

> **Autora:** Jacqueline Abreu do N. T. R. Lopes  
> **Instituição:** ICMC/USP — São Carlos  
> **Contato:** abreujacline@gmail.com
