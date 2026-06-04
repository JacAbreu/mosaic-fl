# MOSAIC-FL

**Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas**

Extensão preditiva do ClinicalPath (Linhares et al., 2023) combinando:
- **Aprendizado Federado (FedProx)** para dados hospitalares fragmentados
- **BEHRT simplificado** para sequências clínicas temporais
- **RAG (ChromaDB + DistilGPT-2)** para justificativa diagnóstica interpretável

> **Nota sobre escopo:** Esta implementação é uma **simulação local** do FL, projetada para validação acadêmica (TCC). Em implantação real, cada `FedProxClient` rodaria em um hospital distinto com comunicação criptografada via TLS, garantindo que os prontuários **nunca saiam da instituição**.

---

## Índice

1. [Arquitetura](#arquitetura)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Instalação](#instalação)
4. [Execução](#execução)
5. [Infraestrutura de Produção](#infraestrutura-de-produção)
6. [Docker](#docker)
7. [Kubernetes (Helm)](#kubernetes-helm)
8. [Experimentos do TCC](#experimentos-do-tcc)
9. [Roadmap de Produção](#roadmap-de-produção)
10. [Solução de Problemas](#solução-de-problemas)
11. [Referências](#referências)

---

## Arquitetura

### Simulação Local (este repositório)

```
┌─────────────────────────────────────────────┐
│           MÁQUINA LOCAL (Dell i7)            │
│                                              │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │   Servidor   │◄──►│  Hospital A (cid=0)│ │
│  │  (server.py) │◄──►│  Hospital B (cid=1)│ │
│  │              │◄──►│  Hospital C (cid=2)│ │
│  │ • Agrega FL  │◄──►│  Hospital D (cid=3)│ │
│  │ • Avalia     │◄──►│  Hospital E (cid=4)│ │
│  │ • RAG        │    └────────────────────┘ │
│  └──────────────┘                            │
│                                              │
│  Dados: split_by_institution() divide o      │
│  dataset FAPESP em 5 partições locais        │
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
│       ├── v1/                         ← experimentos sintéticos TCC
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
│   ├── server/
│   │   ├── server_daemon.py            # Servidor Flower 24/7
│   │   ├── strategy.py                 # FedProx + checkpoint + métricas
│   │   ├── runner.py                   # Orquestrador do servidor
│   │   ├── __init__.py
│   │   ├── __main__.py                 # python -m mosaicfl_server
│   │   └── pyproject.toml             # pacote: mosaicfl-server
│   ├── client/
│   │   ├── client_daemon.py            # Cliente Flower 24/7 (hospital)
│   │   ├── client_daemon_v2.py         # Cliente v2 com datasource flexível
│   │   ├── heartbeat.py               # Registry de status (único ponto)
│   │   ├── runner.py                   # Orquestrador do cliente
│   │   ├── __init__.py
│   │   ├── __main__.py                 # python -m mosaicfl_client
│   │   └── pyproject.toml             # pacote: mosaicfl-client
│   └── scheduler/
│       ├── scheduler_daemon.py         # APScheduler — agenda rounds
│       ├── scheduler_cli.py            # Entrypoint (cron/systemd)
│       ├── schedule_state.py           # Estado persistente entre reinicializações
│       ├── round_training_fl_dispatcher.py  # Monitora e registra rounds
│       └── client_availability_checker.py  # Verifica quais hospitais estão online
│
├── tests/
│   └── test_mosaicfl.py
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

## Execução

### Experimentos v1 — sintéticos (TCC)

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

### Makefile

```bash
make setup   # cria venv e instala dependências
make run     # executa experimentos v1
make test    # roda testes unitários
make clean   # remove venv e caches
```

---

## Infraestrutura de Produção

### Servidor (nuvem)

```bash
# Inicia servidor Flower que fica escutando indefinidamente
python infrastructure/server/server_daemon.py \
  --address 0.0.0.0:8080 \
  --min-clients 3 \
  --rounds 20

# Variáveis de ambiente equivalentes
export FL_SERVER_ADDRESS=0.0.0.0:8080
export FL_MIN_AVAILABLE_CLIENTS=3
export FL_NUM_ROUNDS=20
python infrastructure/server/server_daemon.py
```

### Cliente (hospital)

```bash
# Inicia cliente que reconecta automaticamente ao servidor
python infrastructure/client/client_daemon.py \
  --server 52.67.123.45:8080 \
  --client-id hospital_a

# Com dados reais (PostgreSQL local do hospital)
export FL_SERVER_ADDRESS=52.67.123.45:8080
export FL_CLIENT_ID=hospital_a
export MOSAICFL_DB_URL="postgresql://ehr_user:pass@localhost:5432/prontuarios"
python infrastructure/client/client_daemon.py
```

### Scheduler de rounds (APScheduler)

O scheduler verifica periodicamente quais hospitais estão online e, quando o quórum mínimo (`MIN_FIT_CLIENTS`) é atingido, aguarda a conclusão de um round e registra as métricas.

```bash
# Modo daemon — roda indefinidamente, verifica a cada 6h
python infrastructure/scheduler/scheduler_daemon.py \
  --interval 6 \
  --min-clients 3 \
  --max-rounds 20

# Modo one-shot — ideal para cron (executa 1 ciclo e termina)
python infrastructure/scheduler/scheduler_cli.py --once

# Via crontab (executa às 2h da manhã todos os dias)
# crontab -e
0 2 * * * /path/to/.venv/bin/python /path/to/infrastructure/scheduler/scheduler_cli.py --once
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

## Experimentos do TCC

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

## Roadmap de Produção

| Área | Tarefa | Prioridade |
|---|---|---|
| Dados | Integração HL7 FHIR com EPR dos hospitais | Alta |
| FL | TLS mútuo entre servidor e clientes | Alta |
| FL | Differential Privacy nos pesos | Alta |
| Modelo | Fine-tuning em dados clínicos brasileiros | Alta |
| RAG | Substituir DistilGPT-2 por LLM em português | Média |
| Validação | Estudo retrospectivo + AUC-ROC, sensibilidade | Alta |
| LGPD | Consentimento informado, DPO, auditoria | Alta |
| Regulatório | Submissão ANVISA como SaMD Classe II/III | Alta |

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

- Linhares et al. (2023). ClinicalPath — base do sistema estendido
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
- Li et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx)
- Rasmy et al. (2021). Med-BERT / BEHRT para prontuários eletrônicos
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

---

## Licença

MIT — veja `pyproject.toml` para detalhes.

---

> **Autora:** Jacqueline Abreu do N. T. R. Lopes  
> **Instituição:** ICMC/USP — São Carlos  
> **Contato:** abreujacline@gmail.com
