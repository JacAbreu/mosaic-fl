# MOSAIC-FL — Scheduler de Rounds Federados

## Visão Geral

O scheduler é um **supervisor externo** que orquestra rounds de Aprendizado Federado (FL) em ambiente de produção. Ele não substitui o Flower — complementa-o, adicionando:

- **Agendamento temporal**: rounds só ocorrem em janelas definidas (ex: madrugada)
- **Verificação de disponibilidade**: só inicia se houver clientes suficientes online
- **Persistência de estado**: sobrevive a reinicializações do servidor
- **Detecção de convergência**: para automaticamente quando o modelo estabiliza

## Arquitetura

```
┌─────────────────┐         Internet/VPN          ┌─────────────────┐
│  SERVIDOR       │◄──────────────────────────────►│  HOSPITAL A     │
│  (server_daemon)│    gRPC sobre TLS (porta 8080) │  (client_daemon)│
│  • Fica         │                               │  • Fica         │
│    escutando    │◄──────────────────────────────►│    escutando    │
│  • Agrega       │                               │  • Lê EHR local │
│    pesos        │◄──────────────────────────────►│  • Treina quando│
│  • Avalia       │                               │    solicitado   │
└─────────────────┘                               └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  SCHEDULER (scheduler_daemon.py)                            │
│  • APScheduler com intervalo configurável                   │
│  • Verifica client_registry.json                            │
│  • Decide: iniciar round ou dormir                          │
│  • Persiste estado em scheduler_state.json                  │
└─────────────────────────────────────────────────────────────┘
```

## Arquivos

| Arquivo | Função |
|---|---|
| `scheduler_daemon.py` | Daemon principal com APScheduler. Modo daemon (`--once` ou loop infinito) |
| `scheduler_cli.py` | Entrypoint alternativo para systemd/cron |
| `schedule_state.py` | Estado persistente (rounds, acurácia, convergência) em JSON |
| `client_availability_checker.py` | Verifica quais hospitais estão online |
| `round_dispatcher.py` | Monitora métricas de uma rodada e detecta convergência |

## Instalação

```bash
pip install apscheduler
```

## Uso

### 1. Modo daemon (fica rodando)

```bash
python scheduler_daemon.py
```

O scheduler acorda a cada 6 horas (configurável), verifica clientes e dispara rounds.

### 2. Modo cron (executa 1 vez)

```bash
python scheduler_daemon.py --once
```

Ideal para crontab:

```cron
# Rodar às 2h e 14h todos os dias
0 2,14 * * * cd /opt/mosaic-fl && python scheduler_daemon.py --once >> logs/scheduler_cron.log 2>&1
```

### 3. Variáveis de ambiente

```bash
export FL_SCHEDULER_INTERVAL_HOURS=6      # intervalo entre verificações
export FL_SCHEDULER_MIN_CLIENTS=3         # mínimo de hospitais online
export FL_SCHEDULER_MAX_ROUNDS=20           # máximo de rounds
export FL_SCHEDULER_CONV_THRESHOLD=0.005    # threshold de convergência
export FL_SCHEDULER_CONV_PATIENCE=3         # rodadas estáveis para convergir
export FL_SCHEDULER_TIMEZONE=America/Sao_Paulo
export FL_SCHEDULER_LOG=logs/scheduler.log
```

### 4. Argumentos CLI

```bash
python scheduler_daemon.py --once --interval 1 --min-clients 2 --max-rounds 10
```

| Flag | Descrição |
|---|---|
| `--once` | Executa um ciclo e termina |
| `--interval` | Horas entre ciclos (default: 6) |
| `--min-clients` | Mínimo de clientes para iniciar round (default: 3) |
| `--max-rounds` | Máximo de rounds (default: 20) |

## Estado Persistente

O arquivo `scheduler_state.json` guarda:

```json
{
  "last_run": "2026-06-04T10:30:00",
  "current_round": 0,
  "total_rounds_completed": 12,
  "client_history": {},
  "accuracy_history": [0.65, 0.72, 0.78, 0.81, 0.83, 0.84, 0.845, 0.847],
  "converged": true,
  "convergence_round": 12
}
```

## Como os Clientes Reportam Status

Cada `client_daemon.py` (hospital) deve escrever periodicamente em `logs/client_registry.json`:

```json
{
  "hospital_a": {"last_seen": 1717495800, "status": "ready"},
  "hospital_b": {"last_seen": 1717495820, "status": "training"}
}
```

O scheduler lê este arquivo para decidir se há clientes suficientes.

> **Nota**: Em produção real, substitua o arquivo JSON por:
> - gRPC health check
> - Consul / etcd service discovery
> - Prometheus metrics
> - REST API do servidor Flower

## Diferença: Simulação (TCC) vs. Produção

| Aspecto | TCC (`run_v2_unified.py`) | Produção (scheduler + server_daemon) |
|---|---|---|
| **Cliente** | DataLoader na mesma máquina | Processo separado no hospital |
| **Servidor** | `start_simulation()` ou loop local | `fl.server.start_server()` escutando porta 8080 |
| **Agendamento** | Loop sequencial imediato | APScheduler com intervalos configuráveis |
| **Dados** | CSV carregado na memória | EHR lido sob demanda no hospital |
| **Rede** | Loopback (127.0.0.1) | Internet/VPN com TLS |
| **Persistência** | Nenhuma | scheduler_state.json, logs |

## Integração com o Servidor Flower

O scheduler **não substitui** o servidor Flower. O fluxo de produção é:

1. **Administrador** inicia `server_daemon.py` (Flower escutando na porta 8080)
2. **Hospitais** iniciam `client_daemon.py` (conectam ao servidor via gRPC/TLS)
3. **Scheduler** inicia em horário agendado, verifica `client_registry.json`
4. Se há clientes suficientes: o Flower **já está** gerenciando rounds automaticamente
5. O scheduler apenas **observa** e salva estado; não controla o Flower diretamente

Isso significa que o scheduler é um **supervisor externo**, não um orquestrador de baixo nível.
