# MOSAIC-FL — Helm Chart para Kubernetes

## Visão Geral

Helm Chart para deploy do MOSAIC-FL em cluster Kubernetes (EKS, GKE, AKS, on-premise).

| Componente | Kind Kubernetes | Descrição |
|---|---|---|
| Servidor | Deployment + Service | Flower server 24/7, LoadBalancer exposto |
| Clientes | Deployment (1 por hospital) | Cada hospital é um pod separado |
| Scheduler | CronJob | Executa a cada X horas (padrão: 2h da manhã) |
| PVCs | PersistentVolumeClaim | Dados persistentes (checkpoints, logs, EHR) |

## Instalação

### 1. Adicionar repo (quando publicado)
```bash
helm repo add mosaicfl https://jacabreu.github.io/mosaic-fl
helm repo update
```

### 2. Instalar com valores padrão
```bash
helm install mosaicfl ./helm/mosaicfl
```

### 3. Instalar com valores customizados
```bash
helm install mosaicfl ./helm/mosaicfl -f values-production.yaml
```

### 4. Desinstalar
```bash
helm uninstall mosaicfl
```

## Valores Customizados

### `values-production.yaml` (exemplo)
```yaml
server:
  service:
    type: LoadBalancer
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
  persistence:
    checkpoints:
      size: 50Gi
      storageClass: gp3

client:
  enabled: true
  hospitals:
    - name: hospital-sirio-libanes
      clientId: sirio_libanes
      serverAddress: mosaicfl-server:8080
      resources:
        limits:
          cpu: 2000m
          memory: 4Gi
      persistence:
        data:
          size: 100Gi
          storageClass: gp3

scheduler:
  enabled: true
  schedule: "0 2 * * *"  # 2h da manhã todos os dias
```

## Comandos Úteis

```bash
# Ver status
helm status mosaicfl

# Ver pods
kubectl get pods -l app.kubernetes.io/name=mosaicfl

# Logs do servidor
kubectl logs -f deployment/mosaicfl-server

# Logs de um cliente
kubectl logs -f deployment/mosaicfl-client-hospital-a

# Port-forward para teste local
kubectl port-forward svc/mosaicfl-server 8080:8080

# Escalar servidor (não recomendado, FL server é stateful)
kubectl scale deployment mosaicfl-server --replicas=1
```

## Arquitetura no K8s

```
┌─────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Namespace: mosaicfl                                │   │
│  │                                                     │   │
│  │  ┌──────────────┐      ┌──────────────────────┐   │   │
│  │  │ Service      │      │ Deployment           │   │   │
│  │  │ LoadBalancer │◄────►│ mosaicfl-server      │   │   │
│  │  │ Port: 8080   │      │ Replicas: 1          │   │   │
│  │  └──────────────┘      │ PVC: checkpoints, logs│   │   │
│  │                        └──────────────────────┘   │   │
│  │                                │                    │   │
│  │         ┌─────────────────────┼────────────────────┐  │
│  │         │                     │                    │  │
│  │  ┌──────▼──────┐    ┌────────▼────────┐   ┌──────▼──┐│
│  │  │ Deployment  │    │ Deployment        │   │ CronJob ││
│  │  │ client-a    │    │ client-b          │   │ scheduler││
│  │  │ PVC: data   │    │ PVC: data         │   │         ││
│  │  └─────────────┘    └───────────────────┘   └─────────┘│
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## TLS/SSL

Para produção, configure TLS no ingress ou no gRPC do Flower:

```yaml
server:
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt
      nginx.ingress.kubernetes.io/ssl-redirect: "true"
    hosts:
      - host: mosaicfl.example.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: mosaicfl-tls
        hosts:
          - mosaicfl.example.com
```

## Próximos Passos

1. **Cert-manager**: TLS automático com Let's Encrypt
2. **Istio/Linkerd**: Service mesh para mTLS entre clientes e servidor
3. **Prometheus/Grafana**: Métricas do Flower e do cluster
4. **Velero**: Backup dos PVCs de checkpoints
