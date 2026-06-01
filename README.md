# MOSAIC-FL

**Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas**

Extensão preditiva do ClinicalPath (Linhares et al., 2023) combinando:
- **Aprendizado Federado (FedProx)** para dados hospitalares fragmentados
- **BEHRT simplificado** para sequências clínicas temporais
- **RAG (ChromaDB + DistilGPT-2)** para justificativa diagnóstica interpretável

## Estrutura

| Arquivo | Função |
|---|---|
| `config.py` | Hiperparâmetros globais |
| `preprocess.py` | Padronização FAPESP COVID-19 (Experimento 1) |
| `model.py` | BEHRT com captura de atenção |
| `client.py` | Cliente Flower (FedProx) |
| `server.py` | Servidor de agregação |
| `rag_system.py` | Justificativa clínica via RAG |
| `extract_patterns.py` | Extrai perfis prototípicos do BEHRT |
| `run_experiments.py` | Orquestra os 5 experimentos |

## Instalação

```bash
pip install -r requirements.txt
