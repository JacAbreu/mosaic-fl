# MOSAIC-FL

**Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas**

Extensão preditiva do ClinicalPath (Linhares et al., 2023) combinando:
- **Aprendizado Federado (FedProx)** para dados hospitalares fragmentados
- **BEHRT simplificado** para sequências clínicas temporais
- **RAG (ChromaDB + DistilGPT-2)** para justificativa diagnóstica interpretável

## Estrutura

```
mosaic-fl/
├── run.py                          # Ponto de entrada principal
├── pyproject.toml                  # Metadados e dependências do pacote
├── requirements.txt                # Dependências (referência)
├── setup.sh                        # Script de instalação Linux/macOS
├── setup.bat                       # Script de instalação Windows
├── makefile                        # Atalhos de desenvolvimento
└── src/
    ├── config.py                   # Hiperparâmetros globais
    ├── preprocess.py               # Padronização FAPESP COVID-19 (Experimento 1)
    ├── model.py                    # BEHRT com captura de atenção por camada
    ├── client.py                   # Cliente Flower com FedProx
    ├── server.py                   # Servidor de agregação e avaliação global
    ├── rag_system.py               # Justificativa clínica via RAG (ChromaDB)
    ├── extract_patterns.py         # Extrai perfis prototípicos do BEHRT
    └── experiments/
        ├── runner.py               # Orquestrador dos 5 experimentos
        └── run_experiments.py      # Legado — redireciona para runner.py
```

## Instalação

```bash
pip install -r requirements.txt

## Instalação

### Linux / macOS

```bash
# 1. Clone o repositório
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl

# 2. Execute o script de setup
#    Ele cria o ambiente virtual .venv e instala todas as dependências
bash chmod+x setup.sh
bash ./setup.sh
```

### Windows

```bat
:: 1. Clone o repositório
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl

:: 2. Execute o script de setup
setup.bat
```

O setup cria um ambiente virtual `.venv` na raiz do projeto e instala o pacote
`mosaicfl` em modo editável (`pip install -e .`), tornando todos os módulos
acessíveis sem configuração adicional de `PYTHONPATH`.

---

## Execução

### Linux / macOS

```bash
# Ative o ambiente virtual (necessário uma vez por sessão de terminal)
source .venv/bin/activate

# Execute os 5 experimentos
python run.py
```

### Windows

```bat
:: Ative o ambiente virtual
.venv\Scripts\activate

:: Execute os 5 experimentos
python run.py
```

### Via Makefile (Linux / macOS)

```bash
make setup   # cria o ambiente e instala as dependências
make run     # executa os experimentos
make clean   # remove o ambiente virtual e caches
```

---


## Solução de Problemas

**`externally-managed-environment` ao rodar `pip install`**

Não use `pip install -r requirements.txt` diretamente. Use o `setup.sh` (Linux/macOS)
ou `setup.bat` (Windows), que criam um ambiente virtual isolado automaticamente.

**`ModuleNotFoundError: No module named 'mosaicfl'`**

O pacote não foi instalado ou o ambiente virtual não está ativo. Verifique:

```bash
# O ambiente está ativo? O prompt deve mostrar (.venv)
source .venv/bin/activate

# O pacote está instalado?
pip show mosaicfl

# Se não estiver, instale:
pip install -e .
```

**`python` não encontrado (Linux)**

Algumas distribuições usam `python3` em vez de `python`. Instale o alias:

```bash
sudo apt install python-is-python3
```
