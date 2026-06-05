# Contribuindo com o MOSAIC-FL

Obrigada pelo interesse em contribuir. Este guia cobre o processo completo: ambiente, padrão de código, testes e abertura de PR.

---

## Índice

1. [Configurando o ambiente](#1-configurando-o-ambiente)
2. [Estrutura do projeto](#2-estrutura-do-projeto)
3. [Rodando os testes](#3-rodando-os-testes)
4. [Padrão de código](#4-padrão-de-código)
5. [Padrão de commits](#5-padrão-de-commits)
6. [Abrindo um Pull Request](#6-abrindo-um-pull-request)
7. [O que contribuir](#7-o-que-contribuir)

---

## 1. Configurando o ambiente

### Pré-requisitos

- Python 3.10 ou superior
- Git

### Setup em um comando

```bash
git clone https://github.com/JacAbreu/mosaic-fl.git
cd mosaic-fl
bash setup.sh
source .venv/bin/activate
```

O `setup.sh` cria o `.venv`, instala o pacote em modo editável (`pip install -e .`) e instala os hooks de pre-commit. Qualquer edição em `src/` tem efeito imediato sem reinstalar.

### Setup manual (alternativa)

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

### Verificando a instalação

```bash
make test
# Esperado: 299 passed
```

---

## 2. Estrutura do projeto

```
src/mosaicfl/       — pacote core (modelo, cliente, servidor, RAG)
  v1/               — experimentos sintéticos (legado)
  v2/               — integração com dados reais (versão ativa)

infrastructure/     — daemons de produção
  mosaicfl_server/  — servidor Flower + strategy + config_loader
  mosaicfl_client/  — cliente Flower + heartbeat
  mosaicfl_scheduler/ — APScheduler de rounds

tests/              — suite de testes (299 testes)
benchmark.py        — benchmark de performance
```

**Regra geral:** modificações no comportamento do FL ficam em `src/mosaicfl/v2/`. Modificações nos daemons ficam em `infrastructure/`. Novos comportamentos exigem novos testes.

---

## 3. Rodando os testes

```bash
# Todos os testes
make test

# Com cobertura de código
make test-cov

# Arquivo específico
.venv/bin/python -m pytest tests/test_v2_core.py -v

# Classe ou teste específico
.venv/bin/python -m pytest tests/test_v2_core.py -v -k "TestFedProxClient"
```

### Documentação executável do ciclo FL

O arquivo `tests/test_fl_cycle_explained.py` é o melhor ponto de partida para entender o projeto. Rode com `-s` para ver os logs detalhados de cada fase:

```bash
.venv/bin/python -m pytest tests/test_fl_cycle_explained.py -v -s
```

### Escrevendo novos testes

- Cada arquivo novo em `src/` ou `infrastructure/` deve ter testes correspondentes.
- Use `tmp_path` (fixture do pytest) para arquivos temporários — nunca escreva em caminhos absolutos.
- Use `unittest.mock.patch` e `MagicMock` para dependências externas (ChromaDB, Flower, rede).
- Nomeie os testes descrevendo o comportamento: `test_load_returns_empty_dict_when_file_missing`, não `test_load_1`.

---

## 4. Padrão de código

### Linting e formatação

```bash
make lint    # verifica com ruff (não modifica)
make fmt     # formata com ruff (modifica)
```

O ruff está configurado com `line-length = 100` e regras `E, F, W, I` (erros, pyflakes, warnings, imports).

### Pre-commit (automático)

Os hooks rodam automaticamente em cada `git commit`:

```
ruff lint (auto-fix)     — imports não usados, variáveis não referenciadas
ruff format              — formatação consistente
trailing-whitespace      — espaços no final de linha
end-of-file-fixer        — newline no final de arquivo
check-yaml / check-toml  — sintaxe de arquivos de config
debug-statements         — detecta breakpoint() e pdb.set_trace()
check-added-large-files  — bloqueia arquivos > 500 KB
```

Se um hook falhar, o commit é bloqueado. Corrija o problema e tente novamente.

### Imports

Use imports explícitos — sem `from módulo import *`:

```python
# correto
from mosaicfl.v2.config import VOCAB_SIZE, EMBED_DIM, NUM_LAYERS

# proibido
from mosaicfl.v2.config import *
```

### Type hints

Anote todos os métodos públicos:

```python
def weighted_average_loss(metrics: List[Tuple[int, Dict]]) -> Dict:
    ...
```

### Comentários

Comente apenas o **porquê**, não o **o quê**. Se o nome do método já descreve a operação, o comentário é desnecessário.

```python
# correto — explica uma invariante não óbvia
# ChromaDB metadata só aceita str/int/float/bool — filtrar antes do upsert

# desnecessário — o código já diz isso
# itera sobre os resultados
for result in results:
```

---

## 5. Padrão de commits

Este projeto usa [Conventional Commits](https://www.conventionalcommits.org/pt-br/).

### Formato

```
<tipo>(<escopo opcional>): <descrição curta em imperativo>

<corpo opcional — explica o porquê, não o o quê>
```

### Tipos

| Tipo | Quando usar |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `refactor` | Mudança de código sem alterar comportamento |
| `test` | Adição ou correção de testes |
| `docs` | Documentação (README, CONTRIBUTING, docstrings) |
| `chore` | Configuração, dependências, CI |
| `perf` | Melhoria de performance |

### Exemplos

```bash
# Funcionalidade nova
git commit -m "feat(config_loader): adiciona PostgresConfigLoader como backend"

# Correção de bug
git commit -m "fix(server_v2): corrige KeyError em fit_metrics_aggregation_fn"

# Testes
git commit -m "test(config_loader): cobre _cast com valores inválidos e tipos nativos"

# Refatoração
git commit -m "refactor(server_v2): extrai _weighted_average como implementação privada"

# Documentação
git commit -m "docs: adiciona CONTRIBUTING.md com padrão de commits e setup"
```

### Regras

- Descrição em **português** (padrão do projeto) ou inglês — seja consistente dentro do PR.
- Imperativo: "adiciona", "corrige", "remove" — não "adicionado", "corrigido".
- Linha de assunto com no máximo 72 caracteres.
- Se o commit fecha um bug ou tarefa do TODO, mencione no corpo.

---

## 6. Abrindo um Pull Request

### Antes de abrir

```bash
# 1. Certifique-se de estar em um branch próprio
git checkout -b feat/minha-contribuicao

# 2. Rode os testes
make test

# 3. Rode o linter
make lint

# 4. Verifique se os hooks passam
git add .
git commit -m "feat: minha contribuição"
```

### Checklist do PR

- [ ] Todos os 299 testes passam (`make test`)
- [ ] Nenhum erro de lint (`make lint`)
- [ ] Novo código tem testes correspondentes
- [ ] Docstrings nos métodos públicos novos
- [ ] `CHANGELOG.md` atualizado na seção `## [Unreleased]`
- [ ] `TODO.md` atualizado se o PR fecha algum item

### Descrição do PR

Use o template:

```
## O que este PR faz
<1-3 frases descrevendo a mudança>

## Por que
<motivação — bug encontrado, item do TODO, melhoria de qualidade>

## Como testar
<comandos ou passos para verificar manualmente>

## Checklist
- [ ] Testes passando
- [ ] Lint passando
- [ ] CHANGELOG atualizado
```

---

## 7. O que contribuir

O [`TODO.md`](TODO.md) tem a lista priorizada de trabalho pendente. Os itens mais acessíveis para novos contribuidores:

**Qualidade de código (baixo risco):**
- Unificar os dois `ConvergenceTracker` (um em `server_v2.py`, outro em `strategy.py`)
- Corrigir docstrings com "VERSÃO CORRIGIDA" nos módulos v2
- Adicionar testes de contrato para `fit()` e `evaluate()`

**Infraestrutura (médio risco):**
- Implementar `_save_checkpoint` real em `server_v2.py` com `torch.save`
- Adicionar `PostgresConfigLoader` em `config_loader.py` (o protocolo já existe)
- Structured logging em JSON nos daemons

**Antes de começar um item grande**, abra uma issue ou discuta no PR para evitar trabalho duplicado.

---

## Dúvidas

Abra uma [issue](https://github.com/JacAbreu/mosaic-fl/issues) descrevendo o problema ou a dúvida.

**Autora:** Jacqueline Abreu — abreujacline@gmail.com
