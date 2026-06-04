# Guia de Configuração do SGBD — MOSAIC-FL

Este guia explica como conectar o MOSAIC-FL ao **banco de dados da orientadora** (PostgreSQL, MySQL, SQLite, SQL Server ou Oracle).

O sistema suporta **múltiplas fontes de dados** via Strategy Pattern. Você pode alternar entre CSV e SGBD sem alterar o código do pipeline — apenas configurações.

---

## 🏗️ Arquitetura de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    FONTES DE DADOS SUPORTADAS                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📁 ARQUIVOS LOCAIS              🗄️ SGBD (SQL)               │
│  ├── CSV (.csv)                  ├── PostgreSQL               │
│  ├── Excel (.xlsx)              ├── MySQL                   │
│  ├── JSON (.json)               ├── SQLite                  │
│  └── Parquet (.parquet)         ├── SQL Server               │
│                                  └── Oracle                  │
│                                                             │
│  ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  data_loader.py — Strategy Pattern                   │   │
│  │  • FileDataSource (arquivos)                         │   │
│  │  • DatabaseDataSource (SQL/SQLAlchemy)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Mapeamento de colunas → nomes padronizados         │   │
│  │  Validação de schema                                │   │
│  │  Conversão de desfechos                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  preprocess.py — Limpeza e padronização             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Opção 1: Variáveis de Ambiente (Recomendado)

A forma mais limpa de configurar o SGBD é via **variáveis de ambiente** — não precisa editar código.

### Linux/macOS
```bash
# No terminal, antes de rodar o experimento:
export MOSAICFL_SOURCE_TYPE=postgresql
export MOSAICFL_DB_URL="postgresql://usuario:senha@localhost:5432/nome_do_banco"
export MOSAICFL_DB_QUERY="SELECT * FROM prontuarios_covid WHERE ano >= 2022"

# Depois rode normalmente:
source .venv/bin/activate
python run_v2.py
```

### Windows (CMD)
```bat
set MOSAICFL_SOURCE_TYPE=postgresql
set MOSAICFL_DB_URL=postgresql://usuario:senha@localhost:5432/nome_do_banco
set MOSAICFL_DB_QUERY=SELECT * FROM prontuarios_covid WHERE ano >= 2022

.venv\Scriptsctivate
python run_v2.py
```

### Windows (PowerShell)
```powershell
$env:MOSAICFL_SOURCE_TYPE="postgresql"
$env:MOSAICFL_DB_URL="postgresql://usuario:senha@localhost:5432/nome_do_banco"
$env:MOSAICFL_DB_QUERY="SELECT * FROM prontuarios_covid WHERE ano >= 2022"

.venv\Scriptsctivate
python run_v2.py
```

---

## ⚙️ Opção 2: Editar data_loader.py

Se preferir configuração em arquivo, edite `src/data_loader.py`:

```python
# Linha ~35: tipo de fonte padrão
DEFAULT_SOURCE_TYPE = "postgresql"  # ← altere de "csv" para o seu SGBD

# Linha ~48: connection string
DEFAULT_CONNECTION_STRING = (
    "postgresql://usuario:senha@localhost:5432/nome_do_banco"
    # ou: "mysql+pymysql://usuario:senha@localhost:3306/nome_do_banco"
    # ou: "sqlite:///caminho/para/orientadora.db"
    # ou: "mssql+pyodbc://usuario:senha@host:1433/banco?driver=ODBC+Driver+17+for+SQL+Server"
    # ou: "oracle+cx_oracle://usuario:senha@host:1521/?service_name=XE"
)

# Linha ~55: query padrão
DEFAULT_QUERY = "SELECT * FROM prontuarios_covid WHERE ano >= 2022"
```

---

## ⚙️ Opção 3: Passar diretamente no código

Para testes rápidos ou notebooks:

```python
from src.data_loader import load_clinical_dataset

# PostgreSQL
df = load_clinical_dataset(
    source_type="postgresql",
    connection_string="postgresql://usuario:senha@localhost:5432/prontuarios",
    query="SELECT * FROM pacientes_covid WHERE ano = 2023",
)

# MySQL
df = load_clinical_dataset(
    source_type="mysql",
    connection_string="mysql+pymysql://usuario:senha@localhost:3306/prontuarios",
    query="SELECT * FROM pacientes_covid",
)

# SQLite (dump local do SGBD)
df = load_clinical_dataset(
    source_type="sqlite",
    connection_string="sqlite:///data/orientadora.db",
    query="SELECT * FROM pacientes",
)
```

---

## 🔧 Instalação dos Drivers

Cada SGBD precisa de um driver Python. Instale conforme o seu:

| SGBD | Driver | Comando de instalação |
|---|---|---|
| **PostgreSQL** | psycopg2-binary | `pip install psycopg2-binary` |
| **MySQL** | pymysql | `pip install pymysql` |
| **SQLite** | nativo | Já incluído no Python |
| **SQL Server** | pyodbc | `pip install pyodbc` |
| **Oracle** | cx_oracle | `pip install cx_oracle` |
| **Todos** | SQLAlchemy (base) | `pip install sqlalchemy` |

Instale todos de uma vez (se não souber qual usar):
```bash
pip install sqlalchemy psycopg2-binary pymysql pyodbc
```

---

## 🗺️ Mapeamento de Colunas do Schema do SGBD

O `data_loader.py` precisa saber como as **colunas do seu banco** se chamam para mapeá-las aos nomes internos.

### Exemplo: seu banco tem colunas diferentes

Suponha que o schema da orientadora seja:
```sql
CREATE TABLE prontuarios (
    id_prontuario SERIAL PRIMARY KEY,
    cod_cnes VARCHAR(10),           -- código do hospital
    dt_nascimento DATE,              -- data de nascimento
    vl_idade NUMERIC,                -- idade (já calculada)
    ds_sintomas TEXT,                -- descrição dos sintomas
    ds_exames TEXT,                  -- descrição dos exames
    ds_diagnostico TEXT,             -- diagnóstico
    ds_evolucao VARCHAR(50),         -- "Alta", "Óbito", "UTI"
    dt_atendimento TIMESTAMP
);
```

Edite `COLUMN_MAPPING` em `src/data_loader.py`:

```python
COLUMN_MAPPING = {
    "instituicao": [
        "cod_cnes", "instituicao", "hospital", "unidade", "cnes",
    ],
    "idade": [
        "vl_idade", "idade", "age", "dt_nascimento", "data_nascimento",
    ],
    "sintoma": [
        "ds_sintomas", "sintoma", "sintomas", "queixa_principal",
    ],
    "exame": [
        "ds_exames", "exame", "exames", "resultado_exame",
    ],
    "diagnostico": [
        "ds_diagnostico", "diagnostico", "diagnosis", "cid10",
    ],
    "desfecho": [
        "ds_evolucao", "desfecho", "evolucao", "outcome", "resultado",
    ],
}
```

E os desfechos textuais:
```python
DESFECHO_TEXT_TO_NUMERIC = {
    "Alta": 0,
    "alta": 0,
    "Melhora": 0,
    "Óbito": 1,
    "obito": 1,
    "UTI": 1,
    "Internação": 1,
}
```

---

## 🔍 Diagnóstico da Conexão

Antes de rodar o experimento, teste a conexão:

```bash
source .venv/bin/activate

# Diagnóstico automático (detecta CSV ou SGBD)
python -c "from src.data_loader import diagnose_connection; diagnose_connection()"

# Diagnóstico específico de SGBD
python -c "from src.data_loader import diagnose_connection; \
           diagnose_connection('postgresql', 'postgresql://user:pass@host/db', 'SELECT * FROM tabela')"
```

Saída esperada (sucesso):
```
============================================================
 DIAGNÓSTICO DE CONEXÃO DE DADOS — MOSAIC-FL
============================================================

[Configuração]
  Source type: postgresql
  Connection string: postgresql://user:***@localhost:5432/pr...
  Query: SELECT * FROM prontuarios_covid WHERE ano >= 2022

[Testando fontes...]

[1] SGBD (DatabaseDataSource)
  Connection string (mascarada): postgresql://user:***@localh...
  Query padrão: SELECT * FROM prontuarios_covid WHERE ano >= 2022
  SQLAlchemy disponível: sim
  Conectável: True
  ✓ Conexão OK — 15432 registros, 23 colunas
  Colunas: ['id_prontuario', 'cod_cnes', 'dt_nascimento', ...]

[Resumo]
  Fonte selecionada: DatabaseDataSource
  Disponível: True
  ✓ Dataset acessível: 15432 registros, 23 colunas
============================================================
```

---

## 🔄 Alternando entre CSV e SGBD

Você pode manter **ambas as configurações** e alternar facilmente:

### Cenário A: Desenvolvimento local (CSV)
```bash
# Use CSV local (não precisa de banco rodando)
unset MOSAICFL_DB_URL
export MOSAICFL_SOURCE_TYPE=csv
python run_v2.py
```

### Cenário B: Integração com orientadora (SGBD)
```bash
# Conecta ao banco da orientadora
export MOSAICFL_SOURCE_TYPE=postgresql
export MOSAICFL_DB_URL="postgresql://..."
export MOSAICFL_DB_QUERY="SELECT * FROM prontuarios_covid"
python run_v2.py
```

### Cenário C: Dump do SGBD para SQLite (testes offline)
```bash
# 1. Exporte um subset do banco para SQLite
pg_dump --data-only --table=prontuarios_covid ... | sqlite3 data/orientadora.db

# 2. Rode com SQLite (não precisa PostgreSQL rodando)
export MOSAICFL_SOURCE_TYPE=sqlite
export MOSAICFL_DB_URL="sqlite:///data/orientadora.db"
python run_v2.py
```

---

## 🛡️ Segurança: Nunca commite senhas!

**NUNCA** coloque senhas em arquivos `.py` que vão para o GitHub. Use sempre:

1. **Variáveis de ambiente** (recomendado)
2. **Arquivo `.env`** + `python-dotenv`:
   ```bash
   pip install python-dotenv
   ```
   Crie `.env`:
   ```
   MOSAICFL_DB_URL=postgresql://usuario:senha@localhost:5432/prontuarios
   ```
   Adicione `.env` ao `.gitignore`!

3. **Secrets do sistema operacional** (keyring, vault, etc.)

---

## 📋 Checklist de Integração com SGBD

- [ ] Instalar driver do SGBD (`pip install psycopg2-binary` etc.)
- [ ] Obter connection string da orientadora (host, porta, usuário, senha, banco)
- [ ] Identificar nome da tabela/view (`prontuarios_covid`, `vw_pacientes`, etc.)
- [ ] Mapear colunas do schema real em `COLUMN_MAPPING`
- [ ] Mapear desfechos textuais em `DESFECHO_TEXT_TO_NUMERIC`
- [ ] Testar conexão: `diagnose_connection()`
- [ ] Testar schema: `diagnose_dataset()`
- [ ] Rodar experimento: `python run_v2.py`

---

## ❓ Troubleshooting

### "SQLAlchemy não instalado"
```bash
pip install sqlalchemy psycopg2-binary  # ou pymysql, pyodbc, etc.
```

### "Connection refused"
- Verifique se o PostgreSQL/MySQL está rodando: `sudo systemctl status postgresql`
- Verifique firewall: `sudo ufw allow 5432/tcp` (PostgreSQL)
- Teste com `psql` ou `mysql` CLI primeiro

### "relation 'tabela' does not exist"
- Verifique o nome EXATO da tabela (case-sensitive em PostgreSQL!)
- Use `"SELECT * FROM information_schema.tables WHERE table_schema='public'"` para listar

### "column 'X' does not exist"
- Use `diagnose_connection()` para ver colunas reais
- Atualize `COLUMN_MAPPING` com os nomes exatos do SGBD

### "permission denied for table"
- O usuário precisa de `SELECT` grant: `GRANT SELECT ON prontuarios TO usuario;`

---

## 📚 Exemplos de Connection Strings

| SGBD | Exemplo de connection string |
|---|---|
| PostgreSQL local | `postgresql://usuario:senha@localhost:5432/prontuarios` |
| PostgreSQL remoto | `postgresql://usuario:senha@200.144.10.5:5432/prontuarios` |
| MySQL local | `mysql+pymysql://usuario:senha@localhost:3306/prontuarios` |
| SQLite arquivo | `sqlite:///home/jac/mosaic-fl/data/orientadora.db` |
| SQLite memória | `sqlite:///:memory:` |
| SQL Server | `mssql+pyodbc://usuario:senha@host:1433/banco?driver=ODBC+Driver+17+for+SQL+Server` |
| Oracle | `oracle+cx_oracle://usuario:senha@host:1521/?service_name=XE` |

---

> **Dica:** Se a orientadora não puder liberar acesso direto ao banco de produção, peça um **dump (backup) em SQLite** ou um **CSV exportado** da view `vw_pacientes_anonimizado`. Assim você desenvolve localmente e só conecta ao banco real na fase de validação.
