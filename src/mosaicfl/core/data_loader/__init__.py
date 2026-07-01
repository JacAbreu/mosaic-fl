"""
data_loader.py — Integração Flexível de Fontes de Dados para o MOSAIC-FL.

ATENÇÃO — Dois modos de carregamento coexistem neste projeto:

1. **Modo banco de dados (recomendado para dados reais)**
   Usar diretamente: `SequencePipeline(FL_DB_URL).build_per_hospital()`
   Gera tensores temporais ordenados por dia_relativo, tokenizados semanticamente
   ({analyte}_{HIGH|NORMAL|LOW|NO_REF}), com label de 5 classes de prognóstico clínico
   (curado_pronto, curado_internado, melhora_pronto, melhora_internado_breve, melhora_internado_grave).
   Configurar via variável de ambiente: `FL_DB_URL` (preferencial) ou `MOSAICFL_DB_URL` (legado).

2. **Modo arquivo/sintético (desenvolvimento)**
   Usar: `load_with_fallback()` — tenta CSV e sintetiza se necessário.
   Retorna DataFrame com colunas padronizadas (instituicao, desfecho, sintoma, exame, etc.)
   compatíveis com EHRPreprocessor + split_by_institution.

Guarda de produção:
  Se `FL_ENV=production` e `FL_DB_URL` não estiver configurado, load_with_fallback()
  falha imediatamente com RuntimeError. Dados sintéticos são bloqueados em produção.

Suporta:
  • Arquivos locais: CSV, Excel, JSON, Parquet
  • Bancos de dados: PostgreSQL, MySQL, SQLite, SQL Server, Oracle

Padrão de design: Strategy Pattern — a lógica de conexão é isolada,
mas a interface pública é única por caso de uso.

Submódulos:
  errors.py          — DataLoadError
  settings.py          — constantes e tabelas de mapeamento (editável / via env vars)
  sources.py            — DataSourceStrategy, FileDataSource, DatabaseDataSource, DataSourceFactory
  postprocessing.py      — normalização pós-carregamento (mapeamento de colunas, desfecho, sintético)
  loaders.py              — load_clinical_dataset(), load_with_fallback()
  diagnostics.py           — diagnose_connection(), diagnose_dataset()

Cadeia de fallback (load_with_fallback):
  1. SGBD (se connection_string configurada)     → falha silenciosa, tenta próximo
  2. CSV explícito (se csv_path informado)        → falha com erro claro se não existir
  3. CSV padrão (busca em DATASET_FILENAMES)      → falha silenciosa, tenta próximo
  4. Dados sintéticos (se allow_synthetic=True)   → gera com aviso explícito
  5. DataLoadError                                → falha com diagnóstico completo

Variáveis de ambiente:
  FL_DB_URL         — connection string PostgreSQL (nova; preferencial)
  MOSAICFL_DB_URL   — alias legado, ainda funcional
  FL_ENV            — "production" bloqueia sintético e exige FL_DB_URL

Uso:
    from mosaicfl.core.data_loader import load_with_fallback

    # Tenta tudo automaticamente
    df = load_with_fallback()

    # Sem permitir sintético (falha se nenhuma fonte real disponível)
    df = load_with_fallback(allow_synthetic=False)

    # Fonte específica (sem fallback)
    df = load_clinical_dataset(source_type="postgresql", connection_string="...")

Para diagnóstico:
    python -c "from mosaicfl.core.data_loader import diagnose_connection; diagnose_connection()"
"""
from .diagnostics import diagnose_connection, diagnose_dataset
from .errors import DataLoadError
from .loaders import load_clinical_dataset, load_with_fallback
from .postprocessing import (
    _compute_idade_from_nascimento,
    _convert_desfecho,
    _generate_synthetic_fallback,
    _map_columns,
    _validate_schema,
)
from .settings import (
    COLUMN_MAPPING,
    DATASET_BASE_DIR,
    DATASET_FILENAMES,
    DEFAULT_CONNECTION_STRING,
    DEFAULT_QUERY,
    DEFAULT_SOURCE_TYPE,
    DESFECHO_TEXT_TO_NUMERIC,
    ENCODING_CANDIDATES,
    SEPARATOR_CANDIDATES,
)
from .sources import DatabaseDataSource, DataSourceFactory, DataSourceStrategy, FileDataSource

__all__ = [
    "DataLoadError",
    "DataSourceStrategy",
    "FileDataSource",
    "DatabaseDataSource",
    "DataSourceFactory",
    "load_clinical_dataset",
    "load_with_fallback",
    "diagnose_connection",
    "diagnose_dataset",
    "DEFAULT_SOURCE_TYPE",
    "DATASET_FILENAMES",
    "DATASET_BASE_DIR",
    "DEFAULT_CONNECTION_STRING",
    "DEFAULT_QUERY",
    "ENCODING_CANDIDATES",
    "SEPARATOR_CANDIDATES",
    "COLUMN_MAPPING",
    "DESFECHO_TEXT_TO_NUMERIC",
    "_map_columns",
    "_validate_schema",
    "_convert_desfecho",
    "_compute_idade_from_nascimento",
    "_generate_synthetic_fallback",
]
