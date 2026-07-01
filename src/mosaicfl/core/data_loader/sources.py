"""
sources.py — Strategy Pattern: fontes de dados (arquivo local ou SGBD).

DataSourceStrategy    — interface abstrata
FileDataSource        — CSV, Excel, JSON, Parquet
DatabaseDataSource    — PostgreSQL, MySQL, SQLite, SQL Server, Oracle (via SQLAlchemy)
DataSourceFactory     — seleciona a estratégia correta por nome ou auto-detecção

Nota de teste: DatabaseDataSource e DataSourceFactory leem DEFAULT_CONNECTION_STRING
e DEFAULT_QUERY como variáveis livres de módulo — os testes fazem
monkeypatch/patch em "mosaicfl.core.data_loader.sources.DEFAULT_CONNECTION_STRING"
(não no pacote raiz), porque é este o módulo onde a leitura de fato acontece.
"""
import logging
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import pandas as pd

from .settings import (
    DATASET_BASE_DIR,
    DATASET_FILENAMES,
    DEFAULT_CONNECTION_STRING,
    DEFAULT_QUERY,
    ENCODING_CANDIDATES,
    SEPARATOR_CANDIDATES,
)

try:
    from abc import ABC, abstractmethod
except ImportError:  # pragma: no cover - abc é stdlib, sempre disponível
    raise

logger = logging.getLogger(__name__)


class DataSourceStrategy(ABC):
    """Interface abstrata para qualquer fonte de dados (arquivo ou SGBD)."""

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Carrega dados e retorna DataFrame."""
        pass

    @abstractmethod
    def diagnose(self) -> None:
        """Executa diagnóstico da conexão/fonte."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica se a fonte está acessível."""
        pass


# ───────────────────────────────────────────────────────────────────────────────
# Strategy: Arquivos Locais (CSV, Excel, JSON, Parquet)
# ───────────────────────────────────────────────────────────────────────────────

class FileDataSource(DataSourceStrategy):
    """Carrega dados de arquivos locais."""

    def __init__(self, base_dir: Path = DATASET_BASE_DIR, filenames: List[str] = None):
        self.base_dir = base_dir
        self.filenames = filenames or DATASET_FILENAMES

    def is_available(self) -> bool:
        for filename in self.filenames:
            if (self.base_dir / filename).exists():
                return True
        return False

    def _find_file(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.filenames:
            candidate = self.base_dir / filename
            if candidate.exists():
                logger.info(f"Arquivo encontrado: {candidate.resolve()}")
                return candidate.resolve()

        files_found = list(self.base_dir.iterdir()) if self.base_dir.exists() else []
        files_str = "\n  - ".join([f.name for f in files_found]) if files_found else "(vazio)"
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em: {self.base_dir.resolve()}\n"
            f"Arquivos tentados: {self.filenames}\n"
            f"Arquivos existentes:\n  - {files_str}\n\n"
            f"SOLUÇÃO: Coloque o arquivo em '{self.base_dir}/' ou defina MOSAICFL_SOURCE_TYPE=postgresql"
        )

    def _detect_csv_format(self, file_path: Path) -> Tuple[str, str]:
        for encoding in ENCODING_CANDIDATES:
            for sep in SEPARATOR_CANDIDATES:
                try:
                    df_test = pd.read_csv(file_path, nrows=5, encoding=encoding, sep=sep)
                    if len(df_test.columns) > 1:
                        return encoding, sep
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
        return "utf-8", ","

    def load(self, **kwargs) -> pd.DataFrame:
        file_path = self._find_file()
        suffix = file_path.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            logger.info(f"Carregando Excel: {file_path.name}")
            return pd.read_excel(file_path)
        elif suffix == ".json":
            logger.info(f"Carregando JSON: {file_path.name}")
            return pd.read_json(file_path)
        elif suffix == ".parquet":
            logger.info(f"Carregando Parquet: {file_path.name}")
            return pd.read_parquet(file_path)
        else:
            encoding, sep = self._detect_csv_format(file_path)
            logger.info(f"Carregando CSV: {file_path.name} (encoding={encoding}, sep='{sep}')")
            return pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)

    def diagnose(self) -> None:
        print(f"\n[FileDataSource] Diretório: {self.base_dir.resolve()}")
        print(f"  Arquivos tentados: {self.filenames}")
        print(f"  Disponível: {self.is_available()}")
        if self.is_available():
            path = self._find_file()
            print(f"  Arquivo encontrado: {path.name}")
            df = self.load()
            print(f"  Registros: {len(df)} | Colunas: {len(df.columns)}")
            print(f"  Colunas: {list(df.columns)}")


# ───────────────────────────────────────────────────────────────────────────────
# Strategy: Banco de Dados (SQL via SQLAlchemy)
# ───────────────────────────────────────────────────────────────────────────────

class DatabaseDataSource(DataSourceStrategy):
    """
    Carrega dados de qualquer SGBD suportado pelo SQLAlchemy.

    Drivers necessários (instale conforme seu SGBD):
      PostgreSQL:  pip install psycopg2-binary
      MySQL:       pip install pymysql
      SQL Server:  pip install pyodbc
      Oracle:      pip install cx_oracle
      SQLite:      já incluído no Python
    """

    def __init__(self, connection_string: str = "", query: str = ""):
        self.connection_string = connection_string or DEFAULT_CONNECTION_STRING
        self.query = query or DEFAULT_QUERY
        self._engine = None

    def is_available(self) -> bool:
        if not self.connection_string:
            return False
        try:
            engine = self._get_engine()
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.debug(f"Database não disponível: {e}")
            return False

    def _get_engine(self):
        """Lazy initialization do engine SQLAlchemy."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self.connection_string)
            except ImportError:
                raise ImportError(
                    "SQLAlchemy não instalado. Execute: pip install sqlalchemy\n"
                    "E instale o driver do seu SGBD (ex: pip install psycopg2-binary)"
                )
        return self._engine

    def load(self, query: str = None, **kwargs) -> pd.DataFrame:
        query = query or self.query
        if not query:
            raise ValueError("Query SQL não fornecida. Defina MOSAICFL_DB_QUERY ou passe query='...'")
        if not self.connection_string:
            raise ValueError(
                "Connection string não configurada.\n"
                "Opções:\n"
                "  1. Env var: export MOSAICFL_DB_URL='postgresql://user:pass@host/db'\n"
                "  2. Parâmetro: load_clinical_dataset(connection_string='...')\n"
                "  3. Edite DEFAULT_CONNECTION_STRING em settings.py"
            )

        engine = self._get_engine()
        logger.info(f"Conectando ao SGBD: {self._mask_connection_string()}")
        logger.info(f"Executando query: {query[:100]}{'...' if len(query) > 100 else ''}")

        try:
            df = pd.read_sql(query, engine)
            logger.info(f"Query executada: {len(df)} registros, {len(df.columns)} colunas")
            return df
        except Exception as e:
            raise RuntimeError(f"Erro ao executar query no SGBD: {e}") from e

    def _mask_connection_string(self) -> str:
        """Oculta senha no log."""
        try:
            parsed = urlparse(self.connection_string)
            if parsed.password:
                return self.connection_string.replace(parsed.password, "***")
        except Exception:
            pass
        return self.connection_string[:30] + "..."

    def diagnose(self) -> None:
        print(f"\n[DatabaseDataSource]")
        print(f"  Connection string (mascarada): {self._mask_connection_string()}")
        print(f"  Query padrão: {self.query}")
        print(f"  SQLAlchemy disponível: {'sim' if self._has_sqlalchemy() else 'NÃO'}")
        print(f"  Conectável: {self.is_available()}")

        if self.is_available():
            try:
                df = self.load()
                print(f"  Conexao OK — {len(df)} registros, {len(df.columns)} colunas")
                print(f"  Colunas: {list(df.columns)}")
            except Exception as e:
                print(f"  [ERRO] Erro na query: {e}")

    def _has_sqlalchemy(self) -> bool:
        try:
            import sqlalchemy
            return True
        except ImportError:
            return False

    def list_tables(self) -> List[str]:
        """Lista tabelas disponíveis no banco (útil para descobrir o schema)."""
        if not self.is_available():
            return []
        try:
            from sqlalchemy import inspect
            inspector = inspect(self._get_engine())
            return inspector.get_table_names()
        except Exception as e:
            logger.warning(f"Não foi possível listar tabelas: {e}")
            return []


# ───────────────────────────────────────────────────────────────────────────────
# Factory: Seleciona a estratégia correta
# ───────────────────────────────────────────────────────────────────────────────

class DataSourceFactory:
    """Cria a estratégia de fonte de dados apropriada."""

    _strategies = {
        "csv": FileDataSource,
        "excel": FileDataSource,
        "json": FileDataSource,
        "parquet": FileDataSource,
        "file": FileDataSource,
        "postgresql": DatabaseDataSource,
        "postgres": DatabaseDataSource,
        "mysql": DatabaseDataSource,
        "sqlite": DatabaseDataSource,
        "mssql": DatabaseDataSource,
        "sqlserver": DatabaseDataSource,
        "oracle": DatabaseDataSource,
        "db": DatabaseDataSource,
        "database": DatabaseDataSource,
    }

    @classmethod
    def create(cls, source_type: str, **kwargs) -> DataSourceStrategy:
        source_type = source_type.lower().strip()
        if source_type not in cls._strategies:
            raise ValueError(
                f"Fonte de dados '{source_type}' não suportada.\n"
                f"Opções: {list(cls._strategies.keys())}"
            )
        return cls._strategies[source_type](**kwargs)

    @classmethod
    def auto_detect(cls, **kwargs) -> DataSourceStrategy:
        """Tenta detectar automaticamente a fonte disponível."""
        # 1. Primeiro tenta SGBD se connection string estiver configurada
        if DEFAULT_CONNECTION_STRING:
            db = DatabaseDataSource()
            if db.is_available():
                logger.info("Fonte auto-detectada: SGBD (connection string configurada)")
                return db

        # 2. Senão, tenta arquivo local
        file_src = FileDataSource()
        if file_src.is_available():
            logger.info("Fonte auto-detectada: Arquivo local")
            return file_src

        # 3. Nada encontrado
        raise RuntimeError(
            "Nenhuma fonte de dados detectada automaticamente.\n"
            "Configure uma das opções:\n"
            "  • Arquivo: coloque CSV/Excel em data/\n"
            "  • SGBD: defina MOSAICFL_DB_URL (env var) ou connection_string"
        )
