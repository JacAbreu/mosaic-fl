"""
data_loader.py — Integração Flexível de Fontes de Dados para o MOSAIC-FL.

Suporta:
  • Arquivos locais: CSV, Excel, JSON, Parquet
  • Bancos de dados: PostgreSQL, MySQL, SQLite, SQL Server, Oracle

Padrão de design: Strategy Pattern — a lógica de conexão é isolada,
mas a interface pública é única por caso de uso.

Responsabilidades:
  1. Localizar e conectar à fonte de dados (arquivo OU SGBD)
  2. Executar query SQL ou carregar arquivo
  3. Mapear colunas do schema real → nomes padronizados internos
  4. Validar schema (colunas obrigatórias existem?)
  5. Converter tipos e desfechos textuais → numéricos

Funções públicas principais:
  load_clinical_dataset()   — carrega de uma fonte específica, falha se indisponível
  load_with_fallback()      — tenta SGBD → CSV informado → CSV padrão → sintético → falha

Cadeia de fallback (load_with_fallback):
  1. SGBD (se connection_string configurada)     → falha silenciosa, tenta próximo
  2. CSV explícito (se csv_path informado)        → falha com erro claro se não existir
  3. CSV padrão (busca em DATASET_FILENAMES)      → falha silenciosa, tenta próximo
  4. Dados sintéticos (se allow_synthetic=True)   → gera com aviso explícito
  5. DataLoadError                                → falha com diagnóstico completo

Uso:
    from mosaicfl.core.data_loader import load_with_fallback

    # Tenta tudo automaticamente
    df = load_with_fallback()

    # Com CSV específico como segunda opção
    df = load_with_fallback(csv_path="data/minha_base.csv")

    # Sem permitir sintético (falha se nenhuma fonte real disponível)
    df = load_with_fallback(allow_synthetic=False)

    # Fonte específica (sem fallback)
    df = load_clinical_dataset(source_type="postgresql", connection_string="...")

Para diagnóstico:
    python -c "from mosaicfl.core.data_loader import diagnose_connection; diagnose_connection()"
"""
import os
import re
import random
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from .config import FED_CFG, RUNTIME_CFG

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """
    Levantada quando todas as fontes de dados falharam em load_with_fallback().
    Inclui o histórico de tentativas para facilitar o diagnóstico.
    """
    def __init__(self, message: str, attempts: List[Dict] = None):
        self.attempts = attempts or []
        super().__init__(message)

    def __str__(self):
        base = super().__str__()
        if self.attempts:
            log = "\n".join(
                f"  [{i+1}] {a['fonte']}: {a['erro']}"
                for i, a in enumerate(self.attempts)
            )
            return f"{base}\n\nTentativas realizadas:\n{log}"
        return base


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO GLOBAL — EDITE AQUI OU USE VARIÁVEIS DE AMBIENTE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Fonte de dados padrão ───
# Opções: "csv", "excel", "json", "parquet", "postgresql", "mysql", "sqlite", "mssql", "oracle"
DEFAULT_SOURCE_TYPE = os.getenv("MOSAICFL_SOURCE_TYPE", "csv")

# ─── Configuração de arquivo (modo CSV/Excel) ───
DATASET_FILENAMES = [
    "base_orientadora.csv",
    "dados_covid.csv",
    "fapesp_covid19.csv",
    "dataset.csv",
    "dados.xlsx",
    "dados.parquet",
]
DATASET_BASE_DIR = RUNTIME_CFG.data_path

# ─── Configuração de SGBD (modo database) ───
# Pode ser passada diretamente ou via variável de ambiente MOSAICFL_DB_URL
DEFAULT_CONNECTION_STRING = os.getenv(
    "MOSAICFL_DB_URL",
    # Exemplos (descomente e edite o da sua orientadora):
    # "postgresql://usuario:senha@localhost:5432/nome_do_banco"
    # "mysql+pymysql://usuario:senha@localhost:3306/nome_do_banco"
    # "sqlite:///data/orientadora.db"
    # "mssql+pyodbc://usuario:senha@host:1433/banco?driver=ODBC+Driver+17+for+SQL+Server"
    # "oracle+cx_oracle://usuario:senha@host:1521/?service_name=XE"
    ""
)

# Query padrão (pode ser passada diretamente ou via env)
DEFAULT_QUERY = os.getenv(
    "MOSAICFL_DB_QUERY",
    "SELECT * FROM prontuarios"  # ← SUBSTITUA PELO NOME REAL DA TABELA/VIEW
)

# Encoding para CSVs
ENCODING_CANDIDATES = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
SEPARATOR_CANDIDATES = [";", ","]


# ═══════════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE COLUNAS — ADAPTE ÀS COLUNAS REAIS DO SEU SGBD
# ═══════════════════════════════════════════════════════════════════════════════

COLUMN_MAPPING = {
    "instituicao": [
        "instituicao", "instituição", "hospital", "unidade", "centro", "site",
        "nome_hospital", "cnes", "id_instituicao", "origem", "cod_instituicao",
    ],
    "idade": [
        "idade", "age", "idade_anos", "idade_paciente", "dt_nascimento",
        "data_nascimento", "birth_date",
    ],
    "idade_unidade": [
        "idade_unidade", "unidade_idade", "age_unit", "idade_em", "tipo_idade",
    ],
    "peso": [
        "peso", "weight", "peso_kg", "peso_paciente", "body_weight",
    ],
    "peso_unidade": [
        "peso_unidade", "unidade_peso", "weight_unit", "peso_em",
    ],
    "temperatura": [
        "temperatura", "temp", "temperature", "temp_axilar", "temp_timpanica",
        "temperatura_c", "temperatura_f",
    ],
    "sintoma": [
        "sintoma", "sintomas", "symptoms", "sintoma_principal", "queixa_principal",
        "sintomas_relato", "descricao_sintomas", "ds_sintoma",
    ],
    "exame": [
        "exame", "exames", "exam", "exame_complementar", "resultado_exame",
        "laboratorio", "lab_result", "descricao_exames", "ds_exame",
    ],
    "diagnostico": [
        "diagnostico", "diagnóstico", "diagnosis", "diagnostico_principal",
        "cid10", "cid_10", "diagnostico_entrada", "hipotese_diagnostica", "ds_diagnostico",
    ],
    "desfecho": [
        "desfecho", "outcome", "evolucao", "evolução", "resultado", "desfecho_clinico",
        "obito", "óbito", "alta", "transferencia", "uti", "internacao",
        "evolucao_clinica", "desfecho_final",
    ],
}

# Mapeamento de desfechos textuais → numéricos
DESFECHO_TEXT_TO_NUMERIC = {
    "alta": 0, "Alta": 0, "ALTA": 0, "cura": 0, "melhora": 0,
    "obito": 1, "óbito": 1, "Obito": 1, "Óbito": 1, "morte": 1,
    "uti": 1, "UTI": 1, "internacao": 1, "internação": 1,
    "pneumonia": 1, "Pneumonia": 1, "gravidez": 1, "grave": 1,
    "moderado": 0, "leve": 0, "estavel": 0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PATTERN: Interface base para fontes de dados
# ═══════════════════════════════════════════════════════════════════════════════

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
                "  3. Edite DEFAULT_CONNECTION_STRING em data_loader.py"
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


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE PÓS-PROCESSAMENTO (independentes da fonte)
# ═══════════════════════════════════════════════════════════════════════════════

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas do schema real → nomes padronizados internos."""
    rename_map = {}
    csv_columns_lower = {c.lower().strip(): c for c in df.columns}

    for standard_name, candidates in COLUMN_MAPPING.items():
        for candidate in candidates:
            candidate_clean = candidate.lower().strip()
            if candidate_clean in csv_columns_lower:
                original_name = csv_columns_lower[candidate_clean]
                rename_map[original_name] = standard_name
                logger.info(f"Coluna mapeada: '{original_name}' → '{standard_name}'")
                break
        else:
            logger.warning(f"Coluna não encontrada: '{standard_name}' (tentados: {candidates})")

    return df.rename(columns=rename_map)


def _validate_schema(df: pd.DataFrame) -> None:
    """Valida colunas obrigatórias após mapeamento."""
    required = ["instituicao", "desfecho"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        available = [c for c in df.columns if not c.startswith("Unnamed")]
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing}\n"
            f"Colunas disponíveis: {available}\n\n"
            f"DICA: Edite COLUMN_MAPPING em data_loader.py para mapear "
            f"os nomes reais do seu schema (CSV ou SGBD)."
        )

    if df["desfecho"].dtype == object:
        unique_vals = df["desfecho"].dropna().unique()[:20]
        logger.warning(
            f"'desfecho' é textual. Primeiros valores: {list(unique_vals)}. "
            f"Aplicando mapeamento DESFECHO_TEXT_TO_NUMERIC."
        )


def _convert_desfecho(df: pd.DataFrame) -> pd.DataFrame:
    """Converte desfecho textual → numérico."""
    # Em pandas 4.x, colunas de string têm dtype `str` (StringDtype), não `object`.
    # Usar is_numeric_dtype é robusto para qualquer versão do pandas.
    if pd.api.types.is_numeric_dtype(df["desfecho"]):
        return df

    df = df.copy()
    df["desfecho_original"] = df["desfecho"]
    df["desfecho"] = df["desfecho"].map(DESFECHO_TEXT_TO_NUMERIC)

    n_unmapped = df["desfecho"].isna().sum()
    if n_unmapped > 0:
        unmapped = df.loc[df["desfecho"].isna(), "desfecho_original"].unique()[:10]
        logger.warning(
            f"{n_unmapped} desfechos não mapeados. Valores: {list(unmapped)}. "
            f"Adicione em DESFECHO_TEXT_TO_NUMERIC."
        )

    return df


def _compute_idade_from_nascimento(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula idade a partir da data de nascimento, se disponível."""
    if "idade" in df.columns:
        return df

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["nasc", "birth", "dt_nasc"])]
    if not date_cols:
        return df

    birth_col = date_cols[0]
    logger.info(f"'idade' ausente — calculando a partir de '{birth_col}'")

    try:
        df = df.copy()
        birth_dates = pd.to_datetime(df[birth_col], errors="coerce", dayfirst=True)
        today = pd.Timestamp.now()
        df["idade"] = ((today - birth_dates).dt.days / 365.25).round(1)
        df["idade_unidade"] = "anos"
        logger.info(f"Idade calculada para {df['idade'].notna().sum()} registros.")
    except Exception as e:
        logger.warning(f"Erro ao calcular idade: {e}")

    return df


def _generate_synthetic_fallback(n_samples: int = 1000) -> pd.DataFrame:
    """Gera dados sintéticos para testes. NUNCA use em produção."""
    random.seed(FED_CFG.random_seed)
    np.random.seed(FED_CFG.random_seed)

    institutions = ["HC_usp", "HCFMUSP", "Hospital_Clinicas_SP", "Incor", "Hospital_Sirio"]
    sintomas = ["febre", "tosse", "dispneia", "fadiga", "mialgia", "cefaleia", "anosmia", "diarreia"]
    exames = ["rt_pcr_positivo", "tomografia_normal", "tomografia_vidro_fosco", "rx_consolidacao"]
    diagnosticos = ["covid19_leve", "covid19_moderado", "covid19_grave", "pneumonia_bacteriana"]

    df = pd.DataFrame({
        "instituicao": np.random.choice(institutions, n_samples),
        "idade": np.random.randint(18, 90, n_samples),
        "idade_unidade": np.random.choice(["anos", "meses"], n_samples, p=[0.95, 0.05]),
        "peso": np.random.uniform(50, 120, n_samples),
        "peso_unidade": np.random.choice(["kg", "lb"], n_samples, p=[0.9, 0.1]),
        "temperatura": np.random.uniform(36.0, 40.0, n_samples),
        "sintoma": np.random.choice(sintomas, n_samples),
        "exame": np.random.choice(exames, n_samples),
        "diagnostico": np.random.choice(diagnosticos, n_samples),
        "desfecho": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })

    logger.warning(f"[FALLBACK SINTÉTICO] {n_samples} registros gerados — NÃO SÃO REAIS!")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════════

def load_clinical_dataset(
    source_type: str = None,
    connection_string: str = None,
    query: str = None,
    force_synthetic: bool = False,
    n_synthetic_samples: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """
    Carrega o dataset clínico da orientadora de qualquer fonte de dados.

    Esta é a ÚNICA função que você precisa chamar. Ela abstrai completamente
    se os dados vêm de CSV, Excel, PostgreSQL, MySQL, SQLite, etc.

    Args:
        source_type: Tipo de fonte. Opções:
            - "csv", "excel", "json", "parquet" → arquivo local
            - "postgresql", "mysql", "sqlite", "mssql", "oracle" → SGBD
            - None → auto-detecta (tenta SGBD primeiro, depois arquivo)
        connection_string: URL de conexão do SGBD (ex: postgresql://user:pass@host/db).
            Se None, usa MOSAICFL_DB_URL (env var) ou DEFAULT_CONNECTION_STRING.
        query: Query SQL para o SGBD. Se None, usa MOSAICFL_DB_QUERY ou DEFAULT_QUERY.
        force_synthetic: Se True, ignora fontes reais e gera dados fake (apenas testes).
        n_synthetic_samples: Número de amostras sintéticas.
        **kwargs: Parâmetros extras passados à estratégia (ex: filenames para FileDataSource).

    Returns:
        pd.DataFrame com colunas padronizadas.

    Raises:
        RuntimeError: se nenhuma fonte for detectada.
        FileNotFoundError: se arquivo não for encontrado (modo arquivo).
        ValueError: se schema for inválido após mapeamento.

    Exemplos:
        >>> # Modo arquivo (default)
        >>> df = load_clinical_dataset()

        >>> # Modo PostgreSQL da orientadora
        >>> df = load_clinical_dataset(
        ...     source_type="postgresql",
        ...     connection_string="postgresql://user:pass@localhost:5432/prontuarios",
        ...     query="SELECT * FROM pacientes_covid WHERE ano=2023",
        ... )

        >>> # Modo SQLite (dump do SGBD para testes locais)
        >>> df = load_clinical_dataset(
        ...     source_type="sqlite",
        ...     connection_string="sqlite:///data/orientadora.db",
        ... )
    """

    if force_synthetic:
        logger.warning("=" * 60)
        logger.warning("MODO SINTÉTICO — DADOS NÃO SÃO REAIS!")
        logger.warning("=" * 60)
        return _generate_synthetic_fallback(n_synthetic_samples)

    # ─── PASSO 1: Seleciona fonte de dados ───
    source_type = source_type or DEFAULT_SOURCE_TYPE

    if source_type == "auto":
        source = DataSourceFactory.auto_detect(**kwargs)
    else:
        # Para SGBD, passa connection_string e query se fornecidos
        if source_type in ["postgresql", "postgres", "mysql", "sqlite", "mssql", "sqlserver", "oracle", "db", "database"]:
            source = DataSourceFactory.create(
                source_type,
                connection_string=connection_string,
                query=query,
            )
        else:
            source = DataSourceFactory.create(source_type, **kwargs)

    # ─── PASSO 2: Carrega dados brutos ───
    logger.info(f"Fonte de dados: {source.__class__.__name__}")
    df = source.load(**kwargs)
    logger.info(f"Dados brutos carregados: {len(df)} registros, {len(df.columns)} colunas")
    logger.info(f"Colunas brutas: {list(df.columns)}")

    # ─── PASSO 3: Mapeia colunas ───
    df = _map_columns(df)

    # ─── PASSO 4: Deriva colunas ausentes ───
    df = _compute_idade_from_nascimento(df)

    # ─── PASSO 5: Converte desfecho ───
    df = _convert_desfecho(df)

    # ─── PASSO 6: Valida schema ───
    _validate_schema(df)

    # ─── PASSO 7: Resumo ───
    logger.info("=" * 60)
    logger.info("DATASET INTEGRADO COM SUCESSO")
    logger.info("=" * 60)
    logger.info(f"  Registros:      {len(df)}")
    logger.info(f"  Colunas:        {list(df.columns)}")
    if "instituicao" in df.columns:
        logger.info(f"  Instituições:   {df['instituicao'].nunique()}")
    if "idade" in df.columns:
        logger.info(f"  Idade média:    {df['idade'].mean():.1f} (±{df['idade'].std():.1f})")
    if "desfecho" in df.columns:
        logger.info(f"  Desfecho:       {df['desfecho'].value_counts().to_dict()}")
    logger.info("=" * 60)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO COM FALLBACK AUTOMÁTICO
# ═══════════════════════════════════════════════════════════════════════════════

def load_with_fallback(
    connection_string: str = None,
    query: str = None,
    csv_path: str = None,
    allow_synthetic: bool = True,
    n_synthetic_samples: int = 1000,
) -> pd.DataFrame:
    """
    Carrega o dataset clínico percorrendo uma cadeia de fallback até encontrar
    uma fonte disponível. Falha com diagnóstico completo se nenhuma funcionar.

    Cadeia de tentativas (em ordem):
      1. SGBD         — se connection_string configurada (env ou parâmetro)
      2. CSV explícito — se csv_path informado e o arquivo existir
                         → falha imediata com erro claro se o arquivo NÃO existir
      3. CSV padrão   — busca DATASET_FILENAMES em data/
      4. Sintético    — gera dados fake (só se allow_synthetic=True)
      5. DataLoadError — falha com histórico de todas as tentativas

    Args:
        connection_string: URL de conexão do SGBD. Se None, usa MOSAICFL_DB_URL.
        query:            Query SQL. Se None, usa MOSAICFL_DB_QUERY.
        csv_path:         Caminho explícito para um CSV. Se informado e não
                          encontrado, falha imediatamente (não tenta próximo).
        allow_synthetic:  Se True, gera dados sintéticos como último recurso.
                          Se False, falha se nenhuma fonte real estiver disponível.
        n_synthetic_samples: Número de amostras sintéticas.

    Returns:
        pd.DataFrame com colunas padronizadas e campo '_fonte' indicando a origem:
          'sgbd' | 'csv_explicito' | 'csv_padrao' | 'sintetico'

    Raises:
        FileNotFoundError: se csv_path foi informado mas o arquivo não existe.
        DataLoadError:     se todas as fontes falharam.

    Exemplos:
        >>> # Tenta tudo automaticamente, sintético como último recurso
        >>> df = load_with_fallback()

        >>> # Com CSV específico como segunda opção
        >>> df = load_with_fallback(csv_path="data/base_orientadora.csv")

        >>> # Sem sintético — falha se nenhuma fonte real disponível
        >>> df = load_with_fallback(allow_synthetic=False)

        >>> # SGBD com fallback para CSV
        >>> df = load_with_fallback(
        ...     connection_string="postgresql://user:pass@host/db",
        ...     csv_path="data/backup.csv",
        ...     allow_synthetic=False,
        ... )
    """
    attempts = []   # histórico de tentativas para o DataLoadError

    def _post_process(df: pd.DataFrame, fonte: str) -> pd.DataFrame:
        """Aplica mapeamento, conversão e validação; adiciona campo _fonte."""
        df = _map_columns(df)
        df = _compute_idade_from_nascimento(df)
        df = _convert_desfecho(df)
        _validate_schema(df)
        df["_fonte"] = fonte
        logger.info(f"Dataset carregado via '{fonte}': {len(df)} registros")
        return df

    # ─── TENTATIVA 1: SGBD ────────────────────────────────────────────────────
    conn = connection_string or DEFAULT_CONNECTION_STRING
    if conn:
        logger.info("[Fallback 1/4] Tentando SGBD...")
        try:
            db = DatabaseDataSource(connection_string=conn, query=query or DEFAULT_QUERY)
            if db.is_available():
                df = db.load()
                return _post_process(df, "sgbd")
            else:
                msg = "SGBD configurado mas conexão recusada"
                logger.warning(f"  [FALHOU] {msg}")
                attempts.append({"fonte": "SGBD", "erro": msg})
        except Exception as e:
            logger.warning(f"  [FALHOU] SGBD falhou: {e}")
            attempts.append({"fonte": "SGBD", "erro": str(e)})
    else:
        logger.info("[Fallback 1/4] SGBD pulado (connection_string não configurada)")
        attempts.append({"fonte": "SGBD", "erro": "connection_string não configurada"})

    # ─── TENTATIVA 2: CSV EXPLÍCITO ───────────────────────────────────────────
    if csv_path is not None:
        logger.info(f"[Fallback 2/4] Tentando CSV explícito: {csv_path}")
        path = Path(csv_path)
        if not path.exists():
            # CSV foi explicitamente informado mas não existe → falha imediata,
            # não tenta próximo (intenção clara do chamador)
            raise FileNotFoundError(
                f"CSV informado não encontrado: {path.resolve()}\n"
                f"Verifique o caminho ou omita csv_path para usar o CSV padrão."
            )
        try:
            source = FileDataSource(base_dir=path.parent, filenames=[path.name])
            df = source.load()
            return _post_process(df, "csv_explicito")
        except Exception as e:
            logger.warning(f"  [FALHOU] CSV explicito falhou: {e}")
            attempts.append({"fonte": f"CSV explícito ({csv_path})", "erro": str(e)})
    else:
        logger.info("[Fallback 2/4] CSV explícito pulado (csv_path não informado)")
        attempts.append({"fonte": "CSV explícito", "erro": "csv_path não informado"})

    # ─── TENTATIVA 3: CSV PADRÃO ──────────────────────────────────────────────
    logger.info(f"[Fallback 3/4] Tentando CSV padrão em '{DATASET_BASE_DIR}'...")
    file_src = FileDataSource()
    if file_src.is_available():
        try:
            df = file_src.load()
            return _post_process(df, "csv_padrao")
        except Exception as e:
            logger.warning(f"  [FALHOU] CSV padrao falhou: {e}")
            attempts.append({"fonte": "CSV padrão", "erro": str(e)})
    else:
        msg = f"Nenhum dos arquivos {DATASET_FILENAMES} encontrado em '{DATASET_BASE_DIR}'"
        logger.warning(f"  [FALHOU] {msg}")
        attempts.append({"fonte": "CSV padrão", "erro": msg})

    # ─── TENTATIVA 4: SINTÉTICO ───────────────────────────────────────────────
    if allow_synthetic:
        logger.warning("=" * 60)
        logger.warning("[Fallback 4/4] NENHUMA FONTE REAL DISPONÍVEL")
        logger.warning("Gerando dados SINTÉTICOS — NÃO USE EM PRODUÇÃO")
        logger.warning("=" * 60)
        df = _generate_synthetic_fallback(n_synthetic_samples)
        df["_fonte"] = "sintetico"
        return df
    else:
        attempts.append({
            "fonte": "Sintético",
            "erro": "allow_synthetic=False — geração sintética desabilitada",
        })

    # ─── TODAS AS FONTES FALHARAM ─────────────────────────────────────────────
    raise DataLoadError(
        "Nenhuma fonte de dados disponível.\n"
        "SOLUÇÕES:\n"
        "  1. Coloque um CSV em data/ (nomes aceitos: " + str(DATASET_FILENAMES) + ")\n"
        "  2. Informe csv_path='caminho/para/arquivo.csv'\n"
        "  3. Configure MOSAICFL_DB_URL com a connection string do SGBD\n"
        "  4. Use allow_synthetic=True para dados de teste\n"
        "  5. Diagnóstico: diagnose_connection()",
        attempts=attempts,
    )

def diagnose_connection(
    source_type: str = None,
    connection_string: str = None,
    query: str = None,
) -> None:
    """
    Diagnostica a conexão/fonte de dados sem carregar o pipeline completo.

    Uso:
        python -c "from src.data_loader import diagnose_connection; diagnose_connection()"

    Ou com SGBD específico:
        python -c "from src.data_loader import diagnose_connection; \
                   diagnose_connection('postgresql', 'postgresql://...', 'SELECT * FROM tabela')"
    """
    print("\n" + "=" * 60)
    print(" DIAGNÓSTICO DE CONEXÃO DE DADOS — MOSAIC-FL")
    print("=" * 60)

    source_type = source_type or DEFAULT_SOURCE_TYPE
    print(f"\n[Configuração]")
    print(f"  Source type (env/env): {source_type}")
    print(f"  Connection string: {(connection_string or DEFAULT_CONNECTION_STRING or 'N/A')[:50]}...")
    print(f"  Query: {(query or DEFAULT_QUERY)[:60]}...")

    # Tenta cada fonte
    print(f"\n[Testando fontes...]")

    # 1. SGBD
    if connection_string or DEFAULT_CONNECTION_STRING:
        db = DatabaseDataSource(connection_string=connection_string, query=query)
        print(f"\n[1] SGBD (DatabaseDataSource)")
        db.diagnose()
        if db.is_available():
            tables = db.list_tables()
            if tables:
                print(f"  Tabelas disponíveis: {tables[:10]}{'...' if len(tables) > 10 else ''}")
    else:
        print(f"\n[1] SGBD — pulado (connection string não configurada)")

    # 2. Arquivo
    file_src = FileDataSource()
    print(f"\n[2] Arquivo local (FileDataSource)")
    file_src.diagnose()

    # 3. Resumo
    print(f"\n[Resumo]")
    try:
        source = DataSourceFactory.auto_detect() if source_type == "auto" else DataSourceFactory.create(source_type)
        print(f"  Fonte selecionada: {source.__class__.__name__}")
        print(f"  Disponível: {source.is_available()}")
        if source.is_available():
            df = source.load()
            print(f"  Dataset acessivel: {len(df)} registros, {len(df.columns)} colunas")
    except Exception as e:
        print(f"  [ERRO] {e}")

    print("=" * 60 + "\n")


def diagnose_dataset(df: pd.DataFrame = None) -> None:
    """
    Diagnostica o schema de um DataFrame já carregado.
    Útil para verificar mapeamento de colunas.
    """
    if df is None:
        print("Carregando dataset para diagnóstico...")
        df = load_clinical_dataset()

    print("\n" + "=" * 60)
    print(" DIAGNÓSTICO DO SCHEMA")
    print("=" * 60)
    print(f"\nRegistros: {len(df)} | Colunas: {len(df.columns)}")
    print(f"Colunas: {list(df.columns)}")

    mapped = [c for c in df.columns if c in COLUMN_MAPPING.keys()]
    unmapped = [c for c in df.columns if c not in COLUMN_MAPPING.keys() and not c.startswith("Unnamed")]

    print(f"\nColunas reconhecidas ({len(mapped)}): {mapped}")
    print(f"Colunas nao mapeadas ({len(unmapped)}): {unmapped}")

    required = ["instituicao", "desfecho"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n[ERRO] Colunas obrigatorias ausentes: {missing}")
    else:
        print(f"\nTodas as colunas obrigatorias presentes")

    print("=" * 60 + "\n")