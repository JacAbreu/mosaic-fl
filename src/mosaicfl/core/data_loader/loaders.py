"""
loaders.py — Funções públicas principais de carregamento de dados.

load_clinical_dataset()   — carrega de uma fonte específica, falha se indisponível
load_with_fallback()      — tenta SGBD → CSV informado → CSV padrão → sintético → falha
"""
import logging
from pathlib import Path

import pandas as pd

from ..config import RUNTIME_CFG
from .errors import DataLoadError
from .postprocessing import (
    _compute_idade_from_nascimento,
    _convert_desfecho,
    _generate_synthetic_fallback,
    _map_columns,
    _validate_schema,
)
from .settings import DATASET_BASE_DIR, DATASET_FILENAMES, DEFAULT_CONNECTION_STRING, DEFAULT_QUERY, DEFAULT_SOURCE_TYPE
from .sources import DatabaseDataSource, DataSourceFactory, FileDataSource

logger = logging.getLogger(__name__)


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
    # ─── GUARDA DE PRODUÇÃO ───────────────────────────────────────────────────
    # Em produção, dados sintéticos são proibidos e FL_DB_URL é obrigatório.
    # Essa verificação ocorre antes de qualquer tentativa de carga para falhar
    # rapidamente, sem consumir tempo em fallbacks que nunca serão permitidos.
    if RUNTIME_CFG.env == "production":
        effective_conn = connection_string or RUNTIME_CFG.db_url or DEFAULT_CONNECTION_STRING
        if not effective_conn:
            raise RuntimeError(
                "FL_ENV=production requer FL_DB_URL configurado.\n"
                "Configure: export FL_DB_URL='postgresql://user:pass@host:5432/db'\n\n"
                "Para desenvolvimento com dados sintéticos: export FL_ENV=development"
            )
        if allow_synthetic:
            logger.warning(
                "FL_ENV=production: ignorando allow_synthetic=True — "
                "dados sintéticos são proibidos em ambiente de produção."
            )
            allow_synthetic = False

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
