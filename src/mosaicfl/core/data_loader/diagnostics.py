"""
diagnostics.py — Ferramentas de diagnóstico de conexão e schema, para uso interativo.

diagnose_connection() — testa cada fonte de dados sem carregar o pipeline completo
diagnose_dataset()    — inspeciona o schema de um DataFrame já carregado
"""
from .loaders import load_clinical_dataset
from .settings import COLUMN_MAPPING, DEFAULT_CONNECTION_STRING, DEFAULT_QUERY, DEFAULT_SOURCE_TYPE
from .sources import DatabaseDataSource, DataSourceFactory, FileDataSource


def diagnose_connection(
    source_type: str = None,
    connection_string: str = None,
    query: str = None,
) -> None:
    """
    Diagnostica a conexão/fonte de dados sem carregar o pipeline completo.

    Uso:
        python -c "from mosaicfl.core.data_loader import diagnose_connection; diagnose_connection()"

    Ou com SGBD específico:
        python -c "from mosaicfl.core.data_loader import diagnose_connection; \
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


def diagnose_dataset(df=None) -> None:
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
