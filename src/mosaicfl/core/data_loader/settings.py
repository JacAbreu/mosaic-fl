"""
settings.py — Configuração global e tabelas de mapeamento do data_loader.

Editável diretamente ou via variáveis de ambiente (ver docstring de cada constante).
"""
import os

from ..config import RUNTIME_CFG

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

# Mapeamento de desfechos textuais → 4 classes de prognóstico.
# 0=alta  1=internacao_prolongada  2=uti  3=obito
DESFECHO_TEXT_TO_NUMERIC = {
    "alta": 0, "Alta": 0, "ALTA": 0, "cura": 0, "Cura": 0,
    "melhora": 0, "Melhora": 0, "melhorado": 0, "leve": 0, "estavel": 0,
    "internacao": 1, "internação": 1, "Internacao": 1, "Internação": 1,
    "em atendimento": 1, "moderado": 1, "transferencia": 1, "transferência": 1,
    "uti": 2, "UTI": 2, "Uti": 2, "terapia intensiva": 2, "grave": 2,
    "obito": 3, "óbito": 3, "Obito": 3, "Óbito": 3, "morte": 3, "Morte": 3,
}
