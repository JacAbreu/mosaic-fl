"""schema.py — Definições de tabela (SQLAlchemy Core) para PatientDB.

Schemas:
  clinical  — patient registry, attendances, export paths, FL config (PostgreSQL pure)
  metrics   — time-series exam records, clinical outcomes, risk history (TimescaleDB)
"""
import sqlalchemy as sa


def _build_tables(is_pg: bool):
    clinical = "clinical" if is_pg else None
    metrics  = "metrics"  if is_pg else None

    meta = sa.MetaData()

    patients = sa.Table(
        "patients", meta,
        sa.Column("patient_id",   sa.Text,        primary_key=True),
        sa.Column("sex",          sa.Text,         nullable=False, server_default=sa.text("'M'")),
        sa.Column("age",          sa.Float,        nullable=False, server_default=sa.text("0.0")),
        sa.Column("birth_year",   sa.SmallInteger),
        sa.Column("state_code",   sa.String(2)),
        sa.Column("hospital_id",  sa.Text),
        sa.Column("municipality", sa.Text),
        sa.Column("cep_prefix",   sa.String(5)),
        schema=clinical,
    )
    attendances = sa.Table(
        "attendances", meta,
        sa.Column("attendance_id",   sa.Text, primary_key=True),
        sa.Column("patient_id",      sa.Text, nullable=False),
        sa.Column("hospital_id",     sa.Text),
        sa.Column("attended_at",     sa.Date, nullable=False),
        sa.Column("attendance_type",     sa.Text),
        sa.Column("specialty",           sa.Text),
        sa.Column("clinic_id",           sa.Text),
        sa.Column("suspected_diagnosis", sa.Text),
        sa.Column("confirmed_diagnosis", sa.Text),
        schema=clinical,
    )
    export_paths = sa.Table(
        "export_paths", meta,
        sa.Column("patient_id",  sa.Text, primary_key=True),
        sa.Column("export_path", sa.Text, nullable=False),
        schema=clinical,
    )
    risk_history = sa.Table(
        "risk_history", meta,
        sa.Column("id",         sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Text,    nullable=False),
        sa.Column("date",       sa.Date,    nullable=False),
        sa.Column("risk_score", sa.Float,   nullable=False),
        schema=metrics,
    )
    exam_records = sa.Table(
        "exam_records", meta,
        sa.Column("id",                 sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("patient_id",         sa.Text,  nullable=False),
        sa.Column("analyte",            sa.Text,  nullable=False),
        sa.Column("date",               sa.Date,  nullable=False),
        sa.Column("value",              sa.Float, nullable=False),
        sa.Column("phase",              sa.Text,  nullable=False),
        sa.Column("ref_low",            sa.Float, server_default=sa.text("0.0")),
        sa.Column("ref_high",           sa.Float, server_default=sa.text("0.0")),
        sa.Column("origin",             sa.Text),
        sa.Column("exam_group",         sa.Text),
        sa.Column("value_text",         sa.Text),
        sa.Column("unit",               sa.Text),
        sa.Column("attendance_id",      sa.Text),
        # migration 009 — canonical reference snapshot + clinical classification
        sa.Column("canonical_ref_low",  sa.Float),
        sa.Column("canonical_ref_high", sa.Float),
        sa.Column("classification",     sa.Text),
        schema=metrics,
    )
    clinical_outcomes = sa.Table(
        "clinical_outcomes", meta,
        sa.Column("id",            sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("patient_id",    sa.Text,         nullable=False),
        sa.Column("attendance_id", sa.Text),
        sa.Column("outcome_at",    sa.Date,         nullable=False),
        sa.Column("outcome_text",  sa.Text,         nullable=False),
        sa.Column("outcome_class", sa.SmallInteger, nullable=False),
        schema=metrics,
    )
    # Predições em produção — link via correlation_token para avaliação com ground truth tardio.
    # patient_id_hash: HMAC-SHA256 para audit LGPD sem armazenar identidade real.
    predicted_outcomes = sa.Table(
        "predicted_outcomes", meta,
        sa.Column("id",                   sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("correlation_token",    sa.Text,         nullable=False, unique=True),
        sa.Column("patient_id_hash",      sa.Text,         nullable=False),
        sa.Column("predicted_class",      sa.SmallInteger, nullable=False),
        sa.Column("predicted_label",      sa.Text,         nullable=False),
        sa.Column("class_probabilities",  sa.Text,         nullable=False),  # JSON
        sa.Column("risk_score",           sa.Float,        nullable=False),
        sa.Column("model_round",          sa.Integer),
        sa.Column("model_version",        sa.Text),
        sa.Column("predicted_at",         sa.DateTime,     nullable=False),
        schema=metrics,
    )
    # Desfecho real registrado na alta — gravado pelo hospital, não pelo servidor FL.
    # Nunca trafega pela rede federada: apenas o token efêmero vincula predição ao desfecho.
    outcome_feedback = sa.Table(
        "outcome_feedback", meta,
        sa.Column("id",                sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("correlation_token", sa.Text,         nullable=False, unique=True),
        sa.Column("actual_label",      sa.Text,         nullable=False),
        sa.Column("actual_class",      sa.SmallInteger),
        sa.Column("source",            sa.Text,         nullable=False, server_default=sa.text("'manual'")),
        sa.Column("recorded_at",       sa.DateTime,     nullable=False),
        schema=metrics,
    )

    return meta, patients, attendances, export_paths, risk_history, exam_records, clinical_outcomes, predicted_outcomes, outcome_feedback
