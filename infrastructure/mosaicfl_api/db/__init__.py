"""
db — Persistence layer for mosaicfl_api.

Schemas:
  clinical  — patient registry, attendances, export paths, FL config (PostgreSQL pure)
  metrics   — time-series exam records, clinical outcomes, risk history (TimescaleDB)

Backend selected by FL_DB_URL:
  postgresql://mosaicfl:senha@localhost:5432/mosaicfl  → PostgreSQL
  sqlite:///data/mosaicfl_api.db                       → SQLite (dev/tests)

Constructor accepts Path for backwards compatibility:
  PatientDB(Path("foo.db"))  →  SQLite at foo.db

PatientDB é composta via mixins — a API pública (todos os métodos) é idêntica à
versão anterior de arquivo único; a implementação está dividida por domínio:
  schema.py                    — definições de tabela
  engine.py                      — fábrica de engine SQLAlchemy
  core.py                          — __init__ + upsert statement builders (_PatientDBCore)
  patients_mixin.py                  — pacientes, atendimentos, export paths
  clinical_mixin.py                    — risco, exames, desfechos clínicos
  transactional_mixin.py                 — variantes *_tx (transação explícita)
  prediction_feedback_mixin.py              — predições + desfecho tardio (ground truth)
"""
from .clinical_mixin import _ClinicalMixin
from .core import _PatientDBCore
from .patients_mixin import _PatientsMixin
from .prediction_feedback_mixin import _PredictionFeedbackMixin
from .transactional_mixin import _TransactionalMixin


class PatientDB(
    _PatientsMixin,
    _ClinicalMixin,
    _TransactionalMixin,
    _PredictionFeedbackMixin,
    _PatientDBCore,
):
    """
    Patient data access layer.

    PostgreSQL: clinical schema for registry/config, metrics schema for time-series.
    SQLite:     all tables without schema prefix (dev/tests — identical interface).
    """


__all__ = ["PatientDB"]
