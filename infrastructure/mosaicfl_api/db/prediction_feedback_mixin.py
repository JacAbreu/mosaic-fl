"""prediction_feedback_mixin.py — Predições em produção e desfecho tardio (ground truth), via correlation_token."""
import json
from datetime import datetime as _datetime, timezone as _timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import insert


class _PredictionFeedbackMixin:
    """Requer os atributos definidos em _PatientDBCore (_engine, _predicted_outcomes, _outcome_feedback, _is_pg)."""

    def _stmt_insert_prediction(self, values: dict):
        if self._is_pg:
            return pg_insert(self._predicted_outcomes).values(**values).on_conflict_do_nothing(
                index_elements=["correlation_token"]
            )
        return insert(self._predicted_outcomes).prefix_with("OR IGNORE").values(**values)

    def _stmt_insert_feedback(self, values: dict):
        if self._is_pg:
            return pg_insert(self._outcome_feedback).values(**values).on_conflict_do_nothing(
                index_elements=["correlation_token"]
            )
        return insert(self._outcome_feedback).prefix_with("OR IGNORE").values(**values)

    def store_prediction_tx(
        self,
        conn,
        correlation_token: str,
        patient_id_hash: str,
        predicted_class: int,
        predicted_label: str,
        class_probabilities: dict,
        risk_score: float,
        model_round: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Persiste predição com correlation_token dentro de uma transação existente."""
        conn.execute(self._stmt_insert_prediction({
            "correlation_token":   correlation_token,
            "patient_id_hash":     patient_id_hash,
            "predicted_class":     predicted_class,
            "predicted_label":     predicted_label,
            "class_probabilities": json.dumps(class_probabilities),
            "risk_score":          risk_score,
            "model_round":         model_round,
            "model_version":       model_version,
            "predicted_at":        _datetime.now(_timezone.utc),
        }))

    def record_outcome(
        self,
        correlation_token: str,
        actual_label: str,
        actual_class: Optional[int] = None,
        source: str = "manual",
    ) -> bool:
        """Registra desfecho real. Retorna False se token já registrado (idempotente)."""
        with self._engine.begin() as conn:
            result = conn.execute(self._stmt_insert_feedback({
                "correlation_token": correlation_token,
                "actual_label":      actual_label,
                "actual_class":      actual_class,
                "source":            source,
                "recorded_at":       _datetime.now(_timezone.utc),
            }))
            return (result.rowcount or 0) > 0

    def prediction_exists(self, correlation_token: str) -> bool:
        stmt = select(self._predicted_outcomes.c.id).where(
            self._predicted_outcomes.c.correlation_token == correlation_token
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).first() is not None

    def get_prediction_outcome_pairs(self) -> list[dict]:
        """Retorna pares (predição, desfecho real) para avaliação de production quality."""
        po = self._predicted_outcomes
        of = self._outcome_feedback
        stmt = (
            select(
                po.c.correlation_token,
                po.c.predicted_class,
                po.c.predicted_label,
                po.c.class_probabilities,
                po.c.risk_score,
                po.c.model_round,
                po.c.model_version,
                of.c.actual_label,
                of.c.actual_class,
                of.c.source,
                of.c.recorded_at,
            )
            .join(of, po.c.correlation_token == of.c.correlation_token)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings()
            return [dict(r) for r in rows]
