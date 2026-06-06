"""
ClinicalPath exporter for MOSAIC-FL predictions.

Writes the five plain-text files expected by ClinicalPath v2 for a single patient:

    Patients/{patient_id}/exam-id.txt
    Patients/{patient_id}/timestamp_to_date.txt
    Patients/{patient_id}/time-metadata.txt
    Patients/{patient_id}/node-inline-time.txt
    Patients/{patient_id}/node-inline-time-complete.txt

The FL risk score is appended as a synthetic exam (FL_RISK_SCORE) after all
real exams, using ref_high=0.3 so that days with score > 0.3 are rendered in
ClinicalPath's abnormal colour range.
"""

import logging
from pathlib import Path

from models import (  # noqa: E402 — loaded via sys.path, not a package
    FL_RISK_EXAM_NAME,
    FL_RISK_REF_HIGH,
    FL_RISK_REF_LOW,
    PatientExport,
)

logger = logging.getLogger(__name__)


class ClinicalPathExporter:
    def export(self, patient: PatientExport, output_dir: Path | str) -> Path:
        """Write ClinicalPath files for *patient* under *output_dir*.

        Returns the patient directory path (``output_dir/Patients/{patient_id}``).
        """
        patient_dir = Path(output_dir) / "Patients" / patient.patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        exam_ids = self._build_exam_id_map(patient)
        all_dates = self._collect_dates(patient)

        if not all_dates:
            logger.warning("patient_no_records", extra={"patient_id": patient.patient_id})
            return patient_dir

        timestamp_map: dict = {d: i for i, d in enumerate(sorted(all_dates))}

        self._write_exam_id(patient_dir, exam_ids)
        self._write_timestamp_to_date(patient_dir, timestamp_map)
        self._write_time_metadata(patient_dir, patient, timestamp_map)
        self._write_node_inline_time(patient_dir, patient, exam_ids, timestamp_map)
        self._write_node_inline_time_complete(patient_dir, patient, exam_ids, timestamp_map)

        logger.info(
            "patient_exported",
            extra={
                "patient_id": patient.patient_id,
                "exam_count": len(exam_ids),
                "timestamp_count": len(timestamp_map),
                "record_count": len(patient.exam_records),
                "risk_prediction_count": len(patient.risk_predictions),
            },
        )
        return patient_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_exam_id_map(self, patient: PatientExport) -> dict[str, int]:
        """Return {exam_name: index}. Real exams sorted alphabetically; FL_RISK_SCORE last."""
        names = sorted({r.exam_name for r in patient.exam_records})
        exam_ids = {name: idx for idx, name in enumerate(names)}
        if patient.risk_predictions:
            exam_ids[FL_RISK_EXAM_NAME] = len(exam_ids)
        return exam_ids

    def _collect_dates(self, patient: PatientExport) -> set:
        dates: set = {r.date for r in patient.exam_records}
        dates.update(p.date for p in patient.risk_predictions)
        return dates

    def _write_exam_id(self, patient_dir: Path, exam_ids: dict[str, int]) -> None:
        lines = [
            f"{idx} {name}"
            for name, idx in sorted(exam_ids.items(), key=lambda kv: kv[1])
        ]
        (patient_dir / "exam-id.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_timestamp_to_date(self, patient_dir: Path, timestamp_map: dict) -> None:
        lines = [
            f"{idx} {d.isoformat()}"
            for d, idx in sorted(timestamp_map.items(), key=lambda kv: kv[1])
        ]
        (patient_dir / "timestamp_to_date.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_time_metadata(
        self, patient_dir: Path, patient: PatientExport, timestamp_map: dict
    ) -> None:
        """One row per unique (timestamp_index, status_string) pair, ordered by timestamp."""
        seen: set = set()
        lines: list[str] = []
        all_entries = [
            (r.date, r.phase.status_str) for r in patient.exam_records
        ] + [(p.date, p.phase.status_str) for p in patient.risk_predictions]

        for d, status_str in sorted(all_entries):
            ts = timestamp_map[d]
            key = (ts, status_str)
            if key not in seen:
                seen.add(key)
                lines.append(f"{ts} {status_str}")

        (patient_dir / "time-metadata.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_node_inline_time(
        self,
        patient_dir: Path,
        patient: PatientExport,
        exam_ids: dict[str, int],
        timestamp_map: dict,
    ) -> None:
        lines: list[str] = []
        for r in patient.exam_records:
            lines.append(
                f"{exam_ids[r.exam_name]} {timestamp_map[r.date]} {r.phase.status_code}"
            )
        for p in patient.risk_predictions:
            lines.append(
                f"{exam_ids[FL_RISK_EXAM_NAME]} {timestamp_map[p.date]} {p.phase.status_code}"
            )
        (patient_dir / "node-inline-time.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_node_inline_time_complete(
        self,
        patient_dir: Path,
        patient: PatientExport,
        exam_ids: dict[str, int],
        timestamp_map: dict,
    ) -> None:
        lines: list[str] = []
        for r in patient.exam_records:
            lines.append(
                f"{exam_ids[r.exam_name]} {timestamp_map[r.date]} {r.phase.status_code} "
                f"{r.value} {r.ref_low} {r.ref_high} {r.sex_ref_low} {r.sex_ref_high}"
            )
        for p in patient.risk_predictions:
            eid = exam_ids[FL_RISK_EXAM_NAME]
            ts = timestamp_map[p.date]
            lines.append(
                f"{eid} {ts} {p.phase.status_code} "
                f"{p.risk_score} {FL_RISK_REF_LOW} {FL_RISK_REF_HIGH} "
                f"{FL_RISK_REF_LOW} {FL_RISK_REF_HIGH}"
            )
        (patient_dir / "node-inline-time-complete.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
