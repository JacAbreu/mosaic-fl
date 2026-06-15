"""
loader.py
Entry point for loading FAPESP COVID-19 hospital data into the MOSAIC-FL database.

Reads ZIP archives directly (no extraction to disk) and delegates to the
appropriate extract module based on file name pattern.

Two-phase operation:
  --scan-only   Streams all CSVs, registers unknown analytes in term_dictionary
                as inactive, prints a report. Does not insert data.
                Run this first; let the operator review and activate terms.
  (default)     Full load — inserts patients, exams, outcomes.

Usage:
    # Phase 1 — scan and register analyte terms
    python integration/fapesp/loader.py \\
        --data-dir /path/to/Covid-19 \\
        --db-url   postgresql://mosaicfl:senha@localhost:5432/mosaicfl \\
        --scan-only

    # Phase 2 — load after terms are reviewed
    python integration/fapesp/loader.py \\
        --data-dir /path/to/Covid-19 \\
        --db-url   postgresql://mosaicfl:senha@localhost:5432/mosaicfl \\
        --hospitals HSL BPSP
"""
import argparse
import io
import logging
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.fapesp.patients_extract import load_patients
from integration.fapesp.exams_extract    import load_exams, scan_analytes
from integration.fapesp.outcomes_extract import load_outcomes
from infrastructure.mosaicfl_api.db      import PatientDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hospital ZIP registry
# ---------------------------------------------------------------------------

HOSPITAL_ZIPS = {
    "HSL":  "HSL_Janeiro2021.Zip",
    "HFL":  "GrupoFleury_Janeiro2021.zip",
    "HEI":  "EinsteinAgosto.zip",
    "HCSP": "HC_Janeiro2021.zip",
    "BPSP": "BPSP.zip",
}

_SOURCE = "FAPESP"

_PATIENTS_RE = re.compile(r"paciente",  re.IGNORECASE)
_EXAMS_RE    = re.compile(r"exame",     re.IGNORECASE)
_OUTCOMES_RE = re.compile(r"desfecho",  re.IGNORECASE)
_SKIP_RE     = re.compile(r"dicionario|dictionary|\.xlsx$", re.IGNORECASE)

# Compression type 9 = DEFLATE64 — not supported by Python's zipfile
_DEFLATE64 = 9


# ---------------------------------------------------------------------------
# ZIP helpers
# ---------------------------------------------------------------------------

def _find_zip(data_dir: Path, zip_name: str) -> Path:
    for p in data_dir.iterdir():
        if p.name.lower() == zip_name.lower():
            return p
    raise FileNotFoundError(f"ZIP not found: {zip_name} in {data_dir}")


def _uses_deflate64(zip_path: Path) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        return any(info.compress_type == _DEFLATE64 for info in zf.infolist())


def _list_csv_entries(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [e for e in zf.namelist() if e.lower().endswith(".csv")]


def _open_entry_native(zip_path: Path, entry: str) -> io.TextIOWrapper:
    zf = zipfile.ZipFile(zip_path, "r")
    raw = zf.open(entry)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def _open_entry_unzip(zip_path: Path, entry: str) -> io.TextIOWrapper:
    """Falls back to system unzip for DEFLATE64-compressed entries."""
    proc = subprocess.Popen(
        ["unzip", "-p", str(zip_path), entry],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Per-hospital loader
# ---------------------------------------------------------------------------

def scan_hospital(data_dir: Path, hospital_id: str, db: PatientDB) -> bool:
    """Phase 1 — stream exam CSVs and validate analyte terms. No data inserted.

    Returns True if all analytes are resolved (load can proceed).
    """
    zip_name  = HOSPITAL_ZIPS[hospital_id]
    zip_path  = _find_zip(data_dir, zip_name)
    deflate64 = _uses_deflate64(zip_path)

    logger.info("scan_hospital hospital=%s zip=%s", hospital_id, zip_path.name)

    all_ok = True
    for entry in _list_csv_entries(zip_path):
        if _SKIP_RE.search(entry) or not _EXAMS_RE.search(entry):
            continue

        logger.info("  scanning_file file=%s", entry)
        open_fn = _open_entry_unzip if deflate64 else _open_entry_native
        stream  = open_fn(zip_path, entry)
        try:
            result = scan_analytes(stream, db, hospital_id, source=_SOURCE)
            if not result.ok:
                all_ok = False
        finally:
            stream.close()

    return all_ok


def load_hospital(data_dir: Path, hospital_id: str, db: PatientDB) -> dict:
    """Phase 2 — load patients, exams, outcomes. Run after scan_hospital."""
    zip_name  = HOSPITAL_ZIPS[hospital_id]
    zip_path  = _find_zip(data_dir, zip_name)
    deflate64 = _uses_deflate64(zip_path)

    if deflate64:
        logger.info("loading_hospital hospital=%s zip=%s encoding=deflate64 (using unzip)", hospital_id, zip_path.name)
    else:
        logger.info("loading_hospital hospital=%s zip=%s", hospital_id, zip_path.name)

    stats = {"patients": 0, "exams": 0, "outcomes": 0}

    def _entry_order(name: str) -> int:
        if _PATIENTS_RE.search(name):  return 0
        if _EXAMS_RE.search(name):     return 1
        if _OUTCOMES_RE.search(name):  return 2
        return 9

    entries = sorted(
        [e for e in _list_csv_entries(zip_path) if not _SKIP_RE.search(e)],
        key=_entry_order,
    )

    for entry in entries:
        logger.info("  reading_file file=%s", entry)
        open_fn = _open_entry_unzip if deflate64 else _open_entry_native
        stream  = open_fn(zip_path, entry)
        try:
            if _PATIENTS_RE.search(entry):
                stats["patients"] += load_patients(stream, db, hospital_id)
            elif _OUTCOMES_RE.search(entry):
                stats["outcomes"] += load_outcomes(stream, db, hospital_id)
            elif _EXAMS_RE.search(entry):
                stats["exams"] += load_exams(stream, db, hospital_id, source=_SOURCE)
            else:
                logger.warning("unrecognized_file file=%s — skipped", entry)
        finally:
            stream.close()

    logger.info(
        "hospital_done hospital=%s patients=%d exams=%d outcomes=%d",
        hospital_id, stats["patients"], stats["exams"], stats["outcomes"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load FAPESP COVID-19 hospital data into MOSAIC-FL database."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the hospital ZIP files.",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("FL_DB_URL"),
        help="SQLAlchemy database URL (default: FL_DB_URL env var).",
    )
    parser.add_argument(
        "--hospitals",
        nargs="+",
        choices=list(HOSPITAL_ZIPS.keys()),
        default=list(HOSPITAL_ZIPS.keys()),
        help="Which hospitals to load (default: all).",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "Phase 1: stream exam CSVs, register unknown analytes as inactive, "
            "print report. Does not insert data. "
            "Run this first; review and activate terms before loading."
        ),
    )
    args = parser.parse_args()

    if not args.db_url:
        logger.error("db_url_missing: set --db-url or export FL_DB_URL")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("data_dir_not_found path=%s", data_dir)
        sys.exit(1)

    db = PatientDB(args.db_url)

    if args.scan_only:
        all_ok = True
        for hospital_id in args.hospitals:
            try:
                ok = scan_hospital(data_dir, hospital_id, db)
                all_ok = all_ok and ok
            except FileNotFoundError as e:
                logger.error("zip_not_found hospital=%s error=%s", hospital_id, e)
                all_ok = False

        if all_ok:
            logger.info("scan_complete all_terms_resolved — ready to load")
        else:
            logger.warning(
                "scan_complete pending_terms_found — "
                "review with term_manager before running without --scan-only"
            )
        sys.exit(0 if all_ok else 1)

    totals = {"patients": 0, "exams": 0, "outcomes": 0}
    for hospital_id in args.hospitals:
        try:
            stats = load_hospital(data_dir, hospital_id, db)
            for k in totals:
                totals[k] += stats[k]
        except FileNotFoundError as e:
            logger.error("zip_not_found hospital=%s error=%s", hospital_id, e)
        except Exception as e:
            logger.exception("hospital_load_failed hospital=%s error=%s", hospital_id, e)

    logger.info(
        "load_complete total_patients=%d total_exams=%d total_outcomes=%d",
        totals["patients"], totals["exams"], totals["outcomes"],
    )


if __name__ == "__main__":
    main()
