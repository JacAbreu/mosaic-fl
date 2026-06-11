"""
patch_new_fields.py
Populates municipality, cep_prefix (patients) and clinic_id (attendances)
without touching exam records.

Usage:
    python integration/fapesp/patch_new_fields.py \
        --data-dir /path/to/Covid-19 \
        --db-url   postgresql://mosaicfl:senha@localhost:5432/mosaicfl
"""
import argparse
import io
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.fapesp.loader import HOSPITAL_ZIPS, _find_zip, _list_csv_entries, _uses_deflate64, _PATIENTS_RE, _OUTCOMES_RE, _SKIP_RE, _open_entry_native, _open_entry_unzip
from integration.fapesp.patients_extract import load_patients
from integration.fapesp.outcomes_extract import load_outcomes
from infrastructure.mosaicfl_api.db import PatientDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUTCOMES_HOSPITALS = {"HSL", "BPSP"}


def patch_hospital(data_dir: Path, hospital_id: str, db: PatientDB) -> None:
    zip_name = HOSPITAL_ZIPS[hospital_id]
    zip_path = _find_zip(data_dir, zip_name)
    deflate64 = _uses_deflate64(zip_path)
    open_fn = _open_entry_unzip if deflate64 else _open_entry_native

    entries = [e for e in _list_csv_entries(zip_path) if not _SKIP_RE.search(e)]

    patients_done = False
    outcomes_done = False

    for entry in sorted(entries):
        if _PATIENTS_RE.search(entry) and not patients_done:
            logger.info("patching_patients hospital=%s file=%s", hospital_id, entry)
            stream = open_fn(zip_path, entry)
            try:
                load_patients(stream, db, hospital_id)
            finally:
                stream.close()
            patients_done = True

        elif _OUTCOMES_RE.search(entry) and hospital_id in _OUTCOMES_HOSPITALS and not outcomes_done:
            logger.info("patching_outcomes hospital=%s file=%s", hospital_id, entry)
            stream = open_fn(zip_path, entry)
            try:
                load_outcomes(stream, db, hospital_id)
            finally:
                stream.close()
            outcomes_done = True

    logger.info("patch_done hospital=%s patients=%s outcomes=%s",
                hospital_id, patients_done, outcomes_done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch municipality, cep_prefix and clinic_id fields.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--db-url", default=os.getenv("FL_DB_URL"))
    parser.add_argument("--hospitals", nargs="+", choices=list(HOSPITAL_ZIPS.keys()),
                        default=list(HOSPITAL_ZIPS.keys()))
    args = parser.parse_args()

    db = PatientDB(args.db_url)
    for hospital_id in args.hospitals:
        try:
            patch_hospital(Path(args.data_dir), hospital_id, db)
        except Exception as e:
            logger.exception("patch_failed hospital=%s error=%s", hospital_id, e)


if __name__ == "__main__":
    main()
