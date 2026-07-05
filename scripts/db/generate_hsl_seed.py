"""
generate_hsl_seed.py
Gera um arquivo SQL comprimido (gzip) com os dados do HSL para carregamento
no banco PostgreSQL do equipamento cliente da simulação federada.

Lê os CSVs diretamente do ZIP (sem extração em disco) e grava blocos
COPY FROM stdin para as 4 tabelas do schema MOSAIC-FL:

  clinical.patients          (~8.971 linhas)
  clinical.attendances       (~42.691 linhas)
  metrics.clinical_outcomes  (~42.691 linhas)
  metrics.exam_records       (~1.463.834 linhas)

O arquivo gerado (scripts/db/seeds/hsl_seed.sql.gz) deve ser transferido para
o notebook cliente e carregado com:
    make client-load-hsl

Uso:
    python scripts/db/generate_hsl_seed.py \\
        --data-dir ~/data/Dados/Covid-19 \\
        [--output  scripts/db/seeds/hsl_seed.sql.gz] \\
        [--db-url  postgresql://mosaicfl:senha@localhost:5432/mosaicfl]

    --db-url (opcional): conecta ao banco local para carregar o dicionário de
    termos (alias_cache). Sem ele, nomes de analitos são normalizados sem
    resolução canônica e classification fica NULL (backfill posterior).
"""
import argparse
import csv
import gzip
import io
import logging
import os
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.column_resolver import CLINICAL_SEMANTIC_MAP, ColumnResolver, normalize
from integration.fapesp.transforms import (
    birth_year_to_age,
    classify_outcome,
    extract_numeric,
    infer_phase,
    normalize_optional,
    parse_birth_year,
    parse_date,
    parse_reference_range,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOSPITAL_ID = "HSL"
ZIP_NAME    = "HSL_Janeiro2021.Zip"

_ANON_MARKERS = {"MMMM", "CCCC", "XX", "AAAA", "YYYY", "NULL", "NA", "NAN", ""}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_zip(data_dir: Path) -> Path:
    for p in data_dir.iterdir():
        if p.name.lower() == ZIP_NAME.lower():
            return p
    raise FileNotFoundError(f"{ZIP_NAME} não encontrado em {data_dir}")


def _csv_entries(zip_path: Path) -> dict[str, str]:
    """Retorna {tipo: nome_do_entry} para pacientes, exames e desfechos."""
    entries: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            nl = name.lower()
            if nl.endswith(".csv"):
                if "paciente" in nl:
                    entries["patients"] = name
                elif "exame" in nl:
                    entries["exams"] = name
                elif "desfecho" in nl:
                    entries["outcomes"] = name
    return entries


def _open(zip_path: Path, entry: str) -> io.TextIOWrapper:
    zf = zipfile.ZipFile(zip_path, "r")
    return io.TextIOWrapper(zf.open(entry), encoding="utf-8", errors="replace")


def _val(v) -> Optional[str]:
    """None para valores nulos ou marcadores de anonimização."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s.upper() in _ANON_MARKERS else s


def _csv_write(row: list) -> str:
    """Formata uma linha para o bloco COPY FROM stdin (CSV com NULL = \\N)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["\\N" if v is None else v for v in row])
    return buf.getvalue()


def _load_alias_cache(db_url: str) -> dict[str, str]:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT canonical, alias FROM knowledge.term_dictionary
                WHERE term_type = 'analyte' AND active = TRUE
            """)).fetchall()
        cache: dict[str, str] = {}
        for canonical, alias in rows:
            cache[normalize(alias)]     = canonical
            cache[normalize(canonical)] = canonical
        logger.info("alias_cache carregado: %d entradas", len(cache))
        return cache
    except Exception as e:
        logger.warning("alias_cache indisponível (%s) — analitos sem resolução canônica", e)
        return {}


def _resolve_canonical(raw: str, cache: dict[str, str]) -> str:
    norm = normalize(raw)
    return cache.get(norm, norm.upper())


# ---------------------------------------------------------------------------
# Geradores por tabela
# ---------------------------------------------------------------------------

def _write_copy_block(out: io.BufferedWriter, header: str, rows_iter) -> int:
    """Escreve um bloco COPY ... FROM stdin e retorna o número de linhas."""
    out.write((header + "\n").encode())
    n = 0
    for row in rows_iter:
        out.write(_csv_write(row).encode())
        n += 1
    out.write(b"\\.\n\n")
    return n


def generate_patients(zip_path: Path, entry: str, out: io.BufferedWriter) -> set[str]:
    """Grava clinical.patients e retorna o conjunto de patient_ids inseridos."""
    resolver = ColumnResolver(CLINICAL_SEMANTIC_MAP, required={"patient_id", "sex", "birth_year"})
    patient_ids: set[str] = set()
    rows: list[list] = []

    stream = _open(zip_path, entry)
    try:
        mapping: Optional[dict] = None
        for chunk in __import__("pandas").read_csv(
            stream, sep="|", dtype=str, chunksize=10_000,
            encoding="utf-8", on_bad_lines="warn"
        ):
            if mapping is None:
                mapping = resolver.resolve(list(chunk.columns))
                logger.info("patients columns: %s", mapping)

            def col(c): return mapping.get(c)

            for _, row in chunk.iterrows():
                pid = _val(row.get(col("patient_id")))
                if not pid or pid in patient_ids:
                    continue
                patient_ids.add(pid)

                sex        = (_val(row.get(col("sex"))) or "U")[:1].upper()
                birth_year = parse_birth_year(_val(row.get(col("birth_year"))))
                age        = birth_year_to_age(birth_year)
                state      = normalize_optional(_val(row.get(col("state_code"))))
                muni       = normalize_optional(_val(row.get(col("municipality"))))
                cep_raw    = normalize_optional(_val(row.get(col("cep_prefix"))))
                cep        = cep_raw[:5] if cep_raw else None

                rows.append([pid, sex, age, birth_year, state, HOSPITAL_ID, muni, cep])
    finally:
        stream.close()

    header = (
        "COPY clinical.patients "
        "(patient_id, sex, age, birth_year, state_code, hospital_id, municipality, cep_prefix) "
        "FROM stdin (FORMAT CSV, NULL '\\N');"
    )
    n = _write_copy_block(out, header, iter(rows))
    logger.info("patients gravados: %d", n)
    return patient_ids


def generate_outcomes(
    zip_path: Path, entry: str, out: io.BufferedWriter, patient_ids: set[str]
) -> set[str]:
    """Grava clinical.attendances + metrics.clinical_outcomes.
    Retorna conjunto de attendance_ids inseridos."""
    import pandas as pd
    resolver = ColumnResolver(
        CLINICAL_SEMANTIC_MAP,
        required={"patient_id", "outcome_date", "outcome_text"},
    )

    att_rows: list[list] = []
    out_rows: list[list] = []
    att_ids: set[str] = set()

    stream = _open(zip_path, entry)
    try:
        mapping: Optional[dict] = None
        for chunk in pd.read_csv(
            stream, sep="|", dtype=str, chunksize=10_000,
            encoding="utf-8", on_bad_lines="warn"
        ):
            if mapping is None:
                mapping = resolver.resolve(list(chunk.columns))
                logger.info("outcomes columns: %s", mapping)

            def col(c): return mapping.get(c)

            for _, row in chunk.iterrows():
                pid = _val(row.get(col("patient_id")))
                if not pid or pid not in patient_ids:
                    continue

                att_id      = _val(row.get(col("attendance_id")))
                attended_at = parse_date(_val(row.get(col("attended_at"))))
                att_type    = normalize_optional(_val(row.get(col("attendance_type"))))
                specialty   = normalize_optional(_val(row.get(col("specialty"))))
                clinic_id   = normalize_optional(_val(row.get(col("clinic_id"))))

                if att_id and attended_at and att_id not in att_ids:
                    att_ids.add(att_id)
                    att_rows.append([
                        att_id, pid, HOSPITAL_ID,
                        attended_at.isoformat(),
                        att_type, specialty, clinic_id,
                        None, None,  # suspected_diagnosis, confirmed_diagnosis
                    ])

                outcome_date = parse_date(_val(row.get(col("outcome_date"))))
                if not outcome_date:
                    continue

                outcome_text  = (_val(row.get(col("outcome_text"))) or "").strip()
                outcome_class = classify_outcome(outcome_text)

                out_rows.append([
                    pid,
                    att_id if att_id in att_ids else None,
                    outcome_date.isoformat(),
                    outcome_text,
                    outcome_class,
                ])
    finally:
        stream.close()

    att_header = (
        "COPY clinical.attendances "
        "(attendance_id, patient_id, hospital_id, attended_at, "
        "attendance_type, specialty, clinic_id, suspected_diagnosis, confirmed_diagnosis) "
        "FROM stdin (FORMAT CSV, NULL '\\N');"
    )
    n_att = _write_copy_block(out, att_header, iter(att_rows))
    logger.info("attendances gravados: %d", n_att)

    out_header = (
        "COPY metrics.clinical_outcomes "
        "(patient_id, attendance_id, outcome_at, outcome_text, outcome_class) "
        "FROM stdin (FORMAT CSV, NULL '\\N');"
    )
    n_out = _write_copy_block(out, out_header, iter(out_rows))
    logger.info("clinical_outcomes gravados: %d", n_out)

    return att_ids


def generate_exams(
    zip_path: Path,
    entry: str,
    out: io.BufferedWriter,
    patient_ids: set[str],
    att_ids: set[str],
    alias_cache: dict[str, str],
    chunk_size: int = 50_000,
) -> None:
    """Grava metrics.exam_records em chunks para controlar uso de memória."""
    import pandas as pd
    resolver = ColumnResolver(
        CLINICAL_SEMANTIC_MAP,
        required={"patient_id", "collection_date", "analyte", "result_text"},
    )

    header = (
        "COPY metrics.exam_records "
        "(patient_id, analyte, date, value, phase, ref_low, ref_high, "
        "origin, exam_group, value_text, unit, attendance_id, "
        "canonical_ref_low, canonical_ref_high, classification) "
        "FROM stdin (FORMAT CSV, NULL '\\N');"
    )
    out.write((header + "\n").encode())

    total = 0
    stream = _open(zip_path, entry)
    try:
        mapping: Optional[dict] = None
        for chunk in pd.read_csv(
            stream, sep="|", dtype=str, chunksize=chunk_size,
            encoding="utf-8", on_bad_lines="warn"
        ):
            if mapping is None:
                mapping = resolver.resolve(list(chunk.columns))
                logger.info("exams columns: %s", mapping)

            def col(c): return mapping.get(c)
            def gcol(c):
                c_ = col(c)
                return chunk[c_] if c_ and c_ in chunk.columns else \
                    __import__("pandas").Series([None] * len(chunk), index=chunk.index)

            pid_s    = gcol("patient_id").astype(str).str.strip()
            valid    = pid_s.isin(patient_ids)

            date_s   = gcol("collection_date").where(valid).apply(
                lambda v: parse_date(str(v)) if __import__("pandas").notna(v) else None
            )
            valid   &= date_s.notna()

            ana_s    = gcol("analyte").astype(str).str.strip()
            grp_s    = gcol("exam_group").astype(str).str.strip()
            raw_ana  = ana_s.where(
                ana_s.notna() & ana_s.ne("nan") & ana_s.ne(""), grp_s
            ).fillna("UNKNOWN")

            res_txt  = gcol("result_text").fillna("").astype(str)
            res_num  = gcol("result_num").fillna("").astype(str)

            def _parse_val(num, txt, ana):
                num = num.strip()
                if num and num not in ("nan", "None", ""):
                    try:
                        return float(num.replace(",", "."))
                    except (ValueError, TypeError):
                        pass
                return extract_numeric(txt, ana)

            value_s  = __import__("pandas").Series(
                [_parse_val(n, t, a) for n, t, a in zip(res_num, res_txt, raw_ana)],
                index=chunk.index,
            )
            valid   &= value_s.notna()

            if not valid.any():
                continue

            pid_s    = pid_s[valid]
            date_s   = date_s[valid]
            raw_ana  = raw_ana[valid]
            res_txt  = res_txt[valid]
            value_s  = value_s[valid]

            ref_s    = gcol("reference_range")[valid].fillna("").astype(str).apply(
                parse_reference_range
            )
            ref_low  = ref_s.apply(lambda t: t[0])
            ref_high = ref_s.apply(lambda t: t[1])

            origin_s  = gcol("origin")[valid].apply(
                lambda v: normalize_optional(str(v)) if __import__("pandas").notna(v) else None
            )
            unit_s    = gcol("unit")[valid].apply(
                lambda v: normalize_optional(str(v)) if __import__("pandas").notna(v) else None
            )
            grp_s     = gcol("exam_group")[valid]
            att_s     = gcol("attendance_id")[valid]

            canonical = raw_ana.apply(lambda n: _resolve_canonical(n, alias_cache))
            phase_s   = origin_s.apply(infer_phase)

            for pid, d, rt, v, rl, rh, ori, u, eg, ai, can, ph in zip(
                pid_s, date_s, res_txt, value_s,
                ref_low, ref_high, origin_s, unit_s, grp_s, att_s,
                canonical, phase_s,
            ):
                ai_val = str(ai).strip() if __import__("pandas").notna(ai) and \
                         str(ai) not in ("nan", "None") else None
                if ai_val and ai_val not in att_ids:
                    ai_val = None

                eg_val = str(eg).strip() if __import__("pandas").notna(eg) and \
                         str(eg) not in ("nan", "None") else None

                out.write(_csv_write([
                    pid, can, d.isoformat(), v, ph,
                    rl, rh,  # 0.0 = sem faixa de referência (mesma convenção do bulk_load.py
                             # e do schema NOT NULL DEFAULT 0.0 — nunca converter para None aqui)
                    ori, eg_val, rt.strip() or None, u,
                    ai_val,
                    None, None, None,  # canonical_ref_low/high, classification — backfill posterior
                ]).encode())
                total += 1

            if total % 100_000 == 0 and total > 0:
                logger.info("  exam_records: %d linhas gravadas...", total)
    finally:
        stream.close()

    out.write(b"\\.\n\n")
    logger.info("exam_records gravados: %d", total)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera SQL seed do HSL para o banco cliente da simulação FL."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Diretório com os ZIPs FAPESP (deve conter HSL_Janeiro2021.Zip).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "seeds" / "hsl_seed.sql.gz"),
        help="Arquivo de saída (.sql.gz). Padrão: scripts/db/seeds/hsl_seed.sql.gz",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("FL_DB_URL"),
        help="(Opcional) URL do banco local para carregar alias_cache do term_dictionary.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zip_path = _find_zip(data_dir)
    entries  = _csv_entries(zip_path)

    missing = [k for k in ("patients", "exams", "outcomes") if k not in entries]
    if missing:
        logger.error("CSVs não encontrados no ZIP: %s", missing)
        sys.exit(1)

    alias_cache = _load_alias_cache(args.db_url) if args.db_url else {}
    if not alias_cache:
        logger.warning(
            "alias_cache vazio — analitos gravados sem resolução canônica. "
            "Execute backfill após carregar se necessário."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Gerando seed: %s → %s", zip_path.name, output)

    with gzip.open(output, "wb") as gz:
        gz.write(b"-- MOSAIC-FL HSL Seed\n")
        gz.write(b"-- Hospital: Sirio-Libanes (HSL) | Simulacao cliente FL\n")
        gz.write(b"-- Gerado por scripts/db/generate_hsl_seed.py\n")
        gz.write(b"-- Fonte: USP-FAPESP Data Sharing COVID-19\n\n")
        gz.write(b"SET client_encoding TO 'UTF8';\n")
        gz.write(b"SET datestyle TO 'ISO, YMD';\n\n")

        logger.info("1/3 Gerando clinical.patients...")
        patient_ids = generate_patients(zip_path, entries["patients"], gz)

        logger.info("2/3 Gerando clinical.attendances + metrics.clinical_outcomes...")
        att_ids = generate_outcomes(zip_path, entries["outcomes"], gz, patient_ids)

        logger.info("3/3 Gerando metrics.exam_records (~1,46 M linhas — pode levar alguns minutos)...")
        generate_exams(zip_path, entries["exams"], gz, patient_ids, att_ids, alias_cache)

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Seed gerado: %s (%.1f MB)", output, size_mb)
    logger.info(
        "Transfira para o notebook e execute:\n"
        "    make client-load-hsl"
    )


if __name__ == "__main__":
    main()
