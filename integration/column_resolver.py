"""
column_resolver.py
Resolves DataFrame column names to semantic concepts regardless of case,
accents, or naming conventions.

Usage:
    resolver = ColumnResolver(CLINICAL_SEMANTIC_MAP)
    mapping  = resolver.resolve(df.columns)
    # mapping = {"patient_id": "ID_Paciente", "sex": "IC_Sexo", ...}

    # Access safely:
    patient_id_col = mapping.get("patient_id")
    if patient_id_col:
        df[patient_id_col]

DB-backed usage (preferred — reads from knowledge.term_dictionary):
    from sqlalchemy import create_engine
    engine = create_engine(FL_DB_URL)
    with engine.connect() as conn:
        resolver = ColumnResolver.from_db(conn)
        mapping = resolver.resolve(df.columns)
"""
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy import Connection

logger = logging.getLogger(__name__)


def normalize(name: str) -> str:
    """Lowercase, strip accents, collapse non-alphanumeric runs to underscore."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    return slug.strip("_")


class ColumnResolver:
    """
    Maps semantic concepts to actual column names in a DataFrame.

    Resolution order (first match wins):
      1. Exact match against normalized alias
      2. Starts-with match (normalized alias is prefix of normalized column)
      3. Contains match (normalized alias appears anywhere in normalized column)

    Logs a WARNING for each required concept that could not be resolved.
    """

    def __init__(self, semantic_map: dict[str, list[str]], required: Optional[set[str]] = None):
        self._map = semantic_map
        self._required = required or set()

    @classmethod
    def from_db(
        cls,
        conn: "Connection",
        term_type: str = "column_concept",
        required: Optional[set[str]] = None,
    ) -> "ColumnResolver":
        """Builds a ColumnResolver from knowledge.term_dictionary.

        Falls back to CLINICAL_SEMANTIC_MAP if the table is unreachable or empty,
        so callers are never broken during a migration in progress.
        """
        from sqlalchemy import text

        try:
            rows = conn.execute(
                text("""
                    SELECT canonical, alias
                    FROM knowledge.term_dictionary
                    WHERE term_type = :tt AND active = TRUE
                    ORDER BY canonical, alias
                """),
                {"tt": term_type},
            ).fetchall()
        except Exception:
            logger.warning("term_dictionary unavailable — falling back to CLINICAL_SEMANTIC_MAP")
            return cls(CLINICAL_SEMANTIC_MAP, required)

        if not rows:
            logger.warning("term_dictionary empty for term_type=%s — falling back to CLINICAL_SEMANTIC_MAP", term_type)
            return cls(CLINICAL_SEMANTIC_MAP, required)

        semantic_map: dict[str, list[str]] = {}
        for canonical, alias in rows:
            semantic_map.setdefault(canonical, []).append(alias)

        return cls(semantic_map, required)

    def resolve(self, columns: list[str]) -> dict[str, str]:
        """
        Returns {concept: actual_column_name} for every concept that was found.
        Missing required concepts are logged as warnings.
        """
        normalized_cols = {normalize(c): c for c in columns}
        resolved: dict[str, str] = {}

        for concept, aliases in self._map.items():
            match = self._find(aliases, normalized_cols)
            if match:
                resolved[concept] = match
            elif concept in self._required:
                logger.warning("column_not_found concept=%s aliases=%s", concept, aliases)

        unresolved = [c for c in self._required if c not in resolved]
        if unresolved:
            logger.warning("unresolved_required_concepts concepts=%s", unresolved)

        return resolved

    def _find(self, aliases: list[str], normalized_cols: dict[str, str]) -> Optional[str]:
        norm_aliases = [normalize(a) for a in aliases]

        # 1. Exact match
        for alias in norm_aliases:
            if alias in normalized_cols:
                return normalized_cols[alias]

        # 2. Starts-with
        for col_norm, col_orig in normalized_cols.items():
            for alias in norm_aliases:
                if col_norm.startswith(alias):
                    return col_orig

        # 3. Contains
        for col_norm, col_orig in normalized_cols.items():
            for alias in norm_aliases:
                if alias in col_norm:
                    return col_orig

        return None


# ---------------------------------------------------------------------------
# Analyte name resolution from knowledge.term_dictionary (term_type='analyte')
# ---------------------------------------------------------------------------

def load_analyte_aliases(conn: "Connection") -> dict[str, str]:
    """Loads all analyte aliases from knowledge.term_dictionary into memory.

    Returns {normalized_alias: canonical} for all active analyte entries.
    Call once per ingestion job and pass the result to resolve_analyte_canonical.
    Returns an empty dict (with a warning) if the table is unreachable.
    """
    from sqlalchemy import text

    try:
        rows = conn.execute(
            text("""
                SELECT canonical, alias
                FROM knowledge.term_dictionary
                WHERE term_type = 'analyte' AND active = TRUE
            """)
        ).fetchall()
    except Exception:
        logger.warning("term_dictionary unavailable — analyte resolution will return None for all inputs")
        return {}

    return {normalize(alias): canonical for canonical, alias in rows}


def resolve_analyte_canonical(
    raw_name: str,
    alias_cache: dict[str, str],
) -> Optional[str]:
    """Returns the canonical analyte name for a raw alias using a pre-loaded cache.

    Build the cache once with load_analyte_aliases(conn) before processing a batch.
    Uses exact match only — starts-with is intentionally omitted because shared
    prefixes do not imply clinical equivalence (CREATININA ≠ CREATININA_URINA).
    Aliases must be registered explicitly in knowledge.term_dictionary.

    Returns None if not found.
    """
    return alias_cache.get(normalize(raw_name))


# ---------------------------------------------------------------------------
# Shared semantic map — covers clinical data from multiple hospital sources
# ---------------------------------------------------------------------------

CLINICAL_SEMANTIC_MAP: dict[str, list[str]] = {
    # ── patients ──────────────────────────────────────────────────────────
    "patient_id":      ["id_paciente", "id_patient", "patientid", "cod_paciente", "paciente_id"],
    "sex":             ["ic_sexo", "sexo", "sex", "gender", "genero", "sexo_paciente"],
    "birth_year":      ["aa_nascimento", "ano_nascimento", "birth_year", "anascimento", "aa_nasc"],
    "state_code":      ["cd_uf", "uf", "estado", "state", "cd_estado", "sg_uf"],
    "municipality":    ["cd_municipio", "municipio", "municipality", "nm_municipio", "cd_mun"],
    "cep_prefix":      ["cd_cepreduzido", "cep_reduzido", "cep_prefix", "cd_cep", "cepreduzido"],
    "hospital_id":     ["de_hospital", "hospital", "institution", "hospital_id", "nm_hospital"],
    # ── attendances ───────────────────────────────────────────────────────
    "attendance_id":   ["id_atendimento", "atendimento_id", "attendance_id", "nr_atendimento"],
    "attended_at":     ["dt_atendimento", "data_atendimento", "attended_at", "dt_internacao"],
    "attendance_type": ["de_tipo_atendimento", "tipo_atendimento", "tp_atendimento", "modality"],
    "specialty":       ["de_clinica", "clinica", "specialty", "especialidade", "nm_clinica"],
    "clinic_id":           ["id_clinica", "clinic_id", "id_clinic", "nr_clinica", "cd_clinica"],
    "suspected_diagnosis": ["suspected_diagnosis", "hipotese_diagnostica", "hipotese_diagnostico",
                            "ds_hipotese", "cd_hipotese", "diagnostico_provavel",
                            "probable_diagnosis", "working_diagnosis", "cid_suspeito",
                            "cid_hipotese", "cd_cid_hipotese"],
    "confirmed_diagnosis": ["confirmed_diagnosis", "diagnostico_confirmado", "diagnostico_definitivo",
                            "ds_diagnostico", "cd_diagnostico", "cid_confirmado",
                            "definitive_diagnosis", "final_diagnosis", "cid_principal",
                            "cd_cid_definitivo", "cd_cid_principal"],
    # ── exam records ──────────────────────────────────────────────────────
    "collection_date": ["dt_coleta", "data_coleta", "collection_date", "dt_exame"],
    "origin":          ["de_origem", "origem", "origin", "source", "local_coleta"],
    "exam_group":      ["de_exame", "exame", "exam_name", "grupo_exame", "nm_exame"],
    "analyte":         ["de_analito", "analito", "analyte", "nm_analito", "analise"],
    "result_text":     ["de_resultado", "resultado", "result", "ds_resultado"],
    "result_num":      ["de_resultnum", "resultnum", "valor_numerico", "vl_resultado"],
    "unit":            ["cd_unidade", "unidade", "unit", "ds_unidade"],
    "reference_range": ["de_valor_referencia", "valor_referencia", "referencia", "vl_referencia"],
    # ── outcomes ──────────────────────────────────────────────────────────
    "outcome_text":    ["de_desfecho", "desfecho", "outcome", "evolucao", "ds_desfecho"],
    "outcome_date":    ["dt_desfecho", "data_desfecho", "outcome_date", "dt_alta"],
    "outcome_type":    ["de_tipo_atendimento", "tipo_atendimento", "tp_atendimento"],
}
