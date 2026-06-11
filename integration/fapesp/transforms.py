"""
transforms.py
FAPESP-specific data transformations: date parsing, numeric extraction,
reference range parsing, and outcome classification.
"""
import re
import unicodedata
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(value: str) -> Optional[date]:
    """Parses DD/MM/YYYY or YYYY-MM-DD. Returns None for blank or anonymized values."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if not v or v.upper() in ("AAAA", "YYYY", "NULL", "NA", "NAN"):
        return None
    try:
        if "/" in v:
            day, month, year = v.split("/")
            return date(int(year), int(month), int(day))
        if "-" in v:
            return date.fromisoformat(v[:10])
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Numeric result extraction
# ---------------------------------------------------------------------------

_COVID_KEYWORDS = re.compile(
    r"sars|covid|coronav|anticorp|igm|igg|pcr.*corona|corona.*pcr",
    re.IGNORECASE,
)

# Qualitative COVID result → encoded numeric (mirrors COVID19_Corrige_21_02.sql)
_COVID_RESULT_MAP = {
    -1000: re.compile(r"detect|positiv|reagent|encontr", re.IGNORECASE),
    -1111: re.compile(r"nao.detect|negativ|nao.reagent|ausente", re.IGNORECASE),
    -1234: re.compile(r"inconclu|indet|indetermin", re.IGNORECASE),
}


def extract_numeric(result_text: str, analyte: str = "") -> Optional[float]:
    """
    Extracts a numeric value from a free-text result string.

    For COVID-related analytes, qualitative results are encoded as:
      -1000 = detected / positive / reactive
      -1111 = not detected / negative / non-reactive
      -1234 = inconclusive / indeterminate
      -2222 = other qualitative result

    Returns None when no numeric value can be extracted.
    """
    if not result_text or not isinstance(result_text, str):
        return None

    text = result_text.strip()
    if not text:
        return None

    # COVID qualitative mapping
    if _COVID_KEYWORDS.search(analyte or ""):
        for code, pattern in _COVID_RESULT_MAP.items():
            if pattern.search(text):
                return float(code)

    # Replace Brazilian decimal comma with period
    normalized = text.replace(",", ".")

    # Extract first number (integer or decimal, optionally negative)
    match = re.search(r"-?\d+\.?\d*", normalized)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    # Qualitative COVID fallback if analyte check was inconclusive
    if _COVID_KEYWORDS.search(text):
        for code, pattern in _COVID_RESULT_MAP.items():
            if pattern.search(text):
                return float(code)
        return -2222.0

    return None


# ---------------------------------------------------------------------------
# Reference range parsing
# ---------------------------------------------------------------------------

def parse_reference_range(value: str) -> tuple[float, float]:
    """
    Parses a reference range string into (low, high).
    Returns (0.0, 0.0) when the range is absent or non-numeric.

    Handles:
      "75 a 99"       → (75.0, 99.0)
      "0.5 - 1.5"     → (0.5, 1.5)
      "< 5"           → (0.0, 5.0)
      "> 2"           → (2.0, 0.0)
      "Negativo"      → (0.0, 0.0)
    """
    if not value or not isinstance(value, str):
        return 0.0, 0.0

    v = value.strip().replace(",", ".")

    # Range: "X a Y", "X - Y", "X até Y", "entre X e Y"
    range_match = re.search(r"(-?\d+\.?\d*)\s*(?:a|até|ate|-|–)\s*(-?\d+\.?\d*)", v, re.IGNORECASE)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    # Upper bound: "< X" or "<= X"
    upper_match = re.search(r"<=?\s*(-?\d+\.?\d*)", v)
    if upper_match:
        return 0.0, float(upper_match.group(1))

    # Lower bound: "> X" or ">= X"
    lower_match = re.search(r">=?\s*(-?\d+\.?\d*)", v)
    if lower_match:
        return float(lower_match.group(1)), 0.0

    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Birth year / age
# ---------------------------------------------------------------------------

_REFERENCE_YEAR = 2021  # end of FAPESP data collection window


def parse_birth_year(value) -> Optional[int]:
    """Returns birth year as int, or None for anonymized values (AAAA, YYYY, etc.)."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("AAAA", "YYYY", "NULL", "NAN", "NA", ""):
        return None
    try:
        year = int(float(s))
        return year if 1900 <= year <= _REFERENCE_YEAR else None
    except (ValueError, TypeError):
        return None


def birth_year_to_age(birth_year: Optional[int]) -> float:
    if birth_year is None:
        return 0.0
    age = _REFERENCE_YEAR - birth_year
    return float(max(age, 0))


# ---------------------------------------------------------------------------
# Anonymized field normalization
# ---------------------------------------------------------------------------

_ANON_MARKERS = {"MMMM", "CCCC", "XX", "AAAA", "YYYY", "NULL", "NA", "NAN", ""}


def normalize_optional(value) -> Optional[str]:
    """Returns None for blank or anonymized marker values, otherwise strips the string."""
    if value is None:
        return None
    s = str(value).strip().upper()
    return None if s in _ANON_MARKERS else str(value).strip()


# ---------------------------------------------------------------------------
# Clinical phase inference from FAPESP origin
# ---------------------------------------------------------------------------

_ORIGIN_TO_PHASE = {
    "HOSP": "IN",
    "UTI":  "IN",
    "LAB":  "EX",
}


def infer_phase(origin: Optional[str]) -> str:
    """Maps DE_Origem (LAB/HOSP/UTI) to ClinicalPhase (EX/IN). Defaults to IN."""
    if not origin:
        return "IN"
    return _ORIGIN_TO_PHASE.get(str(origin).strip().upper(), "IN")


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()


_OUTCOME_RULES: list[tuple[int, re.Pattern]] = [
    (6, re.compile(r"obit|morte|falec",                          re.IGNORECASE)),
    (5, re.compile(r"uti|terapia intensiva|semi.?intensiva",     re.IGNORECASE)),
    (4, re.compile(r"em atendimento|em observa|internado(?! uti|.*uti)", re.IGNORECASE)),
    (3, re.compile(r"transfer",                                  re.IGNORECASE)),
    (2, re.compile(r"a pedido|administrativa|evasao|fuga",       re.IGNORECASE)),
    (1, re.compile(r"melhora|melhorado",                         re.IGNORECASE)),
    (0, re.compile(r"curad|cura|alta",                           re.IGNORECASE)),
]


def classify_outcome(outcome_text: str) -> int:
    """
    Maps a free-text outcome description to outcome_class (0–6).

    Scale:
      0 = recovered       (alta médica curado)
      1 = improved        (alta melhorado)
      2 = voluntary       (alta a pedido)
      3 = transferred     (transferência)
      4 = ongoing         (em atendimento)
      5 = icu             (internado em UTI)
      6 = death           (óbito)

    Returns 4 (ongoing) when the text cannot be classified.
    """
    if not outcome_text:
        return 4
    for outcome_class, pattern in _OUTCOME_RULES:
        if pattern.search(outcome_text):
            return outcome_class
    return 4
