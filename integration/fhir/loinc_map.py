"""
loinc_map.py — Mapeamento dos analitos FAPESP para códigos LOINC.

Usado internamente pela tokenização semântica do SequencePipeline.
No contexto FHIR, relevante se o sistema vier a expor Observation resources
(valores de exames). O RiskAssessment atual não os expõe — dados clínicos
ficam no banco local do hospital.

Fonte dos códigos: LOINC® browser (loinc.org). Analitos marcados com
REVISAR não tiveram o código confirmado contra o browser e usam o namespace
urn:mosaicfl como temporário.
"""
from typing import Optional, TypedDict


class LOINCEntry(TypedDict):
    code: str
    system: str
    display: str
    unit: Optional[str]


# Namespace para analitos sem equivalente LOINC confirmado
_MOSAICFL_SYSTEM = "urn:mosaicfl:analyte"
_LOINC_SYSTEM = "http://loinc.org"

LOINC_MAP: dict[str, LOINCEntry] = {
    # ── Hematologia ────────────────────────────────────────────────────────────
    "hemoglobina": {
        "code": "718-7",
        "system": _LOINC_SYSTEM,
        "display": "Hemoglobin [Mass/volume] in Blood",
        "unit": "g/dL",
    },
    "hematocrito": {
        "code": "20570-8",
        "system": _LOINC_SYSTEM,
        "display": "Hematocrit [Volume Fraction] of Blood",
        "unit": "%",
    },
    "leucocitos": {
        "code": "6690-2",
        "system": _LOINC_SYSTEM,
        "display": "Leukocytes [#/volume] in Blood by Automated count",
        "unit": "10*3/uL",
    },
    "plaquetas": {
        "code": "777-3",
        "system": _LOINC_SYSTEM,
        "display": "Platelets [#/volume] in Blood by Automated count",
        "unit": "10*3/uL",
    },
    "neutrofilos": {
        "code": "751-8",
        "system": _LOINC_SYSTEM,
        "display": "Neutrophils [#/volume] in Blood by Automated count",
        "unit": "10*3/uL",
    },
    "linfocitos": {
        "code": "731-0",
        "system": _LOINC_SYSTEM,
        "display": "Lymphocytes [#/volume] in Blood by Automated count",
        "unit": "10*3/uL",
    },

    # ── Bioquímica ─────────────────────────────────────────────────────────────
    "creatinina": {
        "code": "2160-0",
        "system": _LOINC_SYSTEM,
        "display": "Creatinine [Mass/volume] in Serum or Plasma",
        "unit": "mg/dL",
    },
    "ureia": {
        "code": "3094-0",
        "system": _LOINC_SYSTEM,
        "display": "Urea nitrogen [Mass/volume] in Serum or Plasma",
        "unit": "mg/dL",
    },
    "sodio": {
        "code": "2951-2",
        "system": _LOINC_SYSTEM,
        "display": "Sodium [Moles/volume] in Serum or Plasma",
        "unit": "mmol/L",
    },
    "potassio": {
        "code": "2823-3",
        "system": _LOINC_SYSTEM,
        "display": "Potassium [Moles/volume] in Serum or Plasma",
        "unit": "mmol/L",
    },
    "glicose": {
        "code": "2345-7",
        "system": _LOINC_SYSTEM,
        "display": "Glucose [Mass/volume] in Serum or Plasma",
        "unit": "mg/dL",
    },
    "bilirrubina_total": {
        "code": "1975-2",
        "system": _LOINC_SYSTEM,
        "display": "Bilirubin.total [Mass/volume] in Serum or Plasma",
        "unit": "mg/dL",
    },
    "albumina": {
        "code": "1751-7",
        "system": _LOINC_SYSTEM,
        "display": "Albumin [Mass/volume] in Serum or Plasma",
        "unit": "g/dL",
    },

    # ── Marcadores inflamatórios ────────────────────────────────────────────────
    "pcr": {
        "code": "1988-5",
        "system": _LOINC_SYSTEM,
        "display": "C reactive protein [Mass/volume] in Serum or Plasma",
        "unit": "mg/L",
    },
    "ferritina": {
        "code": "2276-4",
        "system": _LOINC_SYSTEM,
        "display": "Ferritin [Mass/volume] in Serum or Plasma",
        "unit": "ng/mL",
    },
    "il6": {
        "code": "26881-3",
        "system": _LOINC_SYSTEM,
        "display": "Interleukin 6 [Mass/volume] in Serum or Plasma",
        "unit": "pg/mL",
    },

    # ── Coagulação ─────────────────────────────────────────────────────────────
    "dimero_d": {
        "code": "48066-5",
        "system": _LOINC_SYSTEM,
        "display": "Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma",
        "unit": "ug/mL{DDU}",
    },
    "tp": {
        "code": "5902-2",
        "system": _LOINC_SYSTEM,
        "display": "Prothrombin time (PT)",
        "unit": "s",
    },

    # ── Enzimas cardíacas e hepáticas ──────────────────────────────────────────
    "ldh": {
        "code": "2532-0",
        "system": _LOINC_SYSTEM,
        "display": "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma",
        "unit": "U/L",
    },
    "troponina": {
        "code": "6598-7",
        "system": _LOINC_SYSTEM,
        "display": "Troponin T.cardiac [Mass/volume] in Serum or Plasma",
        "unit": "ng/mL",
    },
    "tgo": {
        "code": "1920-8",
        "system": _LOINC_SYSTEM,
        "display": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "unit": "U/L",
    },
    "tgp": {
        "code": "1742-6",
        "system": _LOINC_SYSTEM,
        "display": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "unit": "U/L",
    },

    # ── Score FL (sem equivalente LOINC) ──────────────────────────────────────
    "fl_risk_score": {
        "code": "risk-score",
        "system": _MOSAICFL_SYSTEM,
        "display": "MOSAIC-FL Federated Risk Score",
        "unit": None,
    },
}

# Aliases para variações de nome no dataset FAPESP
_ALIASES: dict[str, str] = {
    "hb": "hemoglobina",
    "ht": "hematocrito",
    "leuco": "leucocitos",
    "plaq": "plaquetas",
    "neutro": "neutrofilos",
    "linf": "linfocitos",
    "creat": "creatinina",
    "ur": "ureia",
    "na": "sodio",
    "k": "potassio",
    "gli": "glicose",
    "bili": "bilirrubina_total",
    "alb": "albumina",
    "prot_c_reativa": "pcr",
    "proteina_c_reativa": "pcr",
    "ferrit": "ferritina",
    "d_dimero": "dimero_d",
    "ddimero": "dimero_d",
    "ldh_total": "ldh",
    "trop": "troponina",
    "ast": "tgo",
    "alt": "tgp",
}


def lookup(analyte_name: str) -> Optional[LOINCEntry]:
    """Retorna a entrada LOINC para um analito, ou None se não mapeado.

    A busca é case-insensitive e trata espaços como underscores.
    """
    key = analyte_name.lower().strip().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    return LOINC_MAP.get(key)
