"""
tests/unit/test_audit.py
Testa pseudonimização, fingerprinting e emissão de registros de auditoria.
"""
import hashlib
from unittest.mock import call, patch

from infrastructure.mosaicfl_api import audit


# ---------------------------------------------------------------------------
# Helpers de pseudonimização e fingerprint
# ---------------------------------------------------------------------------

def test_pseudonymize_is_sha256_prefix():
    pid = "paciente-001"
    expected = hashlib.sha256(pid.encode()).hexdigest()[:16]
    assert audit.pseudonymize(pid) == expected


def test_pseudonymize_length():
    assert len(audit.pseudonymize("qualquer-id")) == 16


def test_pseudonymize_different_ids_differ():
    assert audit.pseudonymize("p-001") != audit.pseudonymize("p-002")


def test_pseudonymize_deterministic():
    assert audit.pseudonymize("p-001") == audit.pseudonymize("p-001")


def test_token_fingerprint_is_sha256_prefix():
    token = "Bearer meu-token-secreto"
    expected = hashlib.sha256(token.encode()).hexdigest()[:12]
    assert audit.token_fingerprint(token) == expected


def test_token_fingerprint_length():
    assert len(audit.token_fingerprint("qualquer-token")) == 12


def test_token_fingerprint_deterministic():
    assert audit.token_fingerprint("t") == audit.token_fingerprint("t")


def test_dev_mode_sentinel_is_safe_string():
    # "dev-mode" é uma string inócua, não deve ser igual ao fingerprint de si mesmo
    assert audit.token_fingerprint("dev-mode") != "dev-mode"


# ---------------------------------------------------------------------------
# Emissão de registros — usa patch para evitar I/O de arquivo nos testes
# ---------------------------------------------------------------------------

def _call_extra(mock_logger) -> dict:
    """Extrai o dict 'extra' da última chamada a mock_logger.info."""
    _, kwargs = mock_logger.info.call_args
    return kwargs["extra"]


def test_log_access_emits_one_record():
    with patch.object(audit, "_audit") as mock_log:
        audit.log_access("predict", token_fp="abc123456789", patient_id="pac-001")
    mock_log.info.assert_called_once()


def test_log_access_event_fields():
    with patch.object(audit, "_audit") as mock_log:
        audit.log_access("predict", token_fp="abc123456789", patient_id="pac-001")
    extra = _call_extra(mock_log)
    assert extra["event"] == "patient_data_access"
    assert extra["operation"] == "predict"
    assert extra["token_fingerprint"] == "abc123456789"


def test_log_access_pseudonymizes_patient_id():
    with patch.object(audit, "_audit") as mock_log:
        audit.log_access("predict", token_fp="fp", patient_id="pac-001")
    extra = _call_extra(mock_log)
    assert extra["patient_id_hash"] == audit.pseudonymize("pac-001")
    assert "patient_id" not in extra, "patient_id em texto claro não deve constar no registro"


def test_log_access_without_patient_id_omits_hash():
    with patch.object(audit, "_audit") as mock_log:
        audit.log_access("patient_list", token_fp="fp")
    extra = _call_extra(mock_log)
    assert "patient_id_hash" not in extra
    assert "patient_id" not in extra


def test_log_access_extra_kwargs_forwarded():
    with patch.object(audit, "_audit") as mock_log:
        audit.log_access("ingest", token_fp="fp", patient_id="p1", exam_count=5)
    extra = _call_extra(mock_log)
    assert extra["exam_count"] == 5


def test_log_access_calls_setup():
    """log_access sempre chama _setup() para garantir inicialização do handler."""
    with patch.object(audit, "_setup") as mock_setup, \
         patch.object(audit, "_audit"):
        audit.log_access("predict", token_fp="fp")
    mock_setup.assert_called_once()


def test_log_access_dev_mode_fingerprint():
    """Token sentinel 'dev-mode' é aceito sem exception."""
    with patch.object(audit, "_audit"):
        audit.log_access("predict", token_fp="dev-mode", patient_id="p")
