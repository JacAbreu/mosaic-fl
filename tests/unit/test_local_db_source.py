"""
Testes para mosaicfl.core.class_weighting.load_local_overrides — 2º nível de
prioridade em FedProxClient._compute_class_weights(), entre o valor empurrado
pelo servidor (compartilhado) e o fallback de env var. Lê o banco LOCAL desta
máquina (clinical.fl_orchestration_config), permitindo BPSP e HSL terem pesos
DIFERENTES entre si — ver docs/pesquisa_baseline_implementacao_fontes_
bibliograficas.md, seção 14, e scripts/set_class_weight_overrides.py.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.class_weighting import load_local_overrides


def _mock_engine(row):
    conn = MagicMock()
    conn.execute.return_value.mappings.return_value.first.return_value = row
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


class TestLoadLocalOverrides:
    def test_empty_db_url_returns_none_without_connecting(self):
        assert load_local_overrides("") is None
        assert load_local_overrides(None) is None

    def test_returns_overrides_when_configured(self):
        engine = _mock_engine({"class_weight_overrides_json": {"curado_internado": 25.0}})
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = load_local_overrides("postgresql://x/y")
        assert result == {"curado_internado": 25.0}

    def test_returns_none_when_column_is_null(self):
        engine = _mock_engine({"class_weight_overrides_json": None})
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = load_local_overrides("postgresql://x/y")
        assert result is None

    def test_returns_none_when_no_row(self):
        engine = _mock_engine(None)
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = load_local_overrides("postgresql://x/y")
        assert result is None

    def test_connection_error_returns_none_without_raising(self):
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("conexão recusada")
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = load_local_overrides("postgresql://x/y")  # não deve levantar exceção
        assert result is None
