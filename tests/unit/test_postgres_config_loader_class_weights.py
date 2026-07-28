"""
Testes para PostgreSQLConfigLoader.load()/write() cobrindo class_weight_overrides_json
(migration 028) — o canal que empurra peso de classe explícito (cost-sensitive learning,
Strategy pattern em mosaicfl.core.class_weighting) idêntico pros dois hospitais, sem
precisar sincronizar .env manualmente entre BPSP e HSL. Ver docs/pesquisa_baseline_
implementacao_fontes_bibliograficas.md, seção 14.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_server.config_loader import PostgreSQLConfigLoader


def _make_loader():
    loader = PostgreSQLConfigLoader.__new__(PostgreSQLConfigLoader)
    loader._engine = MagicMock()
    conn = MagicMock()
    loader._engine.connect.return_value.__enter__.return_value = conn
    loader._engine.connect.return_value.__exit__.return_value = False
    loader._engine.begin.return_value.__enter__.return_value = conn
    loader._engine.begin.return_value.__exit__.return_value = False
    return loader, conn


class TestPostgresConfigLoaderClassWeightOverrides:
    def test_load_deserializes_jsonb_back_to_json_string(self):
        loader, conn = _make_loader()
        row = {
            "proximal_mu": None,
            "pause_seconds": 0.0,
            "stop": False,
            # psycopg2 já entrega dict pra coluna JSONB
            "class_weight_overrides_json": {"curado_internado": 25.0},
        }
        conn.execute.return_value.mappings.return_value.first.return_value = row

        result = loader.load(round_num=1)

        assert result["class_weight_overrides_json"] == '{"curado_internado": 25.0}'

    def test_load_omits_key_when_null(self):
        loader, conn = _make_loader()
        row = {
            "proximal_mu": None, "pause_seconds": 0.0, "stop": False,
            "class_weight_overrides_json": None,
        }
        conn.execute.return_value.mappings.return_value.first.return_value = row

        result = loader.load(round_num=1)

        assert "class_weight_overrides_json" not in result

    def test_load_no_row_returns_empty_dict(self):
        loader, conn = _make_loader()
        conn.execute.return_value.mappings.return_value.first.return_value = None

        assert loader.load(round_num=1) == {}

    def test_write_passes_json_string_and_casts_to_jsonb(self):
        loader, conn = _make_loader()
        loader.write({"class_weight_overrides_json": '{"curado_internado": 25.0}'})

        _, params = conn.execute.call_args[0]
        assert params["class_weight_overrides_json"] == '{"curado_internado": 25.0}'
        sql_text = str(conn.execute.call_args[0][0])
        assert "cast(:class_weight_overrides_json as jsonb)" in sql_text

    def test_write_without_override_sends_none(self):
        """Consistente com o comportamento já existente de proximal_mu — write() sem o
        campo reseta pra NULL (linha singleton, sem COALESCE), não preserva o anterior."""
        loader, conn = _make_loader()
        loader.write({"stop": True})

        _, params = conn.execute.call_args[0]
        assert params["class_weight_overrides_json"] is None

    def test_clear_resets_class_weight_overrides(self):
        loader, conn = _make_loader()
        loader.clear()

        sql_text = str(conn.execute.call_args[0][0])
        assert "class_weight_overrides_json = NULL" in sql_text
