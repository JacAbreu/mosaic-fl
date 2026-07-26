"""
Testes para ProductionFedProxStrategy._save_federated_training_id_marker() — grava
FEDERATED_ID_FILE (experiments/last_federated_training_id.txt) ao final do treino,
mesmo mecanismo que já existia só no Caminho A (orchestrator.py). Achado 2026-07-25:
sem isso, a API sempre carregava o último training_id do Caminho A, ignorando
qualquer treino real via SuperLink/SuperNode (Caminho B) — mesmo mais recente e
com accuracy muito melhor.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import infrastructure.mosaicfl_server.strategy.core as core_module
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(training_id=99):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy._training_id = training_id
    return strategy


class TestSaveFederatedTrainingIdMarker:
    def test_writes_training_id_to_marker_file(self, tmp_path, monkeypatch):
        marker = tmp_path / "experiments" / "last_federated_training_id.txt"
        monkeypatch.setattr(core_module, "FEDERATED_ID_FILE", marker)
        strategy = _make_strategy(training_id=70)

        strategy._save_federated_training_id_marker()

        assert marker.read_text(encoding="utf-8") == "70"

    def test_creates_parent_directory_if_missing(self, tmp_path, monkeypatch):
        marker = tmp_path / "nested" / "dir" / "last_federated_training_id.txt"
        monkeypatch.setattr(core_module, "FEDERATED_ID_FILE", marker)
        strategy = _make_strategy(training_id=5)

        strategy._save_federated_training_id_marker()

        assert marker.exists()

    def test_overwrites_stale_previous_value(self, tmp_path, monkeypatch):
        marker = tmp_path / "last_federated_training_id.txt"
        marker.write_text("65", encoding="utf-8")
        monkeypatch.setattr(core_module, "FEDERATED_ID_FILE", marker)
        strategy = _make_strategy(training_id=70)

        strategy._save_federated_training_id_marker()

        assert marker.read_text(encoding="utf-8") == "70"

    def test_never_raises_on_write_failure(self, tmp_path, monkeypatch):
        """Diretório read-only (ou qualquer falha de I/O) não deve propagar — é um
        atalho de conveniência, não uma dependência crítica do treino."""
        unwritable_dir = tmp_path / "readonly"
        unwritable_dir.mkdir(mode=0o444)
        marker = unwritable_dir / "sub" / "last_federated_training_id.txt"
        monkeypatch.setattr(core_module, "FEDERATED_ID_FILE", marker)
        strategy = _make_strategy(training_id=1)

        strategy._save_federated_training_id_marker()  # não deve levantar exceção
