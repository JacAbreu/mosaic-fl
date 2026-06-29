import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.config import NUM_ROUNDS, PROXIMAL_MU, BATCH_SIZE


class TestFedConfig:

    def test_num_rounds_positive(self):
        assert NUM_ROUNDS >= 20

    def test_proximal_mu_small(self):
        assert 0 < PROXIMAL_MU < 1

    def test_batch_size_reasonable(self):
        assert BATCH_SIZE <= 32

    def test_convergence_threshold_positive(self):
        from mosaicfl.core.config import FED_CFG
        assert FED_CFG.convergence_threshold > 0

    def test_convergence_patience_positive(self):
        from mosaicfl.core.config import FED_CFG
        assert FED_CFG.convergence_patience > 0

    def test_fraction_fit_in_range(self):
        from mosaicfl.core.config import FED_CFG
        assert 0.0 < FED_CFG.fraction_fit <= 1.0

    def test_frozen_raises_on_mutation(self):
        from mosaicfl.core.config import FedConfig
        from dataclasses import FrozenInstanceError
        cfg = FedConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.num_rounds = 100

    def test_custom_instance(self):
        from mosaicfl.core.config import FedConfig
        cfg = FedConfig(num_rounds=5, batch_size=8)
        assert cfg.num_rounds == 5
        assert cfg.batch_size == 8
