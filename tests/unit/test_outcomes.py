import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.preprocessor.outcomes import _map_outcome


class TestMapOutcomeDefaultThreshold:
    def test_default_still_10_days(self):
        assert _map_outcome(1, 10, "Internado") == 3   # breve
        assert _map_outcome(1, 11, "Internado") == 4   # grave

    def test_non_internado_and_curado_unaffected_by_threshold(self):
        assert _map_outcome(0, 999, "Ambulatorial") == 0
        assert _map_outcome(0, 999, "Internado") == 1
        assert _map_outcome(1, 999, "Ambulatorial") == 2

    def test_unmapped_outcome_returns_minus_one(self):
        assert _map_outcome(6, 3, "Internado") == -1


class TestMapOutcomeParameterizedThreshold:
    def test_custom_threshold_moves_boundary(self):
        assert _map_outcome(1, 5, "Internado", internado_breve_max_days=5) == 3
        assert _map_outcome(1, 6, "Internado", internado_breve_max_days=5) == 4

    def test_custom_threshold_does_not_affect_other_classes(self):
        assert _map_outcome(0, 3, "Internado", internado_breve_max_days=5) == 1
        assert _map_outcome(1, 3, "Ambulatorial", internado_breve_max_days=5) == 2
