import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.server_v2 import ConvergenceTracker


class TestConvergenceTracker:

    def test_converges_after_patience_stable_rounds(self):
        t = ConvergenceTracker(threshold=0.01, patience=3)
        assert not t.check(0.70)
        assert not t.check(0.71)
        assert not t.check(0.709)
        assert not t.check(0.708)
        assert t.check(0.707)

    def test_does_not_converge_with_large_deltas(self):
        t = ConvergenceTracker(threshold=0.01, patience=3)
        for acc in [0.50, 0.60, 0.55, 0.65, 0.58]:
            t.check(acc)
        assert not t.check(0.70)

    def test_convergence_round_recorded(self):
        t = ConvergenceTracker(threshold=0.05, patience=2)
        t.check(0.80); t.check(0.80); t.check(0.80)
        assert t.converged_round is not None

    def test_convergence_round_not_overwritten(self):
        t = ConvergenceTracker(threshold=0.05, patience=2)
        t.check(0.80); t.check(0.80); t.check(0.80)
        first = t.converged_round
        t.check(0.80)
        assert t.converged_round == first

    def test_reset_clears_all_state(self):
        t = ConvergenceTracker(threshold=0.01, patience=2)
        t.check(0.80); t.check(0.80); t.check(0.80)
        t.reset()
        assert t.history == []
        assert t.stable_count == 0
        assert t.converged_round is None

    def test_single_value_never_converges(self):
        t = ConvergenceTracker(threshold=0.01, patience=1)
        assert not t.check(0.80)

    def test_patience_one_requires_one_stable_round(self):
        t = ConvergenceTracker(threshold=0.01, patience=1)
        t.check(0.80)
        assert t.check(0.801)
