"""Tests for ml/simulate/pitch.py — manual pitch-condition adjustments."""

from __future__ import annotations

import numpy as np
import pytest

from ml.simulate.pitch import OUTCOME_CLASSES, apply_pitch_adjustment

_IDX = {name: i for i, name in enumerate(OUTCOME_CLASSES)}
_BASE = np.array([0.35, 0.25, 0.08, 0.01, 0.12, 0.06, 0.08, 0.03, 0.02])


@pytest.mark.parametrize("condition", [None, "auto", "balanced"])
def test_noop_conditions_unchanged(condition):
    out = apply_pitch_adjustment(_BASE, condition)
    assert np.allclose(out, _BASE)


def test_output_sums_to_one():
    for condition in ["batting_friendly", "bowling_friendly", "spin_friendly", "seam_friendly"]:
        out = apply_pitch_adjustment(_BASE, condition, bowl_type="pace")
        assert abs(out.sum() - 1.0) < 1e-6


def test_batting_friendly_favors_batters():
    out = apply_pitch_adjustment(_BASE, "batting_friendly")
    assert out[_IDX["4"]] / _BASE[_IDX["4"]] > 1.0
    assert out[_IDX["wicket"]] / _BASE[_IDX["wicket"]] < 1.0


def test_bowling_friendly_favors_bowlers():
    out = apply_pitch_adjustment(_BASE, "bowling_friendly")
    assert out[_IDX["wicket"]] / _BASE[_IDX["wicket"]] > 1.0
    assert out[_IDX["4"]] / _BASE[_IDX["4"]] < 1.0


def test_spin_friendly_boosts_spinners_more_than_pace():
    spin_out = apply_pitch_adjustment(_BASE, "spin_friendly", bowl_type="off")
    pace_out = apply_pitch_adjustment(_BASE, "spin_friendly", bowl_type="pace")
    assert spin_out[_IDX["wicket"]] > pace_out[_IDX["wicket"]]


def test_seam_friendly_boosts_pace_more_than_spin():
    pace_out = apply_pitch_adjustment(_BASE, "seam_friendly", bowl_type="pace")
    spin_out = apply_pitch_adjustment(_BASE, "seam_friendly", bowl_type="leg")
    assert pace_out[_IDX["wicket"]] > spin_out[_IDX["wicket"]]


def test_type_conditional_preset_with_no_bowl_type_still_shifts_toward_bowlers():
    """A BattingCard (no single fixed opposing bowler) still gets a milder
    bowling-leaning shift, rather than silently no-op'ing."""
    out = apply_pitch_adjustment(_BASE, "spin_friendly", bowl_type=None)
    assert out[_IDX["wicket"]] > _BASE[_IDX["wicket"]]


def test_unknown_preset_is_noop():
    out = apply_pitch_adjustment(_BASE, "not_a_real_preset")
    assert np.allclose(out, _BASE)


def test_input_array_never_mutated():
    original = _BASE.copy()
    apply_pitch_adjustment(_BASE, "bowling_friendly")
    assert np.array_equal(_BASE, original)
