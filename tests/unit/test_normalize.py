"""Unit tests for ml/ingest/normalize.py."""

from __future__ import annotations

import pytest

from ml.ingest.normalize import _infer_format, _slugify, encode_outcome


class TestEncodeOutcome:
    def test_wide_takes_priority(self):
        assert encode_outcome(0, is_wide=True, is_noball=False, is_wicket=False) == "wide"

    def test_noball_after_wide(self):
        assert encode_outcome(0, is_wide=False, is_noball=True, is_wicket=False) == "noball"

    def test_wicket(self):
        assert encode_outcome(0, is_wide=False, is_noball=False, is_wicket=True) == "wicket"

    def test_dot(self):
        assert encode_outcome(0, is_wide=False, is_noball=False, is_wicket=False) == "dot"

    def test_four(self):
        assert encode_outcome(4, is_wide=False, is_noball=False, is_wicket=False) == "4"

    def test_six(self):
        assert encode_outcome(6, is_wide=False, is_noball=False, is_wicket=False) == "6"

    def test_five_maps_to_six(self):
        # 5 is rare (overthrow); we map it to '6'
        assert encode_outcome(5, is_wide=False, is_noball=False, is_wicket=False) == "6"


class TestInferFormat:
    def test_t20(self):
        assert _infer_format("T20") == "t20"

    def test_odi(self):
        assert _infer_format("ODI") == "odi"

    def test_test(self):
        assert _infer_format("Test") == "test"

    def test_unknown_defaults_to_t20(self):
        assert _infer_format("something") == "t20"


class TestSlugify:
    def test_basic(self):
        assert _slugify("Eden Gardens") == "eden_gardens"

    def test_special_chars(self):
        assert _slugify("Gaddafi Stadium, Lahore") == "gaddafi_stadium_lahore"
