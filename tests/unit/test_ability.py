"""Unit tests for the Empirical Bayes ability model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.models.ability.train import (
    classify_role,
    compute_group_priors,
    shrink,
    apply_shrinkage,
    BAT_METRICS,
    BOWL_METRICS,
)


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

class TestClassifyRole:
    def test_pure_bat(self):
        assert classify_role(bat_balls=100, bowl_balls=10, is_keeper=False) == "pure_bat"

    def test_pure_bowl(self):
        assert classify_role(bat_balls=10, bowl_balls=100, is_keeper=False) == "pure_bowl"

    def test_bat_allrounder(self):
        assert classify_role(bat_balls=100, bowl_balls=50, is_keeper=False) == "bat_allrounder"

    def test_bowl_allrounder(self):
        assert classify_role(bat_balls=50, bowl_balls=100, is_keeper=False) == "bowl_allrounder"

    def test_keeper_bat(self):
        assert classify_role(bat_balls=100, bowl_balls=0, is_keeper=True) == "keeper_bat"

    def test_keeper_needs_bat_volume(self):
        # Keeper with very few batting balls falls to fallback
        assert classify_role(bat_balls=10, bowl_balls=5, is_keeper=True) == "pure_bat"

    def test_fallback_bat_dominant(self):
        assert classify_role(bat_balls=20, bowl_balls=10, is_keeper=False) == "pure_bat"

    def test_fallback_bowl_dominant(self):
        assert classify_role(bat_balls=5, bowl_balls=20, is_keeper=False) == "pure_bowl"

    def test_fallback_equal(self):
        # Equal balls -> defaults to pure_bat
        assert classify_role(bat_balls=15, bowl_balls=15, is_keeper=False) == "pure_bat"

    def test_nan_bat_balls(self):
        assert classify_role(bat_balls=float("nan"), bowl_balls=100, is_keeper=False) == "pure_bowl"

    def test_nan_bowl_balls(self):
        assert classify_role(bat_balls=100, bowl_balls=float("nan"), is_keeper=False) == "pure_bat"

    def test_both_nan(self):
        assert classify_role(bat_balls=float("nan"), bowl_balls=float("nan"), is_keeper=False) == "pure_bat"

    def test_keeper_overrides_pure_bat(self):
        # Even though bat_balls>=60 and bowl_balls<30, keeper takes priority
        assert classify_role(bat_balls=80, bowl_balls=5, is_keeper=True) == "keeper_bat"


# ---------------------------------------------------------------------------
# Shrinkage math
# ---------------------------------------------------------------------------

class TestShrink:
    def test_full_data(self):
        # n >> prior_n -> rating close to player mean
        rating, unc = shrink(player_mean=150.0, n=1000.0, group_mean=120.0, prior_n=30.0)
        assert abs(rating - 150.0) < 2.0
        assert unc < 0.05

    def test_no_data(self):
        # n=0 -> rating == group mean, uncertainty == 1
        rating, unc = shrink(player_mean=np.nan, n=0.0, group_mean=120.0, prior_n=30.0)
        assert rating == 120.0
        assert unc == 1.0

    def test_equal_weight(self):
        # n == prior_n -> equal blend
        rating, unc = shrink(player_mean=150.0, n=30.0, group_mean=120.0, prior_n=30.0)
        assert rating == pytest.approx(135.0)
        assert unc == pytest.approx(0.5)

    def test_exact_formula(self):
        # rating = (n * pm + prior_n * gm) / (n + prior_n)
        rating, unc = shrink(player_mean=200.0, n=60.0, group_mean=100.0, prior_n=40.0)
        expected = (60 * 200 + 40 * 100) / (60 + 40)
        assert rating == pytest.approx(expected)
        assert unc == pytest.approx(40.0 / 100.0)

    def test_nan_player_mean(self):
        rating, unc = shrink(player_mean=np.nan, n=50.0, group_mean=130.0, prior_n=30.0)
        assert rating == 130.0
        assert unc == 1.0

    def test_negative_n_treated_as_no_data(self):
        rating, unc = shrink(player_mean=150.0, n=-5.0, group_mean=120.0, prior_n=30.0)
        assert rating == 120.0
        assert unc == 1.0


# ---------------------------------------------------------------------------
# Group priors
# ---------------------------------------------------------------------------

class TestComputeGroupPriors:
    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "role": ["pure_bat", "pure_bat", "pure_bat", "pure_bowl", "pure_bowl"],
            "format": ["t20", "t20", "t20", "t20", "t20"],
            "bat_sr_wtd": [130.0, 140.0, 150.0, 90.0, 100.0],
            "bat_avg_wtd": [25.0, 30.0, 35.0, 12.0, 15.0],
            "bowl_econ_wtd": [np.nan, np.nan, np.nan, 7.0, 8.0],
            "bowl_sr_wtd": [np.nan, np.nan, np.nan, 18.0, 20.0],
        })

    def test_returns_correct_groups(self, sample_df):
        priors = compute_group_priors(sample_df)
        assert "pure_bat" in priors
        assert "pure_bowl" in priors
        assert "t20" in priors["pure_bat"]

    def test_bat_mean(self, sample_df):
        priors = compute_group_priors(sample_df)
        bat_sr = priors["pure_bat"]["t20"]["bat_sr_wtd"]
        assert bat_sr["mean"] == pytest.approx(140.0)
        assert bat_sr["count"] == 3

    def test_bowl_mean(self, sample_df):
        priors = compute_group_priors(sample_df)
        econ = priors["pure_bowl"]["t20"]["bowl_econ_wtd"]
        assert econ["mean"] == pytest.approx(7.5)
        assert econ["count"] == 2

    def test_nan_excluded_from_count(self, sample_df):
        priors = compute_group_priors(sample_df)
        # pure_bat has no bowling data
        bowl = priors["pure_bat"]["t20"]["bowl_econ_wtd"]
        assert bowl["count"] == 0
        assert bowl["mean"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end apply_shrinkage
# ---------------------------------------------------------------------------

class TestApplyShrinkage:
    def test_high_volume_player_keeps_stats(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "snapshot_month": [pd.Timestamp("2022-06-01")],
            "format": ["t20"],
            "role": ["pure_bat"],
            "bat_balls_wtd": [500.0],
            "bat_sr_wtd": [150.0],
            "bat_avg_wtd": [40.0],
            "bowl_balls_wtd": [0.0],
            "bowl_econ_wtd": [np.nan],
            "bowl_sr_wtd": [np.nan],
        })
        priors = {"pure_bat": {"t20": {
            "bat_sr_wtd": {"mean": 120.0, "var": 100.0, "count": 50},
            "bat_avg_wtd": {"mean": 25.0, "var": 50.0, "count": 50},
            "bowl_econ_wtd": {"mean": 8.0, "var": 4.0, "count": 5},
            "bowl_sr_wtd": {"mean": 20.0, "var": 10.0, "count": 5},
        }}}
        result = apply_shrinkage(df, priors, prior_n=30.0)

        # High-volume batter: rating should be close to raw stats
        bat_expected_sr = (500 * 150 + 30 * 120) / 530
        bat_expected_avg = (500 * 40 + 30 * 25) / 530
        expected_bat = bat_expected_sr * 0.6 + bat_expected_avg * 0.4
        assert result["bat_rating"].iloc[0] == pytest.approx(expected_bat, rel=1e-4)
        assert result["bat_uncertainty"].iloc[0] < 0.1

    def test_zero_volume_gets_prior(self):
        df = pd.DataFrame({
            "player_id": ["p2"],
            "snapshot_month": [pd.Timestamp("2022-06-01")],
            "format": ["t20"],
            "role": ["pure_bowl"],
            "bat_balls_wtd": [0.0],
            "bat_sr_wtd": [np.nan],
            "bat_avg_wtd": [np.nan],
            "bowl_balls_wtd": [0.0],
            "bowl_econ_wtd": [np.nan],
            "bowl_sr_wtd": [np.nan],
        })
        priors = {"pure_bowl": {"t20": {
            "bat_sr_wtd": {"mean": 100.0, "var": 100.0, "count": 50},
            "bat_avg_wtd": {"mean": 15.0, "var": 50.0, "count": 50},
            "bowl_econ_wtd": {"mean": 7.5, "var": 4.0, "count": 50},
            "bowl_sr_wtd": {"mean": 18.0, "var": 10.0, "count": 50},
        }}}
        result = apply_shrinkage(df, priors, prior_n=30.0)

        # Zero-volume: bat_rating should be 100*0.6 + 15*0.4 = 66
        assert result["bat_rating"].iloc[0] == pytest.approx(100.0 * 0.6 + 15.0 * 0.4)
        assert result["bat_uncertainty"].iloc[0] == 1.0
        assert result["bowl_uncertainty"].iloc[0] == 1.0
