"""Tests for the backtest evaluator."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluate.backtest import _compute_features, _fit_logistic, _predict_logistic


def test_fit_logistic_converges():
    np.random.seed(42)
    features = np.random.randn(200, 2)
    w_true = np.array([1.0, -0.5])
    z = features @ w_true + 0.1
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(float)

    w, b = _fit_logistic(features, y, lr=0.1, epochs=1000)
    p = _predict_logistic(features, w, b)
    acc = np.mean((p > 0.5) == y)
    assert acc > 0.8


def test_predict_logistic_bounds():
    features = np.array([[100.0], [-100.0], [0.0]])
    w = np.array([1.0])
    b = 0.0
    p = _predict_logistic(features, w, b)
    assert all(0 < pi < 1 for pi in p)


def test_compute_features_empty_teams():
    t1 = pd.DataFrame()
    t2 = pd.DataFrame({"bat_rating": [50.0], "bowl_rating": [-10.0]})
    assert _compute_features(t1, t2) is None
    assert _compute_features(t2, t1) is None


def test_compute_features_normal():
    t1 = pd.DataFrame(
        {
            "bat_rating": [100.0, 90.0, 80.0, 70.0],
            "bowl_rating": [-10.0, -20.0, -30.0, -40.0],
        }
    )
    t2 = pd.DataFrame(
        {
            "bat_rating": [60.0, 50.0, 40.0, 30.0],
            "bowl_rating": [-15.0, -25.0, -35.0, -45.0],
        }
    )
    feats = _compute_features(t1, t2)
    assert feats is not None
    assert feats["bat_diff"] == pytest.approx(40.0)
    assert feats["n1"] == 4
    assert feats["n2"] == 4
