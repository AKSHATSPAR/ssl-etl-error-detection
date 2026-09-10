"""Tests for windowing and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from etlguard.data.dataset import TEST, TRAIN, ETLData, ETLWindowDataset, RobustScaler


def test_robust_scaler_is_immune_to_a_few_extreme_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(1000, 3, 4)).astype(np.float32)
    X[:5] = 500.0                      # contamination
    s = RobustScaler.fit(X)
    assert np.all(np.abs(s.center) < 0.3), "median must ignore the outliers"
    assert np.all(np.abs(s.scale - 1.0) < 0.3), "IQR-based scale must stay ~1"


def test_robust_scaler_handles_constant_features():
    X = np.zeros((100, 2, 3), dtype=np.float32)
    z = RobustScaler.fit(X).transform(X)
    assert np.isfinite(z).all()
    assert np.allclose(z, 0.0)


def test_scaler_roundtrips_through_dict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2, 3)).astype(np.float32)
    s = RobustScaler.fit(X)
    s2 = RobustScaler.from_dict(s.to_dict())
    assert np.allclose(s.transform(X), s2.transform(X))
