import numpy as np
import pandas as pd
import pytest

from scripts.train_xgboost_v3 import sanitize_features_and_target


class TestSanitizeFeaturesAndTarget:
    def test_sanitize_inf_in_features(self):
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0],
                "feature2": [10.0, 20.0, 30.0],
            }
        )
        y = pd.Series([0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [0, 2]
        assert np.isfinite(X_clean.values).all()

    def test_sanitize_negative_inf_in_features(self):
        X = pd.DataFrame(
            {
                "feature1": [-np.inf, 2.0, 3.0],
                "feature2": [10.0, 20.0, 30.0],
            }
        )
        y = pd.Series([0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [1, 2]
        assert np.isfinite(X_clean.values).all()

    def test_sanitize_nan_in_features(self):
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": [10.0, 20.0, 30.0],
            }
        )
        y = pd.Series([0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [0, 2]
        assert not X_clean.isna().any().any()

    def test_sanitize_inf_in_target(self):
        X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        y = pd.Series([0, np.inf, 1])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [0, 2]

    def test_sanitize_nan_in_target(self):
        X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        y = pd.Series([0, np.nan, 1])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [0, 2]

    def test_sanitize_multiple_bad_rows(self):
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, np.nan],
                "feature2": [10.0, 20.0, 30.0, 40.0],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 2
        assert y_clean.index.tolist() == [0, 2]

    def test_sanitize_preserves_index(self):
        X = pd.DataFrame(
            {"feature1": [1.0, np.inf, 3.0]}, index=pd.Index(["a", "b", "c"])
        )
        y = pd.Series([0, 1, 0], index=pd.Index(["a", "b", "c"]))

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert list(X_clean.index) == ["a", "c"]
        assert list(y_clean.index) == ["a", "c"]

    def test_sanitize_no_inf_or_nan_remain(self):
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0],
                "feature2": [10.0, 20.0, np.nan],
            }
        )
        y = pd.Series([0, 1, 0])

        X_clean, _ = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert np.isfinite(X_clean.values).all()
        assert not X_clean.isna().any().any()

    def test_sanitize_raises_when_bad_ratio_exceeds_threshold(self):
        """Function should raise ValueError if bad ratio > max_bad_ratio."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.inf, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        with pytest.raises(ValueError) as excinfo:
            sanitize_features_and_target(X, y, context="test")

        msg = str(excinfo.value)
        assert "[sanitize_features]" in msg
        assert "exceeds the threshold" in msg
