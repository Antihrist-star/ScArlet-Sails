import pandas as pd

from scripts.train_xgboost_v3 import temporal_split


def test_temporal_split_handles_tzaware_index():
    idx = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    df = pd.DataFrame({"open": range(6)}, index=idx)

    train, val, test = temporal_split(
        df,
        train_start="2021-01-01",
        train_end="2021-01-03",
        val_end="2021-01-05",
        test_end=None,
    )

    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 2
    assert train.index.tz == idx.tz
    assert val.index.tz == idx.tz
    assert test.index.tz == idx.tz
