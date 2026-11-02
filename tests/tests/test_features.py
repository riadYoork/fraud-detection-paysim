import pandas as pd

from src.features import PRE_TX_FEATURES, build_features


def test_full_drain_gives_ratio_near_one():
    df = pd.DataFrame({"step": [25], "type": ["TRANSFER"], "amount": [999.0],
                       "oldbalanceOrg": [999.0], "oldbalanceDest": [0.0]})
    out = build_features(df)
    assert abs(out.loc[0, "amount_ratio"] - 0.999) < 1e-9
    assert out.loc[0, "is_transfer"] == 1
    assert out.loc[0, "hour_of_day"] == 1


def test_all_model_features_are_created():
    df = pd.DataFrame({"step": [1], "type": ["CASH_OUT"], "amount": [10.0],
                       "oldbalanceOrg": [50.0], "oldbalanceDest": [5.0]})
    out = build_features(df)
    assert all(col in out.columns for col in PRE_TX_FEATURES)