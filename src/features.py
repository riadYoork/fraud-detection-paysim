import pandas as pd

RISKY_TYPES = ["CASH_OUT", "TRANSFER"]
PRE_TX_FEATURES = ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest",
                   "amount_ratio", "hour_of_day"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the features available before a transaction completes."""
    out = df.copy()
    out["is_transfer"] = (out["type"] == "TRANSFER").astype(int)
    out["amount_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1)
    out["hour_of_day"] = out["step"] % 24
    return out