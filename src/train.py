import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)

from src.features import PRE_TX_FEATURES, RISKY_TYPES, build_features

ROOT = Path(__file__).resolve().parent.parent
TARGET_RECALL = 0.95


def main() -> None:
    csv_path = next((ROOT / "data" / "raw").glob("*.csv"))
    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(RISKY_TYPES)]
    df = build_features(df)

    cutoff = df["step"].quantile(0.8)
    val_cutoff = df.loc[df["step"] <= cutoff, "step"].quantile(0.85)
    train = df[df["step"] <= val_cutoff]
    val = df[(df["step"] > val_cutoff) & (df["step"] <= cutoff)]
    test = df[df["step"] > cutoff]

    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    model.fit(train[PRE_TX_FEATURES], train["isFraud"])

    val_scores = model.predict_proba(val[PRE_TX_FEATURES])[:, 1]
    prec, rec, thr = precision_recall_curve(val["isFraud"], val_scores)
    i = np.where(rec[:-1] >= TARGET_RECALL)[0][-1]
       # Midpoint to the next lower distinct validation score, so the threshold
       # does not sit exactly on a tied score value
    threshold = float((thr[i] + thr[i - 1]) / 2) if i > 0 else float(thr[i])

    test_scores = model.predict_proba(test[PRE_TX_FEATURES])[:, 1]
    preds = test_scores >= threshold
    metrics = {
        "threshold": threshold,
        "test_pr_auc": float(average_precision_score(test["isFraud"], test_scores)),
        "test_precision": float(precision_score(test["isFraud"], preds)),
        "test_recall": float(recall_score(test["isFraud"], preds)),
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "test_fraud_cases": int(test["isFraud"].sum()),
    }

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "model.joblib")
    (ROOT / "models" / "threshold.json").write_text(json.dumps({"threshold": threshold}))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()