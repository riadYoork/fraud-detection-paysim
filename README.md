# fraud-detection-paysim
# Fraud detection on PaySim

A machine-learning pipeline that flags fraudulent mobile-money transactions,
served through a small FastAPI endpoint. I built it to practise the full path
from raw data to a working service, and to be clear about what the results do
and do not show.

## Data

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a **synthetic**
mobile-money simulation by Edgar Lopez-Rojas (licence CC BY-SA 4.0). It has
6.36 million transactions over 743 hourly steps (about 31 days). 0.129% of them
are fraud.

All fraud in the data is in `CASH_OUT` and `TRANSFER` transactions, so the
model uses only those two types: 2.77 million rows, 0.296% fraud.

## Method

- **Time-based split.** Train on the earliest hours, validate on the next
  slice, and test on the final 20% of the timeline. A random split would let
  the model learn from the future.
- **Metrics.** Precision, recall and PR-AUC. Accuracy is not used: a model
  that never predicts fraud scores about 99.7% on this data.
- **Features.** Only information known before a transaction completes: type,
  amount, sender and receiver balances before the transaction, the amount as a
  fraction of the sender's balance, and hour of day.
- **Model.** Gradient boosting (scikit-learn `HistGradientBoostingClassifier`)
  with class weighting.
- **Threshold.** Chosen on the validation set for 95% recall, placed halfway
  between two adjacent validation scores, then applied once to the test set.

## Results

Validation PR-AUC by model and feature set:

| Model | PR-AUC |
|---|---|
| Logistic regression, raw columns | 0.535 |
| Gradient boosting, raw columns | 0.904 |
| Gradient boosting, engineered features | 0.998 |
| Gradient boosting, pre-transaction features only (final model) | 0.994 |

Final model on the test period (552,504 transactions, 4,258 of them fraud):

| Metric | Value |
|---|---|
| PR-AUC | 0.9985 |
| Precision | 0.9988 |
| Recall | 0.9643 |
| Fraud caught | about 4,106 |
| Fraud missed | about 152 |
| False alarms | about 5 |

The counts are derived from the precision and recall above. The exact figures
are in `reports/metrics.json`.

The same period scored with a one-line rule ("the amount is roughly the whole
sender balance") gives recall 0.972 and precision 0.838, with about 800 false
alarms. The model's advantage over that rule is mostly far fewer false alarms,
not extra fraud caught.

## What the model relies on

Nearly all of the model's skill comes from one feature: the amount relative to
the sender's balance. In the training period, 97.8% of fraud moves the
sender's entire balance in one transaction, against 0.15% of legitimate
transactions. Among transactions that match that pattern, fraud is larger,
goes to an empty destination account more often (65% against 29%), and is more
often a `TRANSFER` (50% against 5.5%).

With the first threshold I tried, the model missed 198 fraud cases in the test
period. Of those, 102 had an amount of exactly 10,000,000 and 13 had an
amount of 0. The 10,000,000 figure looks like a limit built into the simulator
(the largest fraud amount in the data is that value), but I have not
confirmed this from the PaySim documentation.

## Limitations

- **Synthetic data.** PaySim's fraud is simulated. Real fraudsters imitate
  normal behaviour, and these scores would not carry over to a real bank.
- **One dominant pattern.** Fraud that does not empty the account is mostly
  missed. For example, a cash-out of 10,000,000 from a balance of 15,000,000
  is not flagged, and neither is an implausible transaction far outside the
  training range (an amount of 1 billion from a balance of 100,000 scored
  close to zero). A real system would put hard rules in front of a model like
  this.
- **Shift between periods.** The fraud rate is 0.184% in training and 0.771%
  in the test period, so precision is not comparable across periods.
- **Uncalibrated scores.** Class weighting pushes scores close to 1, and many
  transactions share the same top score. Treat the output as a score, not a
  probability.
- **Threshold changed after seeing test results.** My first threshold sat
  exactly on a tied score (4,065 test rows shared it). I moved it to the
  midpoint between two adjacent validation scores, which raised test recall
  from 0.9535 to 0.9643. The test figures above are therefore slightly
  optimistic.

## API

Start the service:

```
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs and call `POST /predict`:

```json
{"step": 100, "type": "TRANSFER", "amount": 500000, "oldbalanceOrg": 500000, "oldbalanceDest": 0}
```

The response contains `fraud_score`, `flagged` and the `threshold`. Only
`CASH_OUT` and `TRANSFER` are accepted; other types, negative amounts and
missing fields are rejected with a 422 error.



## Project layout

```
notebooks/   exploration, feature engineering, model comparison
src/         features.py, train.py, api.py
tests/       unit tests for the feature code
reports/     metrics.json written by the training script
```

## Run it

Tested with Python 3.14 on Windows.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the PaySim CSV from Kaggle into `data/raw/` (the folder is not
tracked by Git), then:

```
python -m src.train
python -m pytest
```

`python -m src.train` saves the model to `models/` and the metrics to
`reports/metrics.json`.

## Data credit

PaySim by Edgar Lopez-Rojas, CC BY-SA 4.0.