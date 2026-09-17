# Model card: customer churn classifier

## Overview

- **Task:** predict whether a telecom customer will churn in the next month.
- **Output:** a probability, a risk band (low / medium / high) and a 0/1 label at a chosen threshold.
- **Selected model:** class-weighted logistic regression (best PR-AUC). Gradient
  boosting is kept in the comparison and is better calibrated (lower Brier).
- **Training data:** 5,600 synthetic customers (80% of 7,000), 38.9% churn rate.

## Metrics (held-out 20%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Baseline (always "stay") | 0.611 | 0.000 | 0.000 | 0.000 | 0.500 | 0.389 | 0.389 |
| Logistic regression | 0.688 | 0.584 | 0.690 | 0.632 | 0.769 | 0.684 | 0.196 |
| Random forest | 0.685 | 0.585 | 0.655 | 0.618 | 0.753 | 0.667 | 0.198 |
| Gradient boosting | 0.718 | 0.661 | 0.565 | 0.609 | 0.770 | 0.681 | 0.187 |

Five-fold cross-validated ROC-AUC on the full dataset: 0.787 / 0.771 / 0.778.

## Intended use

Ranking customers for a retention campaign, and teaching an end-to-end ML workflow.

## Out of scope

- Any decision that affects a person directly (pricing, credit, service refusal).
- Any use on real customers without retraining on real data and re-measuring.

## Limitations

- The data is synthetic. Real churn has seasonality, competitor effects, price
  changes and data-quality problems this generator does not reproduce.
- The label carries deliberate noise, so roughly 0.80 ROC-AUC is the ceiling.
- The model is correlational. "Two-year contract" lowers predicted churn; moving
  someone onto one does not automatically lower their real churn.
- No fairness audit has been run. `senior_citizen` is a feature, and using an age
  proxy to decide who gets a discount needs a policy review first.

## Threshold

The default 0.5 is not special. At 0.4 you catch more churners and waste more
discounts. Pick it from the cost of a false alarm against the value of a saved
customer.

## Retraining

`churn train` retrains from scratch and rewrites `models/churn-model.joblib`. The
model is not committed: it is a few seconds to rebuild and joblib files are tied
to the scikit-learn version that wrote them.
