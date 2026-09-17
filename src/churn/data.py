"""Synthetic telecom-style customer dataset.

Real churn datasets cannot be redistributed, so this generates one with the same
shape and the relationships a telecom analyst would expect: month-to-month
contracts, short tenure, high monthly charges and recent support calls raise the
risk of churn, while long tenure and yearly contracts lower it.

The label is drawn from a logistic model plus noise, so no model can reach 100%
accuracy — which is what makes it useful for comparing models honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fibre optic", "None"]
PAYMENT = ["Bank transfer", "Credit card", "Electronic cheque", "Post"]


def make_dataset(n: int = 7000, seed: int = 42, missing_rate: float = 0.02) -> pd.DataFrame:
    """Return a customer table with a `churned` column (1 = left in the next month)."""
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(2.0, 9.0, n), 0, 72).round().astype(int)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.35, 0.45, 0.20])
    monthly = np.where(internet == "Fibre optic", rng.normal(85, 18, n),
                       np.where(internet == "DSL", rng.normal(58, 14, n), rng.normal(22, 7, n)))
    monthly = np.clip(monthly, 15, 130).round(2)
    support_calls = rng.poisson(np.where(internet == "Fibre optic", 1.3, 0.7), n)
    late_payments = rng.poisson(0.5, n)
    paperless = rng.random(n) < 0.6
    payment = rng.choice(PAYMENT, n, p=[0.22, 0.25, 0.35, 0.18])
    senior = rng.random(n) < 0.16
    dependents = rng.random(n) < 0.3
    extra_services = rng.integers(0, 6, n)
    total = (monthly * np.maximum(tenure, 1) * rng.normal(1.0, 0.03, n)).round(2)

    # log-odds of churn
    z = (
        -1.15
        - 0.045 * tenure
        + 0.016 * (monthly - 60)
        + 0.42 * support_calls
        + 0.30 * late_payments
        + np.where(contract == "Month-to-month", 1.25, np.where(contract == "One year", 0.15, -0.65))
        + np.where(internet == "Fibre optic", 0.55, np.where(internet == "None", -0.35, 0.0))
        + np.where(payment == "Electronic cheque", 0.45, 0.0)
        + 0.25 * senior
        - 0.30 * dependents
        - 0.10 * extra_services
        + 0.20 * paperless
        + rng.normal(0, 0.55, n)  # unexplained variation: nobody can predict this part
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_calls_6m": support_calls,
        "late_payments_12m": late_payments,
        "extra_services": extra_services,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "paperless_billing": paperless,
        "senior_citizen": senior,
        "has_dependents": dependents,
        "churned": churn,
    })

    # A real export always has some holes; the pipeline has to cope with them.
    if missing_rate:
        for col in ("total_charges", "payment_method"):
            idx = rng.choice(n, int(n * missing_rate), replace=False)
            df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "customers.csv"
    out.parent.mkdir(exist_ok=True)
    df = make_dataset()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out} ({df.churned.mean():.1%} churn)")
