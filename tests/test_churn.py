import numpy as np
import pandas as pd
import pytest

from churn.data import make_dataset
from churn.features import FEATURE_COLUMNS, TARGET, add_derived, prepare
from churn.model import (
    MODELS,
    evaluate,
    load_model,
    make_pipeline,
    predict_frame,
    save_model,
    top_features,
    train_all,
)


@pytest.fixture(scope="module")
def df():
    return make_dataset(n=1500, seed=7)


def test_dataset_shape_and_balance(df):
    assert len(df) == 1500
    assert set(df[TARGET].unique()) == {0, 1}
    assert 0.2 < df[TARGET].mean() < 0.6, "churn rate should be realistic, not degenerate"
    assert df.isna().sum().sum() > 0, "the generator must include missing values"
    assert df.customer_id.is_unique


def test_dataset_is_reproducible():
    pd.testing.assert_frame_equal(make_dataset(n=200, seed=3), make_dataset(n=200, seed=3))
    assert not make_dataset(n=200, seed=3).equals(make_dataset(n=200, seed=4))


def test_known_relationships_hold(df):
    by_contract = df.groupby("contract")[TARGET].mean()
    assert by_contract["Month-to-month"] > by_contract["Two year"]
    short = df[df.tenure_months < 6][TARGET].mean()
    long = df[df.tenure_months > 48][TARGET].mean()
    assert short > long


def test_derived_features(df):
    out = add_derived(df)
    assert (out["calls_per_year"] == df["support_calls_6m"] * 2).all()
    assert out["charges_per_month_of_tenure"].notna().all(), "must survive missing total_charges"
    assert set(out["tenure_bucket"].unique()) <= {"0-6m", "6-12m", "1-2y", "2-4y", "4y+"}


def test_prepare_returns_the_agreed_columns(df):
    X = prepare(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(df)
    with pytest.raises(KeyError):
        prepare(df.drop(columns=["contract"]))


def test_pipeline_handles_missing_values(df):
    dirty = df.copy()
    dirty.loc[dirty.index[:50], "total_charges"] = np.nan
    dirty.loc[dirty.index[:50], "payment_method"] = np.nan
    pipe = make_pipeline(MODELS["logistic regression"])
    pipe.fit(prepare(dirty), dirty[TARGET])
    assert pipe.predict_proba(prepare(dirty)).shape == (len(dirty), 2)


def test_pipeline_handles_unseen_category(df):
    pipe = make_pipeline(MODELS["logistic regression"])
    pipe.fit(prepare(df), df[TARGET])
    odd = df.head(5).copy()
    odd["payment_method"] = "Crypto"  # never seen in training
    assert len(predict_frame(pipe, odd)) == 5


def test_models_beat_the_baseline(df):
    scores, fitted, _ = train_all(df, seed=1)
    by_name = {s.name: s for s in scores}
    baseline = by_name["baseline (always stay)"]
    assert baseline.roc_auc == pytest.approx(0.5, abs=0.01)
    for name in ("logistic regression", "random forest", "gradient boosting"):
        assert by_name[name].roc_auc > 0.7, f"{name} should be well above chance"
        assert by_name[name].pr_auc > baseline.pr_auc
        assert by_name[name].brier < baseline.brier


def test_evaluate_matches_hand_computed_metrics():
    y = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    prob = [0.1, 0.6, 0.8, 0.9]
    s = evaluate("t", y, pred, prob)
    assert s.accuracy == 0.75
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == 1.0
    assert s.roc_auc == 1.0
    assert s.confusion == [[1, 1], [0, 2]]


def test_top_features_names_real_columns(df):
    _, fitted, _ = train_all(df, seed=1)
    imp = top_features(fitted["random forest"], k=5)
    assert len(imp) == 5
    assert any("contract" in name for name in imp.index)


def test_predictions_and_round_trip(df, tmp_path):
    _, fitted, _ = train_all(df, seed=1)
    path = tmp_path / "m.joblib"
    save_model(fitted["gradient boosting"], path)
    out = predict_frame(load_model(path), df.head(20))
    assert list(out.columns) == ["customer_id", "churn_probability", "risk", "predicted_churn"]
    assert out.churn_probability.between(0, 1).all()
    assert set(out.risk) <= {"low", "medium", "high"}
    assert set(out.predicted_churn) <= {0, 1}


def test_cli_end_to_end(tmp_path, capsys):
    from churn.cli import main
    data = tmp_path / "c.csv"
    model = tmp_path / "m.joblib"
    assert main(["generate", "-n", "800", "-o", str(data)]) == 0
    assert main(["summary", "-d", str(data)]) == 0
    assert main(["train", "-d", str(data), "-o", str(model)]) == 0
    assert model.exists()
    out = tmp_path / "p.csv"
    assert main(["predict", str(data), "-m", str(model), "-o", str(out), "--top", "3"]) == 0
    assert len(pd.read_csv(out)) == 800
    assert "highest-risk" in capsys.readouterr().out


def test_cli_reports_missing_files(tmp_path, capsys):
    from churn.cli import main
    with pytest.raises(SystemExit):
        main(["train", "-d", str(tmp_path / "nope.csv")])
