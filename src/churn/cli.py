"""churn: train models, score customers and inspect the data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .data import make_dataset
from .features import TARGET
from .model import load_model, predict_frame, save_model, top_features, train_all

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "customers.csv"
DEFAULT_MODEL = ROOT / "models" / "churn-model.joblib"
HEADER = (f"{'model':<22} {'accuracy':>8} {'precision':>9} {'recall':>7} {'f1':>6} "
          f"{'roc_auc':>7} {'pr_auc':>7} {'brier':>6} {'cv_auc':>6}")


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"{path} not found — run `churn generate` first.", file=sys.stderr)
        raise SystemExit(2)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        # The reader went away. Point stdout at the void so the interpreter's
        # own flush on exit does not raise the same error again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="churn", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic customer dataset")
    g.add_argument("-n", type=int, default=7000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("-o", "--output", type=Path, default=DEFAULT_DATA)

    t = sub.add_parser("train", help="train and compare models, save the best")
    t.add_argument("-d", "--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("-o", "--model", type=Path, default=DEFAULT_MODEL)
    t.add_argument("--cv", type=int, default=0, help="also run k-fold cross-validation")
    t.add_argument("--charts", type=Path, default=None, help="write report charts to this folder")

    p = sub.add_parser("predict", help="score customers from a CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top", type=int, default=10, help="print the N riskiest customers")

    s = sub.add_parser("summary", help="describe the dataset")
    s.add_argument("-d", "--data", type=Path, default=DEFAULT_DATA)

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        df = make_dataset(args.n, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"wrote {len(df):,} rows to {args.output} ({df[TARGET].mean():.1%} churned)")
        return 0

    if args.cmd == "summary":
        df = _load(args.data)
        print(f"{len(df):,} customers, churn rate {df[TARGET].mean():.1%}")
        print(f"missing values: {int(df.isna().sum().sum())}")
        print("\nchurn rate by contract:")
        print(df.groupby('contract')[TARGET].agg(['mean', 'count'])
               .rename(columns={'mean': 'churn_rate'}).sort_values('churn_rate', ascending=False)
               .to_string(float_format=lambda v: f"{v:.1%}"))
        print("\nchurn rate by tenure bucket:")
        buckets = pd.cut(df.tenure_months, [-1, 6, 12, 24, 48, 1000], labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"])
        print(df.groupby(buckets, observed=True)[TARGET].mean().to_string(float_format=lambda v: f"{v:.1%}"))
        return 0

    if args.cmd == "train":
        df = _load(args.data)
        scores, fitted, (_, X_test, _, y_test) = train_all(df, cv=args.cv)
        print(HEADER)
        print("-" * len(HEADER))
        for s in scores:
            print(s.row())
        best_name = max((s for s in scores if "baseline" not in s.name), key=lambda s: s.pr_auc).name
        best = fitted[best_name]
        imp = top_features(best)
        print(f"\nbest by PR-AUC: {best_name}")
        print("\ntop features:")
        print(imp.to_string(float_format=lambda v: f"{v:.3f}"))
        save_model(best, args.model)
        print(f"\nsaved {args.model}")
        if args.charts:
            from .plots import save_report_charts
            for p in save_report_charts(fitted, X_test, y_test, imp, args.charts):
                print(f"chart: {p}")
        return 0

    df = _load(args.csv)
    model = load_model(args.model) if args.model.exists() else None
    if model is None:
        print(f"{args.model} not found — run `churn train` first.", file=sys.stderr)
        return 2
    out = predict_frame(model, df, args.threshold)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out):,} predictions to {args.output}")
    print(f"\n{args.top} highest-risk customers:")
    print(out.sort_values("churn_probability", ascending=False).head(args.top).to_string(index=False))
    print(f"\n{int(out.predicted_churn.sum()):,} of {len(out):,} flagged at threshold {args.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
