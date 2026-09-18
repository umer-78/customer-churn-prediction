"""Report charts: ROC, precision-recall, confusion matrix and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (headless: must be set before pyplot)
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

STYLE = {"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
         "axes.spines.right": False, "font.size": 9}


def save_report_charts(fitted: dict, X_test, y_test, importances, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for name, pipe in fitted.items():
            if not hasattr(pipe.named_steps["model"], "predict_proba"):
                continue
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax[0], name=name)
            PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax[1], name=name)
        ax[0].set_title("ROC curve")
        ax[1].set_title("Precision-recall")
        ax[0].plot([0, 1], [0, 1], "--", lw=0.8, color="grey")
        for a in ax:
            a.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "curves.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)

        best = max(fitted.items(), key=lambda kv: getattr(kv[1].named_steps["model"], "n_features_in_", 0))
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ConfusionMatrixDisplay.from_estimator(best[1], X_test, y_test, ax=ax, colorbar=False,
                                              display_labels=["stayed", "churned"])
        ax.set_title(f"Confusion matrix: {best[0]}")
        fig.tight_layout()
        p = out_dir / "confusion-matrix.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)

        if len(importances):
            fig, ax = plt.subplots(figsize=(6, 3.8))
            importances.sort_values().plot.barh(ax=ax, color="#2563eb")
            ax.set_title("Most important features")
            ax.set_xlabel("importance")
            fig.tight_layout()
            p = out_dir / "feature-importance.png"
            fig.savefig(p)
            plt.close(fig)
            written.append(p)
    return written
