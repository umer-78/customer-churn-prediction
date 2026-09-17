"""Customer churn prediction pipeline."""

from .features import build_preprocessor, FEATURE_COLUMNS, TARGET
from .model import evaluate, load_model, train_all

__all__ = ["FEATURE_COLUMNS", "TARGET", "build_preprocessor", "evaluate", "load_model", "train_all"]
__version__ = "1.0.0"
