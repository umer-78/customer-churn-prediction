"""Customer churn prediction pipeline."""

from .features import FEATURE_COLUMNS, TARGET, build_preprocessor
from .model import evaluate, load_model, train_all

__all__ = ["FEATURE_COLUMNS", "TARGET", "build_preprocessor", "evaluate", "load_model", "train_all"]
__version__ = "1.0.0"
