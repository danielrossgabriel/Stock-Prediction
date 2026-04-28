"""
Utility functions for the IEEE-CIS fraud-detection project.

Pulled out of the notebook so they can be reused by the Streamlit app,
the SageMaker inference handler, and any retraining script.
"""

import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
def sample_csv(path: str,
               sample_rate: float = 0.10,
               random_seed: int = 42) -> pd.DataFrame:
    """Read a random subsample of rows from a (potentially huge) CSV.

    Mirrors notebook cell 6 — used to sub-sample the 590k-row
    `train_transaction.csv` down to ~59k rows for tractable training.
    Uses `skiprows` so the full file never has to fit in memory.
    """
    if not (0 < sample_rate <= 1):
        raise ValueError("sample_rate must be in (0, 1]")

    with open(path, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    rng = random.Random(random_seed)
    rows_to_skip = sorted(
        rng.sample(
            range(1, total_rows + 1),
            int(total_rows * (1 - sample_rate)),
        )
    )

    df = pd.read_csv(path, skiprows=rows_to_skip)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Feature engineering helpers (use these inline OR via TransactionFeatureEngineer)
# ─────────────────────────────────────────────────────────────────────
def time_features_from_dt(transaction_dt: pd.Series) -> pd.DataFrame:
    """Return Transaction_hour and Transaction_day from `TransactionDT`.

    `TransactionDT` is seconds since a fixed reference point in the
    IEEE-CIS dataset.
    """
    return pd.DataFrame(
        {
            "Transaction_hour": np.floor(transaction_dt / 3600) % 24,
            "Transaction_day":  np.floor(transaction_dt / (3600 * 24)),
        },
        index=transaction_dt.index,
    )


def card1_amount_stats(df: pd.DataFrame,
                       card_col: str = "card1",
                       amt_col:  str = "TransactionAmt") -> pd.DataFrame:
    """Per-card mean / std / count of transaction amounts.

    Returns a DataFrame indexed by card value with three columns:
    `<card>_TransAmt_mean`, `<card>_TransAmt_std`, `<card>_TransAmt_count`.
    """
    stats = df.groupby(card_col)[amt_col].agg(["mean", "std", "count"])
    stats.columns = [
        f"{card_col}_TransAmt_mean",
        f"{card_col}_TransAmt_std",
        f"{card_col}_TransAmt_count",
    ]
    return stats


# ─────────────────────────────────────────────────────────────────────
# EDA / reporting
# ─────────────────────────────────────────────────────────────────────
def fraud_rate_by(df: pd.DataFrame,
                  group_col: str,
                  target_col: str = "isFraud",
                  top_n: Optional[int] = None) -> pd.Series:
    """Fraud rate (% fraudulent) per group, sorted descending.

    Powers the EDA charts in notebook cells 16, 17, and 19.
    """
    rate = df.groupby(group_col)[target_col].mean() * 100
    rate = rate.sort_values(ascending=False)
    return rate.head(top_n) if top_n else rate


def evaluate_classifier(name: str,
                        model,
                        X_test,
                        y_test,
                        target_names: Tuple[str, str] = ("Legitimate", "Fraudulent")) -> dict:
    """Score a fitted classifier and pretty-print a metrics summary.

    Returns a dict with accuracy, precision, recall, F1, AUC-ROC,
    and the confusion matrix.
    """
    y_pred = model.predict(X_test)
    proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "name":      name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, proba) if proba is not None else None,
        "confusion": confusion_matrix(y_test, y_pred),
    }

    print(f"\n{'=' * 70}")
    print(f"  MODEL: {name}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if metrics["auc_roc"] is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(target_names)))
    return metrics
