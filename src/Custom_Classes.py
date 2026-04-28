"""
Reusable sklearn-compatible transformers for the IEEE-CIS fraud-detection
project. Each transformer wraps one preprocessing step that is currently
done inline in `Project_Final.ipynb` (cells 22-29), so they can be
plugged into a sklearn Pipeline if/when this project moves to an
end-to-end pipeline that the SageMaker endpoint can preprocess on its
own.

Currently the saved model (`finalized_loan_model.joblib`) is a plain
classifier — these classes are NOT yet stitched into a Pipeline. To do
that, replace the inline cells with something like:

    from sklearn.pipeline import Pipeline
    from src.Custom_Classes import (
        DropHighMissingCols, TransactionFeatureEngineer,
        LabelEncodeCategoricals, DropHighCorrelation,
    )

    pipe = Pipeline([
        ("drop_missing", DropHighMissingCols(threshold=0.7)),
        ("feature_eng",  TransactionFeatureEngineer()),
        ("encode_cat",   LabelEncodeCategoricals()),
        ("drop_corr",    DropHighCorrelation(threshold=0.95)),
        ("clf",          gb_tuned),   # or rf_tuned
    ])
    pipe.fit(X_train_raw, y_train)
    joblib.dump(pipe, "finalized_loan_model.joblib")
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


# ─────────────────────────────────────────────────────────────────────
# 1. Drop columns with too many missing values (mirrors notebook cell 22)
# ─────────────────────────────────────────────────────────────────────
class DropHighMissingCols(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-value ratio exceeds `threshold`."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.threshold].index.tolist()
        self.cols_to_keep_ = [c for c in X.columns if c not in self.cols_to_drop_]
        return self

    def transform(self, X):
        check_is_fitted(self, "cols_to_keep_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[[c for c in self.cols_to_keep_ if c in X.columns]].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cols_to_keep_")
        return np.array(self.cols_to_keep_)


# ─────────────────────────────────────────────────────────────────────
# 2. Transaction-level feature engineering (mirrors notebook cell 25)
# ─────────────────────────────────────────────────────────────────────
class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineered features for IEEE-CIS transaction data.

    Adds:
      * Transaction_hour, Transaction_day  — derived from TransactionDT
      * TransactionAmt_log                 — log1p(TransactionAmt)
      * card1_TransAmt_mean / std / count  — per-card1 amount stats
      * TransAmt_deviation                 — TransactionAmt - card1 mean

    Per-card1 stats are learned at fit time and merged in at transform
    time, so the same lookup table is used for training and inference.
    Unseen card1 values fall back to global stats.
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)

        if "card1" in X.columns and "TransactionAmt" in X.columns:
            stats = X.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count"])
            stats.columns = [
                "card1_TransAmt_mean",
                "card1_TransAmt_std",
                "card1_TransAmt_count",
            ]
            self.card1_stats_     = stats
            self.global_amt_mean_ = float(X["TransactionAmt"].mean())
            self.global_amt_std_  = float(X["TransactionAmt"].std())
        else:
            self.card1_stats_     = None
            self.global_amt_mean_ = 0.0
            self.global_amt_std_  = 0.0
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        Xc = X.copy()

        # Time-based features
        if "TransactionDT" in Xc.columns:
            Xc["Transaction_hour"] = np.floor(Xc["TransactionDT"] / 3600) % 24
            Xc["Transaction_day"]  = np.floor(Xc["TransactionDT"] / (3600 * 24))

        # Log of transaction amount (heavy right-skew in the raw data)
        if "TransactionAmt" in Xc.columns:
            Xc["TransactionAmt_log"] = np.log1p(Xc["TransactionAmt"])

        # Per-card1 statistics
        if self.card1_stats_ is not None and "card1" in Xc.columns:
            Xc = Xc.merge(self.card1_stats_, how="left",
                          left_on="card1", right_index=True)
            Xc["card1_TransAmt_mean"]  = Xc["card1_TransAmt_mean"].fillna(self.global_amt_mean_)
            Xc["card1_TransAmt_std"]   = Xc["card1_TransAmt_std"].fillna(self.global_amt_std_)
            Xc["card1_TransAmt_count"] = Xc["card1_TransAmt_count"].fillna(1)
            if "TransactionAmt" in Xc.columns:
                Xc["TransAmt_deviation"] = Xc["TransactionAmt"] - Xc["card1_TransAmt_mean"]

        # Final NaN/Inf cleanup (cells 27 / 28 / 29 in the notebook)
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        Xc.fillna(0, inplace=True)
        return Xc

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        added = [
            "Transaction_hour", "Transaction_day", "TransactionAmt_log",
            "card1_TransAmt_mean", "card1_TransAmt_std",
            "card1_TransAmt_count", "TransAmt_deviation",
        ]
        return np.array(list(self.feature_names_in_) + added)


# ─────────────────────────────────────────────────────────────────────
# 3. Label-encode object/string columns (mirrors notebook cell 26)
# ─────────────────────────────────────────────────────────────────────
class LabelEncodeCategoricals(BaseEstimator, TransformerMixin):
    """Label-encode every object-dtype column, robustly.

    Unlike sklearn's stock `LabelEncoder`, this transformer survives
    unseen categories at inference time by mapping them to a sentinel
    integer (default -1) instead of raising.
    """

    UNSEEN = -1

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)
        self.encoders_ = {}
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        check_is_fitted(self, "encoders_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        Xc = X.copy()
        for col, le in self.encoders_.items():
            if col not in Xc.columns:
                continue
            classes = set(le.classes_)
            col_str = Xc[col].astype(str)
            mask = col_str.isin(classes)
            encoded = np.full(len(col_str), self.UNSEEN, dtype=int)
            if mask.any():
                encoded[mask.values] = le.transform(col_str[mask])
            Xc[col] = encoded
        return Xc

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoders_")
        return self.feature_names_in_


# ─────────────────────────────────────────────────────────────────────
# 4. Drop highly-correlated numeric columns (optional refinement)
# ─────────────────────────────────────────────────────────────────────
class DropHighCorrelation(BaseEstimator, TransformerMixin):
    """Drop one column from each pair whose |Pearson correlation| > threshold.

    Useful for IEEE-CIS where many V- and C-columns are near-duplicates.
    The notebook does not currently apply this — it's available for the
    improved-Pipeline retraining path described in this module's docstring.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)

        numeric = X.select_dtypes(include=[np.number])
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.cols_to_drop_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.cols_to_keep_ = [c for c in X.columns if c not in self.cols_to_drop_]
        return self

    def transform(self, X):
        check_is_fitted(self, "cols_to_keep_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[[c for c in self.cols_to_keep_ if c in X.columns]].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cols_to_keep_")
        return np.array(self.cols_to_keep_)
