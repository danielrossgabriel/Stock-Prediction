"""Feature extraction utilities for the HW6 Streamlit app.

Builds the 12-column feature vector that the HW6 regression model was
trained on in the notebook:

    sentiment_textblob, sentiment_lag1, sentiment_lag2, sentiment_lag3,
    AAPL, MSFT, AMZN, GOOG, WMT, JPM, TSLA, ADBE

Target ticker is NFLX (excluded from the other-ticker columns).
"""

import os
import pandas as pd


# Columns the HW6 model expects, in order
TARGET_TICKER = 'NFLX'
OTHER_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'WMT', 'JPM', 'TSLA', 'ADBE']
SENTIMENT_COLS = ['sentiment_textblob', 'sentiment_lag1', 'sentiment_lag2', 'sentiment_lag3']
FEATURE_COLUMNS = SENTIMENT_COLS + OTHER_TICKERS


def _find_sentiment_csv():
    """Look for DataWithSentimentsResults_HW.csv in a few plausible spots."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    candidates = [
        os.path.join(project_root, 'DataWithSentimentsResults_HW.csv'),
        os.path.join(project_root, 'data', 'DataWithSentimentsResults_HW.csv'),
        os.path.join(project_root, 'HW6', 'DataWithSentimentsResults_HW.csv'),
        os.path.join(current_dir, 'DataWithSentimentsResults_HW.csv'),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _empty_feature_frame():
    """Fallback: a single-row frame of zeros with the right columns."""
    return pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)


def extract_features():
    """Return a single-row DataFrame with the 12 HW6 features.

    Tries to pre-fill with the most recent values from
    DataWithSentimentsResults_HW.csv; if the file isn't available, returns
    a zero-filled row so the app still launches.
    """
    csv_path = _find_sentiment_csv()
    if csv_path is None:
        return _empty_feature_frame()

    try:
        sent = pd.read_csv(csv_path, sep='|')
    except Exception:
        return _empty_feature_frame()

    if not {'ticker', 'date', 'sentiment_textblob'}.issubset(sent.columns):
        return _empty_feature_frame()

    sent['date'] = pd.to_datetime(sent['date'], errors='coerce')
    sent = sent.dropna(subset=['date'])

    # Average sentiment per (ticker, date)
    daily = (
        sent[['ticker', 'date', 'sentiment_textblob']]
        .groupby(['ticker', 'date'])
        .mean()
        .reset_index()
    )

    # Target series: NFLX sentiment over time
    target = (
        daily[daily['ticker'] == TARGET_TICKER]
        .sort_values('date')
        .reset_index(drop=True)
    )
    if target.empty:
        return _empty_feature_frame()

    target['sentiment_lag1'] = target['sentiment_textblob'].shift(1)
    target['sentiment_lag2'] = target['sentiment_textblob'].shift(2)
    target['sentiment_lag3'] = target['sentiment_textblob'].shift(3)

    # Other tickers' sentiment, pivoted wide
    others = (
        daily[daily['ticker'] != TARGET_TICKER]
        .pivot(index='date', columns='ticker', values='sentiment_textblob')
        .reset_index()
    )

    merged = pd.merge(target, others, on='date', how='left').ffill()

    # Make sure every expected column exists, filling with 0 if missing
    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0

    last_row = merged[FEATURE_COLUMNS].tail(1).fillna(0.0).reset_index(drop=True)
    return last_row
