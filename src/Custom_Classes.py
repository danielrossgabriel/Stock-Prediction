import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted
from scipy.stats import skew
from gensim.models import Word2Vec


class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """Apply Yeo-Johnson to numeric columns whose absolute skew exceeds `threshold`."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns)
        self.skewed_cols_ = []
        self.pt_ = PowerTransformer(method='yeo-johnson')

        # Only look at columns that are actually numeric
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self

        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols_ = skewness[abs(skewness) > self.threshold].index.tolist()

        if self.skewed_cols_:
            self.pt_.fit(X[self.skewed_cols_])
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_copy = X.copy()

        if self.skewed_cols_:
            X_copy[self.skewed_cols_] = self.pt_.transform(X_copy[self.skewed_cols_])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return self.feature_names_in_


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Drop columns by missingness, cardinality, and low target correlation."""

    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold  # Ratio of unique values to total rows

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)

        # 1. Missing values filter
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        # 2. High cardinality filter (categoricals only)
        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        denom = max(len(X_filtered), 1)
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / denom
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        # 3. Correlation filter (numerics only, requires target)
        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            # Use a sentinel name to avoid colliding with a real column called 'target'
            temp_df['__target__'] = np.asarray(y)
            correlations = temp_df.corr()['__target__'].abs().drop('__target__')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep_ = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        check_is_fitted(self, 'features_to_keep_')
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.features_to_keep_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'features_to_keep_')
        return np.array(self.features_to_keep_)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Compute EMA / ROC / MOM / RSI / MA indicators across given windows.

    Expects a single-column input (a price series). Use a ColumnTransformer
    upstream if you need to select one column from a multi-column frame.
    """

    def __init__(self, windows=[5, 10, 20]):
        """
        Initialize with a list of windows.
        Example: FeatureEngineer(windows=[5, 14, 30])
        """
        self.windows = windows

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns)

        if X.shape[1] != 1:
            raise ValueError(
                f"FeatureEngineer expects a single-column price series, got {X.shape[1]} columns"
            )

        self.feature_names_out_ = []
        for w in self.windows:
            self.feature_names_out_.extend(
                [f'EMA_{w}', f'ROC_{w}', f'MOM_{w}', f'RSI_{w}', f'MA_{w}']
            )
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X.copy()

        # Explicit single-column extraction (don't rely on squeeze which silently
        # keeps a DataFrame if there are multiple columns)
        data = X_df.iloc[:, 0]
        X_out = pd.DataFrame(index=X_df.index)

        for w in self.windows:

            # 1. Exponential Moving Average
            X_out[f'EMA_{w}'] = data.ewm(span=w, min_periods=w).mean()

            # 2. Rate of Change
            M = data.diff(w - 1)
            N = data.shift(w - 1)
            X_out[f'ROC_{w}'] = (M / N) * 100

            # 3. Price Momentum
            X_out[f'MOM_{w}'] = data.diff(w)

            # 4. Relative Strength Index (RSI)
            delta = data.diff()
            u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
            d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
            avg_gain = u.ewm(com=w - 1, adjust=False).mean()
            # Guard against division-by-zero when there are no down moves
            avg_loss = d.ewm(com=w - 1, adjust=False).mean().replace(0, np.nan)
            rs = avg_gain / avg_loss
            X_out[f'RSI_{w}'] = 100 - (100 / (1 + rs))

            # 5. Simple Moving Average
            X_out[f'MA_{w}'] = data.rolling(w, min_periods=w).mean()

        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_out_')
        return np.array(self.feature_names_out_)


class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    """Rolling OLS of asset A on asset B; produces spread / beta / z-score features.

    Expects exactly two columns: first is asset A, second is asset B.
    """

    def __init__(self, window=60):
        self.window = window

    def fit(self, X, y=None):
        """
        Validates that the input data is sufficient for the window size.
        In scikit-learn, fit must always return self.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if X.shape[1] != 2:
            raise ValueError(
                f"PairFeatureEngineer expects 2 columns (asset A, asset B), got {X.shape[1]}"
            )
        if len(X) < self.window:
            raise ValueError(f"Data length {len(X)} is less than window size {self.window}")

        self.feature_names_in_ = np.array(X.columns)
        self.last_alpha_ = None
        self.last_beta_ = None
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        X: Expected to be a DataFrame or Array with 2 columns: [Price_A, Price_B]
        """
        check_is_fitted(self, 'is_fitted_')

        # Convert to DataFrame if input is a numpy array
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['price_a', 'price_b'])
        else:
            df = X.copy()
            df.columns = ['price_a', 'price_b']

        # 1. Compute rolling spread and beta
        regression_df = self._compute_rolling_regression(df)
        df['spread'] = regression_df['spread'].values
        df['beta'] = regression_df['beta'].values

        # 2. Derive statistics-based features
        df['z_score'] = self._calculate_z_score(df['spread'])
        df['spread_std'] = df['spread'].rolling(self.window).std()
        df['beta_stability'] = df['beta'].rolling(self.window).std()

        return df  # .dropna()

    def _compute_rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas = np.full(len(df), np.nan)

        a_vals = df['price_a'].values
        b_vals = df['price_b'].values

        for i in range(self.window, len(df)):
            y = a_vals[i - self.window:i]
            x = b_vals[i - self.window:i]
            x_with_const = sm.add_constant(x)

            try:
                model = sm.OLS(y, x_with_const).fit()
                params = np.asarray(model.params)
                alpha, beta = float(params[0]), float(params[1])
            except Exception:
                # Skip windows that fail (e.g. singular x with zero variance)
                continue

            betas[i] = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)

            # Update state for live prediction tracking
            self.last_alpha_, self.last_beta_ = alpha, beta

        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window).mean()
        rolling_std = spread_series.rolling(self.window).std()
        return (spread_series - rolling_mean) / rolling_std

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'is_fitted_')
        return np.array([
            'price_a', 'price_b', 'spread', 'beta',
            'z_score', 'spread_std', 'beta_stability'
        ])


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Train Word2Vec on a single text column and return each row's mean word vector."""

    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def _iter_texts(self, X):
        """Yield the text for each row, regardless of input type (DataFrame / ndarray / Series / list)."""
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        elif isinstance(X, np.ndarray):
            series = X[:, 0] if X.ndim == 2 else X
        else:
            series = X  # list-like
        for item in series:
            yield str(item)

    def fit(self, X, y=None):
        sentences = [text.split() for text in self._iter_texts(X)]
        self.model_ = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )
        return self

    def transform(self, X):
        check_is_fitted(self, 'model_')

        def get_mean_vector(text):
            words = text.split()
            # Only use words that exist in the Word2Vec vocabulary
            vectors = [self.model_.wv[w] for w in words if w in self.model_.wv]
            if not vectors:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)

        return np.array([get_mean_vector(text) for text in self._iter_texts(X)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'model_')
        return np.array([f'w2v_{i}' for i in range(self.vector_size)])


# --- Usage Example ---
# extractor = PairFeatureEngineer(window=60)
# features_df = extractor.fit_transform(data[['AAPL', 'MSFT']])
