# engineering new features
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from project_config import PREDICTION_HORIZON_MONTHS

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    _10Y_2Y_SPREAD_COL = 'yield_spread_10Y_2Y'

    def __init__(self, target_col=None, prediction_horizon_months = PREDICTION_HORIZON_MONTHS, lag_periods = [3, 6], rolling_windows = [3, 6]):
        self.target_col = target_col
        self.prediction_horizon_months = prediction_horizon_months
        self.features_to_lag = []
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.exclude_cols = []

    def _add_yield_spread_features(self, df):
        try:
            if '10Y_treasury_yield' in df.columns and '2Y_treasury_yield' in df.columns:
                df[self._10Y_2Y_SPREAD_COL] = df['10Y_treasury_yield'] - df['2Y_treasury_yield']
                logger.info(f"Created yield spread feature: {self._10Y_2Y_SPREAD_COL}")
            else:
                logger.warning("Missing '10Y_treasury_yield' or '2Y_treasury_yield' for 10Y-2Y spread.")
        except Exception as e:
            logger.error(f"Error creating yield spread features: {e}", exc_info=True)

    def fit(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        numeric_cols = X.select_dtypes(include = np.number).columns.tolist()
        self.features_to_lag = [col for col in numeric_cols if col not in self.exclude_cols]
        logger.info(f"[FeatureEngineer] Identified base features for lagging/rolling: {self.features_to_lag}")
        return self

    def transform(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X index must be a DatetimeIndex for time series features.")

        X_transformed = X.copy()
        self._add_yield_spread_features(X_transformed)

        features = self.features_to_lag[:]
        if self._10Y_2Y_SPREAD_COL in X_transformed.columns:
            features.append(self._10Y_2Y_SPREAD_COL)

        for col in features:
            for lag in self.lag_periods:
                X_transformed[f"{col}_lag{lag}"] = X_transformed[col].shift(lag)

        for col in features:
            for window in self.rolling_windows:
                X_transformed[f"{col}_roll_mean{window}"] = X_transformed[col].rolling(window = window).mean().shift(1)
                X_transformed[f"{col}_roll_std{window}"] = X_transformed[col].rolling(window = window).std().shift(1)

        y_aligned = None
        if y is not None:
            y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y.copy()
            y_aligned = y_series.shift(-self.prediction_horizon_months)
            logger.info(f"Shifted target y by -{self.prediction_horizon_months} periods.")

        initial_rows = X_transformed.shape[0]
        if y_aligned is not None:
            combined = pd.concat([X_transformed, y_aligned.rename(self.target_col)], axis = 1)
            cleaned = combined.dropna()
            X_transformed = cleaned.drop(columns = [self.target_col])
            y_aligned = cleaned[self.target_col]
        else:
            X_transformed = X_transformed.dropna()

        rows_dropped = initial_rows - X_transformed.shape[0]
        logger.info(f"Dropped {rows_dropped} rows after feature engineering.")

        if y_aligned is not None:
            if len(X_transformed) != len(y_aligned):
                raise RuntimeError("X and y are not aligned after NaN removal.")
            logger.info(f"Final shapes — X: {X_transformed.shape}, y: {y_aligned.shape}")
            return X_transformed, y_aligned

        logger.info(f"Final shape — X: {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X, y = None, **fit_params):
        logger.info("Calling fit_transform...")
        return self.fit(X, y).transform(X, y)


if __name__ == "__main__":
    def parse_int_list(s):
        try:
            return [int(x.strip()) for x in s.split(',') if x.strip()]
        except ValueError:
            raise argparse.ArgumentTypeError("List values must be comma-separated integers")

    parser = argparse.ArgumentParser(description = "Apply feature engineering to a CSV time series dataset.")
    parser.add_argument("--data_path", type = str, required = True, help = "Full path to input CSV file")
    parser.add_argument("--target_col", type = str, required = True, help = "Target column name")
    parser.add_argument("--date_col", type = str, default = None, help = "Name of the date column (optional)")
    parser.add_argument("--prediction_horizon", type = int, default = 1, help = "Months to predict ahead (default=1)")
    parser.add_argument("--lag_periods", type = parse_int_list, default = [3, 6], help = "Comma-separated lag periods (e.g., 3,6)")
    parser.add_argument("--rolling_windows", type = parse_int_list, default = [3, 6], help = "Comma-separated rolling window sizes (e.g., 3,6)")

    args = parser.parse_args()

    logger.info(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)

    if args.date_col:
        logger.info(f"Parsing date column: {args.date_col}")
        df[args.date_col] = pd.to_datetime(df[args.date_col])
        df.set_index(args.date_col, inplace = True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("No datetime index found and no date_col specified.")

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data.")

    y = df.pop(args.target_col)

    fe = FeatureEngineer(
        target_col = args.target_col,
        prediction_horizon_months = args.prediction_horizon,
        lag_periods = args.lag_periods,
        rolling_windows = args.rolling_windows
    )

    X_fe, y_fe = fe.fit_transform(df, y)

    base = os.path.splitext(os.path.basename(args.data_path))[0]
    X_fe.to_csv(f"{base}_features.csv")
    y_fe.to_csv(f"{base}_target.csv")
    logger.info("Feature engineering complete. Output files saved.")
