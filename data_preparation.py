# data preparation for model consumption

# from project_config import TARGET, PREDICTION_HORIZON_MONTHS, MAIN_TRAIN_TEST_SPLIT_DATE
from feature_engineering import FeatureEngineer
from preprocessor import get_preprocessor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import streamlit as st

TARGET = st.secrets["TARGET"]
PREDICTION_HORIZON_MONTHS = int(st.secrets["PREDICTION_HORIZON_MONTHS"])
MAIN_TRAIN_TEST_SPLIT_DATE = datetime.strptime(st.secrets["MAIN_TRAIN_TEST_SPLIT_DATE"], "%Y-%m-%d")


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(
    df_data: pd.DataFrame,
    split_date = MAIN_TRAIN_TEST_SPLIT_DATE,
    target_col = TARGET,
    prediction_horizon_months = PREDICTION_HORIZON_MONTHS,
    return_preprocessor = False
):
    """
    Given cleaned DataFrame, performs ffill for quarterly columns, feature engineering,
    chronological train-test split, and preprocessor fitting.

    Returns:
        X_train, X_test, y_train, y_test, (optionally preprocessor)
    """

    # --- Data Cleaning ---
    cols_to_clean = ['trade_balance', 'consumer_sentiment', 'nonfarm_business_sector_real_output_per_hour']
    cols = [col for col in cols_to_clean if col in df_data.columns]
    df_data[cols] = df_data[cols].replace(0, np.nan).ffill()
    logger.info(f"Cleaned data loaded. Shape: {df_data.shape}")

    # --- Feature Engineering ---
    fe = FeatureEngineer(target_col = target_col, prediction_horizon_months = prediction_horizon_months)
    X_fe_full, y_aligned_full = fe.fit_transform(df_data.drop(columns = [target_col]), df_data[target_col])
    logger.info(f"Feature Engineering applied. X shape: {X_fe_full.shape}, y shape: {y_aligned_full.shape}")

    # --- Train-Test Split ---
    if not isinstance(X_fe_full.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for date-based splitting.")

    logger.info(f"Main train-test split at date: {split_date}")
    X_train = X_fe_full[X_fe_full.index < split_date]
    X_test = X_fe_full[X_fe_full.index >= split_date]
    y_train = y_aligned_full[y_aligned_full.index < split_date]
    y_test = y_aligned_full[y_aligned_full.index >= split_date]

    logger.info(f"Train from {X_train.index.min().date()} to {X_train.index.max().date()} | Shape: {X_train.shape}")
    logger.info(f"Test from {X_test.index.min().date()} to {X_test.index.max().date()} | Shape: {X_test.shape}")

    # --- Sample Check ---
    train_positive = y_train.sum()
    test_positive = y_test.sum()
    logger.info(f"Positive samples in train: {train_positive} / {len(y_train)}")
    logger.info(f"Positive samples in test: {test_positive} / {len(y_test)}")

    if train_positive == 0:
        logger.error("y_train has no positive samples. Aborting.")
        raise ValueError("y_train must contain at least one positive sample.")
    if test_positive == 0:
        logger.warning("y_test has no positive samples. Evaluation metrics may be unreliable.")

    # --- Preprocessing ---
    preprocessor = get_preprocessor(X_train.copy(), target_col = None)
    preprocessor.fit(X_train)
    logger.info("Preprocessor fit complete.")

    if return_preprocessor:
        return X_train, X_test, y_train, y_test, preprocessor
    else:
        return X_train, X_test, y_train, y_test


def main():
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/cleaned_economic_data_stationary.csv"

    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates = ["date"], index_col = "date")

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, return_preprocessor = True)
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
