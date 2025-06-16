# customer cross validation for time series

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
from datetime import datetime
import logging
import sys
from data_preparation import prepare_data

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_custom_cv_folds(X_data: pd.DataFrame, y_data: pd.Series, validation_periods: List[Tuple[datetime, datetime]]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates train/validation split indices based on specific date ranges for validation sets.
    The training set for each fold will be all data before the validation set's start date.
    This simulates an expanding window or fixed-window validation for time series.
    """
    if not isinstance(X_data.index, pd.DatetimeIndex):
        raise TypeError("X_data index must be a DatetimeIndex for custom date-based CV folds.")
    if not X_data.index.equals(y_data.index):
        raise ValueError("X_data and y_data indices must match.")

    date_to_int_pos = {date: i for i, date in enumerate(X_data.index)}

    for val_start_date, val_end_date in validation_periods:
        # Define the validation set using the specified date range
        val_mask = (X_data.index >= val_start_date) & (X_data.index <= val_end_date)
        val_indices_dates = X_data.index[val_mask]

        if val_indices_dates.empty:
            logger.warning(f"Validation period {val_start_date.date()} to {val_end_date.date()} resulted in an empty validation set (no data in X_data within this range). Skipping this fold.")
            continue

        # Define the training set as all data *before* the validation set's start date
        train_mask = X_data.index < val_start_date
        train_indices_dates = X_data.index[train_mask]

        if train_indices_dates.empty:
            logger.warning(f"Training set for validation period {val_start_date.date()} to {val_end_date.date()} is empty. Skipping this fold. (e.g., if validation starts too early in the overall X_data).")
            continue

        # Convert date indices to integer indices for scikit-learn's CV format
        train_indices_int = np.array([date_to_int_pos[date] for date in train_indices_dates])
        val_indices_int = np.array([date_to_int_pos[date] for date in val_indices_dates])

        # --- IMPORTANT: Check for single-class training sets ---
        train_y_subset = y_data.loc[train_indices_dates]
        train_positive_count = train_y_subset.sum()
        train_negative_count = len(train_y_subset) - train_positive_count

        if train_positive_count == 0 or train_negative_count == 0:
            logger.warning(f"Training set for validation period {val_start_date.date()} to {val_end_date.date()} has only one class (pos: {int(train_positive_count)}, neg: {int(train_negative_count)}). Skipping this fold to avoid classifier errors.")
            continue # Skip this fold if it only has one class

        # Log positive sample counts for the current fold's train/validation sets
        val_y_subset = y_data.loc[val_indices_dates]
        val_positive_count = val_y_subset.sum()
        if val_positive_count == 0:
            logger.warning(f"Validation set for {val_start_date.date()} to {val_end_date.date()} has NO positive samples. This might affect F1-score if zero_division is not handled for this fold.")
        else:
            logger.info(f"Validation set for {val_start_date.date()} to {val_end_date.date()} has {int(val_positive_count)} positive samples (total: {len(val_y_subset)}).")

        logger.info(f"Training set for {val_start_date.date()} to {val_end_date.date()} has {int(train_positive_count)} positive samples (total: {len(train_y_subset)}).")

        yield train_indices_int, val_indices_int


if __name__ == "__main__":

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/cleaned_economic_data_stationary.csv"

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates = ["date"], index_col = "date")

    X_train, X_test, y_train, y_test = prepare_data(df)

    custom_validation_periods = [
        (datetime(1981, 7, 1), datetime(1982, 10, 31)),
        (datetime(1990, 7, 1), datetime(1991, 2, 28)),
        (datetime(2001, 3, 1), datetime(2001, 10, 31)) 
    ]

    logger.info("Generating custom CV folds based on specified validation periods...")
    custom_cv = list(generate_custom_cv_folds(X_train, y_train, custom_validation_periods))
    logger.info(f"Custom CV folds generated: {len(custom_cv)} folds for BayesSearchCV.")

    for fold_idx, (train_idx, val_idx) in enumerate(custom_cv, 1):
        print(f"Fold {fold_idx}:")
        print(f"  Train samples count: {len(train_idx)}")
        print(f"  Validation samples count: {len(val_idx)}")
