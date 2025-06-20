import sys
import logging
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__) # Get a logger instance

# ADF TEST
def adf_test(series, name = '', verbose = True, alpha = 0.05):
    if series.isnull().all():
        if verbose:
            logger.warning(f"Series '{name}' is all NaNs; skipping ADF test.")
        return False
    # Drop NaNs for ADF test as it doesn't handle them
    result = adfuller(series.dropna(), autolag = 'AIC')
    if verbose:
        logger.info(f"ADF Test for {name}: ADF Stat = {result[0]:.3f}, p-value = {result[1]:.4f}")
        logger.info(f"Critical Values: {result[4]}")
    return result[1] <= alpha


def make_stationary_dataset(
    df: pd.DataFrame,
    exclude_cols = None,
    adf_alpha = 0.05,
    verbose = True,
    output_path = "data/cleaned_economic_data_stationary.csv"
) -> pd.DataFrame:
    """
    Checks each numeric column for stationarity using the ADF test.
    Applies first or second differencing as needed (in-place, keeps original column names).
    Retains the original date column.
    Returns a DataFrame with only stationary features (original names) and the date column.
    Saves the result as a CSV.
    """
    if exclude_cols is None:
        # Default columns to exclude from differencing
        exclude_cols = ['date', 'recession']
    else:
        # Ensure 'date' and 'recession' are always in exclude_cols if they exist
        if 'date' not in exclude_cols and 'date' in df.columns:
            exclude_cols.append('date')
        if 'recession' not in exclude_cols and 'recession' in df.columns:
            exclude_cols.append('recession')

    # Store the original date column and set it as index for time series operations
    # Check if 'date' is already the index
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date']) # Ensure 'date' is datetime type
        stationary_df = df.set_index('date').copy()
        logger.info("Set 'date' column as DataFrame index.")
    elif isinstance(df.index, pd.DatetimeIndex):
        stationary_df = df.copy()
        logger.info("DataFrame already has a DatetimeIndex.")
    else:
        # Handle cases where no clear 'date' column or DatetimeIndex exists
        logger.warning("No 'date' column or DatetimeIndex found. Proceeding with current index, which may not be date-based.")
        stationary_df = df.copy()

    # Apply differencing to numeric columns not in exclude_cols
    for col in stationary_df.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(stationary_df[col]):
            continue # Skip non-numeric or excluded columns

        series = stationary_df[col]
        # Skip if series is all NaNs or constant (after dropping NaNs) to prevent ADF test errors
        if series.dropna().empty or series.nunique() <= 1:
            if verbose:
                print(f"{col}: Skipping differencing due to all NaNs or constant values in relevant subset.")
            continue

        # First differencing
        if not adf_test(series, name = col, verbose = verbose, alpha = adf_alpha):
            diff1 = series.diff()
            # Second differencing if still not stationary
            if not adf_test(diff1, name = f"{col}_diff1", verbose = verbose, alpha = adf_alpha):
                diff2 = diff1.diff()
                stationary_df[col] = diff2
                if verbose:
                    print(f"{col}: Second differencing applied (column overwritten).")
            else:
                stationary_df[col] = diff1
                if verbose:
                    print(f"{col}: First differencing applied (column overwritten).")
        else:
            if verbose:
                print(f"{col}: Already stationary.")

    # Drop NaNs introduced by differencing (typically the first 1 or 2 rows).
    # These NaNs are inherent to the differencing operation.
    stationary_df_cleaned = stationary_df.dropna()
    logger.info(f"Dropped {len(stationary_df) - len(stationary_df_cleaned)} rows with NaNs introduced by differencing.")

    # Restore 'date' as a regular column from the index
    if isinstance(stationary_df_cleaned.index, pd.DatetimeIndex):
        stationary_final = stationary_df_cleaned.reset_index()
        # Ensure the date column is named 'date' if reset_index creates 'index'
        if 'index' in stationary_final.columns:
            stationary_final = stationary_final.rename(columns={'index': 'date'})
        logger.info("Restored 'date' column from index.")
    else:
        # If the index was not DatetimeIndex, and 'date' was not explicitly set as index,
        # then stationary_df_cleaned already contains the original date column if it existed.
        stationary_final = stationary_df_cleaned.copy()
        logger.info("Date column was handled via original column retention or was not a DatetimeIndex.")

    # Ensure all original non-numeric columns (like 'recession') are carried over if they exist
    # and were explicitly excluded from differencing.
    # The `exclude_cols` list ensures these columns are not processed.
    # Now, ensure they are in the final DataFrame.
    non_numeric_original_cols = [c for c in df.columns if c in exclude_cols and not pd.api.types.is_numeric_dtype(df[c])]
    for col in non_numeric_original_cols:
        if col not in stationary_final.columns and col in df.columns:
            # Merge back the non-numeric excluded columns using the date index
            # Align by index before merging, then drop index if not desired
            temp_df_with_dates_as_index = df.set_index('date') if 'date' in df.columns else df
            stationary_final = stationary_final.set_index('date').merge(
                temp_df_with_dates_as_index[col],
                left_index=True,
                right_index=True,
                how='left'
            ).reset_index()
            stationary_final = stationary_final.rename(columns={'index': 'date'}) # ensure 'date' is named correctly

            logger.info(f"Re-added non-numeric excluded column: '{col}'.")


    # Final selection of columns: all numeric ones and explicitly excluded ones like 'date', 'recession'
    final_cols_to_keep = [
        c for c in stationary_final.columns
        if pd.api.types.is_numeric_dtype(stationary_final[c]) or c in exclude_cols
    ]
    stationary_final = stationary_final[final_cols_to_keep]

    # Save the result
    stationary_final.to_csv(output_path, index=False)
    if verbose:
        print(f"Stationary dataset saved as {output_path}")

    return stationary_final

# Example usage (from __main__ block):
if __name__ == "__main__":
    input_path = "data/cleaned_economic_data_transformed.csv"
    output_path = "data/cleaned_economic_data_stationary.csv"
    
    # Allow command line arguments for input/output paths
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    logger.info(f"Loading data from {input_path}")
    # Ensure 'date' is parsed as datetime
    df = pd.read_csv(input_path, parse_dates=["date"])

    # Call the modified function
    make_stationary_dataset(df, output_path=output_path, verbose=True)