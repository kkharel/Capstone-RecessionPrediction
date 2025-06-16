import sys
import logging
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")

# ADF TEST
def adf_test(series, name = '', verbose = True, alpha = 0.05):
    if series.isnull().all():
        if verbose:
            logging.warning(f"Series '{name}' is all NaNs; skipping ADF test.")
        return False
    result = adfuller(series.dropna(), autolag = 'AIC')
    if verbose:
        logging.info(f"ADF Test for {name}: ADF Stat = {result[0]:.3f}, p-value = {result[1]:.4f}")
        logging.info(f"Critical Values: {result[4]}")
    return result[1] <= alpha



def make_stationary_dataset(
    df: pd.DataFrame,
    exclude_cols = None,
    adf_alpha = 0.05,
    verbose = True,
    output_path = "data/cleaned_economic_data_stationary.csv"
):
    """
    Checks each numeric column for stationarity using the ADF test.
    Applies first or second differencing as needed (in-place, keeps original column names).
    Returns a DataFrame with only stationary features (original names).
    Saves the result as a CSV.
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'recession']
    stationary_df = df.copy()
    for col in df.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = stationary_df[col]
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
    # Keep only date, recession, and stationary features
    keep_cols = [c for c in stationary_df.columns if c in exclude_cols or pd.api.types.is_numeric_dtype(stationary_df[c])]
    stationary_final = stationary_df[keep_cols].dropna().reset_index(drop=True)
    stationary_final.to_csv(output_path, index=False)
    if verbose:
        print(f"Stationary dataset saved as {output_path}")
    return stationary_final

if __name__ == "__main__":

    input_path = "data/cleaned_economic_data_transformed.csv"
    output_path = "data/cleaned_economic_data_stationary.csv"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, parse_dates = ["date"])
    make_stationary_dataset(df, output_path = output_path, verbose = True)