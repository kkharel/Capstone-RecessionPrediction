import logging
import numpy as np
import pandas as pd

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def detect_exponential_growth(series: pd.Series, time_index = None, corr_threshold = 0.9, var_growth_threshold = 1.5, min_length = 50, verbose = False):
    """
    Detects exponential growth in a time series by:
    1. Checking log(series) correlation with time.
    2. Checking if variance increases over time.
    
    Returns True if exponential growth is likely.
    """
    s = series.dropna()
    if len(s) < min_length or (s <= 0).any():
        if verbose:
            logger.info(f"Skipping {series.name}: too short or non-positive values.")
        return False  # Cannot log non-positive values

    log_s = np.log(s)
    
    # Correlation between log(y) and time
    t = np.arange(len(log_s)) if time_index is None else (pd.to_datetime(time_index) - pd.to_datetime(time_index).min()).dt.days.values
    corr = np.corrcoef(t[:len(log_s)], log_s)[0, 1]
    
    # Variance growth
    split = np.array_split(s.values, 4)
    variances = [np.var(part) for part in split]
    var_growth = variances[-1] / (variances[0] + 1e-8)

    is_exponential = corr > corr_threshold and var_growth > var_growth_threshold

    if verbose:
        logger.info(f"{series.name}: log-corr = {corr:.2f}, var-growth = {var_growth:.2f} => {'EXPONENTIAL' if is_exponential else 'not exponential'}")

    return is_exponential

def detect_exponential_growth_in_df(df: pd.DataFrame, time_col: str = None, corr_threshold = 0.9, var_growth_threshold = 1.5, min_length = 50, verbose = False):
    """
    Detects exponential growth in each numeric column of a DataFrame.
    Uses 'time_col' as the time index for correlation checks if provided.
    Returns a DataFrame with boolean values indicating exponential growth.
    """
    results = {}
    time_index = df[time_col] if time_col and time_col in df.columns else None

    for col in df.columns:
        if col == time_col:
            continue  # Skip time column itself
        if pd.api.types.is_numeric_dtype(df[col]):
            results[col] = detect_exponential_growth(
                df[col],
                time_index = time_index,
                corr_threshold = corr_threshold,
                var_growth_threshold = var_growth_threshold,
                min_length = min_length,
                verbose = verbose
            )
        else:
            results[col] = False  # Skip non-numeric columns
    return pd.DataFrame.from_dict(results, orient = 'index', columns = ['Exponential Growth'])


if __name__ == "__main__":
    logger.info("Loading data...")
    df = pd.read_csv("data/cleaned_economic_data.csv", parse_dates = ["date"])
    logger.info("Loaded data")

    growth_flags = detect_exponential_growth_in_df(df, time_col = 'date', verbose = True)
    logger.info(f"\n{growth_flags}")



