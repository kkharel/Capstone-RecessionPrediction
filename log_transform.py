import logging
import numpy as np
import pandas as pd
from growth_detection import detect_exponential_growth_in_df

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def apply_log_transform(df: pd.DataFrame, growth_flags: pd.DataFrame):
    df_transformed = df.copy()
    for col in growth_flags[growth_flags['Exponential Growth']].index:
        logger.info(f"Applying log transform to {col}")
        df_transformed[col] = np.log(df[col].replace(0, np.nan))
    return df_transformed

if __name__ == "__main__":

    df = pd.read_csv("data/cleaned_economic_data.csv", parse_dates = ["date"])
    growth_flags = detect_exponential_growth_in_df(df, time_col = 'date', verbose = True)

    logger.info("Applying Log Transformation...")
    df_transformed = apply_log_transform(df, growth_flags)
    df_transformed.to_csv("data/cleaned_economic_data_transformed.csv", index=False)
    logger.info("Saved transformed data")
