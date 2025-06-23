import os
import pandas as pd
import logging
from data_pull import pull_economic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_economic_data(
    gold_data_path="data/gold.csv",
    sp500_data_path="data/SP500.csv",
    dji_data_path="data/DJI.csv",
    output_path="data/cleaned_economic_data.csv",
    start_date="1970-01-01",
    end_date="2024-12-31",
    economic_data=None 
):
    """
    Cleans and merges economic, gold, S&P 500, and DJI data.

    Args:
        gold_data_path (str): Path to the gold price CSV.
        sp500_data_path (str): Path to the S&P 500 price CSV.
        dji_data_path (str): Path to the DJI price CSV.
        output_path (str): Path to save the cleaned data. If None, data is not saved.
        start_date (str): Start date for data truncation.
        end_date (str): End date for data truncation.
        economic_data (pd.DataFrame, optional): Pre-fetched economic data.
                                                 If None, it will be pulled.

    Returns:
        pd.DataFrame: The cleaned and merged DataFrame, or None if an error occurs.
    """
    # Pull economic data only if not provided
    if economic_data is None:
        logger.info("Pulling economic data from FRED API...")
        economic_data = pull_economic_data(output_path=None)
    else:
        logger.info("Using pre-fetched economic data...")

    if economic_data is None or economic_data.empty:
        logger.error("No economic data available for cleaning.")
        return None

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Truncate economic data to the specified date range
    truncated_data = economic_data[(economic_data['date'] >= start_date) & (economic_data['date'] <= end_date)].reset_index(drop=True)
    logger.info(f"Shape after truncation: {truncated_data.shape}")
    logger.info(f"NaNs after truncation:\n{truncated_data.isnull().sum()}")

    if truncated_data.empty:
        logger.warning(f"Economic data is empty after truncation to {start_date} - {end_date}.")
        return None

    logger.info("Forward filling quarterly economic indicators...")
    quarterly_cols = ['OPHNFB', 'NETEXP', 'UMCSENT'] 
    truncated_data[quarterly_cols] = truncated_data[quarterly_cols].ffill()
    # Drop rows that still have NaNs after forward filling
    truncated_data.dropna(subset=quarterly_cols, inplace=True)
    logger.info(f"Shape after ffill and dropna on quarterly_cols: {truncated_data.shape}")
    logger.info(f"NaNs after ffill and dropna on quarterly_cols:\n{truncated_data.isnull().sum()}")

    if truncated_data.empty:
        logger.warning("Economic data is empty after forward filling and dropping NaNs in quarterly columns.")
        return None

    logger.info("Reading and merging gold price data...")
    try:
        gold_df = pd.read_csv(gold_data_path, parse_dates=["Date"])
        gold_df.rename(columns={'Date': 'date', 'Price': 'gold'}, inplace=True)
        gold_df = gold_df[['date', 'gold']].dropna() # Drop rows with NaNs in gold data itself
        logger.info(f"Gold data head:\n{gold_df.head()}")
        logger.info(f"Gold data dtypes:\n{gold_df.dtypes}")
    except FileNotFoundError:
        logger.error(f"Gold data file not found at {gold_data_path}.")
        return None
    except Exception as e:
        logger.error(f"Error reading gold data from {gold_data_path}: {e}")
        return None
    
    gold = truncated_data.merge(gold_df, on='date', how='inner')
    logger.info(f"Shape after merge with gold: {gold.shape}")
    logger.info(f"NaNs after merge with gold:\n{gold.isnull().sum()}")

    if gold.empty:
        logger.warning("Merge with gold data resulted in an empty DataFrame. Check date ranges or data availability.")
        return None

    logger.info("Reading and merging S&P 500 data...")
    try:
        sp_data = pd.read_csv(sp500_data_path, parse_dates=["Date"])
        sp_data.rename(columns={'Date': 'date', 'Price': 'SP500'}, inplace=True)
        sp_data = sp_data[['date', 'SP500']].dropna() # Drop rows with NaNs in SP500 data itself
        logger.info(f"SP500 data head:\n{sp_data.head()}")
        logger.info(f"SP500 data dtypes:\n{sp_data.dtypes}")
    except FileNotFoundError:
        logger.error(f"S&P 500 data file not found at {sp500_data_path}.")
        return None
    except Exception as e:
        logger.error(f"Error reading S&P 500 data from {sp500_data_path}: {e}")
        return None

    logger.info("Reading and merging DJI data...")
    try:
        dj_data = pd.read_csv(dji_data_path, parse_dates=["Date"])
        dj_data.rename(columns={'Date': 'date', 'Price': 'DJI'}, inplace=True)
        dj_data = dj_data[['date', 'DJI']].dropna() # Drop rows with NaNs in DJI data itself
        # Handle NaT values before applying replace(day=1)
        dj_data['date'] = dj_data['date'].apply(lambda d: d.replace(day=1) if pd.notna(d) else pd.NaT)
        dj_data.dropna(subset=['date'], inplace=True) # Drop rows where date became NaT after transformation
        logger.info(f"DJI data head:\n{dj_data.head()}")
        logger.info(f"DJI data dtypes:\n{dj_data.dtypes}")
    except FileNotFoundError:
        logger.error(f"DJI data file not found at {dji_data_path}.")
        return None
    except Exception as e:
        logger.error(f"Error reading DJI data from {dji_data_path}: {e}")
        return None

    sp = gold.merge(sp_data, on='date', how='inner')
    logger.info(f"Shape after merge with S&P 500: {sp.shape}")
    logger.info(f"NaNs after merge with S&P 500:\n{sp.isnull().sum()}")

    if sp.empty:
        logger.warning("Merge with S&P 500 data resulted in an empty DataFrame. Check date ranges or data availability.")
        return None

    final_df = sp.merge(dj_data, on='date', how='left')
    logger.info(f"Shape after merge with DJI (before final cleanup): {final_df.shape}")
    logger.info(f"NaNs after merge with DJI (before final cleanup):\n{final_df.isnull().sum()}")
    
    # Robust numeric conversion for 'SP500' and 'DJI'
    # Convert to string, replace commas, then convert to numeric, coercing errors to NaN
    logger.info("Converting SP500 and DJI columns to numeric, coercing errors...")
    # Check current types before conversion attempts
    logger.info(f"SP500 dtype before conversion: {final_df['SP500'].dtype}")
    logger.info(f"DJI dtype before conversion: {final_df['DJI'].dtype}")

    # Use pd.to_numeric for robust conversion, coercing errors
    final_df['SP500'] = pd.to_numeric(final_df['SP500'].astype(str).str.replace(',', ''), errors='coerce')
    final_df['DJI'] = pd.to_numeric(final_df['DJI'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Drop rows with any remaining NaNs in the entire DataFrame after conversions
    logger.info("Dropping any remaining NaNs in the final DataFrame...")
    final_df.dropna(inplace=True) 

    logger.info(f"Shape after all merges and final dropna: {final_df.shape}")
    logger.info(f"NaNs after all merges and final dropna:\n{final_df.isnull().sum()}")
    logger.info(f"Final DataFrame dtypes:\n{final_df.dtypes}")

    if final_df.empty:
        logger.warning("Final DataFrame is empty after all merges and NaN removals. Cannot proceed.")
        return None

    logger.info("Renaming columns for clarity...")
    final_df.rename(columns={
        'INDPRO': 'industrial_production',
        'TCU': 'capacity_utilization',
        'BBKMGDP': 'real_GDP_growth_rate',
        'UNRATE': 'unemployment_rate',
        'PAYEMS': 'nonfarm_payroll',
        'CIVPART': 'laborforce_participation',
        'CLF16OV': 'civilian_labor_force',
        'CPIAUCSL': 'CPI',
        'CPILFESL': 'core_CPI',
        'PPIACO': 'PPI',
        'FEDFUNDS': 'federal_funds_rate',
        'M1SL': 'real_M1_money_supply_growth',
        'M2SL': 'real_M2_money_supply_growth',
        'GS10': '10Y_treasury_yield',
        'GS5': '5Y_treasury_yield',
        'GS7': '7Y_treasury_yield',
        'GS30': '30Y_treasury_yield',
        'GS3': '3Y_treasury_yield',
        'GS2': '2Y_treasury_yield',
        'GS1': '1Y_treasury_yield',
        'PCE': 'personal_consumption_expenditures',
        'DSPIC96': 'real_disposable_personal_income',
        'TOTALSL': 'total_consumer_credit',
        'NETEXP': 'trade_balance', 
        'UMCSENT': 'consumer_sentiment',
        'PERMIT': 'building_permits',
        'HOUST': 'housing_start',
        'MSPNHSUS': 'median_sale_price_new_houses',
        'OPHNFB': 'nonfarm_business_sector_real_output_per_hour',
        'USREC': 'recession',
        'gold': 'gold_price',
        'SP500': 'SP500_price',
        'DJI': 'DJI_price'
    }, inplace=True)

    final_df = final_df.sort_values('date').reset_index(drop=True)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not final_df.empty:
            final_df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        else:
            logger.warning(f"Cleaned DataFrame is empty, not saving to {output_path}")
            return None
    
    logger.info("Data cleaning complete.")

    return final_df

def create_sample_data_files():
    """
    Generates sample CSV files for gold, S&P 500, and DJI data.
    These files are used for testing the data cleaning process.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Generate dates from 2000-01-01 to 2025-03-31
    dates = pd.date_range(start='2000-01-01', end='2025-03-31', freq='MS') # Monthly start
    
    # Gold data
    gold_prices = 400 + 10 * (dates.year - 2000) + 0.5 * dates.month + 50 * (dates.month % 3)
    gold_df = pd.DataFrame({'Date': dates, 'Price': gold_prices.round(2)})
    gold_df.to_csv(os.path.join(data_dir, 'gold.csv'), index=False)
    logger.info("Generated data/gold.csv")

    # SP500 data
    sp500_prices = 1500 + 100 * (dates.year - 2000) + 10 * dates.month + 200 * (dates.month % 2)
    sp500_df = pd.DataFrame({'Date': dates, 'Price': sp500_prices.round(2)})
    sp500_df.to_csv(os.path.join(data_dir, 'SP500.csv'), index=False)
    logger.info("Generated data/SP500.csv")

    # DJI data
    dji_prices = 10000 + 500 * (dates.year - 2000) + 50 * dates.month + 300 * (dates.month % 4)
    dji_df = pd.DataFrame({'Date': dates, 'Price': dji_prices.round(2)})
    dji_df.to_csv(os.path.join(data_dir, 'DJI.csv'), index=False)
    logger.info("Generated data/DJI.csv")


if __name__ == "__main__":
    create_sample_data_files() 
    clean_economic_data()

