# data merging and cleaning file
import os
import pandas as pd
import logging
from data_pull import pull_economic_data
 
logging.basicConfig( level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_economic_data(
    gold_data_path = "data/gold.csv",
    sp500_data_path = "data/SP500.csv",
    dji_data_path = "data/DJI.csv",
    output_path = "data/cleaned_economic_data.csv",
    start_date = "1970-01-01",
    end_date = "2024-12-31",
    economic_data = None 
):
    # Pull economic data only if not provided
    if economic_data is None:
        logger.info("Pulling economic data from FRED API...")
        economic_data = pull_economic_data(output_path = None)
    else:
        logger.info("Using pre-fetched economic data...")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    truncated_data = economic_data[(economic_data['date'] >= start_date) & (economic_data['date'] <= end_date)].reset_index(drop = True)

    logger.info("Forward filling quarterly economic indicators...")
    quarterly_cols = ['OPHNFB', 'NETEXP', 'UMCSENT'] 
    truncated_data[quarterly_cols] = truncated_data[quarterly_cols].ffill()
    truncated_data.dropna(inplace = True)

    logger.info("Reading and merging gold price data...")
    gold_df = pd.read_csv(gold_data_path, parse_dates = ["Date"])
    gold_df.rename(columns={'Date': 'date', 'Price': 'gold'}, inplace = True)
    
    gold = truncated_data.merge(gold_df, on = 'date', how = 'inner')

    logger.info("Reading and merging S&P 500 data...")
    sp_data = pd.read_csv(sp500_data_path, parse_dates = ["Date"])
    sp_data.rename(columns = {'Date': 'date', 'Price': 'SP500'}, inplace = True)
    sp_data = sp_data[['date', 'SP500']]

    logger.info("Reading and merging DJI data...")
    dj_data = pd.read_csv(dji_data_path,  parse_dates = ["Date"])
    dj_data.rename(columns = {'Date': 'date', 'Price': 'DJI'}, inplace = True)
    dj_data = dj_data[['date', 'DJI']]
    dj_data['date'] = dj_data['date'].apply(lambda d: d.replace(day = 1))

    sp = gold.merge(sp_data, on = 'date', how = 'inner')
    final_df = sp.merge(dj_data, on = 'date', how = 'left')
    final_df = final_df.dropna()

    logger.info("Converting SP500 and DJI columns to float...")
    final_df['SP500'] = final_df['SP500'].astype(str).str.replace(',', '').astype(float)
    final_df['DJI'] = final_df['DJI'].astype(str).str.replace(',', '').astype(float)

    logger.info("Renaming columns for clarity...")
    final_df.rename(columns = {
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

    final_df = final_df.sort_values('date').reset_index(drop = True)

    os.makedirs(os.path.dirname(output_path), exist_ok = True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    logger.info("Data cleaning complete.")

    return final_df

if __name__ == "__main__":
    clean_economic_data()
# data merging and cleaning file
import os
import pandas as pd
import logging
# Assuming data_pull is available in the same context or correctly imported
# from data_pull import pull_economic_data # Uncomment if pull_economic_data is from a separate module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_economic_data(
    gold_data_path="data/gold.csv",
    sp500_data_path="data/sp500.csv",
    dji_data_path="data/dji.csv",
    output_path="data/cleaned_economic_data.csv",
    start_date="1970-01-01",
    end_date="2020-12-31",
    economic_data=None # This is the DataFrame from data_pull.pull_economic_data
):
    try:
        # Pull economic data only if not provided
        if economic_data is None:
            logger.info("Pulling economic data from FRED API...")
            # Assuming pull_economic_data is defined or imported correctly
            # If data_pull is a module, it should be `data_pull.pull_economic_data`
            economic_data = data_pull.pull_economic_data(output_path=None) # Pass None for output here as well
        else:
            logger.info("Using pre-fetched economic data...")

        if economic_data is None or economic_data.empty:
            logger.error("Initial economic data is empty or None. Cannot proceed with cleaning.")
            return pd.DataFrame() # Return an empty DataFrame to indicate failure

        # Ensure 'date' column is datetime and sort
        economic_data['date'] = pd.to_datetime(economic_data['date'])
        economic_data = economic_data.sort_values('date').reset_index(drop=True)
        logger.info(f"Economic data shape after initial sort: {economic_data.shape}")
        logger.info(f"Economic data dates: {economic_data['date'].min()} to {economic_data['date'].max()}")


        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter economic data by date range
        truncated_data = economic_data[(economic_data['date'] >= start_date) & (economic_data['date'] <= end_date)].reset_index(drop=True)
        logger.info(f"Shape after initial date truncation to {start_date} - {end_date}: {truncated_data.shape}")
        if truncated_data.empty:
            logger.error("Truncated economic data is empty. Check date range or source data.")
            return pd.DataFrame()

        logger.info("Forward filling quarterly economic indicators...")
        quarterly_cols = ['OPHNFB', 'NETEXP', 'UMCSENT']
        # Apply ffill and then dropna only to relevant columns if possible, or entire df if necessary
        # Ensure 'date' is not affected by dropna based on other columns
        truncated_data[quarterly_cols] = truncated_data[quarterly_cols].ffill()
        truncated_data.dropna(subset=truncated_data.columns.difference(['date']), inplace=True) # Only dropna if non-date columns have NaNs
        logger.info(f"Shape after ffill and dropna: {truncated_data.shape}")
        if truncated_data.empty:
            logger.error("DataFrame empty after ffill and dropna. Too many missing values?")
            return pd.DataFrame()

        logger.info("Reading and merging gold price data...")
        gold_df = pd.read_csv(gold_data_path, parse_dates=["Date"])
        gold_df.rename(columns={'Date': 'date', 'Price': 'gold'}, inplace=True)
        # IMPORTANT: Align daily stock/gold data to the first day of the month for merging with monthly economic data
        gold_df['date'] = gold_df['date'].apply(lambda d: d.replace(day=1))
        gold_df = gold_df.groupby('date').last().reset_index() # Take the last price for the first day of month if multiple
        logger.info(f"Shape of gold_df after parsing and day=1 adjustment: {gold_df.shape}")
        logger.info(f"Dates in gold_df: {gold_df['date'].min()} to {gold_df['date'].max()}")

        gold = pd.merge_asof(
            truncated_data.sort_values('date'),
            gold_df.sort_values('date'),
            on='date',
            direction='nearest' # Use nearest to get closest date if exact match not found (more robust than inner for slight mismatches)
        )
        logger.info(f"Shape after merging with gold: {gold.shape}")
        if gold.empty:
            logger.error("GOLD MERGE RESULTED IN EMPTY DATAFRAME!")
            return pd.DataFrame()

        logger.info("Reading and merging S&P 500 data...")
        sp_data = pd.read_csv(sp500_data_path, parse_dates=["Date"])
        sp_data.rename(columns={'Date': 'date', 'Price': 'SP500'}, inplace=True)
        sp_data = sp_data[['date', 'SP500']]
        # IMPORTANT: Align daily stock/gold data to the first day of the month for merging with monthly economic data
        sp_data['date'] = sp_data['date'].apply(lambda d: d.replace(day=1))
        sp_data = sp_data.groupby('date').last().reset_index() # Take the last price for the first day of month
        logger.info(f"Shape of sp_data after parsing and day=1 adjustment: {sp_data.shape}")
        logger.info(f"Dates in sp_data: {sp_data['date'].min()} to {sp_data['date'].max()}")

        sp = pd.merge_asof(
            gold.sort_values('date'),
            sp_data.sort_values('date'),
            on='date',
            direction='nearest'
        )
        logger.info(f"Shape after merging with SP500: {sp.shape}")
        if sp.empty:
            logger.error("SP500 MERGE RESULTED IN EMPTY DATAFRAME!")
            return pd.DataFrame()

        logger.info("Reading and merging DJI data...")
        dj_data = pd.read_csv(dji_data_path, parse_dates=["Date"])
        dj_data.rename(columns={'Date': 'date', 'Price': 'DJI'}, inplace=True)
        dj_data = dj_data[['date', 'DJI']]
        # This was already present, ensuring consistency
        dj_data['date'] = dj_data['date'].apply(lambda d: d.replace(day=1))
        dj_data = dj_data.groupby('date').last().reset_index() # Take the last price for the first day of month
        logger.info(f"Shape of dj_data after parsing and day=1 adjustment: {dj_data.shape}")
        logger.info(f"Dates in dj_data: {dj_data['date'].min()} to {dj_data['date'].max()}")

        # Use merge_asof for better handling of potentially misaligned dates in real-world data
        # especially if one dataset has dates like '2020-01-01' and another '2020-01-02'
        final_df = pd.merge_asof(
            sp.sort_values('date'),
            dj_data.sort_values('date'),
            on='date',
            direction='nearest'
        )

        logger.info(f"Shape after left merging with DJI (before final dropna): {final_df.shape}")
        # Dropping rows where *any* of the newly merged columns (gold, SP500, DJI) are NaN
        # This is more robust than a generic dropna if you only care about the merged data being complete
        final_df.dropna(subset=['gold', 'SP500', 'DJI'] + quarterly_cols, inplace=True)
        logger.info(f"Final DataFrame shape after specific dropna: {final_df.shape}")
        if final_df.empty:
            logger.error("FINAL DATAFRAME IS EMPTY AFTER DROPPING NaNs. Check data coverage and merge points.")
            return pd.DataFrame()

        logger.info("Converting SP500 and DJI columns to float...")
        final_df['SP500'] = final_df['SP500'].astype(str).str.replace(',', '').astype(float)
        final_df['DJI'] = final_df['DJI'].astype(str).str.replace(',', '').astype(float)

        logger.info("Renaming columns for clarity...")
        final_df.rename(columns={
            'INDPRO': 'industrial_production', 'TCU': 'capacity_utilization',
            'BBKMGDP': 'real_GDP_growth_rate', 'UNRATE': 'unemployment_rate',
            'PAYEMS': 'nonfarm_payroll', 'CIVPART': 'laborforce_participation',
            'CLF16OV': 'civilian_labor_force', 'CPIAUCSL': 'CPI',
            'CPILFESL': 'core_CPI', 'PPIACO': 'PPI', 'FEDFUNDS': 'federal_funds_rate',
            'M1SL': 'real_M1_money_supply_growth', 'M2SL': 'real_M2_money_supply_growth',
            'GS10': '10Y_treasury_yield', 'GS5': '5Y_treasury_yield',
            'GS7': '7Y_treasury_yield', 'GS30': '30Y_treasury_yield',
            'GS3': '3Y_treasury_yield', 'GS2': '2Y_treasury_yield',
            'GS1': '1Y_treasury_yield', 'PCE': 'personal_consumption_expenditures',
            'DSPIC96': 'real_disposable_personal_income', 'TOTALSL': 'total_consumer_credit',
            'NETEXP': 'trade_balance', 'UMCSENT': 'consumer_sentiment',
            'PERMIT': 'building_permits', 'HOUST': 'housing_start',
            'MSPNHSUS': 'median_sale_price_new_houses', 'OPHNFB': 'nonfarm_business_sector_real_output_per_hour',
            'USREC': 'recession', 'gold': 'gold_price', 'SP500': 'SP500_price',
            'DJI': 'DJI_price'
        }, inplace=True)

        final_df = final_df.sort_values('date').reset_index(drop=True)

        # Only save if output_path is provided (not None)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        else:
            logger.info("output_path was None, skipping saving cleaned data to file.")

        logger.info("Data cleaning complete.")
        return final_df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame on error

if __name__ == "__main__":
    # Example usage:
    # This will now print to console if run directly
    cleaned_data = clean_economic_data(output_path="data/test_cleaned_data.csv",
                                        start_date="2000-01-01",
                                        end_date="2023-12-31")
    if not cleaned_data.empty:
        print(f"Cleaned data head:\n{cleaned_data.head()}")
    else:
        print("Data cleaning failed to produce data.")
