# data merging and cleaning file
import os
import pandas as pd
import logging
from data_pull import pull_economic_data
 
logging.basicConfig( level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_economic_data(
    gold_data_path = "data/gold.csv",
    sp500_data_path = "data/sp500.csv",
    dji_data_path = "data/dji.csv",
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
