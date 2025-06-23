import os
import logging
import pandas as pd
import requests
# from project_config import API_KEY
import streamlit as st 

API_KEY = st.secrets["FRED_API_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# List of economic indicators and their FRED series IDs
indicators = {
    "Industrial Production": "INDPRO",
    "Capacity Utilization": "TCU", 
    "Brave-Butters-Kelley Real Gross Domestic Product": "BBKMGDP",  

    # Labor Market Data
    "Unemployment Rate": "UNRATE",
    "Total Nonfarm Employment": "PAYEMS", 
    "Labor Force Participation": "CIVPART", 
    "Civilian Labor Force": "CLF16OV", 

    # Inflation & Prices Data
    "CPI (Inflation)": "CPIAUCSL", 
    "Core CPI": "CPILFESL",
    "PPI (Wholesale Inflation)": "PPIACO", 

    # Monetary & Financial Data
    "Federal Funds Rate": "FEDFUNDS", 
    "Real M1 Money Supply": "M1SL", 
    "Real M2 Money Supply": "M2SL", 

    # Interest Rate Data
    "10-Year Treasury Constant Maturity Rate": "GS10", 
    "5-Year Treasury Constant Maturity Rate": "GS5", 
    "7-Year Treasury Constant Maturity Rate": "GS7", 
    "30-Year Treasury Constant Maturity Rate": "GS30", 
    "3-Year Treasury Constant Maturity Rate": "GS3", 
    "2-Year Treasury Constant Maturity Rate": "GS2", 
    "1-Year Treasury Constant Maturity Rate": "GS1", 
    # "3-Month Treasury Constant Maturity Rate": "GS3M", 

    # Consumer and Business Sentiment Data
    "Consumer Sentiment": "UMCSENT", 
    'Personal Consumption Expenditures': 'PCE',
    'Real Disposable Personal Income': 'DSPIC96',
    "Total Consumer Credit": "TOTALSL", 

    # Trade Data
    "Trade Balance": "NETEXP", 
    # "Exports of Goods and Services": "EXPGS", 
    # "Imports of Goods and Services": "IMPGS", 

    # Housing Data
    "New Private Housing Units Authorized by Building Permits": "PERMIT", 
    "New Private Housing Units Started": "HOUST", 
    "Median Sales Price for New Houses Sold in the United States": "MSPNHSUS",

    # Productivity Data
    "Nonfarm Business Sector: Real Output per Hour of All Persons": "OPHNFB", 


    "NBER based Recession Indicators for the United States from the Period following the Peak through the Trough": "USREC"
}


def fetch_fred_data(series_id: str) -> pd.DataFrame:
    """
    Fetches data for a given FRED series ID.

    Args:
        series_id (str): The FRED series ID.

    Returns:
        pd.DataFrame: A DataFrame with 'date' and the series_id column,
                      or an empty DataFrame if fetching fails.
    """
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.HTTPError as e: # Catch HTTPError specifically
        logger.error(f"HTTP Error {e.response.status_code} fetching {series_id}: {e}. Response text: {e.response.text}")
        return pd.DataFrame(columns=["date", series_id])
    except requests.exceptions.RequestException as e: # Catch other request exceptions
        logger.error(f"Network or general request error fetching {series_id}: {e}")
        return pd.DataFrame(columns=["date", series_id])
    
    data = response.json()

    if "observations" not in data:
        logging.error(f"Error fetching data for series {series_id}: 'observations' key missing in response. Response: {data}") # Log response for debugging
        return pd.DataFrame(columns=["date", series_id])

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    # Replace '.' with NaN, then convert to float. Errors will be coerced to NaN.
    df["value"] = pd.to_numeric(df["value"].replace(".", float('nan')), errors='coerce')

    return df[["date", "value"]].rename(columns={"value": series_id})

def pull_economic_data(output_path: str = None) -> pd.DataFrame:
    """
    Pulls economic data for all defined indicators from FRED.

    Args:
        output_path (str, optional): Path to save the raw economic data.
                                     If None, data is not saved to file.

    Returns:
        pd.DataFrame: A DataFrame containing all fetched economic data,
                      or None if an error occurs.
    """
    # Check for placeholder API key
    if API_KEY == 'f26e07fed4df3480555e31cb4a772df7':
        logger.warning("Using a placeholder FRED API key. Please replace it with your actual key for reliable data fetching.")

    # Initialize raw_economic_data as an empty DataFrame with a 'date' column
    # This ensures that even if the first fetch fails, subsequent merges don't error out
    # and we can still attempt to merge any successfully fetched data.
    raw_economic_data = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]')})

    for name, series in indicators.items():
        logging.info(f"Fetching {name} ({series})...")
        df = fetch_fred_data(series)
        
        # Merge only if the fetched DataFrame is not empty
        if not df.empty:
            # If raw_economic_data is still just the initial empty frame,
            # or if it has some data, merge the new DataFrame.
            # Using outer merge to keep all dates and indicators.
            raw_economic_data = raw_economic_data.merge(df, on="date", how="outer")
        else:
            logger.warning(f"Skipping merge for {name} ({series}) due to empty or failed fetch.")

    # After attempting all merges, check if any data was successfully pulled
    if raw_economic_data.empty:
        logger.error("No economic data was successfully pulled after attempting all FRED series fetches.")
        return None

    # Sort by date before saving/returning
    raw_economic_data = raw_economic_data.sort_values('date').reset_index(drop=True)
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        raw_economic_data.to_csv(output_path, index=False)
        logging.info(f"Data fetched and saved to {output_path}")

    return raw_economic_data

if __name__ == "__main__":
    pull_economic_data(output_path="data/raw_economic_data.csv")
