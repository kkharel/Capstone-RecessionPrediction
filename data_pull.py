# data pull file

import os
import logging
import pandas as pd
import requests
# from project_config import API_KEY
import streamlit as st

API_KEY = st.secrets["API_KEY"]

logging.basicConfig(level = logging.INFO, format = "%(asctime)s — %(levelname)s — %(message)s")
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


def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"HTTP error {response.status_code} for series {series_id}")
        return pd.DataFrame(columns = ["date", series_id])
    data = response.json()

    if "observations" not in data:
        logging.error(f"Error fetching data for series {series_id}: {data}")
        return pd.DataFrame(columns = ["date", series_id])

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].replace(".", None).astype(float)

    return df[["date", "value"]].rename(columns = {"value": series_id})

def pull_economic_data(output_path = None):
    raw_economic_data = None
    for name, series in indicators.items():
        logging.info(f"Fetching {name} ({series})...")
        df = fetch_fred_data(series)
        if raw_economic_data is None:
            raw_economic_data = df
        else:
            raw_economic_data = raw_economic_data.merge(df, on = "date", how = "outer")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        raw_economic_data.to_csv(output_path, index = False)
        logging.info(f"Data fetched and saved to {output_path}")

    return raw_economic_data

if __name__ == "__main__":
    pull_economic_data()    