# --- GitHub fallback CSV ---
GITHUB_FALLBACK_CSV = st.secrets.get(
    "DATA_PATH",
    "https://raw.githubusercontent.com/kkharel/Capstone-RecessionPrediction/main/data/deployment_test.csv"
)

# --- Fetch a FRED series ---
def fetch_fred_data(series_id):
    api_key = st.secrets.get("FRED_API_KEY")
    if not api_key:
        logger.error("❌ FRED_API_KEY not found in st.secrets.")
        return pd.DataFrame(columns=["date", series_id])

    url = (
        f"https://api.stlouisfed.org/fred/series/observations?"
        f"series_id={series_id}&api_key={api_key}&file_type=json"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"HTTP {response.status_code} for series {series_id}")
        return pd.DataFrame(columns=["date", series_id])

    data = response.json()
    if "observations" not in data:
        logger.error(f"No observations in response for {series_id}")
        return pd.DataFrame(columns=["date", series_id])

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].replace(".", None).astype(float)
    return df[["date", "value"]].rename(columns={"value": series_id})


# --- Pull full economic dataset ---
def pull_economic_data(output_path=None):
    raw_economic_data = None
    successful_series = 0

    for name, series in indicators.items():
        logger.info(f"📊 Fetching {name} ({series})...")
        df = fetch_fred_data(series)

        if df.empty:
            logger.warning(f"⚠️ No data returned for {series}")
            continue

        successful_series += 1
        raw_economic_data = df if raw_economic_data is None else raw_economic_data.merge(df, on="date", how="outer")

    if raw_economic_data is not None and not raw_economic_data.empty:
        raw_economic_data.sort_values("date", inplace=True)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            raw_economic_data.to_csv(output_path, index=False)
            logger.info(f"✅ Saved FRED data to {output_path}")
        return raw_economic_data

    # 🔁 Fallback to GitHub CSV
    logger.error("❌ All FRED data fetches failed — switching to fallback CSV.")
    try:
        fallback_df = pd.read_csv(GITHUB_FALLBACK_CSV, parse_dates=["date"])
        logger.info(f"✅ Loaded fallback CSV from GitHub: {GITHUB_FALLBACK_CSV}")
        return fallback_df
    except Exception as e:
        logger.critical(f"❌ Failed to load fallback CSV: {e}")
        return pd.DataFrame()


# --- For local testing ---
if __name__ == "__main__":
    df = pull_economic_data("data/economic_data.csv")
    print(df.head())