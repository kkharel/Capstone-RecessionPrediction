import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import data_pull, data_cleaning, growth_detection, log_transform, make_stationary
from feature_engineering import FeatureEngineer
import tempfile
import requests

# --- CONFIG ---
MODEL_PATH = st.secrets.get("MODEL_PATH", "models/Bagging_Classifier_best_pipeline.pkl")

GOLD_URL = st.secrets.get("GOLD_URL", "https://raw.githubusercontent.com/kkharel/Capstone-RecessionPrediction/main/data/gold.csv")
SP500_URL = st.secrets.get("SP500_URL", "https://raw.githubusercontent.com/kkharel/Capstone-RecessionPrediction/main/data/SP500.csv")
DJI_URL = st.secrets.get("DJI_URL", "https://raw.githubusercontent.com/kkharel/Capstone-RecessionPrediction/main/data/DJI.csv")

START_DATE = st.secrets.get("START_DATE", "2020-10-01")
END_DATE = st.secrets.get("END_DATE", "2025-03-31")
BEST_THRESHOLD = float(st.secrets.get("BEST_THRESHOLD", 0.643535))
PREDICTION_HORIZON_MONTHS = int(st.secrets.get("PREDICTION_HORIZON_MONTHS", 3))

# --- STREAMLIT UI ---
st.set_page_config(page_title="Recession Predictor", layout="centered")
st.title("📉 Recession Forecasting App")
st.markdown("This app predicts upcoming recessions based on macroeconomic indicators.")

@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def download_and_save_temp_csv(url):
    try:
        st.info(f"📥 Downloading: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download from {url}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(response.content)
        tmp.close()
        st.success(f"✅ Downloaded and saved: {tmp.name}")
        return tmp.name
    except Exception as e:
        st.error(f"❌ Failed to download CSV: {e}")
        return None

@st.cache_data
def prepare_data():
    try:
        raw_df = data_pull.pull_economic_data(output_path=None)

        if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            raise ValueError("Raw economic data could not be loaded or is empty.")

        gold_path = download_and_save_temp_csv(GOLD_URL)
        sp500_path = download_and_save_temp_csv(SP500_URL)
        dji_path = download_and_save_temp_csv(DJI_URL)

        if None in [gold_path, sp500_path, dji_path]:
            raise FileNotFoundError("One or more files failed to download.")

        cleaned_df = data_cleaning.clean_economic_data(
            economic_data=raw_df,
            gold_data_path=gold_path,
            sp500_data_path=sp500_path,
            dji_data_path=dji_path,
            output_path=None,
            start_date=START_DATE,
            end_date=END_DATE
        )
        return cleaned_df

    except Exception as e:
        st.error(f"❌ Data preparation failed: {e}")
        return None

def predict_recession(cleaned_df, model):
    try:
        growth_flags = growth_detection.detect_exponential_growth_in_df(cleaned_df, time_col='date', verbose=False)
        df_transformed = log_transform.apply_log_transform(cleaned_df, growth_flags)
        stationary_df = make_stationary.make_stationary_dataset(df_transformed, output_path=None)
        stationary_df.set_index('date', inplace=True)

        fe = FeatureEngineer(
            target_col="recession",
            prediction_horizon_months=PREDICTION_HORIZON_MONTHS,
            lag_periods=[3, 6],
            rolling_windows=[3, 6]
        )
        X_transformed = fe.fit_transform(stationary_df.copy())
        probabilities = model.predict_proba(X_transformed)[:, 1]
        thresholded_preds = (probabilities >= BEST_THRESHOLD).astype(int)

        results = []
        for i, date in enumerate(X_transformed.index):
            pred_date = date + pd.DateOffset(months=PREDICTION_HORIZON_MONTHS)
            results.append({
                "From": date.strftime("%Y-%m"),
                "To": pred_date.strftime("%Y-%m"),
                "Probability": round(probabilities[i], 4),
                "Prediction": "RECESSION" if thresholded_preds[i] == 1 else "No Recession"
            })
        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"❌ Prediction step failed: {e}")
        return None

# --- MAIN FLOW ---
model = load_model(MODEL_PATH)
if model:
    st.success("✅ Model loaded successfully!")
    st.info("🔄 Preparing and predicting...")

    cleaned_df = prepare_data()
    if cleaned_df is not None:
        results_df = predict_recession(cleaned_df, model)
        if results_df is not None and not results_df.empty:
            st.dataframe(results_df, use_container_width=True)

            if (results_df['Prediction'] == "RECESSION").any():
                st.warning("⚠️ Recession likely in some upcoming periods!")
            else:
                st.success("✅ No recession forecasted in the upcoming periods.")

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions as CSV", csv, "recession_predictions.csv", "text/csv")
        else:
            st.error("Prediction failed: No results generated.")
    else:
        st.error("Data preparation failed.")
else:
    st.error("❌ Failed to load model. Please check the model path and try again.")
