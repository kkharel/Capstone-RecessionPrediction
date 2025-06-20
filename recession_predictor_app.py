import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import data_pull, data_cleaning, growth_detection, log_transform, make_stationary
from feature_engineering import FeatureEngineer

# --- CONFIG ---
MODEL_PATH = 'models/Bagging_Classifier_best_pipeline.pkl'
GOLD_PATH = 'data/gold.csv'
SP500_PATH = 'data/sp500.csv'
DJI_PATH = 'data/dji.csv'
START_DATE = "2020-10-01"
END_DATE = "2025-03-31"
BEST_THRESHOLD = 0.643535

st.set_page_config(page_title="Recession Predictor", layout="centered")
st.title("📉 Recession Forecasting App")
st.markdown("This app predicts upcoming recessions based on macroeconomic indicators.")

@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def prepare_data():
    raw_df = data_pull.pull_economic_data(output_path=None)
    cleaned_df = data_cleaning.clean_economic_data(
        economic_data=raw_df,
        gold_data_path=GOLD_PATH,
        sp500_data_path=SP500_PATH,
        dji_data_path=DJI_PATH,
        output_path=None,
        start_date=START_DATE,
        end_date=END_DATE
    )
    return cleaned_df

def predict_recession(cleaned_df, model):
    growth_flags = growth_detection.detect_exponential_growth_in_df(cleaned_df, time_col='date', verbose=False)
    df_transformed = log_transform.apply_log_transform(cleaned_df, growth_flags)
    stationary_df = make_stationary.make_stationary_dataset(df_transformed, output_path=None)
    stationary_df.set_index('date', inplace=True)

    fe = FeatureEngineer(
        target_col="recession",
        prediction_horizon_months=3, # should match training setup
        lag_periods=[3, 6], # should match training setup
        rolling_windows=[3, 6] # should match training setup
    )
    X_transformed = fe.fit_transform(stationary_df.copy())
    probabilities = model.predict_proba(X_transformed)[:, 1]
    thresholded_preds = (probabilities >= BEST_THRESHOLD).astype(int)

    results = []
    for i, date in enumerate(X_transformed.index):
        pred_date = date + pd.DateOffset(months=3)
        results.append({
            "From": date.strftime("%Y-%m"),
            "To": pred_date.strftime("%Y-%m"),
            "Probability": round(probabilities[i], 4),
            "Prediction": "RECESSION" if thresholded_preds[i] == 1 else "No Recession"
        })
    return pd.DataFrame(results)

# MAIN FLOW
model = load_model(MODEL_PATH)
if model:
    st.success("Model loaded successfully!")
    st.info("Generating predictions...")
    try:
        cleaned_df = prepare_data()
        results_df = predict_recession(cleaned_df, model)
        st.dataframe(results_df, use_container_width=True)

        if (results_df['Prediction'] == "RECESSION").any():
            st.warning("Recession likely in some upcoming periods!")
        else:
            st.success("No recession forecasted in the upcoming periods.")

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", csv, "recession_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
