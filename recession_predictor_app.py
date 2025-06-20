import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import data_pull, data_cleaning, growth_detection, log_transform, make_stationary
from feature_engineering import FeatureEngineer

# --- CONFIG ---
MODEL_PATH = st.secrets.get("MODEL_PATH", "models/Bagging_Classifier_best_pipeline.pkl")
GOLD_PATH = st.secrets.get("GOLD_PATH", "data/gold.csv")
SP500_PATH = st.secrets.get("SP500_PATH", "data/sp500.csv")
DJI_PATH = st.secrets.get("DJI_PATH", "data/dji.csv")

START_DATE = st.secrets.get("START_DATE", "2020-10-01")
END_DATE = st.secrets.get("END_DATE", "2025-03-31")

BEST_THRESHOLD = float(st.secrets.get("BEST_THRESHOLD", 0.643535))
PREDICTION_HORIZON_MONTHS = int(st.secrets.get("PREDICTION_HORIZON_MONTHS", 3))

DATA_PATH = st.secrets.get("DATA_PATH", "data/deployment_test.csv")
BEST_THRESHOLD = 0.643535

st.set_page_config(page_title="Recession Predictor", layout="centered")
st.title("📉 Recession Forecasting App")
st.markdown(
    """
    This app predicts upcoming recessions based on macroeconomic indicators.
    Upload or use preloaded deployment data for forecasting.
    """
)

@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Data index is not a DatetimeIndex. Please check date parsing.")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def preprocess_and_predict(df, model, threshold=BEST_THRESHOLD):
    # Exponential growth detection & log transform
    growth_flags = growth_detection.detect_exponential_growth_in_df(df, time_col='date', verbose=False)
    df_transformed = log_transform.apply_log_transform(df, growth_flags)

    # Stationary transformation
    stationary_df = make_stationary.make_stationary_dataset(df_transformed, output_path=None)
    stationary_df.set_index('date', inplace=True)

    # Feature engineering
    fe = FeatureEngineer(
        target_col="recession",  # should be None or valid if recession present
        prediction_horizon_months=3,
        lag_periods=[3,6],
        rolling_windows=[3,6]
    )
    X_transformed = fe.fit_transform(stationary_df.copy())

    # Predict
    probabilities = model.predict_proba(X_transformed)[:,1]
    preds = (probabilities >= threshold).astype(int)

    # Prepare results
    results = []
    for i, date in enumerate(X_transformed.index):
        pred_date = date + pd.DateOffset(months=fe.prediction_horizon_months)
        results.append({
            "From (Features Date)": date.strftime("%Y-%m-%d"),
            "To (Predicted Date)": pred_date.strftime("%Y-%m-%d"),
            "Probability": round(probabilities[i],4),
            "Prediction": "RECESSION likely" if preds[i]==1 else "No Recession"
        })
    return pd.DataFrame(results).set_index("From (Features Date)")

def main():
    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload a deployment CSV (optional)", type=["csv"])

    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()

    if uploaded_file is not None:
        st.info("Using uploaded dataset for predictions")
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["date"], index_col="date")
            if not isinstance(df.index, pd.DatetimeIndex):
                st.error("Uploaded data does not have proper datetime index after parsing.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            st.stop()
    else:
        st.info(f"Loading default deployment data from {DATA_PATH}")
        df = load_data(DATA_PATH)
        if df is None:
            st.stop()

    st.write(f"Data preview ({len(df)} rows):")
    st.dataframe(df.head())

    with st.spinner("Running preprocessing and predictions..."):
        results_df = preprocess_and_predict(df, model, BEST_THRESHOLD)

    st.subheader("Prediction Results")
    st.dataframe(results_df)

    if (results_df['Prediction'] == "RECESSION likely").any():
        st.warning("⚠️ Recession predicted in one or more future periods!")
    else:
        st.success("✅ No recession predicted in the forecast horizon.")

    csv = results_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download prediction results as CSV",
        data=csv,
        file_name="recession_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

