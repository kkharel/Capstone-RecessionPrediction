import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os # Import os for path handling

import data_pull
import data_cleaning
import growth_detection
import log_transform
import make_stationary
from feature_engineering import FeatureEngineer

# --- CONFIG ---

MODEL_PATH = 'models/Bagging_Classifier_best_pipeline.pkl'
GOLD_PATH = 'data/gold.csv'
SP500_PATH = 'data/SP500.csv'
DJI_PATH = 'data/DJI.csv'

START_DATE = "2020-10-01"
END_DATE = "2025-03-31"
BEST_THRESHOLD = 0.643535 # Should match training result
PREDICTION_HORIZON_MONTHS = 3 # Should match training setup

# --- GITHUB LINK ---
st.markdown("🔗 [GitHub Repository](https://github.com/kkharel/Capstone-RecessionPrediction)")

st.set_page_config(page_title="Recession Predictor", layout="centered")
st.title("📉 Recession Forecasting App")
st.markdown("This app predicts upcoming recessions based on macroeconomic indicators.")

# Optional during dev: clear cached data to avoid stale or NoneType bugs
# st.cache_data.clear() # Commented out for production, uncomment for debugging

# ----------------------------- UTILS -----------------------------

@st.cache_resource
def load_model(path):
    """
    Loads the pre-trained machine learning model from the specified path.
    Uses st.cache_resource to cache the model across reruns.
    """
    try:
        model = joblib.load(path)
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: Model file not found at {path}. Please ensure the model exists.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def prepare_data():
    """
    Prepares the economic and market data by pulling, cleaning, and merging.
    Uses st.cache_data to cache the cleaned DataFrame.
    Returns a pandas DataFrame if successful, None otherwise.
    """
    st.info("🔄 Pulling raw economic data...")
    try:
        # data_pull.pull_economic_data will handle fetching from FRED and can return a DataFrame
        raw_df = data_pull.pull_economic_data(output_path=None) # Don't save to file here
        if raw_df is None or raw_df.empty:
            st.error("❌ Failed to pull raw economic data or data is empty.")
            return None

        st.info("🔄 Cleaning and merging data...")
        cleaned_df = data_cleaning.clean_economic_data(
            economic_data=raw_df, # Pass the DataFrame directly
            gold_data_path=GOLD_PATH,
            sp500_data_path=SP500_PATH,
            dji_data_path=DJI_PATH,
            output_path=None, # Don't save cleaned data to file from within Streamlit app
            start_date=START_DATE,
            end_date=END_DATE
        )

        if cleaned_df is None or cleaned_df.empty:
            st.error("❌ Data cleaning failed or resulted in an empty DataFrame. Please check data_cleaning.py logs.")
            return None

        st.success("✅ Data prepared successfully!")
        return cleaned_df

    except Exception as e:
        st.error(f"❌ Data preparation failed: {e}")
        return None


def predict_recession(cleaned_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Performs feature engineering and uses the loaded model to predict recessions.

    Args:
        cleaned_df (pd.DataFrame): The pre-processed and cleaned DataFrame.
        model: The loaded machine learning model.

    Returns:
        pd.DataFrame: A DataFrame containing prediction results.
    """
    if cleaned_df.empty:
        st.error("Cannot predict: Cleaned DataFrame is empty.")
        return pd.DataFrame()

    st.info("📊 Applying transformations and engineering features...")
    try:
        # Step 1: Detect exponential growth
        growth_flags = growth_detection.detect_exponential_growth_in_df(cleaned_df, time_col='date', verbose=False)

        # Step 2: Apply log transformation
        df_transformed = log_transform.apply_log_transform(cleaned_df.copy(), growth_flags)

        # Step 3: Make data stationary
        # make_stationary.make_stationary_dataset internally sets date as index if output_path is None
        stationary_df = make_stationary.make_stationary_dataset(df_transformed.copy(), output_path=None)

        if stationary_df.empty:
            st.error("Stationary DataFrame is empty after transformations. Check make_stationary.py.")
            return pd.DataFrame()

        # Ensure 'date' is the index for FeatureEngineer
        # Check if 'date' is a column; if so, set it as index.
        # Otherwise, assume it's already the index and ensure it's named 'date'.
        if 'date' in stationary_df.columns:
            stationary_df.set_index('date', inplace=True)
        elif stationary_df.index.name != 'date':
            # If 'date' is not a column, but the index is not named 'date',
            # and it's a DatetimeIndex, assume it's the date and rename.
            if isinstance(stationary_df.index, pd.DatetimeIndex):
                stationary_df.index.name = 'date'
            else:
                st.error("Stationary DataFrame index is not a DatetimeIndex and 'date' column is missing.")
                return pd.DataFrame()


        # Step 4: Feature Engineering
        fe = FeatureEngineer(
            target_col="recession",
            prediction_horizon_months=PREDICTION_HORIZON_MONTHS,
            lag_periods=[3, 6], # Ensure these match your model's training setup
            rolling_windows=[3, 6] # Ensure these match your model's training setup
        )
        X_transformed = fe.fit_transform(stationary_df.copy())

        # Align columns if necessary (crucial for model input consistency)
        # This assumes model's feature names are accessible (e.g., via model.feature_names_in_)
        # If model's feature names are known, reindex X_transformed:
        # if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        #     missing_cols = set(model.feature_names_in_) - set(X_transformed.columns)
        #     for c in missing_cols:
        #         X_transformed[c] = 0 # Or the mean/median from training data
        #     X_transformed = X_transformed[model.feature_names_in_]
        # else:
        #     st.warning("Model feature names not accessible. Assuming X_transformed columns match.")


        st.info("🔮 Making predictions...")
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
        st.error(f"❌ Error during prediction pipeline: {e}")
        return pd.DataFrame()


# ----------------------------- MAIN FLOW -----------------------------

# Load the model once
model = load_model(MODEL_PATH)

if model:
    # If model loaded successfully, proceed with data preparation and prediction
    st.info("Starting data preparation and prediction...")
    try:
        cleaned_df = prepare_data()

        # Check if data preparation was successful before proceeding
        if cleaned_df is not None and not cleaned_df.empty:
            results_df = predict_recession(cleaned_df, model)

            if not results_df.empty:
                st.dataframe(results_df, use_container_width=True)

                if (results_df['Prediction'] == "RECESSION").any():
                    st.warning("⚠️ Recession likely in some upcoming periods!")
                else:
                    st.success("✅ No recession forecasted in the upcoming periods.")

                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions as CSV", csv, "recession_predictions.csv", "text/csv")
            else:
                st.error("❌ Prediction process resulted in an empty DataFrame.")
        else:
            st.error("❌ Data preparation failed, cannot proceed with prediction.")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred in the main flow: {e}")
else:
    st.error("❌ Model could not be loaded. Please check the MODEL_PATH and file existence.")

