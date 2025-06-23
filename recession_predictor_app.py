import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import logging

# Set up basic logging for the main app
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Import your local modules
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
# st.cache_data.clear() # Uncomment for debugging, comment out for production

# ----------------------------- UTILS -----------------------------

# Ensure necessary directories exist
@st.cache_resource
def ensure_directories_exist():
    """Ensures data and models directories exist."""
    data_dir = os.path.dirname(GOLD_PATH)
    models_dir = os.path.dirname(MODEL_PATH)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Created directory: {data_dir}")
    else:
        logger.info(f"Directory already exists: {data_dir}")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Created directory: {models_dir}")
    else:
        logger.info(f"Directory already exists: {models_dir}")

# Call this function at the very beginning of the app script
ensure_directories_exist()


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
        st.error(f"❌ Error: Model file not found at {path}. Please ensure you have trained the model and placed '{os.path.basename(path)}' in the '{os.path.dirname(path)}/' directory.")
        logger.error(f"Model file not found: {path}") # Log error
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        logger.error(f"Error loading model from {path}: {e}") # Log error
        return None

@st.cache_data
def prepare_data():
    """
    Prepares the economic and market data by pulling, cleaning, and merging.
    Uses st.cache_data to cache the cleaned DataFrame.
    Returns a pandas DataFrame if successful, None otherwise.
    """
    st.info("🔄 Checking for sample market data files (gold, SP500, DJI)...")
    # Check if market data files exist, if not, generate them
    if not os.path.exists(GOLD_PATH) or not os.path.exists(SP500_PATH) or not os.path.exists(DJI_PATH):
        st.warning("⚠️ Market data files not found. Generating sample data...")
        logger.warning("Market data files not found. Calling create_sample_data_files().")
        data_cleaning.create_sample_data_files()
        st.success("✅ Sample market data generated.")
    else:
        st.info("Market data files found.")

    st.info("🔄 Pulling raw economic data...")
    try:
        # data_pull.pull_economic_data will handle fetching from FRED and can return a DataFrame
        raw_df = data_pull.pull_economic_data(output_path=None) # Don't save to file here
        if raw_df is None or raw_df.empty:
            st.error("❌ Failed to pull raw economic data or data is empty. Check FRED API key and internet connection.")
            logger.error("Raw economic data is None or empty after data_pull.pull_economic_data.") # Log error
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
            st.error("❌ Data cleaning failed or resulted in an empty DataFrame. Please check data_cleaning.py logs for details.")
            logger.error("Cleaned DataFrame is None or empty after data_cleaning.clean_economic_data.") # Log error
            return None

        st.success("✅ Data prepared successfully!")
        return cleaned_df

    except Exception as e:
        st.error(f"❌ Data preparation failed: {e}. Please check logs for detailed error.")
        logger.error(f"Error during data preparation in prepare_data(): {e}", exc_info=True) # Log full traceback
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
        logger.error("Attempted prediction with an empty cleaned DataFrame.")
        return pd.DataFrame()

    logger.info(f"--- Starting prediction pipeline with cleaned_df ---")
    logger.info(f"Cleaned_df shape: {cleaned_df.shape}")
    logger.info(f"Cleaned_df dtypes:\n{cleaned_df.dtypes}")
    logger.info(f"Cleaned_df NaNs:\n{cleaned_df.isnull().sum()}")
    logger.info(f"Cleaned_df head:\n{cleaned_df.head()}")
    logger.info(f"--- End cleaned_df inspection ---")


    st.info("📊 Applying transformations and engineering features...")
    try:
        # Step 1: Detect exponential growth
        # Ensure 'date' column is correctly handled by growth_detection
        logger.info(f"Calling growth_detection.detect_exponential_growth_in_df. date column dtype: {cleaned_df['date'].dtype}")
        growth_flags = growth_detection.detect_exponential_growth_in_df(cleaned_df, time_col='date', verbose=False)
        logger.info(f"Growth flags generated. Shape: {growth_flags.shape}")

        # Step 2: Apply log transformation
        logger.info(f"Calling log_transform.apply_log_transform.")
        df_transformed = log_transform.apply_log_transform(cleaned_df.copy(), growth_flags)
        logger.info(f"df_transformed shape: {df_transformed.shape}")
        logger.info(f"df_transformed dtypes:\n{df_transformed.dtypes}")
        logger.info(f"df_transformed NaNs:\n{df_transformed.isnull().sum()}")


        # Step 3: Make data stationary
        # make_stationary.make_stationary_dataset internally sets date as index if output_path is None
        logger.info(f"Calling make_stationary.make_stationary_dataset.")
        stationary_df = make_stationary.make_stationary_dataset(df_transformed.copy(), output_path=None)

        if stationary_df.empty:
            st.error("Stationary DataFrame is empty after transformations. Check make_stationary.py.")
            logger.error("Stationary DataFrame is empty.")
            return pd.DataFrame()

        # Ensure 'date' is the index for FeatureEngineer
        # Check if 'date' is a column; if so, set it as index.
        # Otherwise, assume it's already the index and ensure it's named 'date'.
        logger.info(f"Stationary_df dtypes before index check:\n{stationary_df.dtypes}")
        if 'date' in stationary_df.columns:
            logger.info("Setting 'date' column as index for stationary_df.")
            stationary_df.set_index('date', inplace=True)
        elif stationary_df.index.name != 'date':
            # If 'date' is not a column, but the index is not named 'date',
            # and it's a DatetimeIndex, assume it's the date and rename.
            if isinstance(stationary_df.index, pd.DatetimeIndex):
                logger.info("Renaming index to 'date' for stationary_df.")
                stationary_df.index.name = 'date'
            else:
                st.error("Stationary DataFrame index is not a DatetimeIndex and 'date' column is missing. Cannot proceed with feature engineering.")
                logger.error("Stationary DataFrame index is not DatetimeIndex and 'date' column is missing.")
                return pd.DataFrame()

        logger.info(f"Stationary_df shape before feature engineering: {stationary_df.shape}")
        logger.info(f"Stationary_df dtypes before feature engineering:\n{stationary_df.dtypes}")
        logger.info(f"Stationary_df NaNs before feature engineering:\n{stationary_df.isnull().sum()}")


        # Step 4: Feature Engineering
        logger.info(f"Calling FeatureEngineer.fit_transform.")
        fe = FeatureEngineer(
            target_col="recession",
            prediction_horizon_months=PREDICTION_HORIZON_MONTHS,
            lag_periods=[3, 6], # Ensure these match your model's training setup
            rolling_windows=[3, 6] # Ensure these match your model's training setup
        )
        X_transformed = fe.fit_transform(stationary_df.copy())
        logger.info(f"X_transformed shape: {X_transformed.shape}")
        logger.info(f"X_transformed dtypes:\n{X_transformed.dtypes}")
        logger.info(f"X_transformed NaNs:\n{X_transformed.isnull().sum()}")


        # Align columns if necessary (crucial for model input consistency)
        # This assumes model's feature names are accessible (e.g., via model.feature_names_in_)
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            # Reindex X_transformed to ensure columns match model's expected features
            # Fill missing with 0 or a sensible default, drop extra columns
            X_transformed = X_transformed.reindex(columns=model.feature_names_in_, fill_value=0)
            logger.info("Aligned X_transformed columns with model's feature names.")
        else:
            st.warning("Model feature names not accessible. Assuming X_transformed columns match model's expected input.")
            logger.warning("Model feature names not accessible. Skipping column alignment.")


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
        st.error(f"❌ Error during prediction pipeline: {e}. Please check logs for detailed error and data state.")
        logger.error(f"Error during prediction pipeline in predict_recession(): {e}", exc_info=True) # Log full traceback
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
                logger.error("Prediction process resulted in an empty DataFrame.") # Log error
        else:
            st.error("❌ Data preparation failed, cannot proceed with prediction.")
            logger.error("Data preparation failed, cleaned_df is None or empty.") # Log error

    except Exception as e:
        st.error(f"❌ An unexpected error occurred in the main flow: {e}")
        logger.error(f"An unexpected error occurred in the main flow: {e}", exc_info=True) # Log full traceback
else:
    st.error("❌ Model could not be loaded. Please ensure the model file is correctly placed.")
    logger.error("Model could not be loaded, exiting main flow.") # Log error
