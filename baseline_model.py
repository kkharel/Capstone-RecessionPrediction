# baseline_model.py

import os
import argparse
import warnings
import joblib
import logging

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import set_config
from preprocessor import get_preprocessor
from utils import (
    display_classification_metrics,
    plot_confusion,
    plot_roc_pr_curves,
    summarize_feature_importance
)
# from project_config import TARGET, RANDOM_STATE
import streamlit as st

# Load secrets
TARGET = st.secrets["TARGET"]
RANDOM_STATE = int(st.secrets["RANDOM_STATE"])


warnings.filterwarnings("ignore")
set_config(display='diagram')

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok = True)



def main(data_path):
    logger.info("Starting baseline model training script.")

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates = ["date"], index_col = "date")

    cols_to_ffill = [
        'trade_balance',
        'consumer_sentiment',
        'nonfarm_business_sector_real_output_per_hour'
    ]
    df[cols_to_ffill] = df[cols_to_ffill].replace(0, np.nan).ffill()
    df.dropna(inplace = True)
    logger.info(f"Data loaded with shape: {df.shape}")

    X = df.drop(columns = [TARGET])
    y = df[TARGET]

    train_end_date = '2006-12-31'
    X_train = X[X.index <= train_end_date]
    y_train = y[y.index <= train_end_date]
    X_test = X[X.index > train_end_date]
    y_test = y[X.index > train_end_date]

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    logger.info("Initializing preprocessor...")
    preprocessor = get_preprocessor(X_train.copy(), target_col = None)

    logger.info("Defining classifier...")
    clf = LogisticRegression(max_iter = 100000, class_weight = "balanced", random_state = RANDOM_STATE)

    logger.info("Building pipeline...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")

    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    logger.info(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    logger.info(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    display_classification_metrics(y_test, y_test_pred, y_test_proba, target_names = ["No Recession", "Recession"])
    plot_confusion(y_test, y_test_pred, title = "Baseline Model Confusion Matrix")
    plot_roc_pr_curves(y_test, y_test_proba, title = "Baseline Model ROC & PR Curves")

    logger.info("Extracting top features...")
    coef = pipeline.named_steps['classifier'].coef_[0]
    summarize_feature_importance(pipeline, [coef], X_train.columns.tolist(), top_n = 20)

    model_path = "models/baseline_logistic_regression_pipeline.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to: {model_path}")
    logger.info("Baseline model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline logistic regression model.")
    parser.add_argument(
        "--data-path",
        type=str,
        required = True,
        help = "Full path to the CSV file (e.g., /absolute/path/xyz.csv)"
    )
    args = parser.parse_args()
    main(data_path=args.data_path)
