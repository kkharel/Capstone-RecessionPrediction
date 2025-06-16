# preprocessing the dataset

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
import argparse

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_preprocessor(X_sample, target_col = None):
    """
    Dynamically infers numeric and categorical columns from sample X.
    
    Parameters:
    - X_sample (pd.DataFrame): DataFrame after feature engineering
    - target_col (str, optional): Target column to exclude from feature lists, if present in X_sample.

    Returns:
    - ColumnTransformer
    """
    if not isinstance(X_sample, pd.DataFrame):
        raise ValueError("X_sample must be a pandas DataFrame.")

    if target_col and target_col in X_sample.columns:
        X_sample = X_sample.drop(columns = [target_col])
        logger.warning(f"'{target_col}' found and dropped from X_sample for preprocessing.")

    numeric_features = X_sample.select_dtypes(include = [np.number]).columns.tolist()
    categorical_features = X_sample.select_dtypes(include = ['object', 'category']).columns.tolist()

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_features)
    ])
    return preprocessor


def main(data_path, target_col = None):
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    
    preprocessor = get_preprocessor(df, target_col = target_col)
    logger.info(f"Preprocessor created with {len(preprocessor.transformers)} transformers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Preprocessor script standalone.")
    parser.add_argument("data_path", type = str, help = "Full path to CSV data file")
    parser.add_argument("--target_col", type = str, default = None, help = "Name of target column to exclude")

    args = parser.parse_args()
    main(args.data_path, args.target_col)