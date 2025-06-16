from datetime import datetime
API_KEY = "f26e07fed4df3480555e31cb4a772df7"

# --- Configuration ---
TARGET = "recession"
PREDICTION_HORIZON_MONTHS = 1 # Predict 1 month before recession starts
MAIN_TRAIN_TEST_SPLIT_DATE = datetime(2007, 1, 1) # Data before this is for training/tuning, after for final test
N_ITER_SEARCH = 100 # Small iterations due to limited data
RANDOM_STATE = 42

