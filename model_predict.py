"""
This script loads a trained Random Forest model and uses it to make predictions 
on new data. It loads the same preprocessing pipeline and feature names used in training 
to ensure perfect feature alignment.

NOTE:
This script uses the same dataset that was originally used for training as a copy renamed as "new data".
Since no unseen dataset is available, this allows the script to demonstrate the full prediction flow
(model loading, preprocessing, and inference) without requiring new inputs.
"""

# Import libraries
import pandas as pd
import joblib
import my_functions
import warnings
import logging
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("logs/prediction.log"),
        logging.StreamHandler()
    ]
)

# Load saved model and preprocessing
model_path = "chosen_model/best_rf_model_resampling.pkl"
cat_preprocessor_path = "chosen_model/cat_preprocessor.pkl"

logging.info("Loading model and preprocessing objects...")
best_rf = joblib.load(model_path)
cat_preprocessor = joblib.load(cat_preprocessor_path)
logging.info("Model and preprocessing loaded successfully.")

# Load new data
logging.info("Loading new data...")
new_data = pd.read_parquet("data/new_data.parquet")
logging.info(f"New data loaded. Shape: {new_data.shape}")

# Preprocess new data
logging.info("Preprocessing new data...")
new_data = my_functions.preprocess_data(new_data)
X_new = my_functions.apply_categorical_preprocessing(new_data, cat_preprocessor, threshold=35)
logging.info(f"New data transformed. Shape: {X_new.shape}")

# Generate predictions
logging.info("Generating predictions...")
y_pred_new = best_rf.predict(X_new)
logging.info("Predictions generated successfully.")

# Save predictions
output_path = "predictions/new_data_predictions.csv"
pd.DataFrame({"prediction": y_pred_new}).to_csv(output_path, index=False)
logging.info(f"Predictions saved to {output_path}")


