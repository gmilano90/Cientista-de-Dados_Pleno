"""
This script trains and evaluates a Random Forest classifier using time-based data splitting.
It performs the following steps:
1. Loads the dataset from a parquet file.
2. Applies preprocessing and feature engineering to create new features.
3. Splits the data into training and testing sets based on timestamps.
4. Applies feature engineering of categorical features.
5. Encodes target labels into integers.
6. Performs hyperparameter tuning using GridSearchCV on a Random Forest model.
7. Evaluates model performance and prints classification metrics.
8. Saves the best trained model to disk.
"""

# Import libraries 
import pandas as pd
import my_functions
import joblib
import warnings
import logging
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting model training pipeline...")

try:
    # Load data
    df = pd.read_parquet("data/sample_data.parquet")
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    # Preprocess data
    df = my_functions.preprocess_data(df)
    logger.info("Data preprocessing completed.")

    # Time-based train-test split
    X_train, X_test, y_train, y_test = my_functions.time_based_train_test_split(df)
    logger.info("Train-test split completed.")

    # Feature Engineering of Categorical Features
    X_train, X_test, final_feature_names, cat_preprocessor = my_functions.prepare_categorical_features(X_train, X_test)
    joblib.dump(cat_preprocessor, "chosen_model/cat_preprocessor.pkl") 
    logger.info("Feature engineering completed and preprocessing artifacts saved.")

    # Transform categorical label into integer
    y_train, y_test = my_functions.encode_target(y_train, y_test)
    logger.info("Target encoding completed.")

    # Random Forest Classifier Training
    grid_search = my_functions.train_random_forest(X_train, y_train)
    logger.info("Random Forest model training completed.")

    # Retrieve grid search results and evaluate
    best_rf, y_pred = my_functions.evaluate_search(grid_search, X_test)
    metrics = my_functions.evaluate_model_performance(y_test, y_pred)
    logger.info(f"Model evaluation completed. Metrics: {metrics}")

    # Save model
    joblib_file = "chosen_model/best_rf_model_resampling.pkl"
    joblib.dump(best_rf, joblib_file)
    logger.info(f"Model saved to {joblib_file}")

except Exception as e:
    logger.exception("An error occurred during model training.")
finally:
    logger.info("Training pipeline finished.")

