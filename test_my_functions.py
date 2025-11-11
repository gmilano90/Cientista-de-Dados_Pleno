"""
This script contains unit tests for functions defined in `my_functions.py`.

It uses pytest fixtures and assertions to verify:
- Data preprocessing (column creation, types, and consistency)
- Time-based train-test splitting
- Target encoding
- Categorical feature preparation and preprocessing
- Random Forest model training via grid search

Each test ensures the integrity of the machine learning pipeline components 
by running small, controlled examples without external dependencies.
"""
# Import libraries
import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import my_functions
import json


@pytest.fixture
def sample_df():
    """Creates a small simulated DataFrame for testing."""
    data = {
        "registry_date": pd.date_range("2023-01-01", periods=6, freq="M"),
        "yearmonth": pd.date_range("2023-01-01", periods=6, freq="M"),
        "country_origin_code": [1, 2, 3, 1, np.nan, 2],
        "consignee_name": ["A", "B", "C", "A", "B", "C"],
        "ncm_code": ["1001", "1002", "1001", "1003", "1002", "1001"],
        "consignee_code": ["X", "Y", "X", "Z", "Y", "X"],
        "channel": ["VERDE", "CINZA", "VERMELHO", "VERDE", "VERMELHO", "CINZA"],
        "clearance_place_entry": ["SP", "RJ", None, "SP", "RJ", "SP"],
        "transport_mode_pt": ["A", "B", "C", None, "A", "B"],
        "shipper_name": ["S1", "S2", "S3", None, "S1", "S2"]
    }
    return pd.DataFrame(data)


def test_preprocess_data(sample_df):
    df_processed = my_functions.preprocess_data(sample_df.copy())
    # Check if derived columns were created
    expected_cols = {"year", "month", "day", "weekday_name", "quarter"}
    assert expected_cols.issubset(df_processed.columns)
    # Check if categorical dtype was applied
    cat_cols = df_processed.select_dtypes("category").columns
    assert len(cat_cols) > 0


def test_time_based_split(sample_df):
    df = my_functions.preprocess_data(sample_df.copy())
    X_train, X_test, y_train, y_test = my_functions.time_based_train_test_split(df)
    # Ensure there is no temporal overlap
    assert X_train["registry_date_ts"].max() <= X_test["registry_date_ts"].min()
    # Ensure roughly correct proportion
    assert 0.6 <= len(X_train) / len(df) <= 0.8


def test_encode_target(sample_df):
    df = my_functions.preprocess_data(sample_df.copy())
    _, _, y_train, y_test = my_functions.time_based_train_test_split(df)
    y_train_enc, y_test_enc = my_functions.encode_target(y_train, y_test)
    # Check that only integers are present
    assert pd.api.types.is_integer_dtype(y_train_enc)


def test_prepare_categorical_features(sample_df):
    df = my_functions.preprocess_data(sample_df.copy())
    X_train, X_test, _, _ = my_functions.time_based_train_test_split(df)
    X_train_prep, X_test_prep, names, preproc = my_functions.prepare_categorical_features(X_train, X_test, threshold=2)
    # Check shapes and consistency
    assert X_train_prep.shape[1] == X_test_prep.shape[1]
    assert len(names) == X_train_prep.shape[1]
    assert isinstance(preproc, ColumnTransformer)


def test_apply_categorical_preprocessing(sample_df):
    df = my_functions.preprocess_data(sample_df.copy())
    X_train, X_test, _, _ = my_functions.time_based_train_test_split(df)
    _, _, _, preproc = my_functions.prepare_categorical_features(X_train, X_test, threshold=2)
    X_new = my_functions.apply_categorical_preprocessing(X_test.copy(), preproc, threshold=2)
    # Check if it returns a DataFrame with expected columns
    assert isinstance(X_new, pd.DataFrame)
    assert len(X_new.columns) > 0


def test_train_random_forest(tmp_path, sample_df):
    """Creates a temporary config file and tests whether GridSearch runs without error."""
    df = my_functions.preprocess_data(sample_df.copy())
    X_train, X_test, y_train, y_test = my_functions.time_based_train_test_split(df)
    config = {
        "random_forest": {
            "n_jobs": 1,
            "verbose": 0,
            "random_state": 42,
            "n_splits": 2,
            "smote_strategy": {"1": 1, "2": 1},
            "undersample_strategy": {"0": 1},
            "param_grid": {"clf__n_estimators": [5], "clf__max_depth": [2]}
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    grid_search = my_functions.train_random_forest(X_train, y_train, config_path=str(config_path))
    assert hasattr(grid_search, "best_estimator_")
