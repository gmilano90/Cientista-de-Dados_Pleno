''' This script contains the functions used by the notebooks and scripts of this project.'''

# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# PREPROCESS DATA
def preprocess_data(df):
    """
    Preprocess the input DataFrame by:
      - Handling missing values and redundant columns
      - Creating time-based features
      - Engineering risk and lag features
      - Optimizing data types for efficiency

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    """

    # Handle missing values and type adjustments
    df['country_origin_code'] = df['country_origin_code'].astype('Int64').astype('object')
    df['registry_date'] = df['registry_date'].fillna(df['yearmonth'])
    df.drop(columns=['yearmonth', 'consignee_name'], inplace=True)

    fill_values = {
        'clearance_place_entry': 'UNKNOWN_ENTRY',
        'transport_mode_pt': 'UNKNOWN_TRANSPORT',
        'ncm_code': 'UNKNOWN_NCM',
        'shipper_name': 'UNKNOWN_SHIPPER',
        'country_origin_code': 'UNKNOWN_COUNTRY'
    }
    df.fillna(value=fill_values, inplace=True)

    # Time-based features
    df['year'] = df['registry_date'].dt.year
    df['month'] = df['registry_date'].dt.month
    df['day'] = df['registry_date'].dt.day
    df['weekday_name'] = df['registry_date'].dt.day_name()
    df['quarter'] = df['registry_date'].dt.quarter
    df['year_month'] = df['registry_date'].dt.to_period('M')
    df['registry_date_ts'] = df['registry_date'].view('int64') // 10**9
    df.drop(columns=['registry_date'], inplace=True)

    # Risk feature creation helper (by NCM, Country and Importer)
    def add_risk_features(df, group_col, prefix):
        risk = df.groupby(group_col)['channel'].value_counts(normalize=True).unstack(fill_value=0)
        for color, label in [('VERMELHO', 'red'), ('CINZA', 'gray')]:
            col_name = f'{prefix}_{label}_risk'
            df = df.merge(risk[[color]], left_on=group_col, right_index=True, how='left')
            df.rename(columns={color: col_name}, inplace=True)
        return df

    # Create risk features 'ncm_red_risk','ncm_gray_risk','country_red_risk','country_gray_risk', 'importer_red_risk' and 'importer_gray_risk'.
    df = add_risk_features(df, 'ncm_code', 'ncm')
    df = add_risk_features(df, 'country_origin_code', 'country')
    df = add_risk_features(df, 'consignee_code', 'importer')

    # Lag features creation helper (NCM and Country ratios)
    def add_lag_ratio(df, group_col, prefix):
        for color, label in [('VERMELHO', 'red'), ('CINZA', 'gray')]:
            ratio_name = f'{prefix}_{label}_ratio'
            lag_name = f'lag1_{prefix}_{label}_ratio'

            stats = (
                df.groupby([group_col, 'year_month'])['channel']
                .apply(lambda x, c=color: (x == c).mean())
                .reset_index(name=ratio_name)
            )
            stats[lag_name] = stats.groupby(group_col)[ratio_name].shift(1)
            df = df.merge(stats[[group_col, 'year_month', lag_name]], on=[group_col, 'year_month'], how='left')
            df[lag_name] = df[lag_name].fillna(0)
        return df

    # Create lag features 'lag1_ncm_red_ratio', 'lag1_ncm_gray_ratio', 'lag1_country_red_ratio' and 'lag1_country_gray_ratio'
    df = add_lag_ratio(df, 'ncm_code', 'ncm')
    df = add_lag_ratio(df, 'country_origin_code', 'country')

    # Create one last Lag feature: importer’s previous channel ('lag1_channel_importer')
    df['lag1_channel_importer'] = (
        df.groupby('consignee_code', group_keys=False)
        .apply(lambda g: g.sort_values('registry_date_ts')['channel'].shift(1))
    ).fillna('UNKNOWN')

    # Derived and final features
    df['year_month'] = df['year_month'].dt.year * 100 + df['year_month'].dt.month
    df['ncm_chapter'] = df['ncm_code'].str[:2]

    if 'document_number' in df.columns:
        df.drop(columns=['document_number'], inplace=True)

    # Optimize dtypes
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].astype('category')

    return df

# TIME-BASED TRAIN-TEST SPLIT
def time_based_train_test_split(df, timestamp_col='registry_date_ts', target_col='channel', train_size=0.7):
    """
    Perform a time-based train-test split based on a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including features and target.
    timestamp_col : str
        Name of the timestamp column (must be numeric).
    target_col : str
        Name of the target column to predict.
    train_size : float
        Proportion of data to use for training (e.g., 0.7 = 70%).

    Returns
    ----------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """

    # Sort data chronologically
    data_sorted = df.sort_values(timestamp_col)

    # Define cutoff index
    split_point = int(len(data_sorted) * train_size)

    # Split data
    train_data = data_sorted.iloc[:split_point]
    test_data = data_sorted.iloc[split_point:]

    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # Print readable date ranges
    print(f"Train period: {pd.to_datetime(X_train[timestamp_col], unit='s').min()} -> {pd.to_datetime(X_train[timestamp_col], unit='s').max()}")
    print(f"Test period:  {pd.to_datetime(X_test[timestamp_col], unit='s').min()} -> {pd.to_datetime(X_test[timestamp_col], unit='s').max()}")

    return X_train, X_test, y_train, y_test

# FEATURE ENGINEERING OF CATEGORICAL FEATURES
def prepare_categorical_features(X_train, X_test, threshold=35):
    """
    Perform categorical feature engineering:
    - Frequency encoding for high-cardinality categorical columns
    - One-hot encoding for low-cardinality columns
    - Pass numeric columns directly

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets
    threshold : int, optional
        Number of unique values above which a column is considered high-cardinality.

    Returns
    -------
    X_train_prepared, X_test_prepared : np.ndarray
        Transformed feature arrays ready for modeling.
    final_feature_names : list
        List of final feature names after transformation.
    """

    # 1. Frequency encoding for high-cardinality categorical columns
    cat_cols = X_train.select_dtypes(include=['category', 'object']).columns
    high_card_cols = [col for col in cat_cols if X_train[col].nunique() > threshold]
    print("High-cardinality columns detected:", high_card_cols)

    # Frequency encode
    for col in high_card_cols:
        freq_map = X_train[col].value_counts(normalize=True)
        X_train[col + '_freq'] = X_train[col].map(freq_map)
        X_test[col + '_freq'] = X_test[col].map(freq_map).fillna(0)

    # Drop original high-cardinality columns
    X_train = X_train.drop(columns=high_card_cols)
    X_test = X_test.drop(columns=high_card_cols)

    # 2. One-hot encoding for remaining low-cardinality categorical columns
    # Identify categorical vs numeric columns
    cat_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Build preprocessing transformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    # Ensure all categorical columns are strings
    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)

    # Fit-transform
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    # Get final feature names for analysis
    final_feature_names = preprocessor.get_feature_names_out()

    print(f"Train shape: {X_train_prepared.shape}")
    print(f"Test shape:  {X_test_prepared.shape}")
    print(f"Features length: {len(final_feature_names)}")

    return X_train_prepared, X_test_prepared, final_feature_names, preprocessor

# TRANSFORM CATEGORICAL LABEL INTO INTEGER
def encode_target (y_train, y_test):
    """
    Encode target labels into integer values.
    
    Parameters
    ----------
    y_train : pandas.Series
        Training target labels.
    y_test : pandas.Series
        Test target labels.
        
    Returns
    -------
    y_train_encoded : pandas.Series
        Encoded training target labels.
    y_test_encoded : pandas.Series
        Encoded test target labels.
    """
    
    # Define the label mapping
    label_map = {'VERDE': 0, 'VERMELHO': 1, 'AMARELO': 2, 'CINZA': 3}
    
    # Apply mapping
    y_train_encoded = y_train.map(label_map).astype(int)
    y_test_encoded = y_test.map(label_map).astype(int)

    print("Encoded class distribution:")
    print("Train:\n", y_train_encoded.value_counts().to_dict())
    print("Test:\n", y_test_encoded.value_counts().to_dict())

    return y_train_encoded, y_test_encoded

# SCALE NUMERICAL FEATURES
def scale_numerical_features(X_train, X_test, feature_names):
    """
    Scale numerical features of X_train and X_test using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Transformed training features (after one-hot encoding/frequency encoding).
    X_test : np.ndarray
        Transformed test features.
    feature_names : list or np.ndarray
        Names of the features corresponding to columns in X_train/X_test.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Scaled training and test feature arrays.
    """
    
    # Identify numeric columns
    numeric_idx = [i for i, name in enumerate(feature_names) if name.startswith('num__')]

    # Initialize scaler
    scaler = StandardScaler()

    # Copy arrays to avoid modifying original
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit scaler on training numeric columns and transform both train and test
    X_train_scaled[:, numeric_idx] = scaler.fit_transform(X_train[:, numeric_idx])
    X_test_scaled[:, numeric_idx] = scaler.transform(X_test[:, numeric_idx])

    print(f"Scaled {len(numeric_idx)} numerical features.")

    return X_train_scaled, X_test_scaled

# EVALUATE LOGISTIC REGRESSION
def evaluate_logistic_regression(y_true, y_pred):
    """
    Evaluate a classifier's performance on test data and print metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels by the model.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing accuracy, macro F1, and balanced accuracy.
    """
    print("--- LOGISTIC REGRESSION RESULTS ---")
    
    # Confusion matrix
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_true, y_pred))
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred))
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Return metrics as a dictionary
    metrics_dict = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc
    }
    
    return metrics_dict

# COMPUTE PERMUTATION IMPORTANCE FOR LOGISTIC REGRESSION
def logistic_regression_permutation_importance(model, X_test, y_test, feature_names, top_n=20, plot=True):
    """
    Retrieve and rank feature importances for logistic regression using permutation importance.
    
    Parameters
    ----------
    model : fitted LogisticRegression
        The trained logistic regression model.
    X : array-like
        Features used for testing.
    y : array-like
        True labels corresponding to X.
    feature_names : list
        Names of the features.
    top_n : int
        Number of top features to display/plot.
    plot : bool
        Whether to plot top features.
    
    Returns
    -------
    perm_importance_df : pd.DataFrame
        Feature importances based on permutation importance, descending.
    """
    
    # Compute permutation importance
    perm_result = permutation_importance(model, X_test, y_test, scoring='f1_macro', n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_result.importances_mean
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # Plot top features
    if plot:
        plt.figure(figsize=(10, 6))
        plt.barh(perm_importance_df['feature'][:top_n][::-1],
                 perm_importance_df['importance'][:top_n][::-1])
        plt.xlabel('Mean Decrease in F1 Macro')
        plt.title('Top Features by Permutation Importance')
        plt.show()
    
    return perm_importance_df

# RANDOM FOREST CLASSIFIER TRAINING WITH RESAMPLING AND GRID SEARCH
def train_random_forest(X_train, y_train, config_path="config.json"):
    """
    Train a Random Forest classifier using a pipeline with SMOTE oversampling,
    undersampling, and GridSearchCV for hyperparameter tuning.
    Model parameters and resampling configurations are loaded from a JSON file.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature set.
    y_train : pd.Series or np.ndarray
        Training target labels.
    config_path : str, optional
        Path to the JSON configuration file. Default is "config.json".

    Returns
    -------
    grid_search : GridSearchCV
        The fitted GridSearchCV object containing the best model and parameters.
    """

    # Load configuration file
    with open(config_path, "r") as f:
        config = json.load(f)
    rf_config = config["random_forest"]

    # Extract parameters
    n_jobs = rf_config.get("n_jobs", 2)
    verbose = rf_config.get("verbose", 2)
    random_state = rf_config.get("random_state", 42)
    n_splits = rf_config.get("n_splits", 5)
    smote_strategy = {int(k): v for k, v in rf_config.get("smote_strategy", {2: 1432, 3: 1432}).items()}
    undersample_strategy = {int(k): v for k, v in rf_config.get("undersample_strategy", {0: 1432}).items()}
    param_grid = rf_config.get("param_grid", {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 20],
        'clf__min_samples_split': [5, 10]
    })

    # Convert to pandas DataFrame if necessary (for consistency in feature checks)
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    # Ensure X_train is numeric (for test safety)
    if X_train.select_dtypes(include=['object', 'category']).shape[1] > 0:
        X_train = pd.get_dummies(X_train, drop_first=True)

    # Filter strategies to include only classes present in y_train
    available_classes = set(y_train.unique())
    smote_strategy = {k: v for k, v in smote_strategy.items() if k in available_classes}
    undersample_strategy = {k: v for k, v in undersample_strategy.items() if k in available_classes}

    # Initialize base model
    rf = RandomForestClassifier(random_state=random_state)

    # Define resamplers
    over = SMOTE(sampling_strategy=smote_strategy, random_state=random_state) if smote_strategy else None
    under = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=random_state) if undersample_strategy else None

    # Stratified K-Folds for balanced cross-validation
    skf = StratifiedKFold(n_splits=n_splits)

    # Build pipeline dynamically
    steps = []
    if over: steps.append(('over', over))
    if under: steps.append(('under', under))
    steps.append(('clf', rf))
    pipeline = Pipeline(steps)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=skf,
        n_jobs=n_jobs,
        verbose=verbose
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    return grid_search

# RETRIEVE GRID SEARCH RESULTS
def evaluate_search(search_obj, X_test):
    """
    Display best parameters and score from GridSearchCV or RandomizedSearchCV,
    retrieve the best model, and make predictions on the test set.
    
    Parameters
    ----------
    search_obj : GridSearchCV or RandomizedSearchCV
        The fitted search object.
    X_test : pandas.DataFrame or numpy.ndarray
        The test feature set.
    
    Returns
    -------
    best_model : estimator
        The best estimator found by the search.
    y_pred : numpy.ndarray
        Predictions on the test set.
    """

    print("--- SEARCH RESULTS ---")
    print(f"Best Parameters: {search_obj.best_params_}")
    print(f"Best Macro F1 Score: {search_obj.best_score_:.4f}")
    
    # Retrieve best model
    best_model = search_obj.best_estimator_
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    return best_model, y_pred

# RETRIEVE MODEL PERFORMANCE METRICS
def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate and print classification performance metrics, including:
    - Confusion matrix
    - Classification report (precision, recall, F1-score per class)
    - Overall metrics: Accuracy, Macro F1, and Balanced Accuracy
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
        
    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, macro F1 score, and balanced accuracy.
    """
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX:")
    print(cm)
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred))
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print("\nOVERALL METRICS:")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro F1 Score:     {macro_f1:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    
    # Return metrics for potential logging or further analysis
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }

# RETRIEVE RANDOM FOREST FEATURE IMPORTANCES
def plot_feature_importances(model, feature_names, top_n=20, plot=True):
    """
    Get feature importances from a fitted RandomForest model,
    whether it's a standalone model or part of a pipeline (e.g., GridSearchCV with resampling).
    
    Parameters
    ----------
    model : fitted model or pipeline
        Model object (can be a RandomForestClassifier or a Pipeline with a classifier named 'clf').
    feature_names : list
        List of feature names corresponding to model input.
    top_n : int, optional
        Number of top features to display. Default is 20.
    plot : bool, optional
        Whether to plot the top features. Default is True.

    Returns
    -------
    feature_importance_df : pd.DataFrame
        DataFrame with features and their importance, sorted descending.
    """
    
    # Detect where feature_importances_ lives
    if hasattr(model, "feature_importances_"):
        # Case 1: model is a standalone RandomForestClassifier
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and "clf" in model.named_steps:
        # Case 2: model is a pipeline with a classifier step
        importances = model.named_steps["clf"].feature_importances_
    else:
        raise AttributeError(
            "The provided model does not have feature_importances_. "
            "Ensure it's a tree-based model or a pipeline with a 'clf' step."
        )
    
    # Create a DataFrame pairing feature names with their importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)  # Sort descending by importance
    feature_importance_df.reset_index(drop=True, inplace=True)  # Reset index
    
    # Display the top_n features in the console
    print(feature_importance_df.head(top_n))
    
    # Optionally, plot the top_n feature importances
    if plot:
        plt.figure(figsize=(10, 6))
        # Horizontal bar chart: reverse order to show highest importance at top
        plt.barh(
            feature_importance_df['feature'][:top_n][::-1],
            feature_importance_df['importance'][:top_n][::-1]
        )
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.show()
    
    # Return the full sorted DataFrame
    return feature_importance_df

# APPLY CATEGORICAL PREPROCESSING TO NEW DATA FOR PREDICTION
def apply_categorical_preprocessing(df, preprocessor, threshold=35):
    """
    Applies the same categorical preprocessing steps used during training:
    1. Frequency encoding for high-cardinality categorical features
    2. Transformation using the saved preprocessor (OneHotEncoder + numeric passthrough)

    Parameters
    ----------
    df : pd.DataFrame
        The new dataset to preprocess.
    preprocessor : sklearn.compose.ColumnTransformer
        The fitted preprocessor loaded from training.
    threshold : int, optional
        Number of unique values above which a column is considered high-cardinality.

    Returns
    -------
    X_prepared : pd.DataFrame
        Fully transformed dataset ready for prediction, with same feature structure as training.
    """

    # Step 1: Frequency encoding for high-cardinality categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > threshold]
    print("High-cardinality columns detected:", high_card_cols)

    for col in high_card_cols:
        freq_map = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq_map).fillna(0)

    # Drop original high-cardinality columns
    df = df.drop(columns=high_card_cols)

    # Step 2: Apply saved preprocessor
    # Ensure categorical columns are strings (as during training)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].astype(str)

    X_transformed = preprocessor.transform(df)
    final_feature_names = preprocessor.get_feature_names_out()

    X_prepared = pd.DataFrame(X_transformed, columns=final_feature_names)
    print(f" Data transformed. Final shape: {X_prepared.shape}")

    return X_prepared

