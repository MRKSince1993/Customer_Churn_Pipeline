# modeling/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from feast import FeatureStore
import pickle
import os
import sqlite3
import logging
from utils.logging_config import setup_logging

# Reduce Feast registry INFO spam
logging.getLogger("feast").setLevel(logging.WARNING)

def train_model():
    """
    Fetches data from Feast, trains multiple models,
    logs experiments with MLflow, and saves the best model.
    """
    logging.info("Starting model training process.")

    # -----------------------------
    # 1. Fetch Features from Feast
    # -----------------------------
    store = FeatureStore(
        repo_path=r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\feature_repo\feature_repo"
    )

    # Get feature view
    feature_view = store.get_feature_view("churn_features_view")
    features = [f"churn_features_view:{f.name}" for f in feature_view.features]

    # Load entity data from SQLite
    conn = sqlite3.connect(
        r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\features.db"
    )
    entity_df = pd.read_sql(
        "SELECT customerID, event_timestamp, Churn_Yes FROM churn_features", conn
    )
    conn.close()

    # --- FIX: Perform targeted data type conversions ---
    # 1. Convert 'event_timestamp' to a proper datetime object. This is required by Feast.
    entity_df["event_timestamp"] = pd.Timestamp.now()

    # 2. Convert 'customerID' to a string to match the offline store's schema.
    # This is the corrected line that targets ONLY the customerID column.
    entity_df["customerID"] = entity_df["customerID"].astype(str)
    # ----------------------------------------------------

    # -----------------------------
    # Diagnostics Section
    # -----------------------------
    logging.info("--- Running Pre-Feast Diagnostics ---")
    logging.info("Entity DF data types:\n" + str(entity_df.dtypes))
    assert pd.api.types.is_datetime64_any_dtype(entity_df['event_timestamp']), \
        "CRITICAL: 'event_timestamp' column is not a datetime type!"
    min_ts = entity_df['event_timestamp'].min()
    max_ts = entity_df['event_timestamp'].max()
    logging.info(f"Entity DF timestamp range: FROM {min_ts} TO {max_ts}")
    if entity_df.empty:
        logging.warning("Entity DF is empty before calling Feast. No features can be retrieved.")
    logging.info("--- Diagnostics Complete ---")
    # -----------------------------

    # Get historical features
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    # Debugging log to check the output from Feast
    logging.info(f"DataFrame shape immediately after Feast call: {training_data.shape}")

    # Merge with label column from the original entity_df
    if "Churn_Yes" not in entity_df.columns:
        raise KeyError("Target column 'Churn_Yes' not found in entity_df!")

    training_data = pd.merge(
        training_data,
        entity_df[["customerID", "Churn_Yes"]],
        on="customerID",
        how="inner"
    ).drop_duplicates()

    logging.info(f"Training data shape after merge: {training_data.shape}")
    logging.info(f"Training data sample:\n{training_data.head().to_string()}")

    # -----------------------------
    # 2. Prepare Data for Modeling
    # -----------------------------
    logging.info("Preparing data for modeling.")

    if "event_timestamp" in training_data.columns:
        training_data.drop(columns=["event_timestamp"], inplace=True)

    if training_data.empty:
        raise ValueError("Training data is empty after preprocessing. Check entity_df and feature store timestamps!")

	    # Correctly separate features (X) and the target variable (y).
    y = training_data["Churn_Yes_y"]
    X = training_data.drop(columns=["customerID", "Churn_Yes_x", "Churn_Yes_y"])

# Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- NEW: Impute missing values (NaNs) ---
    # Many models cannot handle missing values. We use an imputer to fill them
    # with the mean of their respective columns.
    logging.info("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    # -----------------------------------------

    logging.info(f"Train/Test split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # -----------------------------
    # 3. Train and Evaluate Models
    # -----------------------------
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    mlflow.set_experiment("Churn_Prediction_Experiment")

    best_f1 = 0
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logging.info(f"Training model: {name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            if hasattr(model, "predict_proba"):
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                auc = None

            # Log params and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": auc if auc is not None else 0,
            })

            mlflow.sklearn.log_model(model, artifact_path="model")

            logging.info(f"Model: {name}, F1-Score: {f1:.4f}")

            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

    logging.info(f"Best model: {best_model_name} with F1-Score: {best_f1:.4f}")

    # -----------------------------
    # 4. Save the Best Model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_churn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logging.info(f"Best model saved to {model_path}")


if __name__ == "__main__":
    setup_logging()
    train_model()