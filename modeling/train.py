# modeling/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
from feast import FeatureStore
import pickle
import os
import sqlite3
import logging
from utils.logging_config import setup_logging

def train_model():
    """
    Fetches data from Feast, trains multiple models,
    logs experiments with MLflow, and saves the best model.
    """
    logging.info("Starting model training process.")

    # -----------------------------
    # 1. Fetch Features from Feast
    # -----------------------------
    store = FeatureStore(repo_path=r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\feature_repo")

    # Get feature view
    feature_view = store.get_feature_view("churn_features_view")
    features = [f.name for f in feature_view.features]
    feature_service_names = [f"churn_features_view:{name}" for name in features]

    # Load entity data from SQLite
    conn = sqlite3.connect(r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\features.db")
    entity_df = pd.read_sql("SELECT customerID, event_timestamp, Churn FROM churn_features", conn)
    conn.close()

    # Get historical features
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service_names
    ).to_df()

    # Merge with entity_df to get target column
    training_data = pd.merge(training_data, entity_df[['customerID', 'Churn']], on='customerID')

    # -----------------------------
    # 2. Prepare Data for Modeling
    # -----------------------------
    logging.info("Preparing data for modeling.")

    # Drop columns not needed for modeling
    if 'event_timestamp' in training_data.columns:
        training_data.drop(columns=['event_timestamp'], inplace=True)

    # Split features and target
    X = training_data.drop('Churn', axis=1)
    y = training_data['Churn']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 3. Train and Evaluate Models
    # -----------------------------
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
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

            # ROC AUC: check if model has predict_proba
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
                "roc_auc": auc if auc is not None else 0
            })

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            logging.info(f"Model: {name}, F1-Score: {f1:.4f}")

            # Update best model
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
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info(f"Best model saved to {model_path}")


if __name__ == "__main__":
    setup_logging()
    train_model()
 