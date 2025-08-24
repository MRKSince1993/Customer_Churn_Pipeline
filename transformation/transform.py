# transformation/transform.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sqlite3
import os
from utils.logging_config import setup_logging
import logging

def transform_data(input_path: str, db_path: str):
    """
    Loads cleaned data, engineers features, transforms them, and stores
    the result in an SQLite database ready for Feast.
    """
    logging.info(f"Starting data transformation from {input_path}.")

    # Load the cleaned parquet file
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded data with columns: {df.columns.tolist()}")

    # --- Feature Engineering ---
    logging.info("Engineering new features.")

    # Tenure groups
    max_tenure = df['tenure'].max()
    bins = [0, 12, 48, max(max_tenure, 48) + 1]  # ensure monotonically increasing
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=bins,
        labels=['New', 'Established', 'Loyal'],
        right=False
    )
    df['tenure_group'] = df['tenure_group'].cat.add_categories(['Unknown']).fillna('Unknown')

    # Optional service columns
    optional_services = [
        'OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies'
    ]
    existing_optional_services = [col for col in optional_services if col in df.columns]

    # Count number of optional services a customer has
    if existing_optional_services:
        df['num_optional_services'] = df[existing_optional_services].apply(
            lambda row: sum(s == 'Yes' for s in row), axis=1
        )
    else:
        df['num_optional_services'] = 0

    # Save customerID separately
    if 'customerID' not in df.columns:
        raise ValueError("customerID column missing in input data!")
    customer_ids = df['customerID'].copy()

    # Drop optional service columns for transformation
    df_trans = df.drop(columns=existing_optional_services, errors='ignore')

    # Feature: monthly charge per tenure
    df_trans['monthly_charge_per_tenure'] = df_trans['MonthlyCharges'] / (df_trans['tenure'] + 1)

    # Binary columns mapping Yes/No to 1/0
    binary_cols = ['PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_trans.columns:
            df_trans[col] = df_trans[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Gender mapping
    if 'gender' in df_trans.columns:
        df_trans['gender'] = df_trans['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Identify categorical and numerical features excluding customerID
    categorical_features = df_trans.select_dtypes(include=['object','category']).columns.drop('customerID', errors='ignore')
    numerical_features = df_trans.select_dtypes(include=np.number).columns.tolist()

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Fit and transform
    transformed_data = preprocessor.fit_transform(df_trans)

    # Get final column names
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_columns = numerical_features + list(ohe_cols)

    # Convert to DataFrame
    final_df = pd.DataFrame(transformed_data, columns=final_columns, index=df_trans.index)

    # Reattach customerID and add event_timestamp for Feast
    final_df['customerID'] = customer_ids.values
    final_df['event_timestamp'] = pd.Timestamp.now()

    # --- Storage in SQLite ---
    logging.info(f"Storing transformed data to SQLite DB at {db_path}.")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    final_df.to_sql('churn_features', conn, if_exists='replace', index=False)
    conn.close()

    logging.info("Transformation and storage complete.")
    print(f"Transformed data stored in SQLite DB at {db_path}")


# --- Main ---
if __name__ == "__main__":
    setup_logging()
    transform_data(
        input_path=r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\cleaned_churn_data.parquet",
        db_path=r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\features.db"
    )
 