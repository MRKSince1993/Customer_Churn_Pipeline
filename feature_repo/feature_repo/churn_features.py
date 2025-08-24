# feature_repo/churn_features.py

from datetime import timedelta, datetime
import pandas as pd
from feast import Entity, FeatureView, ValueType, FileSource
import sqlite3
import os

# --- Step 1: Load features from SQLite and add event_timestamp ---
sqlite_path = r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\features.db"
features_table = "churn_features"
parquet_path = r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\silver\cleaned_churn_data.parquet"

# Connect to SQLite
conn = sqlite3.connect(sqlite_path)
features_df = pd.read_sql_query(f"SELECT * FROM {features_table}", conn)
conn.close()

# Add an event timestamp column (required by Feast)
features_df['event_timestamp'] = datetime.now()

# Export to Parquet for Feast
os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
features_df.to_parquet(parquet_path, index=False)

# --- Step 2: Define Feast Entity ---
customer = Entity(
    name="customerID",
    value_type=ValueType.STRING
)

# --- Step 3: Define Feast FileSource ---
feature_source = FileSource(
    path=parquet_path,
    event_timestamp_column="event_timestamp"
)

# --- Step 4: Define FeatureView (modern Feast syntax) ---
churn_feature_view = FeatureView(
    name="churn_features_view",
    entities=[customer],
    ttl=timedelta(days=365),
    source=feature_source,
    online=True,
)

print("Feast FeatureView created successfully!")
print(f"Entity: {customer.name}")
print(f"Feature source path: {feature_source.path}")
 