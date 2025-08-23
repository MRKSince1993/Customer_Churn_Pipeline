# ingestion/ingest_hf_data.py

import pandas as pd
import numpy as np
import sqlite3
import logging
from utils.logging_config import setup_logging
import os
from datasets import load_dataset
from datetime import datetime

# Configure logging

def get_customer_ids_from_source(source_csv_path: str) -> list:
    """Helper function to get customer IDs from the main CSV dataset."""
    try:
        df = pd.read_csv(source_csv_path)
        logging.info(f"Loaded {len(df)} customer IDs from {source_csv_path}")
        return df["customerID"].tolist()
    except FileNotFoundError:
        logging.error(f"Customer ID source file not found: {source_csv_path}")
        return

def ingest_and_process_hf_data(customer_ids: list, db_path: str):
    """
    Ingests data from Hugging Face, simulates interaction features,
    and stores the result in an SQLite database.

    Args:
        customer_ids (list): A list of customer IDs to generate data for.
        db_path (str): The path to the SQLite database for storage.
    """
    try:
        # 1. Load dataset from Hugging Face Hub
        logging.info("Loading 'Kaludi/Customer-Support-Responses' dataset from Hugging Face.")
        hf_dataset = load_dataset("Kaludi/Customer-Support-Responses")
        # Convert to pandas DataFrame for easier manipulation
        interactions_df = hf_dataset['train'].to_pandas()
        logging.info(f"Successfully loaded {len(interactions_df)} sample interactions.")

        # 2. Simulate interaction features for our customers
        logging.info("Simulating interaction features for each customer.")
        
        # Set a seed for reproducibility
        np.random.seed(42)
        
        simulated_data = []
        for cid in customer_ids:
            # Simulate number of support calls (e.g., between 0 and 5)
            num_calls = np.random.randint(0, 6)
            
            # Simulate a satisfaction score (e.g., 1 to 5)
            satisfaction = np.random.randint(1, 6) if num_calls < 3 else np.random.randint(1, 4)
            
            # Simulate days since last interaction (e.g., 1 to 120 days)
            last_interaction = np.random.randint(1, 121)
            
            simulated_data.append({
                "customerID": cid,
                "support_calls": num_calls,
                "satisfaction_score": satisfaction,
                "last_interaction_days_ago": last_interaction
            })
            
        final_df = pd.DataFrame(simulated_data)
        logging.info(f"Generated simulated interaction data for {len(final_df)} customers.")

        # 3. Store the data in SQLite database
        logging.info(f"Storing processed data into SQLite DB at {db_path}")
        
        # Create directory for DB if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        final_df.to_sql('customer_interactions', conn, if_exists='replace', index=False)
        conn.close()
        
        logging.info("Successfully stored interaction data in the database.")
        print(f"Interaction data has been successfully ingested and stored in {db_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during Hugging Face data ingestion: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    TELCO_CSV_PATH = r"C:\Users\mrakkuma\OneDrive - Magna\BITS\Sem2\DMML\Assignment\Assignment\customer_churn_pipeline\data\raw/Telco-Customer-Churn.csv"
    DB_STORAGE_PATH = r"C:\Users\mrakkuma\OneDrive - Magna\BITS\Sem2\DMML\Assignment\Assignment\customer_churn_pipeline\data\silver\interactions.db"
    
    customer_ids = get_customer_ids_from_source(TELCO_CSV_PATH)
    if customer_ids:
        ingest_and_process_hf_data(customer_ids, DB_STORAGE_PATH)
