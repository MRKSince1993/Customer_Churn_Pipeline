# ingestion/ingest_csv.py

import pandas as pd
import logging
from utils.logging_config import setup_logging
import argparse
import os
from datetime import datetime

# Configure logging

def ingest_csv_data(file_path: str, output_dir: str):
    """
    Ingests data from a CSV file and saves it to a specified output directory.

    Args:
        file_path (str): The path to the source CSV file.
        output_dir (str): The directory to save the ingested data.
    """
    try:
        logging.info(f"Starting ingestion from CSV file: {file_path}")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at {file_path}")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read {len(df)} rows from {file_path}")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the output file path with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(output_dir, f"telco_churn_{timestamp}.csv")
        
        # Save the DataFrame to the output directory
        df.to_csv(output_file, index=False)
        
        logging.info(f"Successfully ingested data and saved to {output_file}")
        print(f"Data saved to {output_file}")

    except FileNotFoundError as e:
        logging.error(f"Ingestion failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV ingestion: {e}")
        raise

if __name__ == "__main__":
    # CORRECT: Call the setup_logging() function at the start.
    setup_logging()

    parser = argparse.ArgumentParser(description="Ingest data from a CSV file.")
    parser.add_argument("--source_file", 
                        default=r"C:\Users\mrakkuma\OneDrive - Magna\BITS\Sem2\DMML\Src_Files\WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                        help="Path to the source CSV file.")
    parser.add_argument("--output_dir", 
                        default=r"C:\Users\mrakkuma\OneDrive - Magna\BITS\Sem2\DMML\Assignment\Assignment\customer_churn_pipeline\data\raw", 
                        help="Directory to save the ingested raw data.")
    
    args = parser.parse_args()
    
    ingest_csv_data(args.source_file, args.output_dir)
