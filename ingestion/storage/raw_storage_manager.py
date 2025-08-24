# storage/raw_storage_manager.py


import os
import shutil
from datetime import datetime
import logging
from utils.logging_config import setup_logging


# Configure logging


def store_raw_data(source_file_path: str, data_source_name: str, base_dir: str):
    """
    Stores a raw data file in a partitioned data lake structure.


    Args:
        source_file_path (str): The path to the temporary ingested file.
        data_source_name (str): The name of the data source (e.g., 'telco_csv').
        base_dir (str): The base directory of the data lake (e.g., 'data/bronze').
    """
    try:
        logging.info(f"Storing raw data for source: {data_source_name}")
       
        # Get current date for partitioning
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
       
        # Construct the partitioned path
        partition_path = os.path.join(
            base_dir,
            f"source={data_source_name}",
            f"year={year}",
            f"month={month}",
            f"day={day}"
        )
       
        # Create the directory structure
        os.makedirs(partition_path, exist_ok=True)
       
        # Move the file to the final destination
        destination_path = os.path.join(partition_path, os.path.basename(source_file_path))
        shutil.copy(source_file_path, destination_path)
       
        logging.info(f"Successfully stored file at: {destination_path}")
        print(f"File stored at: {destination_path}")


    except Exception as e:
        logging.error(f"Failed to store raw data: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    # This is an example of how the function would be called by an ingestion script.
    # First, create a dummy file to simulate ingestion output.
    temp_ingestion_dir = "temp_ingestion"
    os.makedirs(temp_ingestion_dir, exist_ok=True)
   
    csv_file = os.path.join(temp_ingestion_dir, r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline\data\raw\telco_churn_20250823.csv")
   



    # Call the storage function for the ingested file
    store_raw_data(csv_file, "telco_csv", "data/bronze")
