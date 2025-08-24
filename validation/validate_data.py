# validation/validate_data.py

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors
import logging
import os
from datetime import datetime
from utils.logging_config import setup_logging

def validate_raw_data():
    """
    Validates the raw Telco churn data using a Pandera schema.
    """
    try:
        logging.info("Starting raw data validation.")

        # Find the path to the latest ingested CSV file
        now = datetime.now()
        data_path = os.path.join(
            "data", "bronze", "source=telco_csv",
            f"year={now.strftime('%Y')}", f"month={now.strftime('%m')}", f"day={now.strftime('%d')}"
        )

        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Partition directory not found for today: {data_path}")

        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {data_path} for today.")

        # Pick the most recently modified CSV
        latest_csv_path = max(
            [os.path.join(data_path, f) for f in csv_files],
            key=os.path.getmtime
        )
        logging.info(f"Found latest CSV for validation: {latest_csv_path}")
        df = pd.read_csv(latest_csv_path)

        # Define the validation schema using Pandera
        schema = pa.DataFrameSchema({
            "customerID": pa.Column(str, nullable=False, unique=True),
            "gender": pa.Column(str, checks=pa.Check.isin(["Male", "Female"]), nullable=False),
            "SeniorCitizen": pa.Column(int, checks=pa.Check.isin([0,1]), nullable=False),
            "Partner": pa.Column(str, checks=pa.Check.isin(["Yes","No"]), nullable=False),
            "Dependents": pa.Column(str, checks=pa.Check.isin(["Yes","No"]), nullable=False),
            "tenure": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0), nullable=False),
            "MonthlyCharges": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0), nullable=False),
            "TotalCharges": pa.Column(str, nullable=True),  # Keep nullable True to handle empty/space values
            "Contract": pa.Column(str, checks=pa.Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False),
            "Churn": pa.Column(str, checks=pa.Check.isin(["Yes","No"]), nullable=False),
        })
 

        logging.info("Applying validation schema to the dataframe.")
        schema.validate(df, lazy=True)
        logging.info("Data validation successful. All checks passed.")
        print("Data validation successful.")

    except SchemaErrors as err:
        logging.error("Data validation failed!")
        os.makedirs("validation", exist_ok=True)
        report_path = os.path.join("validation", "data_quality_report.txt")
        with open(report_path, "w") as report_file:
            report_file.write(f"Data Quality Validation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write("="*50 + "\n")
            report_file.write(str(err.failure_cases))

        logging.error(f"Validation failure report saved to {report_path}")
        print(f"Data validation failed. See report at {report_path}")
        raise

    except Exception as e:
        logging.error(f"An unexpected error occurred during validation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    setup_logging()
    validate_raw_data()

 