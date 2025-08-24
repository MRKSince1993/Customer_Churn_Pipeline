# utils/logging_config.py

import logging
import os
from datetime import datetime

def setup_logging():
    """
    Configures the logging to output to both a file and the console.
    """
    # Create a 'logs' directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Define the log file path with a timestamp for daily logs
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

    # Configure the root logger
    # This setup will apply to all logging calls in your application
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

