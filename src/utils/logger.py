"""
Simple logger for training
"""

import os
from datetime import datetime


class Logger:
    def __init__(self, log_file):
        """
        Initialize logger

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Write header
        with open(self.log_file, "w") as f:
            f.write("=== Training Log ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

    def log(self, message):
        """Log message to file and print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        print(log_message)

        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
