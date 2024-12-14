
import logging
import os

class SingletonLogger:
    _instance = None  # Holds the single instance of the logger

    def __new__(cls, log_dir="logs", log_file="training.log"):
        if cls._instance is None:
            # Create a new instance if it doesn't exist
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.setup_logger(log_dir, log_file)
        return cls._instance

    def setup_logger(self, log_dir, log_file):
        # Create log directory if it doesn’t exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up logging format
        log_path = os.path.join(log_dir, log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("training_logger")

    def get_logger(self):
        return self.logger
