import os
import sys
from dataclasses import dataclass

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging
from src.my_project.utils import read_sql_data


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")

        try:

            # Read training data
            train_df = read_sql_data("train")
            logging.info("Training data successfully loaded from MySQL")

            # Read testing data
            test_df = read_sql_data("test")
            logging.info("Testing data successfully loaded from MySQL")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path ), exist_ok=True)

            # Save training data
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # Save testing data
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Train and test data saved to artifacts folder")
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logging.error(
                "Exception occurred during data ingestion")
            raise CustomException(e, sys) from e