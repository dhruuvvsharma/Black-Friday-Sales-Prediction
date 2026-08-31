from src.my_project.logger import logging
from src.my_project.exceptions import CustomException
from src.my_project.components.data_ingestion import DataIngestion
import sys


if __name__ == "__main__":

    logging.info("Starting data ingestion...")

    try:

        data_ingestion = DataIngestion()

        train_data_path, test_data_path = ( data_ingestion.initiate_data_ingestion())

        logging.info(f"Train data saved at: {train_data_path}")
        logging.info(f"Test data saved at: {test_data_path}")

    except Exception as e:

        logging.error( "Error occurred during data ingestion")

        raise CustomException(e, sys) from e