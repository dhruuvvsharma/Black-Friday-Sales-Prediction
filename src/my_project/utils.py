import os
import sys
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


load_dotenv()

host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
port = os.getenv("MYSQL_PORT", "3306")


def read_sql_data(table_name):

    logging.info(f"Reading data from MySQL table: {table_name}")

    try:

        connection_string = (
            f"mysql+pymysql://{user}:{quote_plus(password)}"
            f"@{host}:{port}/{database}"
        )

        engine = create_engine(connection_string)

        logging.info("Successfully connected to MySQL database")

        df = pd.read_sql_query(f"SELECT * FROM {table_name}",con=engine)

        logging.info(f"Successfully read data from {table_name} table")

        engine.dispose()

        return df

    except Exception as e:

        logging.error(f"Error reading data from MySQL: {e}")

        raise CustomException(e, sys) from e