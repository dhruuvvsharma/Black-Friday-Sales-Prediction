import os
import pickle
import sys
from dataclasses import dataclass

import pandas as pd

from src.my_project.components.data_transformation import DataTransformation
from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


@dataclass
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    frequency_maps_path: str = os.path.join("artifacts", "frequency_maps.pkl")
    model_path: str = os.path.join("artifacts", "model.pkl")
    kaggle_test_path: str = os.path.join("artifacts", "test.csv")
    submission_path: str = os.path.join("artifacts", "submission.csv")


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()

    @staticmethod
    def _load_object(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def initiate_prediction(self):
        """
        it runs the original Kaggle test.csv (233k rows, no 'Purchase') through
        the exact same feature engineering + preprocessing + best model used
        in training, then writes submission.csv.

        Critical note for myself : uses preprocessor.transform() (never fit_transform) and the
        train-fitted frequency maps saved during data_transformation.py, so
        this never leaks test-set information into the encoding.
        """
        try:
            logging.info("Loading preprocessor, frequency maps, and trained model")
            preprocessor = self._load_object(self.config.preprocessor_path)
            freq_maps = self._load_object(self.config.frequency_maps_path)
            model = self._load_object(self.config.model_path)

            product_frequency = freq_maps["product_frequency"]
            user_frequency = freq_maps["user_frequency"]

            logging.info(f"Reading Kaggle test data from {self.config.kaggle_test_path}")
            test_df = pd.read_csv(self.config.kaggle_test_path)

            # additional_features() drops User_ID/Product_ID — keep them here
            # so the submission file can still identify each row
            ids = test_df[["User_ID", "Product_ID"]].copy()

            additional_df = DataTransformation.additional_features(test_df, product_frequency, user_frequency)

            logging.info("Applied feature engineering to Kaggle test data")

            X_transformed = preprocessor.transform(additional_df)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            logging.info("Transformed Kaggle test data with saved preprocessor")

            predictions = model.predict(X_transformed)

            submission_df = ids.copy()
            submission_df["Purchase"] = predictions

            submission_df.to_csv(self.config.submission_path, index=False)

            logging.info(f"Saved submission file to {self.config.submission_path}")

            return self.config.submission_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    path = PredictionPipeline().initiate_prediction()
    print("Submission saved to:", path)