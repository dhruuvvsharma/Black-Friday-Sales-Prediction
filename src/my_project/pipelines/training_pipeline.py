import sys

from src.my_project.components.data_ingestion import DataIngestion
from src.my_project.components.data_transformation import DataTransformation
from src.my_project.components.model_trainer import ModelTrainer
from src.my_project.exceptions import CustomException
from src.my_project.logger import logging


class TrainingPipeline:
    """ evaluation flow:
    Data Ingestion ---> Data Transformation ----> Model Training ----> Hyperparameter Tuning
    """

    def run(self, tune: bool = True, n_iter: int = 40, cv: int = 3):
        try:
            logging.info("Training pipeline started...")

            # S1: Data Ingestion pulls MySQL train/test tables to artifacts/
            logging.info("Starting data ingestion..")
            train_path, _test_path = DataIngestion().initiate_data_ingestion()

            # Note: _test_path is the ingested Kaggle test.csv (no 'Purchase').
            # It is intentionally unused here — reserved for prediction_pipeline.py.

            # S2 : Data Transformation splits train_path internally (440k/110k),
            # applies feature engineering + preprocessing, saves preprocessor.pkl and frequency_maps.pkl.
            logging.info("Starting data transformation")
            train_arr, test_arr, preprocessor_path = DataTransformation().initiate_data_transformation(train_path)
            logging.info(f"Preprocessor saved at {preprocessor_path}")

            # S3: Model Training — trains and compares all candidate models,
            # saves the best one (by R2) to artifacts/model.pkl.
            logging.info("Starting model training")
            trainer = ModelTrainer()
            best_model_name, best_score, report = trainer.initiate_model_training(train_arr, test_arr)
            logging.info(f"Best model before tuning: {best_model_name} (R2={best_score:.4f})")

            result = {
                "best_model_before_tuning": best_model_name,
                "best_score_before_tuning": best_score,
                "comparison_report": report,
            }

            # S4: Hyperparameter Tuning — scoped to XGBoost only, overwrites artifacts/model.pkl only if it improves on the saved baseline.
            if tune:
                logging.info("Starting hyperparameter tuning")
                best_params, tuned_metrics, improved = trainer.tune_xgboost(train_arr, test_arr, n_iter=n_iter, cv=cv)
                logging.info(f"Tuning complete. R2={tuned_metrics['R2 Score']:.4f}, improved={improved}")

                result["tuned_params"] = best_params
                result["tuned_metrics"] = tuned_metrics
                result["tuning_improved"] = improved

            logging.info("Training pipeline completed successfully.")

            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    result = TrainingPipeline().run()

    print("\n<---Training Pipeline Result--->")
    print("Best model (pre-tuning):", result["best_model_before_tuning"], "R2:", result["best_score_before_tuning"])

    if "tuned_metrics" in result:
        print("Tuned R2:", result["tuned_metrics"]["R2 Score"])
        print("Tuning improved saved model:", result["tuning_improved"])

    print("\nFinal model saved at: artifacts/model.pkl")