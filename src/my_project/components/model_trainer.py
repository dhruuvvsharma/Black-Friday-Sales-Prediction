import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging
from src.my_project.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @staticmethod
    def evaluate_model(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info("Splitting train and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Decision Tree": DecisionTreeRegressor(max_depth=20, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42, n_jobs=-1),

                "Gradient Boosting (Hist)": HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42),
                "XGBoost": XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),

                "CatBoost": CatBoostRegressor(
                    iterations=300,
                    depth=6,
                    learning_rate=0.1,
                    loss_function="RMSE",
                    random_state=42,
                    thread_count=-1,
                    early_stopping_rounds=30,
                    verbose=False,
                ),
            }

            report = {}
            fitted_models = {}

            for name, model in models.items():
                logging.info(f"Training {name}...")
                start = time.time()

                if name == "CatBoost":
                    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
                else:
                    model.fit(X_train, y_train)

                duration = time.time() - start

                y_pred = model.predict(X_test)
                metrics = self.evaluate_model(y_test, y_pred)
                metrics["train_time_sec"] = round(duration, 2)

                report[name] = metrics
                fitted_models[name] = model

                logging.info(
                    f"{name} -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, "
                    f"R2: {metrics['R2 Score']:.4f}, time: {duration:.1f}s"
                )

            best_model_name = max(report, key=lambda name: report[name]["R2 Score"])
            best_model = fitted_models[best_model_name]
            best_score = report[best_model_name]["R2 Score"]

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Saved best model")

            return best_model_name, best_score, report

        except Exception as e:
            raise CustomException(e, sys)


    # Now, Hyperparameter tuning for XGBoost using RandomizedSearchCV
    def tune_xgboost(self, train_array, test_array, n_iter: int = 40, cv: int = 3): 

        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            param_distributions = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [4, 5, 6, 7, 8, 9, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "gamma": [0, 0.1, 0.2, 0.3],
            }

            base_model = XGBRegressor(objective="reg:squarederror",random_state=42,tree_method="hist",device="cuda",)

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="r2",
                cv=cv,
                verbose=2,
                random_state=42,
                n_jobs=1,  
            )

            logging.info(f"Starting RandomizedSearchCV: n_iter={n_iter}, cv={cv}")
            start = time.time()
            search.fit(X_train, y_train)
            duration = time.time() - start
            logging.info(f"RandomizedSearchCV completed in {duration:.1f}s")

            tuned_model = search.best_estimator_
            best_params = search.best_params_

            y_pred = tuned_model.predict(X_test)
            tuned_metrics = self.evaluate_model(y_test, y_pred)
            tuned_metrics["train_time_sec"] = round(duration, 2)

            logging.info(f"Best params: {best_params}")
            logging.info(
                f"Tuned XGBoost -> MAE: {tuned_metrics['MAE']:.2f}, "
                f"RMSE: {tuned_metrics['RMSE']:.2f}, R2: {tuned_metrics['R2 Score']:.4f}"
            )

            # Comparing against whatever is currently saved before overwriting
            baseline_r2 = None
            if os.path.exists(self.model_trainer_config.trained_model_file_path):
                import pickle

                with open(self.model_trainer_config.trained_model_file_path, "rb") as f:
                    current_model = pickle.load(f)
                baseline_pred = current_model.predict(X_test)
                baseline_r2 = r2_score(y_test, baseline_pred)

            if baseline_r2 is None or tuned_metrics["R2 Score"] > baseline_r2:
                save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=tuned_model,)

                logging.info(f"Tuned model improved R2 ({baseline_r2} -> {tuned_metrics['R2 Score']:.4f}), saved.")
                improved = True

            else:
                logging.info(f"Tuned model did not beat saved baseline "f"({tuned_metrics['R2 Score']:.4f} <= {baseline_r2:.4f}), keeping original.")
                improved = False

            return best_params, tuned_metrics, improved

        except Exception as e:
            raise CustomException(e, sys)