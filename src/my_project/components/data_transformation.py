import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.my_project.exceptions import CustomException
from src.my_project.logger import logging
from src.my_project.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    frequency_maps_file_path: str = os.path.join("artifacts", "frequency_maps.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            numerical_columns = [
                "Product_Frequency",
                "User_Frequency",
                "Product_Category_2_Missing",
                "Product_Category_3_Missing",
            ]

            categorical_columns = [
                "Gender",
                "Age",
                "Occupation",
                "City_Category",
                "Stay_In_Current_City_Years",
                "Marital_Status",
                "Product_Category_1",
                "Product_Category_2",
                "Product_Category_3",
                "Age_Gender",
                "Age_City",
                "Occupation_City",
                "Occupation_Age",
            ]

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()),])

            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore")),])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def additional_features(df: pd.DataFrame, product_frequency: pd.Series, user_frequency: pd.Series) -> pd.DataFrame:

        df = df.copy()

        # Missing-value indicator flags (created before imputation, on raw NaNs)
        df["Product_Category_2_Missing"] = df["Product_Category_2"].isnull().astype(int)
        df["Product_Category_3_Missing"] = df["Product_Category_3"].isnull().astype(int)

        # Frequency encodings
        df["Product_Frequency"] = df["Product_ID"].map(product_frequency).fillna(0)
        df["User_Frequency"] = df["User_ID"].map(user_frequency).fillna(0)

        # Combo categorical features
        df["Age_Gender"] = df["Age"].astype(str) + "_" + df["Gender"].astype(str)
        df["Age_City"] = df["Age"].astype(str) + "_" + df["City_Category"].astype(str)
        df["Occupation_City"] = df["Occupation"].astype(str) + "_" + df["City_Category"].astype(str)
        df["Occupation_Age"] = df["Occupation"].astype(str) + "_" + df["Age"].astype(str)

        # Drop id columns now that they've been used for frequency mapping
        df = df.drop(columns=["User_ID", "Product_ID"])

        return df

    def initiate_data_transformation(self, train_path: str):

        try:
            df = pd.read_csv(train_path)

            logging.info("Read ingested train data completed")

            target_column = "Purchase"

            X = df.drop(columns=[target_column])
            y = df[target_column]

            input_feature_train_df, input_feature_test_df, target_feature_train_df, target_feature_test_df = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info(
                f"Train/validation split done. Train: {input_feature_train_df.shape}, "
                f"Validation: {input_feature_test_df.shape}"
            )

            # Frequency maps are fit on TRAIN only, then reused on test
            product_frequency = input_feature_train_df["Product_ID"].value_counts()
            user_frequency = input_feature_train_df["User_ID"].value_counts()

            input_feature_train_df = self.additional_features(input_feature_train_df, product_frequency, user_frequency)
            input_feature_test_df = self.additional_features(input_feature_test_df, product_frequency, user_frequency)

            logging.info("Applied feature engineering (missing flags, frequency encodings, combo features)")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing dataframes")

            train_arr = np.c_[
                input_feature_train_arr.toarray() if hasattr(input_feature_train_arr, "toarray") else input_feature_train_arr,
                np.array(target_feature_train_df),
            ]
            test_arr = np.c_[
                input_feature_test_arr.toarray() if hasattr(input_feature_test_arr, "toarray") else input_feature_test_arr,
                np.array(target_feature_test_df),
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            """frequency maps must be persisted separately as they're fit on the
             train split only and prediction_pipeline.py needs the exact same
             mapping when it runs on the Kaggle test.csv later. Without this,
             unseen data would either crash or silently get wrong frequency
             encodings computed from the wrong data."""

            save_object(file_path=self.data_transformation_config.frequency_maps_file_path,
                        obj={"product_frequency": product_frequency, "user_frequency": user_frequency},)

            logging.info("Saved preprocessing object and frequency maps")

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path,)

        except Exception as e:
            raise CustomException(e, sys)
        
