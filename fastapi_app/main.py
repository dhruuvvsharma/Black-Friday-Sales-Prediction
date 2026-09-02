import logging
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from src.my_project.pipelines.prediction_pipeline import PredictionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("black_friday_api")

app = FastAPI(title="Black Friday Sales Prediction API")

""" Load the pipeline once at startup, not on every request.If the model/preprocessor files 
    are missing or corrupt, fail loudly at startup instead of on the first user request."""
try:
    prediction_pipeline = PredictionPipeline()
except Exception as e:
    logger.exception("Failed to load prediction pipeline at startup")
    prediction_pipeline = None
    _startup_error = str(e)
else:
    _startup_error = None


class PredictionInput(BaseModel):
    User_ID: int = Field(..., ge=1000000, le=1010000, description="Known training range")
    Product_ID: str = Field(..., pattern=r"^P\d{8}$")
    Gender: Literal["M", "F"]
    Age: Literal["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    Occupation: int = Field(..., ge=0, le=20)
    City_Category: Literal["A", "B", "C"]
    Stay_In_Current_City_Years: Literal["0", "1", "2", "3", "4+"]
    Marital_Status: Literal[0, 1]
    Product_Category_1: int = Field(..., ge=1, le=20)
    Product_Category_2: Optional[int] = Field(default=None, ge=1, le=20)
    Product_Category_3: Optional[int] = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def check_category_cascade(self):
        """Category 3 should never be set without Category 2 — this mirrors the actual structure of 
           the training data and catches bad frontend input before it reaches the model."""

        if self.Product_Category_3 is not None and self.Product_Category_2 is None:
            raise ValueError(
                "Product_Category_3 cannot be provided without Product_Category_2"
            )
        return self


@app.get("/")
def home():
    return {"message": "Black Friday Sales Prediction API is running"}


@app.get("/health")
def health():
    if prediction_pipeline is None:
        return JSONResponse(status_code=503,content={"status": "unhealthy", "detail": _startup_error},)
    return {"status": "OK"}


@app.post("/predict")
def predict(data: PredictionInput):
    if prediction_pipeline is None:
        """ Model failed to load at startup — don't let every request crash with a stack trace, return one clear error instead."""

        raise HTTPException(status_code=503,detail="Model is not available. Check server logs.",)

    input_data = data.model_dump()

    try:
        input_df = pd.DataFrame([input_data])
        prediction = prediction_pipeline.predict(input_df)
    except ValueError as e:
        """ Typically an unseen category (e.g. a Product_ID the encoder never saw during training) — this is a client input problem."""

        logger.warning(f"Prediction input rejected: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid input for model: {e}")

    except Exception as e:
        """" Anything else (missing artifact, shape mismatch, etc.) is a server-side problem, not the caller's fault."""
        
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    predicted_value = float(prediction[0])

    """Purchase amounts in this dataset are always positive. 
    a negative or zero prediction means something upstream is broken, not a valid answer."""
    if predicted_value <= 0:
        logger.warning(f"Model returned non-positive prediction: {predicted_value}")

    return {"predicted_purchase": round(predicted_value, 2)}