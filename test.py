import pandas as pd

from src.my_project.pipelines.prediction_pipeline import PredictionPipeline


input_data = pd.DataFrame([{
    "User_ID": 100001,
    "Product_ID": "P00069042",
    "Gender": "M",
    "Age": "26-35",
    "Occupation": 10,
    "City_Category": "A",
    "Stay_In_Current_City_Years": "2",
    "Marital_Status": 0,
    "Product_Category_1": 3,
    "Product_Category_2": 5,
    "Product_Category_3": 14
}])


pipeline = PredictionPipeline()

prediction = pipeline.predict(input_data)

print("Predicted Purchase:", prediction[0])