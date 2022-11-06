from fastapi import FastAPI
from data_schema import InputData
from income_prediction.make_inference import make_inference
import pandas as pd
import yaml


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Hello!"}


@app.post("/")
async def predict_income(input_data: InputData):

    input_data = input_data.dict()
    df_input = pd.DataFrame.from_records([input_data])
    all_features = config["features"]["categorical_features"] + config["features"]["continious_features"]
    df_input = df_input[all_features]
    prediction = make_inference(df_input, config["features"]["categorical_features"])

    return {"prediction": prediction}
