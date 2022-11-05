import pandas as pd
from income_prediction.train_model import train_and_save_model
from income_prediction.model import validate_performance_on_sliced_data
from joblib import load
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=os.path.join("logs/ex.log"),
)


# train and save model
input_data_path = 'data/census.csv'
model_file_path = 'artifacts'
train_and_save_model(input_data_path, model_file_path)

# validate model performance on sliced data
model = load('artifacts/model.joblib')
encoder = load('artifacts/encoder.joblib')
lb = load('artifacts/lb.joblib')
data = pd.read_csv(input_data_path)
stats_path = 'logs'
df_train, df_test = train_test_split(data, test_size=0.3)
validate_performance_on_sliced_data(model, encoder, lb, df_train, stats_path)