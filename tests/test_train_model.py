from income_prediction.train_model import train_and_save_model
import os


def test_train_and_save_model():

    input_data_path = './data/census.csv'
    model_file_path = './artifacts'
    train_and_save_model(input_data_path, model_file_path)

    assert os.path.isfile('artifacts/model.joblib')
    assert os.path.isfile('artifacts/lb.joblib')
    assert os.path.isfile('artifacts/encoder.joblib')
