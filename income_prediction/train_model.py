# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import process_data
from .model import train_model, compute_model_metrics, inference
import logging
from joblib import dump
import os


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=os.path.join(root_path, "logs/train_and_save_model.log"),
)


# Add code to load in the data.
def train_and_save_model(input_data_path, model_file_path):

    data = pd.read_csv(input_data_path)

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Proces the test data with the process_data function.
    model = train_model(X_train, y_train)

    y_test_hat = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_hat)
    logging.info(
        f"precision: {precision: .3f}, recall: {recall: .3f}, fbeta: {fbeta :.3f}"
    )

    dump(
        model,
        open(f"{model_file_path}/model.joblib", "wb"),
    )
    dump(
        encoder,
        open(f"{model_file_path}/encoder.joblib", "wb"),
    )
    dump(
        lb,
        open(f"{model_file_path}/lb.joblib", "wb"),
    )

    logging.info("Save trained model")
