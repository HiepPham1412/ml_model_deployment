# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import logging
import pickle

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="ex.log",
)

# Add code to load in the data.
data = pd.read_csv(
    "/Users/hieppham/Documents/mlops/model_deployment/ml_model_deployment/starter/data/census.csv"
)
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
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

y_test_hat = inference(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_hat)
logging.info(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")

pickle.dump(
    model,
    open(
        "/Users/hieppham/Documents/mlops/model_deployment/ml_model_deployment/starter/model/model.pkl",
        'wb'),
)

logging.info("Save trained model")
