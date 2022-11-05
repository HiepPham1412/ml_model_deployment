from joblib import load
from .model import inference
from .data import process_data
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

def make_inference(X_raw, categorical_features):
    """make inference from input raw fearure data

    Args:
        data (pd.DataFrame): pandas data frame containing features
        categorical_features (list): list of categorical features

    Returns:
        _type_: _description_
    """

    model = load(os.path.join(root_path, 'artifacts/model.joblib'))
    encoder = load(os.path.join(root_path, 'artifacts/encoder.joblib'))
    lb = load(os.path.join(root_path, 'artifacts/lb.joblib'))

    X, _, _, _ = process_data(X=X_raw, 
                              categorical_features=categorical_features, 
                              label=None, 
                              training=True, 
                              encoder=encoder, 
                              lb=lb)
    prediction = inference(model, X)
    prediction = lb.inverse_transform(prediction)[0]

    return prediction


