from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, **kwargs):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    kwargs: GradientBoosting model hyper parameters
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = GradientBoostingClassifier(**kwargs)
    model.fit(X=X_train, y=y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : trained sklearn model object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds

def validate_performance_on_sliced_data(model, encoder, lb, data, stats_path):
    """_summary_

    Args:
        model (_type_): model picke file
        encoder (_type_): encoder picke file
        lb (_type_): label transformer picke file
        data (_type_): pandas data frame
        stats_path (str): path to store model stats on sliced data
    """
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    with open(f'{stats_path}/slice_output.txt', 'w') as file:
        for feat in categorical_features:
            for cls in data[feat].unique():
                df_tmp = data[data[feat] == cls]
                X_test, y_test, _, _ = process_data(df_tmp,
                                                    categorical_features=categorical_features,
                                                    label="salary",
                                                    training=False,
                                                    encoder=encoder,
                                                    lb=lb,
                                                )
                y_test_hat = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, y_test_hat)
                performance_metrics = f"precision: {precision: .3f}, recall: {recall:.3f}, fbeta: {fbeta: .3f}"
                file.write(performance_metrics + '\n')
    


