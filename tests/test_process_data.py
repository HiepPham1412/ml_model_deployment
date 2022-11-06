from income_prediction.data import process_data


def test_process_data(clean_data):
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
    X, y, encoder, lb = process_data(X=clean_data,
                                     categorical_features=cat_features,
                                     label='salary',
                                     training=True,
                                     encoder=None,
                                     lb=None)

    assert X.shape[0] == clean_data.shape[0]
    assert X.shape[1] > clean_data.shape[1]
