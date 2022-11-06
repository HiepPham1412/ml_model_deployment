from income_prediction.make_inference import make_inference


def test_make_inference(clean_data):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]
    clean_data = clean_data.drop(columns=['salary'])
    clean_data = clean_data.head(1)
    prediction = make_inference(clean_data, cat_features)

    assert prediction is not None
