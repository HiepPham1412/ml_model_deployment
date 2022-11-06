def test_status(client):

    results = client.get('/')
    assert results.status_code == 200
    assert results.json() == {"message": "Hello!"}


def test_post_correct(client):
    request = client.post("/", json={
                                    "education": "Bachelors",
                                    "marital_status": "Never-married",
                                    "occupation": "Adm-clerical",
                                    "relationship": "Not-in-family",
                                    "race": "White",
                                    "sex": "Male",
                                    "hours_per_week": 40,
                                    "native_country": "United-States",
                                    "age": 25,
                                    "workclass": "State-gov",
                                    "fnlgt": 12240
                                    })
    assert request.status_code == 200
    assert request.json() == {"prediction": " <=50K"}


def test_post_wrong_params(client):
    request = client.post("/", json={
                                    "education": "",
                                    "marital_status": "Never-married",
                                    "occupation": "Adm-clerical",
                                    "relationship": "Not-in-family",
                                    "race": "Asian",
                                    "sex": "Male",
                                    "hours_per_week": 40,
                                    "native_country": "United-States",
                                    "age": 25,
                                    "workclass": "State-gov",
                                    "fnlgt": 12240
                                    })
    assert request.status_code == 422
