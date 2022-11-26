def test_status(client):

    results = client.get('/')
    assert results.status_code == 200
    assert results.json() == {"message": "Hello!"}


def test_post_low(client):
    request = client.post("/", json={
                                    "workclass": "State-gov",
                                    "education": "Bachelors",
                                    "marital_status": "Never-married",
                                    "occupation": "Adm-clerical",
                                    "relationship": "Not-in-family",
                                    "race": "White",
                                    "sex": "Male",
                                    "native_country": "United-States",
                                    "hours_per_week": 40,
                                    "age": 25,
                                    "fnlgt": 12240
                                    })
    assert request.status_code == 200
    assert request.json() == {"prediction": " <=50K"}


def test_post_high(client):
    request = client.post("/", json={
                                    "workclass": "Private",
                                    "education": 'Masters',
                                    "marital_status": 'Married-civ-spouse',
                                    "occupation": 'Exec-managerial',
                                    "relationship": 'Husband',
                                    "race": "White",
                                    "sex": "Male",
                                    "hours_per_week": 60,
                                    "native_country": 'United-States',
                                    "age": 40,
                                    "fnlgt": 500000
                                    })
    assert request.status_code == 200
    assert request.json() == {"prediction": " >50K"}


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
