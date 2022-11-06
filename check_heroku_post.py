import requests


json_data = {"education": "Bachelors",
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
             }
response = requests.post('https://us-income-classification.herokuapp.com/', json=json_data)
print(response)

