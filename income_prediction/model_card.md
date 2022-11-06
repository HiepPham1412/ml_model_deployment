# Model Card

## Model Details
This is an income prediction model using training data from US census

## Intended Use
The model predicts whether a person with a certain profile will have income lower or higher than 50k

## Training Data
80% data from US census (https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data
20% data from US census (https://archive.ics.uci.edu/ml/datasets/census+income)

## Metrics
The model was evaluated using Accuracy score, F1 beta score, Precision and Recall. Their values on the test set are: precision: 0.729, recall: 0.531, fbeta: 0.614

## Ethical Considerations
We try to make model performance for different gender and race visible and aim to reduce the bias when training the model

## Caveats and Recommendations
Performance of the model is quite extreme on certain cohorts with thin data