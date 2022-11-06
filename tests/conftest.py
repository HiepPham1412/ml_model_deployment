from fastapi.testclient import TestClient
from main import app
import pytest
import pandas as pd


@pytest.fixture
def clean_data():
    df = pd.read_csv('data/census.csv')
    df.columns = [c.strip() for c in df.columns]
    return df


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client
