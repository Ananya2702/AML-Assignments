import os
import pytest
import requests
import joblib
from score import score,preprocessing
from app import app
import subprocess
import time
import json
import numpy as np
import pandas as pd

trained_model = joblib.load('best_model.pkl')

@pytest.fixture(scope="module")
def client():
    with app.test_client() as client:
        yield client

def test_score():
    # Smoke test
    text = "This is a test"
    text = preprocessing(text)
    threshold = 0.55
    prediction, propensity = score(text, trained_model, threshold)
    assert prediction is not None
    assert propensity is not None
    assert isinstance(propensity, float)
    print('Smoke Test: Success')
    
    # Format test
    assert 0 <= propensity <= 1
    print('Format Test and propensity: Success')
    
    # Prediction value test
    assert prediction in [False, True]
    print('Prediction Value Test: Success')
    
    # Threshold test
    assert score(text, trained_model, 0)[0] == True
    assert score(text, trained_model, 1)[0] == False
    print('Threshold Test: Success')


    # Obvious spam input test
    spam_text = "Subject: Congratulations! You've Won $1,000,000 in our Exclusive Prize Draw!"
    spam_text = preprocessing(spam_text)
    spam_prediction, _ = score(spam_text, trained_model, threshold)
    assert spam_prediction == True
    print("Obvious spam text: Success")

    # Obvious non-spam input test
    non_spam_text = "Subject: Meeting Agenda for Friday's Team Meeting"
    non_spam_text = preprocessing(non_spam_text)
    non_spam_prediction, _ = score(non_spam_text, trained_model, threshold)
    assert non_spam_prediction == False
    print("Obvious Ham text: Success")


def test_flask():
    # Start Flask app using subprocess
    flask_process = subprocess.Popen(["python", "app.py"])
    # Allow some time for the Flask server to start
    time.sleep(5)

    try:
        # Load test data
        test_data = pd.read_csv("test.csv")
        text = test_data.sample(n=1)["text"].iloc[0]

        # Prepare request data
        data = {"text": text}
        url = "http://127.0.0.1:5000/score"
        headers = {"Content-Type": "application/json"}
        json_data = json.dumps(data)

        # Send POST request to Flask app
        response = requests.post(url, data=json_data, headers=headers)

        # Assert response status code and fields
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response
        assert "propensity" in json_response
    finally:
        # Terminate Flask app subprocess
        flask_process.terminate()
