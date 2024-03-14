import os
import pytest
import requests
import joblib
from score import score,preprocessing
from app import app

trained_model = joblib.load('best_model.pkl')


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
    # Start Flask app in a separate process
    flask_process = os.system("python app.py &")

    # Ensure the Flask app is running
    assert os.system("lsof -i:5000") == 0

    # Endpoint testing
    text = "Subject: only our software is guaranteed 100 % legal . name - brand software at low , low , low , low prices everything comes to him who hustles while he waits . many would be cowards if they had courage enough ."
    payload = {'text': text}
    response = requests.post("http://127.0.0.1:5000/score", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'propensity' in data
    assert isinstance(data['prediction'], bool)
    assert isinstance(data['propensity'], float)

    # Close the Flask app
    os.system("kill -9 $(lsof -t -i:5000)")

    # Ensure the Flask app has stopped
    assert os.system("lsof -i:5000") != 0
