from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from score import score,preprocessing

app = Flask(__name__)

# Load the trained model and vectorizer
trained_model = joblib.load('best_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')  

@app.route('/score', methods=['POST'])
def score_text():
    try:
        # Check if request is JSON
        if request.headers['Content-Type'] != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Extract text from JSON request
        text = request.json['text']
        
        X_test = preprocessing(text)
        
        # Set threshold for classification
        threshold = 0.55
        
        # Get prediction and propensity score
        prediction, propensity = score(X_test, trained_model, threshold)
        
        # Prepare response data
        response_data = {
            'prediction': bool(prediction),
            'propensity': float(propensity)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
