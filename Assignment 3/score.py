from pandas import DataFrame
from sklearn.base import BaseEstimator 
from typing import Tuple
import joblib
import numpy as np

def score(text: DataFrame, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    propensity = model.predict_proba(text)[0][1]  # Probability of the positive class
    if propensity>threshold:
        return (1,propensity)
    else:
        return (0,propensity)
    

def preprocessing(text):
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')  
    # Calculate TF-IDF features for the text
    tfidf_features = tfidf_vectorizer.transform([text]).toarray()
    
    # Calculate length and word count of the text
    text_length = len(text)
    word_count = len(text.split())
    
    # Scale the length and word count features
    length_words_scaled = scaler.transform(np.array([[text_length, word_count]]))
    
    # Concatenate TF-IDF features with scaled length and word count features
    X_test = np.hstack((tfidf_features, length_words_scaled))
    
    return X_test