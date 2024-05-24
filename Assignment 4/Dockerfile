FROM python:3.9-slim

WORKDIR /app
COPY . /app
COPY . /score.py 
COPY . /best_model.pkl
COPY . /scaler.pkl
COPY . /tfidf_vectorizer.pkl


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
