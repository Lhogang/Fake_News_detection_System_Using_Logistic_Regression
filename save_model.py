# Joblib saves the trained files and  TF_IDF vectorized

import joblib

# import model training
from model_training import model

# import tfidf vectorize
from tfidf_vectorization import tfidf



# saves model and vectorizer

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorization.pkl")

print("Model and Vectorization are saved successfully")