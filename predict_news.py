import joblib

# Load Model and Tfidf
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorization.pkl")

def predict_news(news_text):
    # predict whether the news article is real or fake
    
    # convert text into lowercase
    news_text = news_text.lower()
    
    # Transform using the same tfidf vectorization
    vectorized_text = tfidf.transform([news_text])
    
    # predict
    prediction = model.predict(vectorized_text)
    return "Real News " if prediction [0]==0 else "Fake News "

    
# Test with sample
sample_news = "The government announced new economic reforms to boost employment and growth."


print("Sample Prediction")
print(predict_news(sample_news))