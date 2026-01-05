from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopword if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# load saved model and Tfidf vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorization.pkl")


# initialization flask app
app = Flask(__name__)

# Function to clean text

def clean_text(text):
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Home page route
@app.route ('/')
def home():
    return render_template('index.html')

# prediction route
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    result = "Most Likely Real News" if prediction == 0 else "Most Likely Fake News"

    return render_template(
        'index.html',
        prediction=result,
        news_text=news_text
    )



# Run the app

if __name__ == "__main__":
    app.run(debug=True)
    
