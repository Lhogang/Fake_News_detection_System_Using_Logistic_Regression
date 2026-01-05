# import tran_test, tf-idf_vectorization file

from train_test import x_train, x_test, y_train, y_test
from tfidf_vectorization import tfidf

# import Machine Learning Model
from sklearn.linear_model import LogisticRegression

# import evaluation metrices
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# initialize Logistic Regression
model = LogisticRegression(max_iter=1000)

# train the model
model.fit(x_train, y_train)

print("Model training completed successfully")


# Evaluate the model
#
# predict on test data

y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: ", accuracy)

# classification report
print("Classification report ")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix ")
print(confusion_matrix(y_test, y_pred))


# Test with the new News

def predict_news(news_text):
    # predict whether the news article is real or fake
    
    # convert text into lowercase
    news_text = news_text.lower()
    
    # Transform using the same tfidf vectorization
    vectorized_text = tfidf.transform([news_text])
    
    # predict
    prediction = model.predict(vectorized_text)
    return "Real News " if prediction [0]==0 else "Fake News "
    

sample_news = "The government announced new economic reforms to boost employment and growth."


print("Sample Prediction")
print(predict_news(sample_news))



