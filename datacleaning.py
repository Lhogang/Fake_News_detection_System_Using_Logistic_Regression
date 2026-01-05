# importing datasets from the loaded datasets
from load_datasets import df

# used for regular expression to clean text
import re

# importing natural language processing library
import nltk

# common words like a an the  and is in  which do not help classification
from nltk.corpus import stopwords

# load stopwords in a set
stop_words = set(stopwords.words('english'))


# Define text cleaning Function
def clean_text(text):
    # convert text into lowercase
    text = str(text).lower()
    
    # Remove URLS
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove Extra Space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove Stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # join clean wordss back
    return ' '.join(words)

# Applying cleaning to entire datasets
df['clean_text'] = df['text'].apply(clean_text)

# displayed clean data
print(df[['clean_text', 'labels']].head())








