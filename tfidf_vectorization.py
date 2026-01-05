from datacleaning import df
from sklearn.feature_extraction.text import TfidfVectorizer

# define tf-idf vectorize

tfidf = TfidfVectorizer(
    # keep top 5000 maximum words
    max_features = 5000,
    # use unigrams and biagrams
    ngram_range=(1,2)
)

# Applied Tf-idf into cleaned text
# numerical representation of text
x = tfidf.fit_transform(df['clean_text'])

# target labels(fake/real)
y = df['labels']

print("TD-IDF matrix shape: ", x.shape)
print(tfidf.get_feature_names_out()[:20])

