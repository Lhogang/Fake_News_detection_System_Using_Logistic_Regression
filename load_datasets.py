import pandas as pd

# load new dataset
df = pd.read_csv("datasets/fake_news.csv ")  # path to your new dataset

# check first rows and columns
print(df.head())
print(df.info())
print(df['labels'].value_counts())  # see distribution of 0 (true) and 1 (fake)
