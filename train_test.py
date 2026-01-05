from tfidf_vectorization import x, y
from sklearn.model_selection import train_test_split

# Split into Training and Testing Sets 
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape: ", x_train.shape)
print("Testing set shape: ", x_test.shape)
