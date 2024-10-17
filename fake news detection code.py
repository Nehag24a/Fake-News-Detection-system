# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
# Dataset must have two columns: 'text' (news articles) and 'label' (fake/real labels)
data = pd.read_csv('news.csv')

# Extract labels and features
labels = data['label']  # The 'label' column should contain 'FAKE' or 'REAL' labels
X_train, X_test, y_train, y_test = train_test_split(data['text'], labels, test_size=0.2, random_state=7)

# Initialize a TF-IDF vectorizer and transform the dataset
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training set and transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set
y_pred = pac.predict(tfidf_test)

# Evaluate the model
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

# Confusion matrix to see performance
confusion = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print('Confusion Matrix:')
print(confusion)