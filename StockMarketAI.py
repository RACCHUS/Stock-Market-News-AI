import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the CSV file
df = pd.read_csv("news_headlines.csv")

# Split the data into training and testing sets
train_headlines, test_headlines, train_labels, test_labels = train_test_split(df["headline"], df["direction"], test_size=0.2)

# Convert the headlines into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_headlines)
test_features = vectorizer.transform(test_headlines)

# Train a Logistic Regression classifier on the training data
classifier = LogisticRegression()
classifier.fit(train_features, train_labels)

# Make predictions on the test data
predictions = classifier.predict(test_features)

# Evaluate the accuracy of the model
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save the trained model to a file
dump(classifier, 'market_predictor.joblib')