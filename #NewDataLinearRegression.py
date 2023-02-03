from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from StockMarketAI import vectorizer

# Load the saved model from a file
classifier = load('market_predictor.joblib')

# New data to make predictions on
new_headlines = [
    "Fed Meeting Preview: Powell Won't Break S&P 500 Rally; Wage Growth Eases",
    "Interest rates expected to increase again next year",
    "Global economy shows signs of slowing down",
    "New product launch drives sales for tech company",
    "Natural disaster causes damage, market drops"
]

# Convert the new headlines into numerical features using the TF-IDF vectorizer
new_features = vectorizer.transform(new_headlines)

# Use the trained model to make predictions on the new data
new_predictions = classifier.predict(new_features)

# Print the predictions
print("Predictions:", new_predictions)