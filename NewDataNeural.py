import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from TrainModelStock import tokenizer, model, encoder

# load the training data from a CSV file
data = pd.read_csv('news_headlines.csv')

# load the saved model
loaded_model = tf.keras.models.load_model("market_direction_model.h5")

headline = "GDP growth expected to remain low but higher than last year"
headline_sequence = tokenizer.texts_to_sequences([headline])
padded_headline_sequence = tf.keras.preprocessing.sequence.pad_sequences(headline_sequence, maxlen=100)

prediction = model.predict(padded_headline_sequence)

direction = encoder.inverse_transform(prediction.round().astype(int))

# format the prediction
prediction = tf.round(prediction)
prediction = int(prediction[0][0].numpy())

if prediction == 1:
    print("Market is predicted to go up.")
else:
    print("Market is predicted to go down.")