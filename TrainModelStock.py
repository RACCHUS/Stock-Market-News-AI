import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load the csv file into a pandas dataframe
df = pd.read_csv("news_headlines.csv")

# encode the direction column
encoder = LabelEncoder()
df["direction"] = encoder.fit_transform(df["direction"])

# split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# create a dataset from the dataframe
train_dataset = tf.data.Dataset.from_tensor_slices((train_df["headline"].values, train_df["direction"].values))
test_dataset = tf.data.Dataset.from_tensor_slices((test_df["headline"].values, test_df["direction"].values))

# create a tokenizer to preprocess the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)

# fit the tokenizer on the training data
tokenizer.fit_on_texts(train_df["headline"].values)

# preprocess the text data into sequences of integers
train_sequences = tokenizer.texts_to_sequences(train_df["headline"].values)
test_sequences = tokenizer.texts_to_sequences(test_df["headline"].values)

# pad the sequences to make them all the same length
train_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=100)
test_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=100)

# create a dataset from the padded sequences
train_dataset = tf.data.Dataset.from_tensor_slices((train_padded_sequences, train_df["direction"].values))
test_dataset = tf.data.Dataset.from_tensor_slices((test_padded_sequences, test_df["direction"].values))

# create a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(train_dataset.batch(32), epochs=10, validation_data=test_dataset.batch(32))

# evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_dataset.batch(32))

print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)

# save the model
model.save("market_direction_model.h5")

# load the saved model
loaded_model = tf.keras.models.load_model("market_direction_model.h5")

headline = "Fed expected to raise interest rates again this quarter"
headline_sequence = tokenizer.texts_to_sequences([headline])
padded_headline_sequence = tf.keras.preprocessing.sequence.pad_sequences(headline_sequence, maxlen=100)

prediction = model.predict(padded_headline_sequence)

direction = encoder.inverse_transform(prediction.round().astype(int))

#Python: 3.7 or 3.8
#TensorFlow: 2.x (up to 2.5)
#scikit-learn: 0.22.x or 0.23.x
#pandas: 1.0.x or 1.1.x