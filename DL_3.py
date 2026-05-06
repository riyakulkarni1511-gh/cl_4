#Part A
#LSTM Time Series Prediction
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Create sine wave data
data = np.sin(np.linspace(0, 20, 100))

# Prepare data
X = []
y = []

for i in range(5, len(data)):
    X.append(data[i-5:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(10, input_shape=(5,1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=10, verbose=0)

# Prediction
pred = model.predict(X)

# Plot graph
plt.plot(y, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.show()

#=========================================================

#PART B: GRU Sentiment Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Dense

# Load dataseta
data = pd.read_csv("IMDB Dataset.csv")

# Convert sentiment into numbers
data['sentiment'] = data['sentiment'].map({
    'positive':1,
    'negative':0
})

# Input and output
reviews = data['review']
labels = data['sentiment']

# Convert text into numbers
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)
X = tokenizer.texts_to_sequences(reviews)

# Equal length sequences
X = pad_sequences(X, maxlen=100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Build GRU model
model = Sequential([
    Embedding(5000, 32, input_length=100),
    GRU(32),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=2, batch_size=64)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("GRU Model Loss :", loss)
print("GRU Model Accuracy :", accuracy)

# Test review
test_review = ["This movie was amazing"]

# Preprocess review
test_seq = tokenizer.texts_to_sequences(test_review)
test_pad = pad_sequences(test_seq, maxlen=100)

# Prediction
pred = model.predict(test_pad)

if pred > 0.5:
    print("Positive Review")
else:
    print("Negative Review")