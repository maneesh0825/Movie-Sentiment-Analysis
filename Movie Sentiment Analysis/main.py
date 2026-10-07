"""
PROJECT TITLE: Movie Sentiment Analysis using RNN
ALGORITHM: Simple RNN
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, SimpleRNN, Dense

sentences = [
    "I love this movie",
    "I really like this movie",
    "This movie is amazing",
    "I enjoyed this movie",
    "This is a wonderful movie",
    "I hate this movie",
    "I really dislike this movie",
    "This movie is terrible",
    "I did not enjoy this movie",
    "This is a boring movie"
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

vectorizer = TextVectorization(
    max_tokens=1000,
    output_sequence_length=6
)

vectorizer.adapt(sentences)

X = vectorizer(sentences)

model = Sequential([
    Embedding(input_dim=1000, output_dim=16),
    SimpleRNN(32, activation="tanh"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, labels, epochs=20, verbose=0)

test = vectorizer(["I love this wonderful movie"])
prediction = model.predict(test, verbose=0)[0][0]

print("Positive probability:", prediction)
print("Sentiment:", "Positive" if prediction >= 0.5 else "Negative")
