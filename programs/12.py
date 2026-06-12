import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = [
    "i want to reset my password",
    "how to get refund",
    "my account is locked",
    "payment failed"
]

labels = [0, 1, 2, 3]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    sequences,
    maxlen=10
)

y = np.array(labels)

# Build Bi-LSTM Model
model = Sequential()

model.add(
    Embedding(
        input_dim=1000,
        output_dim=64,
        input_length=10
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(
    Dense(
        32,
        activation='relu'
    )
)

model.add(
    Dense(
        4,
        activation='softmax'
    )
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(
    X,
    y,
    epochs=10,
    verbose=1
)

# Prediction Function
def predict(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=10
    )

    pred = model.predict(
        padded,
        verbose=0
    )

    return np.argmax(pred)

# Test
print(
    predict("i forgot my password")
)