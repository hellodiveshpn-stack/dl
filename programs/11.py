import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset
text = [
    "the sun rises in the east",
    "the sun sets in the west",
    "machine learning is powerful",
    "deep learning is interesting"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

total_words = len(tokenizer.word_index) + 1

# Create sequences
seq = []

for sentence in text:
    tokens = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokens)):
        seq.append(tokens[:i + 1])

# Padding
max_len = max(len(x) for x in seq)

seq = pad_sequences(
    seq,
    maxlen=max_len,
    padding='pre'
)

# Split into X and y
X = seq[:, :-1]
y = np.eye(total_words)[seq[:, -1]]

# Build RNN Model
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_len - 1
    ),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

# Prediction Function
def next_word(text_input):
    token = tokenizer.texts_to_sequences([text_input])[0]

    token = pad_sequences(
        [token],
        maxlen=max_len - 1,
        padding='pre'
    )

    pred = model.predict(token, verbose=0)

    return tokenizer.index_word[np.argmax(pred)]

# Test
print("Next word:", next_word("the sun rises in the"))