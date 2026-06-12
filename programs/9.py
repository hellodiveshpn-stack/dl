import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Download tokenizer
nltk.download('punkt')

# Sample dataset
text = [
    "I love machine learning",
    "Word2Vec is a useful technique",
    "I love deep learning",
    "Natural language processing is interesting"
]

# Tokenization
data = [word_tokenize(sentence.lower()) for sentence in text]

# Train Word2Vec model
model = Word2Vec(
    sentences=data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Get word vector
print("Vector for 'learning':")
print(model.wv['learning'])

# Find similar words
print("\nWords similar to 'learning':")
print(model.wv.most_similar('learning'))