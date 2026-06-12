from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download tokenizer
nltk.download('punkt')

# Sample documents
documents = [
    "I like machine learning",
    "Doc2Vec learns document embeddings",
    "Deep learning is powerful",
    "Natural language processing is fun"
]

# Tag documents
tagged_data = [
    TaggedDocument(
        words=word_tokenize(doc.lower()),
        tags=[str(i)]
    )
    for i, doc in enumerate(documents)
]

# Train Doc2Vec model
model = Doc2Vec(
    tagged_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=40
)

# Get vectors of two documents
vec1 = model.dv['0']
vec2 = model.dv['2']

# Compute similarity
sim = cosine_similarity([vec1], [vec2])

print("Similarity between Document 0 and 2:")
print(sim[0][0])

# Compare new document
new_doc = "I enjoy learning NLP"

new_vec = model.infer_vector(
    word_tokenize(new_doc.lower())
)

sim_new = cosine_similarity(
    [new_vec],
    [model.dv['0']]
)

print("\nSimilarity between new document and Document 0:")
print(sim_new[0][0])

# Find most similar documents
print("\nMost similar documents to Document 0:")
print(model.dv.most_similar('0'))