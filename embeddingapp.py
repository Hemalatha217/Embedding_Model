from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input two sentences
sentence1 = input("Enter Sentence 1: ")
sentence2 = input("Enter Sentence 2: ")

# Generate embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# Display embeddings
print("\n===== EMBEDDINGS =====")
print("\nSentence 1:", sentence1)
print("Embedding 1:", embedding1)
print("Dimension:", len(embedding1))

print("\nSentence 2:", sentence2)
print("Embedding 2:", embedding2)
print("Dimension:", len(embedding2))

# Calculate similarity
similarity = cosine_similarity(
    [embedding1],
    [embedding2]
)

# Display similarity score
print("\n===== SIMILARITY SCORE =====")
print("Similarity:", similarity[0][0])

# Interpretation
if similarity[0][0] > 0.8:
    print("The sentences are very similar.")
elif similarity[0][0] > 0.5:
    print("The sentences are somewhat similar.")
else:
    print("The sentences are not very similar.")