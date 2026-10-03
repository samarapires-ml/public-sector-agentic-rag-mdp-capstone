from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHUNKS_PATH = Path(
    "data/processed/chunks.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

QUERY = (
    "What do I need to register "
    "a vehicle in Alberta?"
)

TOP_K = 5


# --------------------------------------------------
# 2. Load chunk corpus
# --------------------------------------------------

with CHUNKS_PATH.open(
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


print(f"Chunks loaded: {len(chunks)}")


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Extract chunk text
# --------------------------------------------------

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# --------------------------------------------------
# 5. Create embeddings
# --------------------------------------------------

print("\nCreating chunk embeddings...")

chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)

print(
    f"Embedding matrix shape: "
    f"{tuple(chunk_embeddings.shape)}"
)


# --------------------------------------------------
# 6. Embed test query
# --------------------------------------------------

query_embedding = model.encode(
    QUERY,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# --------------------------------------------------
# 7. Calculate cosine similarity
# --------------------------------------------------

scores = cos_sim(
    query_embedding,
    chunk_embeddings
)[0]


# --------------------------------------------------
# 8. Rank chunks
# --------------------------------------------------

ranked_indices = scores.argsort(
    descending=True
)[:TOP_K]


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n================================")
print("SEMANTIC RETRIEVAL TEST")
print("================================")

print(f"\nQuery: {QUERY}\n")


for rank, index in enumerate(
    ranked_indices,
    start=1
):

    index = int(index)

    chunk = chunks[index]

    score = float(scores[index])

    print("--------------------------------")
    print(f"Rank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Document: {chunk['title']}")
    print("--------------------------------")

    print(chunk["text"])

    print()