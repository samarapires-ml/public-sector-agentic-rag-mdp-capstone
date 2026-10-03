from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VECTOR_DB_PATH = Path("data/vector_store")

COLLECTION_NAME = "alberta_public_services"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# 3. Connect to persistent ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Connected to collection: {COLLECTION_NAME}"
)

print(
    f"Chunks available: {collection.count()}"
)


# --------------------------------------------------
# 4. Retrieval function
# --------------------------------------------------

def retrieve_chunks(query, top_k=TOP_K):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    for i in range(
        len(results["ids"][0])
    ):

        retrieved_chunks.append(
            {
                "chunk_id":
                    results["ids"][0][i],

                "text":
                    results["documents"][0][i],

                "metadata":
                    results["metadatas"][0][i],

                "distance":
                    results["distances"][0][i]
            }
        )

    return retrieved_chunks


# --------------------------------------------------
# 5. Test retrieval
# --------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    results = retrieve_chunks(query)

    print("\n==============================")
    print("RETRIEVAL RESULTS")
    print("==============================")

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print("--------------------------------")
        print(f"Rank: {rank}")
        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )
        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )
        print(
            f"Document: "
            f"{metadata['title']}"
        )
        print(
            f"Doc ID: "
            f"{metadata['doc_id']}"
        )
        print(
            f"Source: "
            f"{metadata['source_url']}"
        )
        print("--------------------------------")

        print(result["text"])
        print()