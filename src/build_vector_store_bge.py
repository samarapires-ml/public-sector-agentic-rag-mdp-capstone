from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHUNKS_PATH = Path("data/processed/chunks.json")

VECTOR_DB_PATH = Path("data/vector_store")

COLLECTION_NAME = "alberta_public_services_bge"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


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

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Create persistent Chroma client
# --------------------------------------------------

VECTOR_DB_PATH.mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)


# --------------------------------------------------
# 5. Rebuild collection
# --------------------------------------------------

# Delete the existing collection if this script
# has been run before.
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
    print(
        f"Existing collection "
        f"'{COLLECTION_NAME}' removed."
    )

except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description":
            "Alberta public-service policy corpus"
    }
)


# --------------------------------------------------
# 6. Prepare chunk data
# --------------------------------------------------

ids = []
documents = []
metadatas = []

for chunk in chunks:

    ids.append(
        chunk["chunk_id"]
    )

    documents.append(
        chunk["text"]
    )

    metadatas.append(
        {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source_url": chunk["source_url"],
            "source_type": chunk["source_type"]
        }
    )


# --------------------------------------------------
# 7. Create embeddings
# --------------------------------------------------

print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    f"Embeddings created: "
    f"{embeddings.shape}"
)


# --------------------------------------------------
# 8. Add chunks to ChromaDB
# --------------------------------------------------

print("\nAdding chunks to ChromaDB...")

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings.tolist()
)


# --------------------------------------------------
# 9. Validate stored collection
# --------------------------------------------------

stored_count = collection.count()

print("\n------------------------------")
print("VECTOR STORE BUILD SUMMARY")
print("------------------------------")

print(
    f"Chunks in source corpus: {len(chunks)}"
)

print(
    f"Chunks stored in Chroma: {stored_count}"
)

print(
    f"Embedding dimensions: "
    f"{embeddings.shape[1]}"
)

print(
    f"Collection: {COLLECTION_NAME}"
)

print(
    f"Database path: {VECTOR_DB_PATH}"
)


if stored_count == len(chunks):

    print(
        "\nVector Store v1.0 "
        "successfully created."
    )

else:

    print(
        "\nERROR: Stored chunk count "
        "does not match source corpus."
    )