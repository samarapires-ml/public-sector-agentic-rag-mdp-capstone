from pathlib import Path
import json
from collections import Counter


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

documents_directory = Path(
    "data/processed/documents"
)

chunks_directory = Path(
    "data/processed/chunks"
)

combined_chunks_path = Path(
    "data/processed/chunks.json"
)

CHUNK_SIZE = 1200


# --------------------------------------------------
# 2. Load processed documents
# --------------------------------------------------

document_files = sorted(
    documents_directory.glob("*.json")
)

documents = {}

for document_path in document_files:

    with document_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    documents[document["doc_id"]] = document


# --------------------------------------------------
# 3. Load combined chunk corpus
# --------------------------------------------------

with combined_chunks_path.open(
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


individual_chunk_files = list(
    chunks_directory.glob("*.json")
)


print("------------------------------")
print("CHUNK CORPUS VALIDATION")
print("------------------------------")

print(f"Documents found:       {len(documents)}")
print(f"Combined chunks found: {len(chunks)}")
print(f"Chunk files found:     {len(individual_chunk_files)}")

print()


# --------------------------------------------------
# 4. Validation
# --------------------------------------------------

errors = []

required_fields = [
    "chunk_id",
    "doc_id",
    "title",
    "source_url",
    "source_type",
    "chunk_index",
    "text"
]


# --------------------------------------------------
# 5. Check every chunk
# --------------------------------------------------

chunk_ids = []

for chunk in chunks:

    chunk_id = chunk.get(
        "chunk_id",
        "<missing chunk_id>"
    )

    # Required fields
    for field in required_fields:

        if field not in chunk:

            errors.append(
                f"{chunk_id}: "
                f"missing field '{field}'"
            )


    # Document reference
    doc_id = chunk.get("doc_id")

    if doc_id not in documents:

        errors.append(
            f"{chunk_id}: "
            f"unknown doc_id '{doc_id}'"
        )

        continue


    source_document = documents[doc_id]


    # Metadata consistency
    for field in [
        "title",
        "source_url",
        "source_type"
    ]:

        if (
            chunk.get(field)
            != source_document.get(field)
        ):

            errors.append(
                f"{chunk_id}: "
                f"{field} does not match "
                f"source document"
            )


    # Text checks
    text = chunk.get("text")

    if not isinstance(text, str):

        errors.append(
            f"{chunk_id}: text is not a string"
        )

    elif not text.strip():

        errors.append(
            f"{chunk_id}: text is empty"
        )

    elif len(text) > CHUNK_SIZE:

        errors.append(
            f"{chunk_id}: "
            f"text exceeds chunk size "
            f"({len(text)} > {CHUNK_SIZE})"
        )


    # Chunk ID format
    chunk_index = chunk.get("chunk_index")

    if isinstance(chunk_index, int):

        expected_chunk_id = (
            f"{doc_id}-CH-{chunk_index:03d}"
        )

        if chunk_id != expected_chunk_id:

            errors.append(
                f"{chunk_id}: "
                f"expected ID "
                f"'{expected_chunk_id}'"
            )

    else:

        errors.append(
            f"{chunk_id}: "
            "chunk_index is not an integer"
        )


    chunk_ids.append(chunk_id)


# --------------------------------------------------
# 6. Duplicate chunk IDs
# --------------------------------------------------

id_counts = Counter(chunk_ids)

duplicate_ids = [
    chunk_id
    for chunk_id, count in id_counts.items()
    if count > 1
]

for chunk_id in duplicate_ids:

    errors.append(
        f"{chunk_id}: duplicate chunk_id"
    )


# --------------------------------------------------
# 7. Check every document has chunks
# --------------------------------------------------

chunk_doc_ids = {
    chunk["doc_id"]
    for chunk in chunks
    if "doc_id" in chunk
}

for doc_id in documents:

    if doc_id not in chunk_doc_ids:

        errors.append(
            f"{doc_id}: document has no chunks"
        )


# --------------------------------------------------
# 8. Check individual chunk files
# --------------------------------------------------

if len(individual_chunk_files) != len(chunks):

    errors.append(
        "Number of individual chunk files "
        "does not match combined chunk corpus"
    )


for chunk in chunks:

    expected_path = (
        chunks_directory
        / f"{chunk['chunk_id']}.json"
    )

    if not expected_path.exists():

        errors.append(
            f"{chunk['chunk_id']}: "
            "individual JSON file missing"
        )


# --------------------------------------------------
# 9. Report chunks per document
# --------------------------------------------------

chunks_per_document = Counter(
    chunk["doc_id"]
    for chunk in chunks
)


print("Chunks per document:\n")

for doc_id in sorted(documents):

    print(
        f"{doc_id}: "
        f"{chunks_per_document[doc_id]}"
    )


# --------------------------------------------------
# 10. Summary
# --------------------------------------------------

print()
print("------------------------------")
print("VALIDATION SUMMARY")
print("------------------------------")

print(f"Documents:        {len(documents)}")
print(f"Chunks:           {len(chunks)}")
print(f"Chunk files:      {len(individual_chunk_files)}")
print(f"Duplicate IDs:    {len(duplicate_ids)}")
print(f"Validation errors:{len(errors):>5}")


if errors:

    print("\nErrors:\n")

    for error in errors:

        print(f"- {error}")

    print(
        "\nWARNING: Chunk corpus "
        "requires review."
    )

else:

    print(
        "\nChunk Corpus v1.0 "
        "successfully validated."
    )