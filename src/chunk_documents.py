from pathlib import Path
import json


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DOCUMENT_DIRECTORY = Path(
    "data/processed/documents"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/chunks"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Natural boundary helper
# --------------------------------------------------

def find_natural_start(
    text,
    target_start,
    previous_end
):

    if target_start <= 0:
        return 0

    newline = text.find(
        "\n",
        target_start,
        previous_end
    )

    if newline != -1:
        return newline + 1

    sentence_end = text.find(
        ". ",
        target_start,
        previous_end
    )

    if sentence_end != -1:
        return sentence_end + 2

    return target_start


# --------------------------------------------------
# 3. Chunking function
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200
):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length
        )

        end = target_end

        if target_end < text_length:

            newline = text.rfind(
                "\n",
                start + (chunk_size // 2),
                target_end
            )

            if newline != -1:
                end = newline

            else:

                sentence_end = text.rfind(
                    ". ",
                    start + (chunk_size // 2),
                    target_end
                )

                if sentence_end != -1:
                    end = sentence_end + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        desired_start = max(
            0,
            end - overlap
        )

        next_start = find_natural_start(
            text,
            desired_start,
            end
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# --------------------------------------------------
# 4. Load processed documents
# --------------------------------------------------

document_files = sorted(
    DOCUMENT_DIRECTORY.glob("*.json")
)

print(
    f"Processed documents found: "
    f"{len(document_files)}"
)

print("\nCreating chunks:\n")


# --------------------------------------------------
# 5. Process corpus
# --------------------------------------------------

all_chunks = []

successful_documents = 0
failed_documents = 0


for document_path in document_files:

    try:

        with document_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            document = json.load(file)

        text = document["text"]

        chunks = chunk_text(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )


        # ------------------------------------------
        # Create metadata-rich chunks
        # ------------------------------------------

        for index, chunk_text_value in enumerate(
            chunks,
            start=1
        ):

            chunk_id = (
                f"{document['doc_id']}"
                f"-CH-{index:03d}"
            )

            chunk_document = {
                "chunk_id": chunk_id,
                "doc_id": document["doc_id"],
                "title": document["title"],
                "source_url": document["source_url"],
                "source_type": document["source_type"],
                "chunk_index": index,
                "text": chunk_text_value
            }

            all_chunks.append(
                chunk_document
            )


        successful_documents += 1

        print(
            f"{document['doc_id']}: PASS "
            f"({len(chunks)} chunks)"
        )


    except Exception as error:

        failed_documents += 1

        print(
            f"{document_path.name}: "
            f"FAIL - {error}"
        )


# --------------------------------------------------
# 6. Save one JSON file per chunk
# --------------------------------------------------

for chunk in all_chunks:

    output_path = (
        OUTPUT_DIRECTORY
        / f"{chunk['chunk_id']}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunk,
            file,
            ensure_ascii=False,
            indent=2
        )


# --------------------------------------------------
# 7. Save complete chunk corpus
# --------------------------------------------------

corpus_path = Path(
    "data/processed/chunks.json"
)

with corpus_path.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# 8. Summary
# --------------------------------------------------

print("\n------------------------------")
print("CHUNKING SUMMARY")
print("------------------------------")

print(
    f"Documents processed: "
    f"{successful_documents}"
)

print(
    f"Documents failed:    "
    f"{failed_documents}"
)

print(
    f"Total chunks:        "
    f"{len(all_chunks)}"
)

print(
    f"Chunk size:          "
    f"{CHUNK_SIZE}"
)

print(
    f"Target overlap:      "
    f"{CHUNK_OVERLAP}"
)


if (
    successful_documents == len(document_files)
    and failed_documents == 0
):

    print(
        "\nChunk Corpus v1.0 "
        "successfully created."
    )

else:

    print(
        "\nWARNING: Chunk corpus "
        "requires review."
    )