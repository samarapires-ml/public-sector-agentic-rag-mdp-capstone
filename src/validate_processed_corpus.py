from pathlib import Path
import csv
import json


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

manifest_path = Path("docs/corpus_manifest.csv")
processed_directory = Path("data/processed/documents")


# --------------------------------------------------
# 2. Read manifest
# --------------------------------------------------

with manifest_path.open(
    "r",
    encoding="utf-8-sig"
) as file:

    reader = csv.DictReader(file)
    sources = list(reader)


expected_ids = {
    source["doc_id"]
    for source in sources
}


# --------------------------------------------------
# 3. Find processed JSON documents
# --------------------------------------------------

json_files = list(
    processed_directory.glob("*.json")
)


print("------------------------------")
print("PROCESSED CORPUS VALIDATION")
print("------------------------------")

print(f"Expected documents: {len(expected_ids)}")
print(f"JSON files found:   {len(json_files)}")

print()


# --------------------------------------------------
# 4. Validate each expected document
# --------------------------------------------------

passed = 0
failed = 0

seen_ids = set()


for source in sources:

    expected_id = source["doc_id"]

    json_path = (
        processed_directory
        / f"{expected_id}.json"
    )

    errors = []


    # ----------------------------------------------
    # File existence
    # ----------------------------------------------

    if not json_path.exists():

        print(
            f"{expected_id}: FAIL - "
            "JSON file missing"
        )

        failed += 1
        continue


    # ----------------------------------------------
    # Read JSON
    # ----------------------------------------------

    try:

        with json_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            document = json.load(file)

    except Exception as error:

        print(
            f"{expected_id}: FAIL - "
            f"invalid JSON: {error}"
        )

        failed += 1
        continue


    # ----------------------------------------------
    # Required fields
    # ----------------------------------------------

    required_fields = [
        "doc_id",
        "title",
        "source_url",
        "source_type",
        "text"
    ]

    for field in required_fields:

        if field not in document:

            errors.append(
                f"missing field '{field}'"
            )


    # ----------------------------------------------
    # Metadata validation
    # ----------------------------------------------

    if document.get("doc_id") != expected_id:

        errors.append(
            "doc_id does not match manifest"
        )


    if document.get("title") != source["title"]:

        errors.append(
            "title does not match manifest"
        )


    if (
        document.get("source_url")
        != source["source_url"]
    ):

        errors.append(
            "source_url does not match manifest"
        )


    if (
        document.get("source_type", "")
        .strip()
        .upper()
        != source["source_type"]
        .strip()
        .upper()
    ):

        errors.append(
            "source_type does not match manifest"
        )


    # ----------------------------------------------
    # Text validation
    # ----------------------------------------------

    text = document.get("text", "")

    if not isinstance(text, str):

        errors.append(
            "text is not a string"
        )

    elif not text.strip():

        errors.append(
            "text is empty"
        )


    # ----------------------------------------------
    # Duplicate ID validation
    # ----------------------------------------------

    doc_id = document.get("doc_id")

    if doc_id in seen_ids:

        errors.append(
            "duplicate doc_id"
        )

    else:

        seen_ids.add(doc_id)


    # ----------------------------------------------
    # Result
    # ----------------------------------------------

    if errors:

        failed += 1

        print(
            f"{expected_id}: FAIL"
        )

        for error in errors:

            print(
                f"  - {error}"
            )

    else:

        passed += 1

        print(
            f"{expected_id}: PASS "
            f"({len(text):,} characters)"
        )


# --------------------------------------------------
# 5. Check for unexpected JSON files
# --------------------------------------------------

actual_ids = {
    path.stem
    for path in json_files
}

unexpected_ids = (
    actual_ids - expected_ids
)

missing_ids = (
    expected_ids - actual_ids
)


# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print()
print("------------------------------")
print("VALIDATION SUMMARY")
print("------------------------------")

print(f"Expected documents: {len(expected_ids)}")
print(f"Passed:             {passed}")
print(f"Failed:             {failed}")
print(f"Missing files:      {len(missing_ids)}")
print(f"Unexpected files:   {len(unexpected_ids)}")


if missing_ids:

    print(
        "\nMissing document IDs:"
    )

    for doc_id in sorted(missing_ids):

        print(f"  - {doc_id}")


if unexpected_ids:

    print(
        "\nUnexpected document IDs:"
    )

    for doc_id in sorted(unexpected_ids):

        print(f"  - {doc_id}")


if (
    passed == len(expected_ids)
    and failed == 0
    and not missing_ids
    and not unexpected_ids
):

    print(
        "\nProcessed Corpus v1.0 "
        "successfully validated."
    )

else:

    print(
        "\nWARNING: Processed corpus "
        "requires review."
    )