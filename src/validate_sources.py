from pathlib import Path
import csv

manifest_path = Path("docs/corpus_manifest.csv")

with manifest_path.open("r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    sources = list(reader)

passed = 0
failed = 0

for source in sources:

    doc_id = source["doc_id"]
    source_type = source["source_type"].strip().upper()

    if source_type == "HTML":
        file_path = Path(f"data/raw/html/{doc_id}.html")

    elif source_type == "PDF":
        file_path = Path(f"data/raw/pdf/{doc_id}.pdf")

    else:
        print(f"{doc_id}: FAIL - unsupported source type")
        failed += 1
        continue

    # Check 1: Does the file exist?
    if not file_path.exists():
        print(f"{doc_id}: FAIL - file missing")
        failed += 1
        continue

    # Check 2: Is the file empty?
    if file_path.stat().st_size == 0:
        print(f"{doc_id}: FAIL - file is empty")
        failed += 1
        continue

    # Check 3: Does the file look like the expected format?
    if source_type == "HTML":

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).lower()

        if "<html" not in text:
            print(f"{doc_id}: FAIL - does not appear to be HTML")
            failed += 1
            continue

    elif source_type == "PDF":

        with file_path.open("rb") as file:
            header = file.read(5)

        if header != b"%PDF-":
            print(f"{doc_id}: FAIL - does not appear to be a PDF")
            failed += 1
            continue

    print(f"{doc_id}: PASS")
    passed += 1


print("\n------------------------------")
print("CORPUS VALIDATION SUMMARY")
print("------------------------------")
print(f"Expected sources: {len(sources)}")
print(f"Passed:           {passed}")
print(f"Failed:           {failed}")

if failed == 0 and passed == len(sources):
    print("\nRaw Corpus v1.0 successfully validated.")
else:
    print("\nCorpus validation failed. Review the errors above.")