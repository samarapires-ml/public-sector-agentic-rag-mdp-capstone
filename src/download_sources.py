from pathlib import Path
import csv
import urllib.request

manifest_path = Path("docs/corpus_manifest.csv")

with manifest_path.open("r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    sources = list(reader)

for source in sources:

    doc_id = source["doc_id"]
    url = source["source_url"]
    source_type = source["source_type"].strip().upper()

    print(f"Reading {doc_id} from manifest...")

    if source_type == "HTML":
        output_path = Path(f"data/raw/html/{doc_id}.html")

    elif source_type == "PDF":
        output_path = Path(f"data/raw/pdf/{doc_id}.pdf")

    else:
        print(f"SKIPPED {doc_id}: unsupported source type '{source_type}'")
        continue

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()

        output_path.write_bytes(content)

        print(f"SUCCESS: Downloaded {doc_id} to {output_path}")

    except Exception as error:
        print(f"FAILED: {doc_id}")
        print(f"Reason: {error}")

print("\nDownload attempt complete.")