from pathlib import Path
import csv
import json
import re

from pypdf import PdfReader


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

manifest_path = Path("docs/corpus_manifest.csv")
raw_pdf_directory = Path("data/raw/pdf")
output_directory = Path("data/processed/documents")

output_directory.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Read corpus manifest
# --------------------------------------------------

with manifest_path.open(
    "r",
    encoding="utf-8-sig"
) as file:

    reader = csv.DictReader(file)
    sources = list(reader)


pdf_sources = [
    source
    for source in sources
    if source["source_type"].strip().upper() == "PDF"
]


print(f"Total sources in manifest: {len(sources)}")
print(f"PDF sources found: {len(pdf_sources)}")

print("\nProcessing PDF corpus:\n")


# --------------------------------------------------
# 3. Counters
# --------------------------------------------------

successful = 0
failed = 0


# --------------------------------------------------
# 4. Process PDFs
# --------------------------------------------------

for source in pdf_sources:

    doc_id = source["doc_id"]

    input_path = raw_pdf_directory / f"{doc_id}.pdf"
    output_path = output_directory / f"{doc_id}.json"

    try:

        # ------------------------------------------
        # Read PDF
        # ------------------------------------------

        reader = PdfReader(input_path)

        pages_text = []


        # ------------------------------------------
        # Extract each page
        # ------------------------------------------

        for page in reader.pages:

            page_text = page.extract_text() or ""

            pages_text.append(page_text)


        # ------------------------------------------
        # Combine pages
        # ------------------------------------------

        text = "\n\n".join(pages_text)


        # ------------------------------------------
        # Basic PDF cleanup
        # ------------------------------------------

        # Remove common PDF replacement/bullet artifact
        text = text.replace("�", "•")

        # Remove "Classification: Public"
        text = re.sub(
            r"Classification:\s*Public",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Remove trailing page-number text such as
        # "Page 1 of 1"
        text = re.sub(
            r"Page\s+\d+\s+of\s+\d+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Normalize excessive spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Normalize excessive blank lines
        text = re.sub(
            r"\n\s*\n\s*\n+",
            "\n\n",
            text
        )

        text = text.strip()


        # ------------------------------------------
        # QA
        # ------------------------------------------

        if not text:

            print(
                f"{doc_id}: FAIL - "
                "no text extracted"
            )

            failed += 1
            continue


        # ------------------------------------------
        # Build processed document
        # ------------------------------------------

        document = {
            "doc_id": source["doc_id"],
            "title": source["title"],
            "source_url": source["source_url"],
            "source_type": source["source_type"],
            "text": text
        }


        # ------------------------------------------
        # Save JSON
        # ------------------------------------------

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                document,
                file,
                ensure_ascii=False,
                indent=2
            )


        successful += 1

        print(
            f"{doc_id}: PASS "
            f"({len(reader.pages)} page(s), "
            f"{len(text):,} cleaned characters) "
            f"-> {output_path}"
        )


    except Exception as error:

        failed += 1

        print(
            f"{doc_id}: FAIL - {error}"
        )


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------

print("\n------------------------------")
print("PDF PROCESSING SUMMARY")
print("------------------------------")

print(f"Expected PDF sources: {len(pdf_sources)}")
print(f"Successfully processed: {successful}")
print(f"Failed:                 {failed}")


if successful == len(pdf_sources) and failed == 0:

    print(
        "\nAll PDF sources successfully processed."
    )

else:

    print(
        "\nWARNING: Some PDF sources failed processing."
    )