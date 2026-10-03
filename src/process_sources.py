from pathlib import Path
import csv
import json

from bs4 import BeautifulSoup


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

manifest_path = Path("docs/corpus_manifest.csv")
output_directory = Path("data/processed/documents")

# Make sure the output folder exists
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


# Keep HTML sources only for this stage
html_sources = [
    source
    for source in sources
    if source["source_type"].strip().upper() == "HTML"
]


print(f"Total sources in manifest: {len(sources)}")
print(f"HTML sources found: {len(html_sources)}")

print("\nProcessing HTML corpus:\n")


# --------------------------------------------------
# 3. Process each HTML source
# --------------------------------------------------

successful = 0
failed = 0


for source in html_sources:

    doc_id = source["doc_id"]

    input_path = Path(
        f"data/raw/html/{doc_id}.html"
    )

    output_path = output_directory / f"{doc_id}.json"

    try:

        # ------------------------------------------
        # Read raw HTML
        # ------------------------------------------

        raw_bytes = input_path.read_bytes()
        html = raw_bytes.decode("utf-8")


        # ------------------------------------------
        # Parse HTML
        # ------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )


        # ------------------------------------------
        # Remove non-content elements
        # ------------------------------------------

        for element in soup.find_all(
            ["script", "style", "noscript"]
        ):
            element.decompose()


        # ------------------------------------------
        # Find main page content
        # ------------------------------------------

        main_content = soup.find("main")

        if main_content is None:

            print(
                f"{doc_id}: FAIL - "
                "<main> section not found"
            )

            failed += 1
            continue


        # ------------------------------------------
        # Remove Alberta.ca skip links
        # ------------------------------------------

        for skip_link in main_content.find_all(
            class_="goa-skip-link"
        ):
            skip_link.decompose()


        # ------------------------------------------
        # Remove "Explore pages in:" navigation
        # ------------------------------------------

        explore_heading = None

        for heading in main_content.find_all("h2"):

            heading_text = heading.get_text(
                " ",
                strip=True
            )

            if heading_text == "Explore pages in:":
                explore_heading = heading
                break


        if explore_heading is not None:

            nav_list = explore_heading.find_next("ul")

            if nav_list is not None:
                nav_list.decompose()

            explore_heading.decompose()


        # ------------------------------------------
        # Remove Next navigation
        # ------------------------------------------

        for next_button in main_content.find_all(
            "div",
            class_="goa-next-btn"
        ):
            next_button.decompose()


        # ------------------------------------------
        # Remove Previous navigation
        # ------------------------------------------

        for previous_button in main_content.find_all(
            "div",
            class_="goa-previous-btn"
        ):
            previous_button.decompose()


        # ------------------------------------------
        # Extract cleaned text
        # ------------------------------------------

        text = main_content.get_text(
            separator="\n",
            strip=True
        )


        # ------------------------------------------
        # Basic QA
        # ------------------------------------------

        if not text:

            print(
                f"{doc_id}: FAIL - "
                "no text extracted"
            )

            failed += 1
            continue


        # ------------------------------------------
        # Create processed document
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
            f"({len(text):,} cleaned characters) "
            f"-> {output_path}"
        )


    except Exception as error:

        failed += 1

        print(
            f"{doc_id}: FAIL - {error}"
        )


# --------------------------------------------------
# 4. Processing summary
# --------------------------------------------------

print("\n------------------------------")
print("HTML PROCESSING SUMMARY")
print("------------------------------")

print(f"Expected HTML sources: {len(html_sources)}")
print(f"Successfully processed: {successful}")
print(f"Failed:                 {failed}")

if successful == len(html_sources) and failed == 0:

    print(
        "\nAll HTML sources successfully processed."
    )

else:

    print(
        "\nWARNING: Some HTML sources failed processing."
    )