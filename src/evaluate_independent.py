
"""
Independent evaluation of the public-sector agentic RAG prototype.

Input:
    data/evaluation/independent_test_questions.csv

Outputs:
    data/evaluation/independent_results.csv
    data/evaluation/independent_evidence.json
    data/evaluation/independent_summary.json

Run from the project root:
    python src/evaluate_independent.py
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agent import run_agent


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

QUESTIONS_FILE = EVAL_DIR / "independent_test_questions.csv"
RESULTS_FILE = EVAL_DIR / "independent_results.csv"
EVIDENCE_FILE = EVAL_DIR / "independent_evidence.json"
SUMMARY_FILE = EVAL_DIR / "independent_summary.json"

EXPECTED_CATEGORIES = {
    "supported": "answered",
    "insufficient_evidence": "abstain",
    "ambiguous": "clarify",
    "out_of_scope": "out_of_scope",
}

RESULT_FIELDS = [
    "test_id",
    "question",
    "category",
    "expected_status",
    "actual_status",
    "status_correct",
    "answer",
    "reference_doc_ids",
    "retrieved_chunk_ids",
    "retrieved_doc_ids",
    "top_score",
    "supporting_chunks",
    "gate_sufficient",
    "error",
]


# --------------------------------------------------
# Load and validate questions
# --------------------------------------------------

def load_questions():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "test_id",
            "question",
            "category",
            "expected_status",
            "reference_doc_ids",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        questions = list(reader)

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    seen_ids = set()

    for row in questions:
        test_id = row["test_id"].strip()
        question = row["question"].strip()
        category = row["category"].strip()
        expected = row["expected_status"].strip()

        if not test_id or not question:
            raise ValueError(
                "Every question needs a test_id and question."
            )

        if test_id in seen_ids:
            raise ValueError(f"Duplicate test ID: {test_id}")

        seen_ids.add(test_id)

        if category not in EXPECTED_CATEGORIES:
            raise ValueError(
                f"{test_id}: Unknown category '{category}'"
            )

        required_status = EXPECTED_CATEGORIES[category]

        if expected != required_status:
            raise ValueError(
                f"{test_id}: Category '{category}' should "
                f"have expected_status='{required_status}', "
                f"not '{expected}'."
            )

    return questions


# --------------------------------------------------
# JSON conversion for evidence preservation
# --------------------------------------------------

def json_safe(value):
    """Convert common numeric and container types to JSON."""
    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        return json_safe(value.item())

    return str(value)


# --------------------------------------------------
# Execute independent evaluation
# --------------------------------------------------

def evaluate():
    questions = load_questions()

    print("\n======================================")
    print("INDEPENDENT AGENT EVALUATION")
    print("======================================")
    print(f"Total questions: {len(questions)}")

    results = []
    evidence_records = []

    for index, item in enumerate(questions, start=1):

        test_id = item["test_id"].strip()
        question = item["question"].strip()
        category = item["category"].strip()
        expected = item["expected_status"].strip()

        print("\n--------------------------------------")
        print(f"Test {index}/{len(questions)}: {test_id}")
        print(f"Category: {category}")
        print(f"Question: {question}")

        actual = "error"
        answer = ""
        sources = []
        evidence = []
        assessment = {}
        error = ""

        try:
            response = run_agent(question)

            actual = response.get("status", "unknown")
            answer = response.get("answer", "")
            sources = response.get("sources") or []
            evidence = response.get("evidence") or []
            assessment = (
                response.get("evidence_assessment") or {}
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {error}")

        status_correct = actual == expected

        chunk_ids = [
            str(chunk.get("chunk_id", ""))
            for chunk in evidence
        ]

        doc_ids = sorted({
            str(chunk.get("metadata", {}).get("doc_id", ""))
            for chunk in evidence
            if chunk.get("metadata", {}).get("doc_id")
        })

        results.append({
            "test_id": test_id,
            "question": question,
            "category": category,
            "expected_status": expected,
            "actual_status": actual,
            "status_correct": int(status_correct),
            "answer": answer,
            "reference_doc_ids": item.get(
                "reference_doc_ids", ""
            ),
            "retrieved_chunk_ids": "; ".join(chunk_ids),
            "retrieved_doc_ids": "; ".join(doc_ids),
            "top_score": assessment.get("top_score"),
            "supporting_chunks": assessment.get(
                "supporting_chunks"
            ),
            "gate_sufficient": assessment.get(
                "sufficient"
            ),
            "error": error,
        })

        # Preserve exact evidence for later manual review.
        evidence_records.append({
            "test_id": test_id,
            "question": question,
            "category": category,
            "expected_status": expected,
            "actual_status": actual,
            "answer": answer,
            "reference_doc_ids": item.get(
                "reference_doc_ids", ""
            ),
            "sources": json_safe(sources),
            "evidence": json_safe(evidence),
            "evidence_assessment": json_safe(assessment),
            "error": error,
        })

        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        print(
            f"Result:   {'PASS' if status_correct else 'FAIL'}"
        )

    return results, evidence_records


# --------------------------------------------------
# Calculate status-selection metrics
# --------------------------------------------------

def calculate_summary(results):

    total = len(results)
    correct = sum(
        row["status_correct"] for row in results
    )

    categories = defaultdict(list)

    for row in results:
        categories[row["category"]].append(row)

    category_metrics = {}

    for category, rows in categories.items():
        category_correct = sum(
            row["status_correct"] for row in rows
        )

        category_metrics[category] = {
            "total": len(rows),
            "correct": category_correct,
            "accuracy": category_correct / len(rows),
        }

    expected_counts = Counter(
        row["expected_status"] for row in results
    )

    actual_counts = Counter(
        row["actual_status"] for row in results
    )

    confusion_matrix = {}

    for expected in sorted(expected_counts):
        confusion_matrix[expected] = dict(
            Counter(
                row["actual_status"]
                for row in results
                if row["expected_status"] == expected
            )
        )

    return {
        "evaluation_type": "independent_status_evaluation",
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_questions": total,
        "correct_statuses": correct,
        "status_accuracy": correct / total if total else 0,
        "errors": sum(
            row["actual_status"] == "error"
            for row in results
        ),
        "category_metrics": category_metrics,
        "expected_status_counts": dict(expected_counts),
        "actual_status_counts": dict(actual_counts),
        "confusion_matrix": confusion_matrix,
        "note": (
            "Status accuracy does not establish factual "
            "correctness, completeness, or citation support. "
            "These require separate manual assessment."
        ),
    }


# --------------------------------------------------
# Save results
# --------------------------------------------------

def save_outputs(results, evidence_records, summary):

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=RESULT_FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    with EVIDENCE_FILE.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            evidence_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with SUMMARY_FILE.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


# --------------------------------------------------
# Print evaluation summary
# --------------------------------------------------

def print_summary(summary):

    print("\n======================================")
    print("INDEPENDENT EVALUATION SUMMARY")
    print("======================================")

    print(f"Total: {summary['total_questions']}")
    print(f"Correct: {summary['correct_statuses']}")
    print(
        f"Status accuracy: "
        f"{summary['status_accuracy'] * 100:.2f}%"
    )
    print(f"Errors: {summary['errors']}")

    print("\nCategory breakdown:")

    for category, metrics in (
        summary["category_metrics"].items()
    ):
        print(
            f"  {category}: "
            f"{metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy'] * 100:.2f}%)"
        )

    print("\nFiles saved:")
    print(f"  {RESULTS_FILE}")
    print(f"  {EVIDENCE_FILE}")
    print(f"  {SUMMARY_FILE}")

    print("\nNext: Manual factual and citation assessment.")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    results, evidence_records = evaluate()

    summary = calculate_summary(results)

    save_outputs(
        results,
        evidence_records,
        summary,
    )

    print_summary(summary)
