import csv
import sys
from pathlib import Path
from collections import Counter

from agent import run_agent


# --------------------------------------------------
# Agent Evaluation v1.0
# --------------------------------------------------
#
# Purpose:
# Evaluate the complete public-sector RAG agent
# against a manually constructed benchmark dataset.
#
# Evaluation categories:
#
#   1. Supported queries
#   2. Ambiguous queries
#   3. Out-of-scope queries
#   4. Insufficient-evidence queries
#
# --------------------------------------------------


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_questions.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_results.csv"
)


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

def load_evaluation_questions():
    """
    Load the manually constructed evaluation
    benchmark from CSV.
    """

    questions = []

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            questions.append(
                {
                    "test_id": row["test_id"],
                    "question": row["question"],
                    "category": row["category"],
                    "expected_status":
                        row["expected_status"],
                }
            )

    return questions


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

def evaluate_agent():
    """
    Run every benchmark question through the
    complete agent and compare the actual routing
    decision with the expected status.
    """

    questions = load_evaluation_questions()

    results = []

    total = len(questions)

    print("\n================================")
    print("END-TO-END AGENT EVALUATION")
    print("================================")

    print(
        f"\nEvaluation questions: {total}"
    )


    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_status = (
            item["expected_status"]
        )

        category = item["category"]

        print("\n--------------------------------")
        print(
            f"Test {number}/{total}"
        )
        print(
            f"ID: {item['test_id']}"
        )
        print(
            f"Category: {category}"
        )
        print(
            f"Question: {question}"
        )


        # ------------------------------------------
        # Run complete agent
        # ------------------------------------------

        try:

            response = run_agent(
                question
            )

            actual_status = response.get(
                "status",
                "unknown"
            )

            answer = response.get(
                "answer",
                ""
            )

            assessment = response.get(
                "evidence_assessment"
            ) or {}

            top_score = assessment.get(
                "top_score"
            )

            supporting_chunks = assessment.get(
                "supporting_chunks",
                0
            )

            error = ""


        except Exception as exc:

            actual_status = "error"

            answer = ""

            top_score = None

            supporting_chunks = 0

            error = str(exc)


        # ------------------------------------------
        # Determine pass / fail
        # ------------------------------------------

        passed = (
            actual_status
            == expected_status
        )


        print(
            f"Expected: {expected_status}"
        )

        print(
            f"Actual:   {actual_status}"
        )

        print(
            "Result:   "
            + (
                "PASS"
                if passed
                else "FAIL"
            )
        )


        if top_score is not None:

            print(
                "Top Score: "
                f"{top_score:.4f}"
            )


        # ------------------------------------------
        # Save result
        # ------------------------------------------

        results.append(
            {
                "test_id":
                    item["test_id"],

                "question":
                    question,

                "category":
                    category,

                "expected_status":
                    expected_status,

                "actual_status":
                    actual_status,

                "pass":
                    passed,

                "top_score":
                    (
                        top_score
                        if top_score is not None
                        else ""
                    ),

                "supporting_chunks":
                    supporting_chunks,

                "answer":
                    answer.replace(
                        "\n",
                        " "
                    ),

                "error":
                    error,
            }
        )


    return results


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

def save_results(results):
    """
    Save individual evaluation results to CSV.
    """

    fieldnames = [
        "test_id",
        "question",
        "category",
        "expected_status",
        "actual_status",
        "pass",
        "top_score",
        "supporting_chunks",
        "answer",
        "error",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(results):
    """
    Calculate overall and category-level accuracy.
    """

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["pass"]
    )

    failed = total - passed

    overall_accuracy = (
        passed / total
        if total
        else 0
    )


    # ----------------------------------------------
    # Category metrics
    # ----------------------------------------------

    categories = sorted(
        set(
            result["category"]
            for result in results
        )
    )

    category_metrics = {}

    for category in categories:

        category_results = [
            result
            for result in results
            if result["category"]
            == category
        ]

        category_total = len(
            category_results
        )

        category_passed = sum(
            1
            for result in category_results
            if result["pass"]
        )

        category_accuracy = (
            category_passed
            / category_total
            if category_total
            else 0
        )

        category_metrics[
            category
        ] = {
            "total":
                category_total,

            "passed":
                category_passed,

            "failed":
                (
                    category_total
                    - category_passed
                ),

            "accuracy":
                category_accuracy,
        }


    # ----------------------------------------------
    # Expected vs actual status counts
    # ----------------------------------------------

    expected_counts = Counter(
        result["expected_status"]
        for result in results
    )

    actual_counts = Counter(
        result["actual_status"]
        for result in results
    )


    return {
        "total":
            total,

        "passed":
            passed,

        "failed":
            failed,

        "overall_accuracy":
            overall_accuracy,

        "category_metrics":
            category_metrics,

        "expected_counts":
            expected_counts,

        "actual_counts":
            actual_counts,
    }


# --------------------------------------------------
# Print evaluation summary
# --------------------------------------------------

def print_summary(metrics):
    """
    Print a readable evaluation report.
    """

    print("\n")
    print("================================")
    print("EVALUATION SUMMARY")
    print("================================")

    print(
        f"\nTotal tests: "
        f"{metrics['total']}"
    )

    print(
        f"Passed: "
        f"{metrics['passed']}"
    )

    print(
        f"Failed: "
        f"{metrics['failed']}"
    )

    print(
        "\nOverall accuracy: "
        f"{metrics['overall_accuracy'] * 100:.2f}%"
    )


    print("\n--------------------------------")
    print("CATEGORY ACCURACY")
    print("--------------------------------")


    for category, values in (
        metrics[
            "category_metrics"
        ].items()
    ):

        print(
            f"\n{category}"
        )

        print(
            f"  Passed: "
            f"{values['passed']}"
            f"/{values['total']}"
        )

        print(
            f"  Accuracy: "
            f"{values['accuracy'] * 100:.2f}%"
        )


    print("\n--------------------------------")
    print("STATUS DISTRIBUTION")
    print("--------------------------------")

    print("\nExpected:")

    for status, count in (
        metrics[
            "expected_counts"
        ].items()
    ):

        print(
            f"  {status}: {count}"
        )


    print("\nActual:")

    for status, count in (
        metrics[
            "actual_counts"
        ].items()
    ):

        print(
            f"  {status}: {count}"
        )


    print("\n--------------------------------")
    print("FAILED TESTS")
    print("--------------------------------")

    if metrics["failed"] == 0:

        print(
            "\nNo failed tests."
        )

    else:

        print(
            "\nSee evaluation_results.csv "
            "for failed cases."
        )


    print("\n================================")
    print(
        "Results saved to:"
    )
    print(
        RESULTS_FILE
    )
    print("================================")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    results = evaluate_agent()

    save_results(
        results
    )

    metrics = calculate_metrics(
        results
    )

    print_summary(
        metrics
    )