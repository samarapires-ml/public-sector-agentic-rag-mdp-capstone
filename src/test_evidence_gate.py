from retrieval_pipeline import retrieve_evidence
from evidence_gate import assess_evidence


# --------------------------------------------------
# Real-corpus Evidence Gate Test
# --------------------------------------------------

TEST_QUERIES = [

    # ----------------------------------------------
    # Questions that SHOULD be supported
    # ----------------------------------------------

    {
        "question":
            "What do I need to register a vehicle in Alberta?",
        "expected": "SUPPORTED"
    },

    {
        "question":
            "What information do I need to transfer land?",
        "expected": "SUPPORTED"
    },

    {
        "question":
            "How can I get a vehicle information report?",
        "expected": "SUPPORTED"
    },


    # ----------------------------------------------
    # Questions OUTSIDE our corpus
    # ----------------------------------------------

    {
        "question":
            "How do I apply for an Alberta health card?",
        "expected": "UNSUPPORTED"
    },

    {
        "question":
            "How do I apply for a Canadian passport?",
        "expected": "UNSUPPORTED"
    },

    {
        "question":
            "What documents do I need to get married in Alberta?",
        "expected": "UNSUPPORTED"
    },

    {
        "question":
            "How do I start a business in Alberta?",
        "expected": "UNSUPPORTED"
    },
]


print("\n================================")
print("REAL-CORPUS EVIDENCE GATE TEST")
print("================================")


for number, test in enumerate(
    TEST_QUERIES,
    start=1
):

    question = test["question"]

    evidence = retrieve_evidence(
        question
    )

    assessment = assess_evidence(
        evidence
    )

    actual = (
        "SUPPORTED"
        if assessment["sufficient"]
        else "UNSUPPORTED"
    )

    passed = (
        actual == test["expected"]
    )

    print("\n--------------------------------")

    print(
        f"Test {number}: {question}"
    )

    print(
        f"Expected: {test['expected']}"
    )

    print(
        f"Actual: {actual}"
    )

    print(
        f"Result: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    print(
        f"Top reranker score: "
        f"{assessment['top_score']}"
    )

    print(
        f"Supporting chunks: "
        f"{assessment['supporting_chunks']}"
    )

    if evidence:

        print(
            f"Top retrieved chunk: "
            f"{evidence[0]['chunk_id']}"
        )

        print(
            f"Top retrieved document: "
            f"{evidence[0]['metadata']['title']}"
        )


print("\n================================")
print("REAL-CORPUS TEST COMPLETE")
print("================================")