from sentence_transformers import CrossEncoder

from retrieve_bge import retrieve_chunks


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieve more candidates first, then let the
# CrossEncoder choose the best final results.
INITIAL_RETRIEVAL_K = 10
FINAL_TOP_K = 5


# --------------------------------------------------
# 2. Load reranker
# --------------------------------------------------

print(
    f"Loading reranker: {RERANKER_MODEL}"
)

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded successfully.")


# --------------------------------------------------
# 3. Evaluation dataset
# --------------------------------------------------

TEST_CASES = [
    {
        "question": "What do I need to register a vehicle in Alberta?",
        "expected_doc_id": "VR-02"
    },
    {
        "question": "How do I transfer my vehicle registration to another vehicle?",
        "expected_doc_id": "VR-03"
    },
    {
        "question": "How can I renew my vehicle registration?",
        "expected_doc_id": "VR-04"
    },
    {
        "question": "How do I cancel my vehicle registration?",
        "expected_doc_id": "VR-05"
    },
    {
        "question": "How can I get a vehicle information report?",
        "expected_doc_id": "MVI-02"
    },
    {
        "question": "How can protected motor vehicle information be requested?",
        "expected_doc_id": "MVI-03"
    },
    {
        "question": "How is personal driving and motor vehicle information protected?",
        "expected_doc_id": "MVI-04"
    },
    {
        "question": "How do I change ownership of a land title?",
        "expected_doc_id": "LT-03"
    },
    {
        "question": "How do I register a land title document?",
        "expected_doc_id": "LT-04"
    },
    {
        "question": "What information do I need to transfer land?",
        "expected_doc_id": "LT-05"
    },
]


# --------------------------------------------------
# 4. Reranking function
# --------------------------------------------------

def retrieve_and_rerank(
    question,
    initial_k=INITIAL_RETRIEVAL_K,
    final_k=FINAL_TOP_K
):

    # First-stage semantic retrieval using BGE
    candidates = retrieve_chunks(
        question,
        top_k=initial_k
    )

    if not candidates:
        return []

    # Create query-document pairs for CrossEncoder
    pairs = [
        (
            question,
            candidate["text"]
        )
        for candidate in candidates
    ]

    # Score every candidate against the question
    reranker_scores = reranker.predict(
        pairs
    )

    # Attach reranker score to each result
    reranked_results = []

    for candidate, score in zip(
        candidates,
        reranker_scores
    ):

        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        reranked_results.append(
            result
        )

    # Higher CrossEncoder score = more relevant
    reranked_results.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True
    )

    return reranked_results[:final_k]


# --------------------------------------------------
# 5. Evaluation counters
# --------------------------------------------------

top_1_hits = 0
top_3_hits = 0
top_5_hits = 0


print("\n================================")
print("BGE + RERANKER EVALUATION")
print("================================")

print(
    f"\nQuestions: {len(TEST_CASES)}"
)

print(
    f"Initial candidates: "
    f"{INITIAL_RETRIEVAL_K}"
)

print(
    f"Final results: {FINAL_TOP_K}\n"
)


# --------------------------------------------------
# 6. Run evaluation
# --------------------------------------------------

for number, test_case in enumerate(
    TEST_CASES,
    start=1
):

    question = test_case["question"]

    expected_doc_id = test_case[
        "expected_doc_id"
    ]

    results = retrieve_and_rerank(
        question
    )

    retrieved_doc_ids = [
        result["metadata"]["doc_id"]
        for result in results
    ]

    top_1 = (
        expected_doc_id
        in retrieved_doc_ids[:1]
    )

    top_3 = (
        expected_doc_id
        in retrieved_doc_ids[:3]
    )

    top_5 = (
        expected_doc_id
        in retrieved_doc_ids[:5]
    )

    if top_1:
        top_1_hits += 1

    if top_3:
        top_3_hits += 1

    if top_5:
        top_5_hits += 1


    # ----------------------------------------------
    # Display individual result
    # ----------------------------------------------

    print("--------------------------------")

    print(
        f"Test {number}: {question}"
    )

    print(
        f"Expected: {expected_doc_id}"
    )

    print(
        "Reranked:",
        " -> ".join(retrieved_doc_ids)
    )

    print(
        f"Top-1: {'PASS' if top_1 else 'FAIL'}"
    )

    print(
        f"Top-3: {'PASS' if top_3 else 'FAIL'}"
    )

    print(
        f"Top-5: {'PASS' if top_5 else 'FAIL'}"
    )


# --------------------------------------------------
# 7. Calculate metrics
# --------------------------------------------------

total = len(TEST_CASES)

top_1_accuracy = (
    top_1_hits / total
) * 100

top_3_accuracy = (
    top_3_hits / total
) * 100

top_5_accuracy = (
    top_5_hits / total
) * 100


# --------------------------------------------------
# 8. Display summary
# --------------------------------------------------

print("\n================================")
print("RERANKING EVALUATION SUMMARY")
print("================================")

print(
    f"Total questions: {total}"
)

print(
    f"Top-1 hits: {top_1_hits}/{total}"
)

print(
    f"Top-3 hits: {top_3_hits}/{total}"
)

print(
    f"Top-5 hits: {top_5_hits}/{total}"
)

print()

print(
    f"Top-1 Accuracy: "
    f"{top_1_accuracy:.1f}%"
)

print(
    f"Top-3 Accuracy: "
    f"{top_3_accuracy:.1f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{top_5_accuracy:.1f}%"
)

print(
    "\nBGE + CrossEncoder "
    "reranking evaluation complete."
)