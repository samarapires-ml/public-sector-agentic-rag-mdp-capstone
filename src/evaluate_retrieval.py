from retrieve import retrieve_chunks


# --------------------------------------------------
# 1. Evaluation dataset
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
# 2. Evaluation counters
# --------------------------------------------------

top_1_hits = 0
top_3_hits = 0
top_5_hits = 0


print("\n================================")
print("RETRIEVAL EVALUATION v1.0")
print("================================")

print(f"\nQuestions: {len(TEST_CASES)}\n")


# --------------------------------------------------
# 3. Run evaluation
# --------------------------------------------------

for number, test_case in enumerate(
    TEST_CASES,
    start=1
):

    question = test_case["question"]

    expected_doc_id = test_case[
        "expected_doc_id"
    ]

    results = retrieve_chunks(
        question,
        top_k=5
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
        "Retrieved:",
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
# 4. Calculate metrics
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
# 5. Display summary
# --------------------------------------------------

print("\n================================")
print("EVALUATION SUMMARY")
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

print("\nRetrieval evaluation complete.")