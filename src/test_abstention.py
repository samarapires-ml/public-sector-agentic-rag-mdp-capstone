from answer_generator import generate_answer


# --------------------------------------------------
# Abstention / Evidence Sufficiency Test
# --------------------------------------------------
#
# These questions deliberately mention supported
# domains but ask for information that may not be
# supported by the prototype corpus.
#
# Goal:
# Determine whether the evidence gate correctly
# prevents unsupported answers.
# --------------------------------------------------


TEST_QUERIES = [

    {
        "question":
            "How long does a land title transfer take to process?",
        "expected_behavior":
            "ABSTAIN_IF_UNSUPPORTED"
    },

    {
        "question":
            "Can I register my vehicle at 3 AM?",
        "expected_behavior":
            "ABSTAIN_IF_UNSUPPORTED"
    },

    {
        "question":
            "What is the penalty for lying on a vehicle registration application?",
        "expected_behavior":
            "ABSTAIN_IF_UNSUPPORTED"
    },

    {
        "question":
            "Can I transfer land ownership using cryptocurrency?",
        "expected_behavior":
            "ABSTAIN_IF_UNSUPPORTED"
    },
]


print("\n================================")
print("ABSTENTION TEST")
print("================================")


for number, test in enumerate(
    TEST_QUERIES,
    start=1
):

    question = test["question"]

    print("\n================================")

    print(
        f"TEST {number}"
    )

    print("================================")

    print(
        f"\nQuestion: {question}\n"
    )


    result = generate_answer(
        question
    )


    print(
        f"Status: {result['status']}"
    )


    assessment = result.get(
        "evidence_assessment"
    )


    if assessment:

        print(
            f"Evidence sufficient: "
            f"{assessment['sufficient']}"
        )

        print(
            f"Reason: "
            f"{assessment['reason']}"
        )

        print(
            f"Top score: "
            f"{assessment['top_score']}"
        )

        print(
            f"Supporting chunks: "
            f"{assessment['supporting_chunks']}"
        )


    print("\nResponse:\n")

    print(
        result["answer"]
    )


    # ----------------------------------------------
    # Show retrieved evidence for diagnosis
    # ----------------------------------------------

    evidence = result.get(
        "evidence",
        []
    )


    if evidence:

        print("\nTop Retrieved Evidence:")

        for rank, item in enumerate(
            evidence[:3],
            start=1
        ):

            print(
                f"\n{rank}. "
                f"{item['chunk_id']}"
            )

            print(
                f"   Document: "
                f"{item['metadata']['title']}"
            )

            print(
                f"   Score: "
                f"{item['reranker_score']:.4f}"
            )


print("\n================================")
print("ABSTENTION TEST COMPLETE")
print("================================")