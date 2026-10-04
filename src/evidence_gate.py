# --------------------------------------------------
# Evidence Sufficiency Gate v1.0
# --------------------------------------------------
#
# Purpose:
# Decide whether retrieved evidence is sufficiently
# relevant to support grounded answer generation.
#
# This prevents the system from answering merely
# because the vector store returned some chunks.
# --------------------------------------------------


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Conservative prototype threshold.
# We will test and tune this empirically rather than
# treating it as a universal confidence probability.

MIN_RERANKER_SCORE = 0.0

MIN_SUPPORTING_CHUNKS = 1


# --------------------------------------------------
# Evidence assessment
# --------------------------------------------------

def assess_evidence(evidence):
    """
    Assess whether retrieved evidence is sufficient
    to proceed to grounded generation.

    Parameters
    ----------
    evidence : list
        Reranked evidence returned by the retrieval
        pipeline.

    Returns
    -------
    dict
        Structured evidence assessment.
    """

    # ----------------------------------------------
    # No evidence
    # ----------------------------------------------

    if not evidence:

        return {
            "sufficient": False,
            "reason": "no_evidence",
            "top_score": None,
            "supporting_chunks": 0
        }


    # ----------------------------------------------
    # Extract reranker scores
    # ----------------------------------------------

    scores = [
        result["reranker_score"]
        for result in evidence
    ]

    top_score = max(scores)

    supporting_chunks = sum(
        score >= MIN_RERANKER_SCORE
        for score in scores
    )


    # ----------------------------------------------
    # Weak evidence
    # ----------------------------------------------

    if top_score < MIN_RERANKER_SCORE:

        return {
            "sufficient": False,
            "reason": "low_relevance",
            "top_score": top_score,
            "supporting_chunks":
                supporting_chunks
        }


    # ----------------------------------------------
    # Insufficient supporting evidence
    # ----------------------------------------------

    if (
        supporting_chunks
        < MIN_SUPPORTING_CHUNKS
    ):

        return {
            "sufficient": False,
            "reason":
                "insufficient_support",
            "top_score": top_score,
            "supporting_chunks":
                supporting_chunks
        }


    # ----------------------------------------------
    # Evidence accepted
    # ----------------------------------------------

    return {
        "sufficient": True,
        "reason": "sufficient_evidence",
        "top_score": top_score,
        "supporting_chunks":
            supporting_chunks
    }


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_CASES = [

        {
            "name": "Strong evidence",

            "evidence": [
                {
                    "reranker_score": 5.8
                },
                {
                    "reranker_score": 3.3
                }
            ]
        },

        {
            "name": "Weak evidence",

            "evidence": [
                {
                    "reranker_score": -4.2
                },
                {
                    "reranker_score": -6.1
                }
            ]
        },

        {
            "name": "No evidence",

            "evidence": []
        }
    ]


    print("\n================================")
    print("EVIDENCE SUFFICIENCY GATE TEST")
    print("================================")


    for number, test in enumerate(
        TEST_CASES,
        start=1
    ):

        result = assess_evidence(
            test["evidence"]
        )

        print("\n--------------------------------")

        print(
            f"Test {number}: {test['name']}"
        )

        print(
            f"Sufficient: "
            f"{result['sufficient']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

        print(
            f"Top score: "
            f"{result['top_score']}"
        )

        print(
            f"Supporting chunks: "
            f"{result['supporting_chunks']}"
        )


    print("\n================================")
    print("EVIDENCE GATE TEST COMPLETE")
    print("================================")