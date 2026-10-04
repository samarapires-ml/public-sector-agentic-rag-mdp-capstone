# --------------------------------------------------
# Public-Sector Agent v1.0
# --------------------------------------------------
#
# Purpose:
# Provide a single orchestration entry point for the
# complete agentic RAG prototype.
#
# Pipeline:
#
# User Query
#     -> Query Processing / Scope Routing
#     -> Clarification OR Out-of-Scope
#     -> BGE Retrieval
#     -> CrossEncoder Reranking
#     -> Evidence Sufficiency Gate
#     -> Grounded Gemini Generation
#     -> Citation Traceability
#     -> Answer OR Abstention / Escalation
#
# --------------------------------------------------


from answer_generator import generate_answer


# --------------------------------------------------
# Agent entry point
# --------------------------------------------------

def run_agent(query):
    """
    Run the complete public-sector agentic RAG
    pipeline for a user query.

    Parameters
    ----------
    query : str
        User question.

    Returns
    -------
    dict
        Structured agent response containing:
        - status
        - query
        - answer
        - sources
        - evidence
        - evidence_assessment
    """

    # ----------------------------------------------
    # 1. Validate input
    # ----------------------------------------------

    if query is None:

        return {
            "status": "clarify",
            "query": "",
            "answer": (
                "Please enter a question about "
                "vehicle registration, motor vehicle "
                "information, or land titles."
            ),
            "sources": [],
            "evidence": [],
            "evidence_assessment": None
        }


    cleaned_query = query.strip()


    if not cleaned_query:

        return {
            "status": "clarify",
            "query": cleaned_query,
            "answer": (
                "Please enter a question about "
                "vehicle registration, motor vehicle "
                "information, or land titles."
            ),
            "sources": [],
            "evidence": [],
            "evidence_assessment": None
        }


    # ----------------------------------------------
    # 2. Run complete RAG pipeline
    # ----------------------------------------------

    result = generate_answer(
        cleaned_query
    )


    # ----------------------------------------------
    # 3. Return standardized agent response
    # ----------------------------------------------

    return {
        "status": result["status"],
        "query": cleaned_query,
        "answer": result["answer"],
        "sources": result.get(
            "sources",
            []
        ),
        "evidence": result.get(
            "evidence",
            []
        ),
        "evidence_assessment": result.get(
            "evidence_assessment"
        )
    }


# --------------------------------------------------
# Local end-to-end orchestration test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_QUERIES = [

        # Supported
        "What do I need to register a vehicle in Alberta?",

        # Supported
        "What information do I need to transfer land?",

        # Ambiguous
        "How do I transfer it?",

        # Out of scope
        "How do I apply for an Alberta health card?",

        # In scope but unsupported
        "Can I register my vehicle at 3 AM?",
    ]


    print("\n================================")
    print("PUBLIC-SECTOR AGENT v1.0")
    print("END-TO-END ORCHESTRATION TEST")
    print("================================")


    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n================================")
        print(f"TEST {number}")
        print("================================")

        print(
            f"\nUser Query:\n{query}\n"
        )


        result = run_agent(
            query
        )


        print(
            f"Status: {result['status']}"
        )

        print(
            f"\nAgent Response:\n"
            f"{result['answer']}"
        )


        # ------------------------------------------
        # Source summary
        # ------------------------------------------

        if result["sources"]:

            print("\nSources:")

            seen_sources = set()

            for source in result["sources"]:

                source_key = (
                    source["doc_id"],
                    source["url"]
                )

                if source_key in seen_sources:
                    continue

                seen_sources.add(
                    source_key
                )

                print(
                    f"- {source['title']} "
                    f"({source['doc_id']})"
                )

                print(
                    f"  {source['url']}"
                )


        # ------------------------------------------
        # Evidence assessment
        # ------------------------------------------

        assessment = result[
            "evidence_assessment"
        ]

        if assessment:

            print("\nEvidence Assessment:")

            print(
                f"- Sufficient: "
                f"{assessment['sufficient']}"
            )

            print(
                f"- Reason: "
                f"{assessment['reason']}"
            )

            print(
                f"- Top Score: "
                f"{assessment['top_score']}"
            )

            print(
                f"- Supporting Chunks: "
                f"{assessment['supporting_chunks']}"
            )


    print("\n================================")
    print("AGENT ORCHESTRATION TEST COMPLETE")
    print("================================")