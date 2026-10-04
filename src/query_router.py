from query_processor import process_query
from retrieval_pipeline import retrieve_evidence


# --------------------------------------------------
# Query Router v2.0
# --------------------------------------------------
#
# Purpose:
# Route a user query through the query processor.
#
# Clear + supported queries:
#     -> proceed to retrieval
#
# Ambiguous queries:
#     -> return clarification question
#
# Clear but unsupported queries:
#     -> stop before retrieval
#
# --------------------------------------------------


def route_query(query):
    """
    Process a user query and decide whether to
    retrieve evidence, request clarification, or
    stop because the query is outside the supported
    prototype scope.

    Parameters
    ----------
    query : str
        User question.

    Returns
    -------
    dict
        Structured routing result.
    """

    # ----------------------------------------------
    # 1. Process query
    # ----------------------------------------------

    query_result = process_query(
        query
    )

    status = query_result["status"]


    # ----------------------------------------------
    # 2. Clarification required
    # ----------------------------------------------

    if status == "clarify":

        return {
            "status": "clarify",
            "query": query,
            "domain": query_result["domain"],
            "clarification_question":
                query_result[
                    "clarification_question"
                ],
            "evidence": []
        }


    # ----------------------------------------------
    # 3. Query is outside supported scope
    # ----------------------------------------------

    if status == "out_of_scope":

        return {
            "status": "out_of_scope",
            "query": query,
            "domain": None,
            "clarification_question": None,
            "evidence": []
        }


    # ----------------------------------------------
    # 4. Query is supported -> retrieve evidence
    # ----------------------------------------------

    evidence = retrieve_evidence(
        query
    )

    return {
        "status": "proceed",
        "query": query,
        "domain": query_result["domain"],
        "clarification_question": None,
        "evidence": evidence
    }


# --------------------------------------------------
# Local integration test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_QUERIES = [

        # ------------------------------------------
        # Ambiguous
        # ------------------------------------------

        "How do I transfer it?",

        "How do I renew it?",


        # ------------------------------------------
        # Supported
        # ------------------------------------------

        "What do I need to register a vehicle in Alberta?",

        "How do I transfer ownership of land in Alberta?",

        "How can I get a vehicle information report?",

        "How do I cancel my vehicle registration?",

        "What documents do I need for a caveat?",


        # ------------------------------------------
        # Out of scope
        # ------------------------------------------

        "How do I apply for an Alberta health card?",

        "How do I apply for a Canadian passport?",

        "What documents do I need to get married in Alberta?",

        "How do I start a business in Alberta?",
    ]


    print("\n================================")
    print("QUERY ROUTER v2.0 INTEGRATION TEST")
    print("================================")


    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n--------------------------------")

        print(
            f"Test {number}"
        )

        print(
            f"Query: {query}"
        )


        result = route_query(
            query
        )


        print(
            f"Status: {result['status']}"
        )

        print(
            f"Domain: {result['domain']}"
        )


        # ------------------------------------------
        # Clarification path
        # ------------------------------------------

        if result["status"] == "clarify":

            print(
                "Clarification: "
                f"{result['clarification_question']}"
            )

            print(
                "Retrieval performed: NO"
            )


        # ------------------------------------------
        # Out-of-scope path
        # ------------------------------------------

        elif result["status"] == "out_of_scope":

            print(
                "Query is outside supported "
                "prototype scope."
            )

            print(
                "Retrieval performed: NO"
            )


        # ------------------------------------------
        # Retrieval path
        # ------------------------------------------

        elif result["status"] == "proceed":

            print(
                "Retrieval performed: YES"
            )

            print(
                f"Evidence chunks returned: "
                f"{len(result['evidence'])}"
            )


            if result["evidence"]:

                top_result = (
                    result["evidence"][0]
                )

                metadata = (
                    top_result["metadata"]
                )


                print(
                    f"Top Chunk: "
                    f"{top_result['chunk_id']}"
                )

                print(
                    f"Top Document: "
                    f"{metadata['title']}"
                )

                print(
                    f"Top Doc ID: "
                    f"{metadata['doc_id']}"
                )

                print(
                    "Top Reranker Score: "
                    f"{top_result['reranker_score']:.4f}"
                )


    print("\n================================")
    print("QUERY ROUTER TEST COMPLETE")
    print("================================")