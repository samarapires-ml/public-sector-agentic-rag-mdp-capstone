from sentence_transformers import CrossEncoder

from retrieve_bge import retrieve_chunks


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_RETRIEVAL_K = 10
FINAL_TOP_K = 5


# --------------------------------------------------
# Load reranker once
# --------------------------------------------------

print(
    f"Loading reranker: {RERANKER_MODEL}"
)

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded successfully.")


# --------------------------------------------------
# Improved retrieval pipeline
# --------------------------------------------------

def retrieve_evidence(
    query,
    initial_k=INITIAL_RETRIEVAL_K,
    final_k=FINAL_TOP_K
):
    """
    Retrieve candidate chunks using BGE semantic
    retrieval and rerank them using a CrossEncoder.

    Parameters
    ----------
    query : str
        User question.

    initial_k : int
        Number of chunks retrieved from the vector
        store before reranking.

    final_k : int
        Number of reranked evidence chunks returned.

    Returns
    -------
    list
        Ranked evidence chunks with reranker scores.
    """

    # ----------------------------------------------
    # 1. First-stage BGE retrieval
    # ----------------------------------------------

    candidates = retrieve_chunks(
        query,
        top_k=initial_k
    )

    if not candidates:
        return []


    # ----------------------------------------------
    # 2. Create query-document pairs
    # ----------------------------------------------

    pairs = [
        (
            query,
            candidate["text"]
        )
        for candidate in candidates
    ]


    # ----------------------------------------------
    # 3. CrossEncoder relevance scoring
    # ----------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # ----------------------------------------------
    # 4. Attach reranker scores
    # ----------------------------------------------

    reranked_results = []

    for candidate, score in zip(
        candidates,
        scores
    ):
        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        reranked_results.append(
            result
        )


    # ----------------------------------------------
    # 5. Sort by reranker relevance
    # ----------------------------------------------

    reranked_results.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True
    )


    # ----------------------------------------------
    # 6. Return final evidence
    # ----------------------------------------------

    return reranked_results[:final_k]


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    test_query = (
        "What information do I need "
        "to transfer land?"
    )

    print("\n================================")
    print("IMPROVED RETRIEVAL PIPELINE TEST")
    print("================================")

    print(
        f"\nQuery: {test_query}\n"
    )

    results = retrieve_evidence(
        test_query
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        metadata = result["metadata"]

        print("--------------------------------")
        print(f"Rank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(
            f"Document: {metadata['title']}"
        )
        print(
            f"Doc ID: {metadata['doc_id']}"
        )
        print(
            f"Source: {metadata['source_url']}"
        )
        print(
            "Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )
        print("--------------------------------")

        print(result["text"])
        print()

    print("================================")
    print("IMPROVED RETRIEVAL TEST COMPLETE")
    print("================================")