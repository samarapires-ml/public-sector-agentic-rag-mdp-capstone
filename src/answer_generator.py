import os

from dotenv import load_dotenv
from google import genai

from query_router import route_query
from evidence_gate import assess_evidence


# --------------------------------------------------
# Answer Generator v2.1
# --------------------------------------------------
#
# Purpose:
# Generate grounded answers using retrieved evidence
# while supporting:
#
#   1. Clarification
#   2. Out-of-scope handling
#   3. Evidence sufficiency / abstention
#   4. Grounded answer generation
#   5. Chunk-level citation traceability
#   6. Generation-level abstention
#
# --------------------------------------------------


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "gemini-3.5-flash-lite"

ABSTENTION_MESSAGE = (
    "The available sources do not provide enough "
    "information to answer this question reliably."
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# Build evidence context
# --------------------------------------------------

def build_evidence_context(evidence):
    """
    Convert retrieved evidence chunks into structured
    context with stable chunk-level citation labels.
    """

    context_blocks = []

    for result in evidence:

        metadata = result["metadata"]
        chunk_id = result["chunk_id"]

        block = (
            f"[{chunk_id}]\n"
            f"Document: {metadata['title']}\n"
            f"Document ID: {metadata['doc_id']}\n"
            f"Source URL: {metadata['source_url']}\n"
            f"Evidence:\n{result['text']}"
        )

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


# --------------------------------------------------
# Build traceable source list
# --------------------------------------------------

def build_source_list(evidence):
    """
    Build a source record for every evidence chunk
    returned by the retrieval pipeline.
    """

    sources = []

    for result in evidence:

        metadata = result["metadata"]

        sources.append(
            {
                "chunk_id": result["chunk_id"],
                "doc_id": metadata["doc_id"],
                "title": metadata["title"],
                "url": metadata["source_url"],
                "reranker_score":
                    result["reranker_score"]
            }
        )

    return sources


# --------------------------------------------------
# Generate grounded answer
# --------------------------------------------------

def generate_answer(query):
    """
    Route a query, assess evidence sufficiency, and
    generate a grounded answer only when appropriate.

    Possible statuses:
        clarify
        out_of_scope
        abstain
        answered
    """

    # ----------------------------------------------
    # 1. Route query
    # ----------------------------------------------

    routed = route_query(
        query
    )


    # ----------------------------------------------
    # 2. Clarification path
    # ----------------------------------------------

    if routed["status"] == "clarify":

        return {
            "status": "clarify",
            "answer":
                routed["clarification_question"],
            "sources": [],
            "evidence": [],
            "evidence_assessment": None
        }


    # ----------------------------------------------
    # 3. Out-of-scope path
    # ----------------------------------------------

    if routed["status"] == "out_of_scope":

        return {
            "status": "out_of_scope",
            "answer": (
                "This question is outside the scope "
                "of the current prototype. I can "
                "currently provide information about "
                "vehicle registration, motor vehicle "
                "information, and land titles."
            ),
            "sources": [],
            "evidence": [],
            "evidence_assessment": None
        }


    # ----------------------------------------------
    # 4. Retrieve routed evidence
    # ----------------------------------------------

    evidence = routed["evidence"]


    # ----------------------------------------------
    # 5. Assess evidence sufficiency
    # ----------------------------------------------

    assessment = assess_evidence(
        evidence
    )


    # ----------------------------------------------
    # 6. Evidence-gate abstention / escalation
    # ----------------------------------------------

    if not assessment["sufficient"]:

        return {
            "status": "abstain",
            "answer": (
                "I could not find sufficiently "
                "relevant evidence in the available "
                "official sources to answer this "
                "question reliably. Please consult "
                "the appropriate Government of "
                "Alberta service or a registry agent "
                "for further assistance."
            ),
            "sources": [],
            "evidence": evidence,
            "evidence_assessment": assessment
        }


    # ----------------------------------------------
    # 7. Build evidence context
    # ----------------------------------------------

    evidence_context = build_evidence_context(
        evidence
    )


    # ----------------------------------------------
    # 8. Grounded generation prompt
    # ----------------------------------------------

    prompt = f"""
You are a public-sector information assistant.

Answer the user's question using ONLY the retrieved
evidence provided below.

GROUNDING RULES:

1. Use only information explicitly supported by the
   retrieved evidence.

2. Do not use outside knowledge.

3. Do not invent requirements, forms, fees, deadlines,
   procedures, exceptions, or legal conclusions.

4. Cite factual claims using the exact chunk ID shown
   with the evidence.

5. Citation format must be:
   [LT-05-CH-001]
   [VR-02-CH-002]
   etc.

6. Never create a citation ID that does not appear in
   the retrieved evidence.

7. When multiple chunks support a claim, cite each
   relevant chunk.

8. Preserve important conditions and exceptions from
   the evidence.

9. If the evidence does not actually contain enough
   information to answer the user's specific question,
   respond with exactly this sentence:

   "{ABSTENTION_MESSAGE}"

10. Do not add anything before or after that sentence
    when abstaining.

11. Do not attempt to fill missing information using
    your own knowledge.

12. Keep supported answers concise, clear, and
    practical.


USER QUESTION:

{query}


RETRIEVED EVIDENCE:

{evidence_context}


ANSWER:
"""


    # ----------------------------------------------
    # 9. Gemini generation
    # ----------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text.strip()


    # ----------------------------------------------
    # 10. Detect generation-level abstention
    # ----------------------------------------------

    normalized_answer = (
        answer
        .strip()
        .strip('"')
        .strip()
    )

    if (
        ABSTENTION_MESSAGE.lower()
        in normalized_answer.lower()
    ):

        return {
            "status": "abstain",
            "answer": ABSTENTION_MESSAGE,
            "sources": build_source_list(
                evidence
            ),
            "evidence": evidence,
            "evidence_assessment": assessment
        }


    # ----------------------------------------------
    # 11. Build evidence trace
    # ----------------------------------------------

    sources = build_source_list(
        evidence
    )


    # ----------------------------------------------
    # 12. Return grounded answer
    # ----------------------------------------------

    return {
        "status": "answered",
        "answer": answer,
        "sources": sources,
        "evidence": evidence,
        "evidence_assessment": assessment
    }


# --------------------------------------------------
# Local end-to-end test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_QUERIES = [

        # ------------------------------------------
        # Supported
        # ------------------------------------------

        "What information do I need to transfer land?",

        "What do I need to register a vehicle in Alberta?",


        # ------------------------------------------
        # Ambiguous
        # ------------------------------------------

        "How do I transfer it?",


        # ------------------------------------------
        # Out of scope
        # ------------------------------------------

        "How do I apply for an Alberta health card?",

        "What documents do I need to get married in Alberta?",


        # ------------------------------------------
        # In scope but unsupported by evidence
        # ------------------------------------------

        "Can I register my vehicle at 3 AM?",
    ]


    print("\n================================")
    print("ANSWER GENERATOR v2.1 TEST")
    print("================================")


    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        print("\n================================")
        print(f"TEST {number}")
        print("================================")

        print(
            f"\nQuestion: {query}\n"
        )


        result = generate_answer(
            query
        )


        print(
            f"Status: {result['status']}"
        )

        print("\nAnswer:\n")

        print(
            result["answer"]
        )


        # ------------------------------------------
        # Evidence gate result
        # ------------------------------------------

        assessment = result[
            "evidence_assessment"
        ]

        if assessment:

            print("\nEvidence Assessment:")

            print(
                f"Sufficient: "
                f"{assessment['sufficient']}"
            )

            print(
                f"Reason: "
                f"{assessment['reason']}"
            )

            print(
                f"Top Score: "
                f"{assessment['top_score']}"
            )

            print(
                f"Supporting Chunks: "
                f"{assessment['supporting_chunks']}"
            )


        # ------------------------------------------
        # Evidence trace
        # ------------------------------------------

        if result["sources"]:

            print("\n--------------------------------")
            print("EVIDENCE TRACE")
            print("--------------------------------")

            for source in result["sources"]:

                print()

                print(
                    f"Citation: "
                    f"[{source['chunk_id']}]"
                )

                print(
                    f"Document: "
                    f"{source['title']}"
                )

                print(
                    f"Doc ID: "
                    f"{source['doc_id']}"
                )

                print(
                    f"Reranker Score: "
                    f"{source['reranker_score']:.4f}"
                )

                print(
                    f"Source: "
                    f"{source['url']}"
                )


    print("\n================================")
    print("ANSWER GENERATOR TEST COMPLETE")
    print("================================")