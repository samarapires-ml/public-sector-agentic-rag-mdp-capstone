# --------------------------------------------------
# Query Processor v2.0
# --------------------------------------------------
#
# Purpose:
# Determine whether a user query should:
#
#   1. Proceed to retrieval
#   2. Ask for clarification
#   3. Be identified as outside the POC scope
#
# This prototype focuses on three service domains:
#
#   1. Vehicle registration
#   2. Motor vehicle information
#   3. Land titles
#
# --------------------------------------------------


# --------------------------------------------------
# Domain keywords
# --------------------------------------------------

DOMAIN_KEYWORDS = {
    "vehicle_registration": [
        "vehicle registration",
        "register vehicle",
        "register a vehicle",
        "registration",
        "renew registration",
        "cancel registration",
        "transfer registration",
        "vehicle ownership",
        "licence plate",
        "license plate",
        "register my vehicle",
        "register your vehicle",
        "registering a vehicle",
    ],

    "motor_vehicle_information": [
        "vehicle information",
        "vehicle information report",
        "motor vehicle information",
        "protected motor vehicle information",
        "driver information",
        "driving information",
        "vehicle report",
    ],

    "land_titles": [
        "land title",
        "land titles",
        "property title",
        "transfer land",
        "transfer of land",
        "transfer ownership of land",
        "ownership of land",
        "land ownership",
        "change ownership of land",
        "change land ownership",
        "title ownership",
        "property ownership",
        "caveat",
    ],
}


# --------------------------------------------------
# Supported prototype domains
# --------------------------------------------------

SUPPORTED_DOMAINS = [
    "vehicle registration",
    "motor vehicle information",
    "land titles",
]


# --------------------------------------------------
# Ambiguous expressions
# --------------------------------------------------

AMBIGUOUS_TERMS = [
    "it",
    "this",
    "that",
    "something",
]


# --------------------------------------------------
# Detect domain
# --------------------------------------------------

def detect_domain(query):
    """
    Identify which supported corpus domain is most
    clearly represented in the query.

    Returns
    -------
    str or None
        Detected domain name, or None if no supported
        domain can be identified.
    """

    query_lower = query.lower()

    domain_scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in query_lower:
                score += 1

        domain_scores[domain] = score


    best_domain = max(
        domain_scores,
        key=domain_scores.get
    )


    if domain_scores[best_domain] == 0:
        return None


    return best_domain


# --------------------------------------------------
# Detect vague references
# --------------------------------------------------

def contains_ambiguous_reference(query):
    """
    Detect simple vague references such as
    'it', 'this', 'that', or 'something'.
    """

    words = (
        query.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .split()
    )

    return any(
        term in words
        for term in AMBIGUOUS_TERMS
    )


# --------------------------------------------------
# Detect clearly ambiguous query
# --------------------------------------------------

def is_clearly_ambiguous(query):
    """
    Detect short queries that rely on vague references
    and therefore do not contain enough context to
    identify the intended service.

    Example:
        "How do I transfer it?"
    """

    words = (
        query.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .split()
    )

    contains_vague_term = any(
        term in words
        for term in AMBIGUOUS_TERMS
    )

    return (
        contains_vague_term
        and len(words) <= 7
    )


# --------------------------------------------------
# Main query-processing function
# --------------------------------------------------

def process_query(query):
    """
    Decide whether the query should proceed to
    retrieval, request clarification, or be classified
    as outside the supported prototype scope.

    Returns
    -------
    dict
        Structured query-processing result.
    """

    cleaned_query = query.strip()


    # ----------------------------------------------
    # Empty query
    # ----------------------------------------------

    if not cleaned_query:

        return {
            "status": "clarify",
            "domain": None,
            "query": cleaned_query,
            "clarification_question": (
                "What Alberta public service "
                "would you like help with?"
            )
        }


    # ----------------------------------------------
    # Detect supported domain
    # ----------------------------------------------

    domain = detect_domain(
        cleaned_query
    )

    vague_reference = (
        contains_ambiguous_reference(
            cleaned_query
        )
    )


    # ----------------------------------------------
    # No supported domain identified
    # ----------------------------------------------

    if domain is None:

        # Short vague question:
        # ask the user to clarify rather than
        # assuming it is outside the scope.

        if is_clearly_ambiguous(
            cleaned_query
        ):

            return {
                "status": "clarify",
                "domain": None,
                "query": cleaned_query,
                "clarification_question": (
                    "Could you clarify what service "
                    "you are asking about? For example, "
                    "is your question about vehicle "
                    "registration, motor vehicle "
                    "information, or land titles?"
                )
            }


        # Clear question, but it does not belong to
        # one of the prototype's supported domains.

        return {
            "status": "out_of_scope",
            "domain": None,
            "query": cleaned_query,
            "clarification_question": None
        }


    # ----------------------------------------------
    # Supported domain exists, but query still uses
    # a vague object/reference.
    # ----------------------------------------------

    if (
        vague_reference
        and len(cleaned_query.split()) <= 7
    ):

        return {
            "status": "clarify",
            "domain": domain,
            "query": cleaned_query,
            "clarification_question": (
                "Could you clarify exactly what "
                "you want to transfer, change, "
                "renew, register, or request?"
            )
        }


    # ----------------------------------------------
    # Query is sufficiently clear and supported
    # ----------------------------------------------

    return {
        "status": "proceed",
        "domain": domain,
        "query": cleaned_query,
        "clarification_question": None
    }


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    TEST_QUERIES = [

        # Ambiguous queries

        "How do I transfer it?",

        "How do I renew it?",


        # Supported queries

        "How do I transfer ownership of land in Alberta?",

        "What do I need to register a vehicle in Alberta?",

        "How can I get a vehicle information report?",

        "How do I cancel my vehicle registration?",

        "What documents do I need for a caveat?",


        # Out-of-scope queries

        "How do I apply for an Alberta health card?",

        "How do I apply for a Canadian passport?",

        "What documents do I need to get married in Alberta?",

        "How do I start a business in Alberta?",
    ]


    print("\n================================")
    print("QUERY PROCESSOR v2.0 TEST")
    print("================================")


    for number, query in enumerate(
        TEST_QUERIES,
        start=1
    ):

        result = process_query(
            query
        )

        print("\n--------------------------------")

        print(
            f"Test {number}"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Status: {result['status']}"
        )

        print(
            f"Domain: {result['domain']}"
        )


        if result[
            "clarification_question"
        ]:

            print(
                "Clarification: "
                f"{result['clarification_question']}"
            )


    print("\n================================")
    print("QUERY PROCESSOR TEST COMPLETE")
    print("================================")