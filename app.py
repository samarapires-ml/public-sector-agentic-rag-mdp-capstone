import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alberta Public-Service Research Assistant",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>

    /* Main page */

    .stApp {
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 6rem;
    }


    /* Hide Streamlit UI */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Header */

    .assistant-title {
        text-align: center;
        color: #102a43;
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.02em;
    }

    .assistant-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }


    /* Prototype badge */

    .prototype-badge-container {
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .prototype-badge {
    display: inline-block;
    background-color: #e8f2fb;
    color: #185d91;
    border: 1px solid #c9dfef;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}


    /* Scope text */

    .scope-container {
        text-align: center;
        margin-bottom: 2.5rem;
    }

    .scope-pill {
        display: inline-block;
        background-color: white;
        border: 1px solid #dce5ed;
        color: #42566a;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 600;
    }


    /* Divider */

    .header-divider {
        height: 1px;
        background-color: #e0e7ef;
        margin-bottom: 2rem;
    }


    /* Chat messages */

    div[data-testid="stChatMessage"] {
        background-color: white;
        border: 1px solid #e1e8ef;
        border-radius: 16px;
        padding: 0.35rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.035);
    }


    /* Chat input */

    div[data-testid="stChatInput"] {
        border-radius: 14px;
    }


    /* Expander */

    div[data-testid="stExpander"] {
        background-color: #fafcff;
        border: 1px solid #e1e8ef;
        border-radius: 10px;
    }


    /* Status labels */

    .status-grounded {
        display: inline-block;
        background-color: #eaf8ef;
        color: #187443;
        border: 1px solid #bfe5cd;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    .status-clarify {
        display: inline-block;
        background-color: #fff7e5;
        color: #805800;
        border: 1px solid #ecd79d;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    .status-abstain {
        display: inline-block;
        background-color: #fff4e8;
        color: #8a4b08;
        border: 1px solid #efd0aa;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    .status-scope {
        display: inline-block;
        background-color: #eef3f8;
        color: #52677b;
        border: 1px solid #d7e0e8;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }


    /* Small footer */

    .small-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 2.5rem;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="prototype-badge-container">'
    '<span class="prototype-badge">'
    'MDP CAPSTONE · AGENTIC RAG PROTOTYPE'
    '</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="assistant-title">'
    'Alberta Public-Service Research Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="assistant-subtitle">'
    'Ask questions and receive evidence-grounded answers '
    'from selected official Alberta public-service sources.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="scope-container">'
    '<span class="scope-pill">🚗 Vehicle Registration</span>'
    '<span class="scope-pill">📄 Motor Vehicle Information</span>'
    '<span class="scope-pill">🏠 Land Titles</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="header-divider"></div>',
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can help you research selected Alberta "
                "public-service information.\n\n"
                "For example, you can ask:\n\n"
                "- What do I need to register a vehicle in Alberta?\n"
                "- How can I get a vehicle information report?\n"
                "- What information do I need to transfer land?\n"
                "- What documents do I need for a caveat?"
            ),
            "result": None,
        }
    ]


# ============================================================
# HELPER — DISPLAY TECHNICAL TRACE
# ============================================================

def display_trace(result):

    sources = result.get("sources", [])
    evidence = result.get("evidence", [])
    assessment = result.get("evidence_assessment")

    if not sources and not evidence and not assessment:
        return

    with st.expander("View evidence & sources"):

        # ----------------------------------------------------
        # Evidence assessment
        # ----------------------------------------------------

        if assessment:

            st.markdown("#### Evidence assessment")

            col1, col2, col3 = st.columns(3)

            with col1:

                sufficient = assessment.get(
                    "sufficient",
                    False
                )

                st.metric(
                    "Sufficient",
                    "Yes" if sufficient else "No"
                )

            with col2:

                score = assessment.get(
                    "top_score"
                )

                if score is not None:
                    st.metric(
                        "Top score",
                        f"{score:.2f}"
                    )
                else:
                    st.metric(
                        "Top score",
                        "N/A"
                    )

            with col3:

                st.metric(
                    "Supporting chunks",
                    assessment.get(
                        "supporting_chunks",
                        0
                    )
                )

            reason = assessment.get(
                "reason"
            )

            if reason:

                st.caption(
                    f"Gate decision: {reason}"
                )

            st.divider()


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if sources:

            st.markdown("#### Official sources")

            seen = set()

            for source in sources:

                doc_id = source.get(
                    "doc_id",
                    ""
                )

                title = source.get(
                    "title",
                    "Official source"
                )

                url = source.get(
                    "url"
                )

                key = (
                    doc_id,
                    url
                )

                if key in seen:
                    continue

                seen.add(key)

                st.markdown(
                    f"**{title}**  \n"
                    f"Document ID: `{doc_id}`"
                )

                if url:

                    st.link_button(
                        "Open official source ↗",
                        url
                    )

            st.divider()


        # ----------------------------------------------------
        # Retrieved evidence
        # ----------------------------------------------------

        if evidence:

            st.markdown("#### Retrieved evidence")

            for index, item in enumerate(
                evidence,
                start=1
            ):

                metadata = item.get(
                    "metadata",
                    {}
                )

                title = metadata.get(
                    "title",
                    "Evidence"
                )

                chunk_id = item.get(
                    "chunk_id",
                    "Unknown"
                )

                score = item.get(
                    "reranker_score"
                )

                st.markdown(
                    f"**{index}. {title}**"
                )

                if score is not None:

                    st.caption(
                        f"Chunk: {chunk_id} · "
                        f"Reranker score: {score:.4f}"
                    )

                else:

                    st.caption(
                        f"Chunk: {chunk_id}"
                    )

                evidence_text = item.get(
                    "text"
                )

                if evidence_text:

                    st.write(
                        evidence_text
                    )

                if index != len(evidence):

                    st.divider()


# ============================================================
# DISPLAY EXISTING CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        result = message.get(
            "result"
        )

        if result:

            status = result.get(
                "status"
            )

            if status == "answered":

                st.markdown(
                    '<span class="status-grounded">'
                    '✓ Evidence grounded'
                    '</span>',
                    unsafe_allow_html=True,
                )

            elif status == "clarify":

                st.markdown(
                    '<span class="status-clarify">'
                    '? Clarification required'
                    '</span>',
                    unsafe_allow_html=True,
                )

            elif status == "abstain":

                st.markdown(
                    '<span class="status-abstain">'
                    '⚠ Insufficient evidence'
                    '</span>',
                    unsafe_allow_html=True,
                )

            elif status == "out_of_scope":

                st.markdown(
                    '<span class="status-scope">'
                    'Outside prototype scope'
                    '</span>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            message["content"]
        )

        if result:

            display_trace(
                result
            )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask a question about an Alberta public service..."
)


# ============================================================
# PROCESS NEW QUESTION
# ============================================================

if prompt:

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "result": None,
        }
    )


    # Display user message

    with st.chat_message("user"):

        st.markdown(prompt)


    # Run agent

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching official sources..."
        ):

            try:

                result = run_agent(
                    prompt
                )

            except Exception as error:

                st.error(
                    "The assistant encountered an error "
                    "while processing your question."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(
                        error
                    )

                st.stop()


        status = result.get(
            "status"
        )

        answer = result.get(
            "answer",
            "No response was generated."
        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if status == "answered":

            st.markdown(
                '<span class="status-grounded">'
                '✓ Evidence grounded'
                '</span>',
                unsafe_allow_html=True,
            )

        elif status == "clarify":

            st.markdown(
                '<span class="status-clarify">'
                '? Clarification required'
                '</span>',
                unsafe_allow_html=True,
            )

        elif status == "abstain":

            st.markdown(
                '<span class="status-abstain">'
                '⚠ Insufficient evidence'
                '</span>',
                unsafe_allow_html=True,
            )

        elif status == "out_of_scope":

            st.markdown(
                '<span class="status-scope">'
                'Outside prototype scope'
                '</span>',
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        display_trace(
            result
        )


    # Save assistant message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "result": result,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="small-footer">'
    'Research prototype · Not an official Government of Alberta service'
    '</div>',
    unsafe_allow_html=True,
)