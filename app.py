import hashlib

import streamlit as st

from rag import (
    answer_question,
    create_uploaded_vectorstore
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NyayaSetu AI",
    page_icon="⚖️",
    layout="centered"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #777777;
        margin-bottom: 1.5rem;
    }

    .feature-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #17191f;
        border: 1px solid #2b2e36;
        margin-bottom: 20px;
    }

    .disclaimer {
        padding: 14px 18px;
        border-radius: 10px;
        background-color: #f4f4f4;
        color: #555555;
        font-size: 0.85rem;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ NyayaSetu AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Indian Legal Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None


if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚖️ NyayaSetu AI")

    mode = st.radio(
        "Choose assistance mode",
        [
            "Legal Knowledge Base",
            "Analyze My Document"
        ]
    )


    st.divider()


    if mode == "Legal Knowledge Base":

        st.subheader("📚 Legal Knowledge Base")

        st.write(
            "Ask questions using the curated Indian legal "
            "documents available in the knowledge base."
        )

        st.markdown(
            """
            - 🇮🇳 Constitution of India
            - 📄 RTI Act, 2005
            - 🛒 Consumer Protection Act, 2019
            - 🔐 Digital Personal Data Protection Act, 2023
            """
        )


    else:

        st.subheader("📄 Document Analysis")

        st.write(
            "Upload a legal PDF to understand, summarize "
            "and analyze its contents."
        )


    st.divider()


    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DOCUMENT UPLOAD MODE
# ============================================================

if mode == "Analyze My Document":

    st.markdown(
        """
        <div class="feature-box">

        <h3>📄 Analyze Your Legal Document</h3>

        Upload a PDF such as an agreement, contract,
        insurance policy, terms & conditions, or other
        legal document.

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        help="Upload a PDF document for AI-assisted analysis."
    )


    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        current_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        # Build vector store only when a new document is uploaded
        if (
            st.session_state.uploaded_hash
            != current_hash
        ):

            with st.spinner(
                "Reading and indexing your document..."
            ):

                st.session_state.uploaded_vectorstore = (
                    create_uploaded_vectorstore(
                        uploaded_file
                    )
                )

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )

                st.session_state.uploaded_hash = (
                    current_hash
                )

                st.session_state.messages = []


        st.success(
            f"Document ready: {uploaded_file.name}"
        )


        st.caption(
            "You can now ask questions about this document."
        )


    else:

        st.info(
            "Upload a PDF above to start document analysis."
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(
        "user" if role == "user"
        else "assistant"
    ):

        if role == "user":

            st.markdown(
                message["content"]
            )

        else:

            answer = message["content"]

            sources = message.get(
                "sources",
                []
            )


            formatted_answer = answer


            for source in sources:

                label = source["label"]

                document = source["document"]

                page = source["page"]


                citation = (
                    f"**Source: {document}, "
                    f"p. {page}**"
                )


                formatted_answer = (
                    formatted_answer.replace(
                        f"[{label}]",
                        citation
                    )
                )


                formatted_answer = (
                    formatted_answer.replace(
                        f"[{label.replace(' ', '')}]",
                        citation
                    )
                )


            st.markdown(
                formatted_answer
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a legal question..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Document mode validation
    # --------------------------------------------------------

    if (
        mode == "Analyze My Document"
        and st.session_state.uploaded_vectorstore is None
    ):

        st.warning(
            "Please upload a PDF document first."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # --------------------------------------------------------
    # Previous conversation
    # --------------------------------------------------------

    chat_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }

        for message
        in st.session_state.messages[:-1]
    ]


    # --------------------------------------------------------
    # Select source
    # --------------------------------------------------------

    if mode == "Analyze My Document":

        active_vectorstore = (
            st.session_state.uploaded_vectorstore
        )

    else:

        active_vectorstore = None


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing the legal information..."
        ):

            answer, sources = answer_question(
                user_question,
                chat_history,
                active_vectorstore
            )


        formatted_answer = answer


        # Replace internal source labels
        for source in sources:

            label = source["label"]

            document = source["document"]

            page = source["page"]


            citation = (
                f"**Source: {document}, "
                f"p. {page}**"
            )


            formatted_answer = (
                formatted_answer.replace(
                    f"[{label}]",
                    citation
                )
            )


            formatted_answer = (
                formatted_answer.replace(
                    f"[{label.replace(' ', '')}]",
                    citation
                )
            )


        st.markdown(
            formatted_answer
        )


    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    ⚖️ <b>Legal AI Assistant:</b>
    NyayaSetu AI provides AI-assisted legal research,
    document understanding, summarization and analysis.
    Its output should be reviewed by a qualified legal
    professional before being relied upon for significant
    legal decisions.

    </div>
    """,
    unsafe_allow_html=True
)