import os
import tempfile
import uuid

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

CHROMA_FOLDER = "chroma_db"

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file "
        "or Streamlit Secrets."
    )


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# EXISTING LEGAL KNOWLEDGE BASE
# ============================================================

legal_vectorstore = Chroma(
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are NyayaSetu AI, an AI-powered Indian Legal Assistant.

Your role is to help users understand Indian legal information and
analyze legal documents using the retrieved source material.

You may assist with:
- Indian legal research
- Legal document understanding
- Contract and agreement analysis
- Clause explanations
- Summarization
- Identifying important obligations, rights, restrictions and
  potentially important clauses
- Explaining legal language in simple terms

==================================================
SOURCE ACCURACY
==================================================

1. Use ONLY the retrieved context for factual or legal claims.

2. Do not invent laws, Sections, Articles, clauses, penalties,
rights, obligations, procedures or interpretations.

3. Do not assume that information exists in a document when it
was not retrieved.

4. If the retrieved context does not contain enough information,
clearly say that the information could not be found in the
available source material.

5. You may simplify legal language, but do not change its meaning.

==================================================
DOCUMENT ANALYSIS
==================================================

6. When analyzing an uploaded document, focus specifically on
that document.

7. You may identify:
- important clauses
- obligations
- restrictions
- deadlines
- fees or penalties
- termination conditions
- unusual or potentially concerning clauses
- rights and responsibilities

8. When identifying something as potentially risky or concerning,
explain WHY it may deserve attention. Do not claim that a clause
is legally invalid unless the retrieved source supports that claim.

9. Do not invent missing information.

==================================================
ANSWERING STYLE
==================================================

10. Answer the user's actual question directly.

11. Use simple and professional language.

12. If the user asks for a simple explanation, explain it as if
you are helping a person who does not have a legal background.

13. If the user asks for detailed analysis, provide structured
analysis.

14. Avoid unnecessary background information.

15. Use headings, bullets and numbered lists when they improve
clarity.

16. Do not repeat the same point unnecessarily.

==================================================
FOLLOW-UP QUESTIONS
==================================================

17. If the user asks a follow-up such as:
"explain this",
"what does this mean",
"tell me more",
use the previous conversation to understand what "this" refers to.

18. Previous conversation helps understand the question but is
NOT a source of new legal facts.

==================================================
CITATIONS
==================================================

19. Retrieved documents are labelled as [Source 1], [Source 2],
etc.

20. Use these labels when an important claim needs verification.

21. Never invent source numbers.

22. Never invent document names or page numbers.

==================================================
PROFESSIONAL USE
==================================================

23. You are designed as an AI legal assistant for legal research
and document understanding.

24. Your output should be treated as AI-assisted analysis and
reviewed by a qualified legal professional before being relied
upon for significant legal decisions.

25. Do not falsely claim to be a lawyer or law firm.

"""
        ),
        (
            "human",
            """
SOURCE MATERIAL:

{context}

PREVIOUS CONVERSATION:

{history}

CURRENT USER QUESTION:

{question}

Answer the user's question directly and professionally.

Use source labels such as [Source 1] when they help verify
important information.
"""
        )
    ]
)


# ============================================================
# BUILD VECTOR STORE FOR UPLOADED PDF
# ============================================================

def create_uploaded_vectorstore(uploaded_file):

    file_bytes = uploaded_file.getvalue()
    original_name = uploaded_file.name

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    try:

        temp_file.write(file_bytes)
        temp_file.close()

        loader = PyMuPDFLoader(temp_file.name)

        documents = loader.load()

        # Replace temporary filename with actual uploaded filename
        for document in documents:
            document.metadata["source"] = original_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        collection_name = (
            "upload_" + uuid.uuid4().hex[:12]
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )

        return vectorstore

    finally:

        try:
            os.unlink(temp_file.name)
        except OSError:
            pass


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question,
    chat_history=None,
    document_vectorstore=None
):

    if chat_history is None:
        chat_history = []


    # --------------------------------------------------------
    # Improve retrieval for follow-up questions
    # --------------------------------------------------------

    retrieval_query = question

    previous_user_questions = [
        message["content"]
        for message in chat_history
        if message.get("role") == "user"
    ]

    if previous_user_questions:

        recent_question = previous_user_questions[-1]

        retrieval_query = (
            f"Previous question: {recent_question}\n"
            f"Current question: {question}"
        )


    # --------------------------------------------------------
    # Select knowledge source
    # --------------------------------------------------------

    if document_vectorstore is not None:

        active_vectorstore = document_vectorstore

    else:

        active_vectorstore = legal_vectorstore


    retriever = active_vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    documents = retriever.invoke(
        retrieval_query
    )


    if not documents:

        return (
            "I couldn't find enough relevant information in "
            "the available source material to answer this question.",
            []
        )


    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    context_parts = []
    source_details = []


    for index, document in enumerate(
        documents,
        start=1
    ):

        source_label = f"Source {index}"

        source = os.path.basename(
            document.metadata.get(
                "source",
                "Unknown document"
            )
        )

        page = document.metadata.get(
            "page",
            0
        ) + 1


        context_parts.append(
            f"[{source_label}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content}"
        )


        source_details.append(
            {
                "label": source_label,
                "document": source,
                "page": page
            }
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    if chat_history:

        history_text = ""

        for message in chat_history[-6:]:

            role = message.get(
                "role",
                "user"
            )

            content = message.get(
                "content",
                ""
            )

            history_text += (
                f"{role.capitalize()}: "
                f"{content}\n"
            )

    else:

        history_text = (
            "No previous conversation."
        )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "history": history_text,
            "question": question
        }
    )


    response = llm.invoke(
        messages
    )


    answer = response.content.strip()


    return answer, source_details