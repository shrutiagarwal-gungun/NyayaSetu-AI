import os
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"


def load_documents():
    documents = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_FOLDER, filename)

            print(f"Loading: {filename}")

            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())

    return documents


def main():

    # Load all PDF documents
    documents = load_documents()

    print(f"\nTotal pages loaded: {len(documents)}")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # Create embedding model
    print("\nCreating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove old Chroma database
    if os.path.exists(CHROMA_FOLDER):
        shutil.rmtree(CHROMA_FOLDER)

    # Create Chroma vector database
    print("\nCreating ChromaDB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    print("\n===================================")
    print("ChromaDB created successfully!")
    print("===================================")
    print(f"Documents stored in: {CHROMA_FOLDER}")


if __name__ == "__main__":
    main()