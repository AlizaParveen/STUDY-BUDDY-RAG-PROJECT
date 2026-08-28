"""
ingest.py
---------
This script builds the "knowledge base" for the RAG assistant.

What it does, step by step:
1. Loads every PDF from the `data/` folder.
2. Splits each PDF into small overlapping chunks (so the LLM gets
   focused, relevant context instead of a whole 40-page PDF).
3. Converts each chunk into a vector embedding (a list of numbers that
   captures the *meaning* of the text).
4. Stores those vectors in a local ChromaDB database on disk, so we
   only have to do this once per document.

Run this whenever you add new PDFs:
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "study_notes"

# Chunk size / overlap are tunable. 1000 chars ~ 200-250 tokens.
# Overlap ensures we don't cut a sentence in half between two chunks
# and lose context that spans a chunk boundary.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Local embedding model served by Ollama. Must be pulled first with:
#   ollama pull nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"


def load_documents():
    """Load every PDF inside data/ into LangChain Document objects."""
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(
            f"No files found in '{DATA_DIR}/'. Put your notes/PDFs there first."
        )

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs in '{DATA_DIR}/'.")
    return docs


def split_documents(docs):
    """Break long documents into small, semantically coherent chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph breaks first
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_vector_store(chunks):
    """Embed each chunk and persist it to a local Chroma vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store built and saved to '{CHROMA_DIR}/'.")
    return vector_store


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("\n✅ Ingestion complete! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
