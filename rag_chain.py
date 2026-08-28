"""
rag_chain.py
------------
This is the "brain" of the app. Given a user question, it:

1. Embeds the question into a vector.
2. Searches ChromaDB for the most semantically similar chunks
   (this is the "Retrieval" in RAG).
3. Stuffs those chunks into a prompt as context.
4. Sends the prompt to the LLM, which generates an answer grounded
   in the retrieved context (this is the "Generation" in RAG).
5. Returns the answer along with the source chunks used, so the UI
   can show citations.

Keeping this separate from app.py means the retrieval/LLM logic can
be reused later in the AI Agent or MLOps projects.
"""

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "study_notes"

# Must match the models pulled via `ollama pull <model>`
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"  # small, low-RAM model — good for 4-8GB machines

# Number of chunks retrieved per query. Higher = more context but
# more tokens/cost and more chance of irrelevant info confusing the LLM.
TOP_K = 4

# This prompt is deliberately strict: it tells the model to answer
# ONLY from the provided context and to admit when it doesn't know,
# rather than making things up (hallucinating).
SYSTEM_PROMPT = """You are a helpful study assistant. Answer the student's \
question using ONLY the context below, which comes from their own class notes.

Rules:
- If the answer is in the context, explain it clearly and in enough \
detail to help the student understand it for an exam.
- If the context does NOT contain the answer, say plainly: \
"I couldn't find this in your uploaded notes." Do not make up information.
- When helpful, use short bullet points or numbered steps.
- Keep the tone simple and exam-focused.

Context:
{context}

Question: {question}

Answer:"""


def get_vector_store():
    """Load the existing persisted Chroma vector store from disk."""
    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(
            "No vector store found. Run `python ingest.py` first to "
            "process your PDFs."
        )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def format_docs(docs):
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """
    Build and return:
      - the runnable chain that produces the final answer
      - the retriever (used separately to fetch source docs for citation)
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask_question(question: str):
    """
    Convenience function used by the Streamlit app.
    Returns a dict with the answer text and the source chunks
    (with filename + page number) that were used to generate it.
    """
    chain, retriever = get_rag_chain()

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    sources = []
    for doc in source_docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:250] + "...",
        })

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python rag_chain.py
    q = input("Ask a question about your notes: ")
    result = ask_question(q)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    for s in result["sources"]:
        print(f"- {s['source']} (page {s['page']})")
