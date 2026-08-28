# 📚 Study Buddy — RAG-based Notes Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions **strictly from your own uploaded notes/PDFs**, instead of relying on the LLM's general (and sometimes hallucinated) knowledge.

Built as a portfolio project demonstrating practical LLM application development: document ingestion, chunking, embeddings, vector search, and grounded generation.

---

## 🧠 How It Works (Architecture)

```
                ┌─────────────────┐
                │   PDF Notes      │
                └────────┬─────────┘
                         │  1. Load
                         ▼
                ┌─────────────────┐
                │  Text Splitter   │  (1000-char chunks, 200 overlap)
                └────────┬─────────┘
                         │  2. Chunk
                         ▼
                ┌─────────────────┐
                │ Ollama Embeddings│  (nomic-embed-text, runs locally)
                └────────┬─────────┘
                         │  3. Embed
                         ▼
                ┌─────────────────┐
                │   ChromaDB       │  (local vector store, persisted)
                └────────┬─────────┘
                         │
        User Question ──┤  4. Similarity Search (top-k=4)
                         ▼
                ┌─────────────────┐
                │  Retrieved Chunks│
                └────────┬─────────┘
                         │  5. Stuffed into prompt as context
                         ▼
                ┌─────────────────┐
                │ Llama 3.2 (Ollama)│  6. Generates grounded answer, runs locally
                └────────┬─────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Streamlit Chat UI│  7. Answer + source citations
                └─────────────────┘
```

## ✨ Features

- **Upload PDFs directly in the browser** — no manual file handling needed
- **Source citation** — every answer shows which document/page it came from, reducing hallucination risk
- **Grounded generation** — the LLM is explicitly instructed to say "I don't know" if the answer isn't in the notes, rather than making things up
- **Persistent vector store** — process your notes once, query them anytime without re-embedding
- **Chat history** within a session

## 🛠️ Tech Stack

| Component        | Choice                          | Why |
|-------------------|----------------------------------|-----|
| LLM               | Qwen2.5 (0.5B) via Ollama (local) | Free, runs entirely offline, works on low-RAM machines |
| Orchestration     | LangChain (LCEL)                 | Industry-standard for chaining retrieval + generation |
| Embeddings        | nomic-embed-text via Ollama      | Free local embedding model, good quality |
| Vector Store      | ChromaDB (local)                 | Free, no signup, persists to disk |
| Frontend          | Streamlit                        | Fast to build, good for demos |
| PDF Parsing       | PyPDF                            | Simple, reliable for text-based PDFs |

> **Note:** This project can also run on OpenAI's API (GPT-4o-mini + text-embedding-3-small) instead of Ollama — swap `langchain_ollama` imports for `langchain_openai` in `ingest.py` and `rag_chain.py` if you'd rather use a hosted model.

## 🚀 Setup & Run

### 0. Install Ollama (one-time, only needed once)
Download from [ollama.com](https://ollama.com) and install it — it's a small app that runs LLMs locally on your machine.

Then pull the two models this project needs:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```
> This project defaults to `qwen2.5:0.5b`, a small model that runs comfortably on low-RAM machines (4-8GB). If you have 16GB+ RAM, you can use a stronger model like `llama3.2` instead — just update `LLM_MODEL` in `rag_chain.py` and pull it with `ollama pull llama3.2`.

Make sure Ollama is running in the background (it usually starts automatically after install; otherwise run `ollama serve`).

### 1. Project setup
```bash
# Clone and enter the project
git clone <your-repo-url>
cd study-buddy-rag

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Option A: Ingest via command line
#   Put PDFs in data/ folder first
python ingest.py

# Option B: Or just run the app and upload PDFs from the UI
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

No API key, no signup, no cost — everything runs on your machine.

## 📂 Project Structure

```
study-buddy-rag/
├── app.py           # Streamlit UI
├── ingest.py         # PDF loading, chunking, vector store creation
├── rag_chain.py       # Retrieval + prompt + LLM chain logic
├── requirements.txt
├── data/             # Your uploaded PDFs land here
└── chroma_db/         # Persisted vector database (auto-created)
```

## 🎯 What This Project Demonstrates

- Understanding of **RAG architecture** end-to-end (not just calling an API)
- **Chunking strategy** decisions (size/overlap trade-offs)
- **Prompt engineering** to reduce hallucination and enforce grounded answers
- **Vector similarity search** fundamentals
- Building a usable **end-user interface**, not just a notebook

## 🔮 Possible Extensions

- Swap ChromaDB for Pinecone/Weaviate for cloud-hosted, production-scale search
- Add conversation memory so follow-up questions understand prior context
- Add re-ranking (e.g. Cohere Rerank) to improve retrieval quality
- Support DOCX/PPTX notes, not just PDFs
- Deploy backend as a FastAPI service + React frontend (see companion "Full-Stack AI SaaS" project)

## 📝 Resume Bullet (ready to use)

> Built a fully local RAG-based document Q&A assistant using LangChain, ChromaDB, and Ollama (Llama 3.2); implemented semantic chunking and source-grounded retrieval to reduce hallucination, with a Streamlit interface for PDF upload and chat-based querying.
