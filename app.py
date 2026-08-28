"""
app.py
------
Streamlit front-end for the Study Buddy RAG Assistant.

Features:
- Upload PDFs directly from the browser (no need to touch the terminal)
- Chat-style Q&A over your notes
- Shows which notes/pages an answer came from (source citation)
- Keeps chat history for the session

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from ingest import load_documents, split_documents, build_vector_store, DATA_DIR
from rag_chain import ask_question

st.set_page_config(page_title="Study Buddy — RAG Assistant", page_icon="📚", layout="wide")

# ---------- Sidebar: upload + rebuild knowledge base ----------
with st.sidebar:
    st.header("📁 Your Notes")
    st.caption("Upload PDF notes, then click 'Process' to build the knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploaded_files:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to '{DATA_DIR}/'.")

    if st.button("⚙️ Process notes into knowledge base", use_container_width=True):
        with st.spinner("Reading PDFs, chunking, and building vector store..."):
            try:
                docs = load_documents()
                chunks = split_documents(docs)
                build_vector_store(chunks)
                st.success("Knowledge base ready! Ask a question below. 🎉")
            except Exception as e:
                st.error(
                    f"Something went wrong: {e}\n\n"
                    "Make sure Ollama is running (`ollama serve`) and you've "
                    "pulled the models (`ollama pull nomic-embed-text` and "
                    "`ollama pull llama3.2`)."
                )

    st.divider()
    st.caption(
        "💡 Tip: upload subject-wise notes (e.g. Cryptography.pdf, "
        "ComputerNetworks.pdf) and ask exam-style questions to revise faster."
    )

# ---------- Main chat area ----------
st.title("📚 Study Buddy — Ask Your Notes Anything")
st.caption("A RAG-powered assistant that answers strictly from your uploaded notes.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, sources)

# Render past messages
for role, content, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("📎 Sources used for this answer"):
                for s in sources:
                    st.markdown(
                        f"**{os.path.basename(s['source'])}** (page {s['page']})\n\n"
                        f"> {s['snippet']}"
                    )

# Chat input
question = st.chat_input("Ask something from your notes, e.g. 'Explain AES encryption'")

if question:
    st.session_state.chat_history.append(("user", question, None))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not os.path.isdir("chroma_db"):
            answer = (
                "⚠️ No knowledge base found yet. Upload PDFs in the sidebar and "
                "click 'Process notes into knowledge base' first."
            )
            sources = []
            st.markdown(answer)
        else:
            with st.spinner("Searching your notes..."):
                try:
                    result = ask_question(question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
            st.markdown(answer)
            if sources:
                with st.expander("📎 Sources used for this answer"):
                    for s in sources:
                        st.markdown(
                            f"**{os.path.basename(s['source'])}** (page {s['page']})\n\n"
                            f"> {s['snippet']}"
                        )

    st.session_state.chat_history.append(("assistant", answer, sources))
