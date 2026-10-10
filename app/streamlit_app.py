from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import ask  # noqa: E402


st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Document Q&A Chatbot")
st.caption("RAG + LangChain + LangGraph + FAISS")

with st.sidebar:
    st.header("How it works")
    st.write("1. Retrieve relevant document chunks")
    st.write("2. Pass them as context to the LLM")
    st.write("3. Generate a grounded answer")
    st.divider()
    st.write("Add PDFs/TXT/MD files to `data/documents/` and run:")
    st.code("python -m src.ingest")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                result = ask(question)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write(f"• {source}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
        except Exception as exc:
            st.error(str(exc))
