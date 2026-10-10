from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local embedding model used for both indexing and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents():
    """Load PDFs and text-like files from data/documents/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported documents found in {DATA_DIR}. "
            "Add at least one PDF, TXT, or MD file."
        )

    documents = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        loaded = loader.load()
        for doc in loaded:
            # Keep a simple source field for citations in the UI.
            doc.metadata["source"] = path.name
        documents.extend(loaded)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vectorstore() -> None:
    """Load -> chunk -> embed -> index -> save."""
    print("1/4 Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document pages/files.")

    print("2/4 Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3/4 Creating embeddings and FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("4/4 Saving FAISS index...")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Done. Vector store saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
