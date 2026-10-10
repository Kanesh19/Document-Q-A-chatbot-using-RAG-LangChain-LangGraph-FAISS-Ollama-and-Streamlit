from __future__ import annotations

from langchain_community.vectorstores import FAISS

from src.config import EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR
from src.ingest import get_embeddings


def get_vectorstore() -> FAISS:
    """Load the FAISS index created by src.ingest."""
    index_file = VECTORSTORE_DIR / "index.faiss"
    docstore_file = VECTORSTORE_DIR / "index.pkl"

    if not index_file.exists() or not docstore_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python -m src.ingest` after adding documents."
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever():
    """Return a Runnable retriever for top-k semantic search."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
