from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the project root
load_dotenv(BASE_DIR / ".env")

# Document and vector store directories
DATA_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

# OpenAI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")

# Local embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)