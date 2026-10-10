# Document Q&A Chatbot with RAG

Beginner-level RAG project using Python, LangChain, LangGraph, FAISS, Hugging Face embeddings, OpenAI, and Streamlit.

## 1. Architecture

```text
                    USER QUESTION
                          |
                          v
                  +---------------+
                  |   Retriever   |
                  | FAISS search  |
                  +-------+-------+
                          |
                  Top-k relevant chunks
                          |
                          v
                  +---------------+
                  |   Generator   |
                  |   LLM +       |
                  | grounded      |
                  | prompt        |
                  +-------+-------+
                          |
                          v
                       ANSWER
                          |
                          v
                    Streamlit UI

INGESTION (runs before chat):
PDF/TXT/MD -> Load -> Chunk -> Embeddings -> FAISS -> save locally
```

## 2. Folder structure

```text
rag-langchain-langgraph-chatbot/
├── data/
│   └── documents/                 # Put your PDFs/TXT/MD files here
├── vectorstore/
│   └── faiss_index/               # Generated FAISS index
├── src/
│   ├── __init__.py
│   ├── config.py                  # Settings
│   ├── ingest.py                  # Load -> chunk -> embed -> index
│   ├── retriever.py               # Load FAISS + top-k search
│   ├── chains.py                  # Prompt + LLM generation
│   └── graph.py                   # LangGraph retrieve -> generate
├── app/
│   └── streamlit_app.py           # Chat UI
├── tests/
│   └── test_retriever.py
├── notebooks/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Windows + VS Code setup

Use Python 3.11 or another version supported by the packages you install.

### Create the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure the API key

```powershell
copy .env.example .env
```

Open `.env` and replace `your_api_key_here` with your OpenAI API key. Never commit `.env` to GitHub.

### Add documents

Put PDF, TXT, or MD files inside:

```text
data/documents/
```

The repository includes `sample.txt`, so you can test the pipeline immediately.

### Build the FAISS vector store

```powershell
python -m src.ingest
```

### Start the chatbot

```powershell
streamlit run app/streamlit_app.py
```

## 4. How the code maps to RAG concepts

### Ingestion
`src/ingest.py` loads documents with LangChain loaders, splits them with `RecursiveCharacterTextSplitter`, creates embeddings, and stores the resulting vectors in FAISS.

### Retrieval
`src/retriever.py` reloads the FAISS index and exposes a top-k semantic retriever.

### Generation
`src/chains.py` builds a prompt that injects the retrieved context into the LLM request. The system message explicitly tells the model not to invent information outside the context.

### Orchestration
`src/graph.py` defines a LangGraph state with two nodes:

```text
START -> retrieve -> generate -> END
```

The graph keeps the question, retrieved documents, answer, and source labels in state.

### Interface
`app/streamlit_app.py` provides a simple chat UI and displays the answer plus the retrieved document sources.

## 5. First questions to test

Try these after indexing `sample.txt`:

- What does RAG stand for?
- What is the purpose of chunking?
- Why are embeddings used?
- What does FAISS do in this project?
- What is LangGraph doing here?

Then ask something that is not in the document, such as:

- Who is the CEO of Microsoft?

The prompt should make the system say that the information is not in the provided documents instead of guessing.

## 6. Important beginner tuning parameters

Start with:

- `CHUNK_SIZE=800`
- `CHUNK_OVERLAP=120`
- `TOP_K=4`

Change one value at a time and compare retrieval quality. The most important part of a RAG system is whether the retrieved chunks are actually relevant to the question.

## 7. Next upgrade: relevance checking

The current graph intentionally stays simple:

```text
retrieve -> generate
```

A natural next step is:

```text
retrieve -> check relevance -> generate
                    |
                  retry
                    |
                 retrieve
```

That version can use a grader node to decide whether retrieval is good enough before generation.
