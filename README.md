# Document Q&A Chatbot using RAG

A document question-answering application built with Retrieval-Augmented Generation (RAG). It retrieves relevant passages from a document collection and uses a locally running Ollama language model to generate answers grounded in that context.

## Features

- Load PDF and text documents from a local folder
- Split documents into smaller chunks
- Generate embeddings with Hugging Face Sentence Transformers
- Store and search document embeddings with FAISS
- Retrieve relevant context for a user's question
- Generate answers using Ollama (`llama3.2:3b`)
- Build prompts and answer chains with LangChain
- Organize the workflow with LangGraph
- Provide a browser-based interface with Streamlit
- Display source information for retrieved documents

## Tech Stack

- **Language:** Python
- **LLM runtime:** Ollama
- **Language model:** `llama3.2:3b`
- **LLM orchestration:** LangChain
- **Workflow:** LangGraph
- **Embeddings:** Hugging Face Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector search:** FAISS
- **UI:** Streamlit

## How RAG Works

1. **Load documents** from `data/documents/`.
2. **Split text** into manageable chunks.
3. **Create embeddings** for each chunk.
4. **Index embeddings** in a FAISS vector store.
5. **Retrieve relevant chunks** for a user's question.
6. **Generate an answer** using the retrieved context and the local Ollama model.
7. **Display the answer and sources** in the Streamlit interface.

## Project Structure

```text
rag-langchain-langgraph-chatbot/
├── app/
│   └── streamlit_app.py
├── data/
│   └── documents/           # Add your own supported documents here
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── chains.py
│   └── graph.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Files may vary slightly as the project evolves. The generated FAISS index and local Ollama model files should not be committed to Git.

## Prerequisites

- Windows, macOS, or Linux
- Python version compatible with the packages in `requirements.txt`
- Git
- [Ollama](https://ollama.com/download)
- Enough disk space to download the Ollama model and embedding model

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kanesh19/Document-Q-A-chatbot-using-RAG-LangChain-LangGraph-FAISS-Ollama-and-Streamlit.git
cd Document-Q-A-chatbot-using-RAG-LangChain-LangGraph-FAISS-Ollama-and-Streamlit
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv env
source env/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Ollama integration is not listed in `requirements.txt`, install it with:

```bash
python -m pip install langchain-ollama
```

If you install it separately, add `langchain-ollama` to `requirements.txt` so other users can install it too.

### 4. Start Ollama and download the model

Install Ollama, then run:

```bash
ollama pull llama3.2:3b
```

Confirm that the model is available:

```bash
ollama list
```

Ollama should be running locally and accessible at its default address, `http://localhost:11434`.

## Add Documents and Build the Index

Place documents you are permitted to use in:

```text
data/documents/
```

Build the FAISS vector index:

```bash
python -m src.ingest
```

The vector store is generated locally and is intentionally excluded from Git. Run ingestion again after adding or changing documents.

## Run the Application

From the project root, with the Python virtual environment activated and Ollama running:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser.

## Configuration

Review `src/config.py` for settings such as chunk size, chunk overlap, retrieval count, embedding model, and vector-store path. Keep credentials and other private configuration in a local `.env` file when required.

- Never commit `.env` or API keys.
- `.env.example` should contain placeholders only, not real credentials.
- Do not commit private documents or confidential information.

## Deployment Notes

The application uses Ollama on the local machine by default. A hosted Streamlit application cannot automatically connect to Ollama running on your personal laptop.

To deploy publicly, configure a model backend that is reachable and authorized from the hosting environment, and ensure the document collection and FAISS index are available there. Do not expose your local Ollama server directly to the public internet. Review hosting limits, model requirements, document privacy, and any provider costs before deployment.

## Troubleshooting

- **`ollama` is not recognized:** Install Ollama and reopen your terminal.
- **Connection refused / server unavailable:** Open Ollama and check that `http://localhost:11434` is reachable.
- **Model not found:** Run `ollama pull llama3.2:3b`.
- **Vector store missing:** Add documents and run `python -m src.ingest`.
- **Missing Python module:** Activate the project virtual environment and install `requirements.txt`.

## Limitations

- Answer quality depends on document quality, retrieval relevance, and the local model.
- Instructions to answer only from context reduce but do not completely eliminate hallucinations.
- The vector store is not included in the repository; it must be generated from the documents available to the running application.

## License

No license is specified yet. Add a suitable license file before permitting reuse or redistribution.
