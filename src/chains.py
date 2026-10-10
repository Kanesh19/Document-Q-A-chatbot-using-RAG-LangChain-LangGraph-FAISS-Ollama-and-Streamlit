from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question using ONLY the supplied context.
Do not invent facts or use outside knowledge.
If the context does not contain the answer, say:
"I don't have that information in the provided documents."

Keep the answer clear and concise.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)

# Run the language model locally through Ollama.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

def format_documents(documents: list[Document]) -> str:
    """Turn retrieved LangChain Documents into prompt-ready context."""
    blocks = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""

        blocks.append(
            f"[Source {i}: {source}{page_text}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


answer_chain = prompt | llm | StrOutputParser()


def generate_answer(question: str, documents: list[Document]) -> str:
    """Generate an answer using the retrieved documents."""
    context = format_documents(documents)

    return answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )


def extract_sources(documents: list[Document]) -> list[str]:
    """Return unique human-readable source labels."""
    sources: list[str] = []

    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")

        label = (
            f"{source} (page {page + 1})"
            if isinstance(page, int)
            else source
        )

        if label not in sources:
            sources.append(label)

    return sources