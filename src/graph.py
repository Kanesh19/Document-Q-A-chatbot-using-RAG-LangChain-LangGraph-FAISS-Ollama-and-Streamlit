from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.chains import extract_sources, generate_answer
from src.retriever import get_retriever


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
    sources: list[str]


retriever = get_retriever()


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    documents = retriever.invoke(question)
    return {"documents": documents}


def generate_node(state: RAGState) -> RAGState:
    documents = state.get("documents", [])
    answer = generate_answer(state["question"], documents)
    return {
        "answer": answer,
        "sources": extract_sources(documents),
    }


def build_graph():
    """Simple baseline graph: retrieve -> generate."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


graph = None


def ask(question: str) -> dict:
    """Run the RAG graph for one user question."""
    global graph
    if graph is None:
        graph = build_graph()

    result = graph.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
