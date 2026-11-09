from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .retrieval import retrieve_and_rerank
from .functions import route_query, answer_question_e5, direct_answer_func, python_tool_func, build_context

class RAGState(TypedDict):
    query: str
    route: str
    answer: str
    sources: list[dict]


def router_node(state: RAGState) -> dict:
    route = route_query(state["query"])
    return {"route": route}

def route_function(state: RAGState) -> str:
    return state["route"]

def rag_search_node(state: RAGState) -> dict:
    result = answer_question_e5(state["query"])
    return {
        'answer': result['answer'],
        'sources': result['sources']
        }

def python_tool_node(state: RAGState) -> dict:
    code_result = python_tool_func(state["query"])
    return {
        "answer": code_result['answer'],
        "sources": code_result['sources']
        }

def direct_answer_node(state: RAGState) -> dict:
    answer_result = direct_answer_func(state["query"])
    return {
        "answer": answer_result['answer'],
        "sources": answer_result['sources']
        }
    
    
def router_node(state: RAGState) -> dict:
    route = route_query(state["query"])
    return {"route": route}

def route_function(state: RAGState) -> str:
    return state["route"]

def rag_search_node(state: RAGState) -> dict:
    result = answer_question_e5(state["query"])
    return {
        'answer': result['answer'],
        'sources': result['sources']
        }

def python_tool_node(state: RAGState) -> dict:
    code_result = python_tool_func(state["query"])
    return {
        "answer": code_result['answer'],
        "sources": code_result['sources']
        }

def direct_answer_node(state: RAGState) -> dict:
    answer_result = direct_answer_func(state["query"])
    return {
        "answer": answer_result['answer'],
        "sources": answer_result['sources']
        }

builder = StateGraph(RAGState)

builder.add_node('router', router_node)
builder.add_node('rag_search', rag_search_node)
builder.add_node('python_tool', python_tool_node)
builder.add_node('direct_answer', direct_answer_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_function,
    {
        "rag_search": "rag_search",
        "python_tool": "python_tool",
        "direct_answer": "direct_answer"
    }
)

builder.add_edge("rag_search", END)
builder.add_edge("python_tool", END)
builder.add_edge("direct_answer", END)

rag_e5_graph = builder.compile()