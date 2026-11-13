from dotenv import load_dotenv
from .database import get_client
from .graph import rag_e5_graph
from langfuse import get_client as get_langfuse_client
from langfuse.langchain import CallbackHandler

load_dotenv()
langfuse_handler = CallbackHandler()

def run_query(query: str) -> dict:
    result = rag_e5_graph.invoke(
        {"query": query},
        config={
            "callbacks": [langfuse_handler],
            "run_name": "Agentic-RAG-OTUS"
        }
    )

    get_langfuse_client().flush()

    return result

def main():
    tests = [
        "Привет!",
        "Посчитай precision, recall и F1, если TP=85, FP=15, FN=5",
        "Чем отличаются model.train() и model.eval() в PyTorch?"
    ]

    try:
        for query in tests:
            result = run_query(query)

            print("Маршрут:", result["route"])
            print("Ответ:", result["answer"])
            print("Источники:", result["sources"])
    
    finally:
        get_client().close()
        get_client.cache_clear()
    

if __name__ == "__main__":
    main()