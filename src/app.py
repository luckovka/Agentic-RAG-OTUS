from .database import get_client
from .graph import rag_e5_graph

def main():
    tests = [
        "Привет!",
        "Посчитай precision, recall и F1, если TP=85, FP=15, FN=5",
        "Чем отличаются model.train() и model.eval() в PyTorch?"
    ]

    try:
        for query in tests:
            result = rag_e5_graph.invoke({"query": query})

            print("Маршрут:", result["route"])
            print("Ответ:", result["answer"])
            print("Источники:", result["sources"])

    finally:
        # Закрываем клиент только после окончания всех запросов
        get_client().close()
        get_client.cache_clear()


if __name__ == "__main__":
    main()