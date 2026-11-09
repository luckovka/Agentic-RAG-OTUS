from .database import get_client
client = get_client()

def retrieve_and_rerank(query: str):
    embedding = e5_model.encode(
        "query: " + query,
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=e5_collection,
        query=embedding.tolist(),
        limit=5
    ).points

    pairs = [
        (query, point.payload["text"])
        for point in results
    ]

    rerank_scores = reranker.predict(pairs)

    return sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )