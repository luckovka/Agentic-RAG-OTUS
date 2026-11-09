from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers.util import cos_sim
from .database import get_client
from .config import e5_model, reranker, e5_collection 


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(e5_model)

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(reranker)

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

