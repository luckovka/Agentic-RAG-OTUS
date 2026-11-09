from qdrant_client import QdrantClient
from functools import lru_cache
from .config import QDRANT_PATH

@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_PATH))