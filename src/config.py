
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import json
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

data_dir = PROJECT_ROOT / "data"
QDRANT_PATH = SRC_DIR / "qdrant_data"

e5_model = SentenceTransformer("intfloat/multilingual-e5-small")
e5_collection = "rag_e5_small"
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)
