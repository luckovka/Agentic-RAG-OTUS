
from qdrant_client import QdrantClient
import pymupdf
import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers import CrossEncoder
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ollama import chat
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
QDRANT_PATH = SRC_DIR / "qdrant_data"

e5_model = SentenceTransformer("intfloat/multilingual-e5-small")
e5_collection = "rag_e5_small"
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)
