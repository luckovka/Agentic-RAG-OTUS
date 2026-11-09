import re
import pymupdf
import fitz
from langchain_core.documents import Document
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct
from .config import data_dir
from .config import e5_model, e5_collection 
from .database import get_client

client = get_client()

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(e5_model)

files = [
    path for path in data_dir.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {".pdf", ".txt", ".md"}
]

def load_document(path):
    loaded_documents = []

    if path.suffix.lower() == ".pdf":
        pdf = fitz.open(path)

        for page_num, page in enumerate(pdf):
            text = page.get_text()

            if text.strip():
                loaded_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "category": path.parent.name,
                            "page": page_num + 1,
                        }
                    )
                )

        pdf.close()

    elif path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")

        if text.strip():
            loaded_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "category": path.parent.name,
                    }
                )
            )

    return loaded_documents

documents = []

for path in files:
    documents.extend(load_document(path))

def clean_text(text):
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text

def split_text(text, chunk_size=1000):
    text_chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            split_pos = text.rfind("\n", start, end)

            if split_pos == -1:
                split_pos = text.rfind(" ", start, end)

            if split_pos != -1 and split_pos > start:
                end = split_pos + 1

        chunk = text[start:end].strip()

        if chunk:
            text_chunks.append(chunk)

        start = end

    return text_chunks

all_chunks = []
for document in documents:
    text = clean_text(document.page_content)
    page_chunks = split_text(text)

    for chunk_num, chunk_text in enumerate(page_chunks):
        chunk_data = {
            "source": document.metadata['source'],
            'category': document.metadata['category'],
            "page": document.metadata.get("page"),
            "chunk_number": chunk_num + 1,
            "text": chunk_text
        }
        all_chunks.append(chunk_data)

def ingest_documents():
    e5_texts = [
    "passage: " + chunk["text"]
    for chunk in all_chunks
    ]

    e5_embeddings = e5_model.encode(
        e5_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


    if not client.collection_exists(e5_collection):
        client.create_collection(
            collection_name=e5_collection,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    e5_points = []

    for i, chunk in enumerate(all_chunks):
        e5_points.append(
            PointStruct(
                id=i,
                vector=e5_embeddings[i].tolist(),
                payload={
                    "source": chunk["source"],
                    "category": chunk["category"],
                    "page": chunk["page"],
                    "chunk_number": chunk["chunk_number"],
                    "text": chunk["text"]
                }
            )
        )

    client.upsert(
        collection_name=e5_collection,
        points=e5_points
    )

if __name__ == "__main__":
    ingest_documents()
