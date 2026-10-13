# %%
from qdrant_client import QdrantClient
client = QdrantClient(path="qdrant_data")

# %%
import pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers import CrossEncoder
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz
from langchain_core.documents import Document

# %%
data_dir = Path("..") / "data"

files = [
    path for path in data_dir.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {".pdf", ".txt", ".md"}
]

for path in files:
    print(path)



# %%


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


# %%

documents = []

for path in files:
    documents.extend(load_document(path))

print("Files:", len(files))
print("Documents:", len(documents))

# %%



# %%


# %%
collection_name = "ml_knowledge"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print("Collection created")
else:
    print("Collection already exists")

# %%
# ingestion & chunking

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

# %%

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# sample_text = all_chunks[0]["text"]

# embedding = model.encode(sample_text)

texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts)


# %%
query = "Как работает линейная регрессия?"


#  %%
query_embedding = model.encode(query)

# %%
points = []

for i, chunk in enumerate(all_chunks):
    point = PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"]
        }
    )

    points.append(point)

# %%

client.upsert(
    collection_name=collection_name,
    points=points
)
# %%
results = client.query_points(
    collection_name=collection_name,
    query=query_embedding.tolist(),
    limit=5
).points

# %%
for result in results:
    print("Score:", result.score)
    print("Page:", result.payload["page"])
    print("Chunk:", result.payload["chunk_number"])
    print("Source:", result.payload["source"])
    print(result.payload["text"])
    print("----------------")

# %%

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)
# reranker = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2"
# )

# %%
pairs = [
    (query, result.payload["text"])
    for result in results
]

# %%
rerank_scores = reranker.predict(pairs)

print(rerank_scores)

# %%
reranked = list(zip(results, rerank_scores))

# for result, rerank_score in reranked:
#     print(result.score, rerank_score, result.payload["text"][:100])

# %%
reranked_sorted = sorted(
    zip(results, rerank_scores),
    key=lambda x: x[1],
    reverse=True
)
    
# %%
# for result, score in reranked_sorted:
    

#     print("Rerank score:", score)
#     print("Page:", result.payload["page"])
#     print("Chunk:", result.payload["chunk_number"])
#     print(result.payload["text"][:500])
#     print("----------------")

for result, rerank_score in reranked_sorted:
    
    print("Qdrant score:", result.score)
    print("Rerank score:", rerank_score)
    print("Page:", result.payload["page"])
    print("Chunk:", result.payload["chunk_number"])
    print("Source:", result.payload["source"])
    print("Text:", result.payload["text"])
    print("---")
# %%
# build context

def build_context(reranked_sorted, top_k=3):
    context_parts = []

    for i, (result, rerank_score) in enumerate(reranked_sorted[:top_k], start=1):
        payload = result.payload

        source = payload["source"]
        page = payload.get("page")
        text = payload["text"]

        source_info = f"Источник {i}: {source}"

        if page is not None:
            source_info += f", страница {page}"

        context_parts.append(
            f"{source_info}\n"
            f"Содержание:\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)
# %%
context = build_context(reranked, top_k=3)

print(context[:2000])
print(type(context))

# %%
print(len(context))
print(context[-500:])