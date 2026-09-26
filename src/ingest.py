# %%
import pymupdf
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers import CrossEncoder

# %%
# ingestion & chunking

pdf_path = "D:\LLM\RAG-project\Agentic-RAG-OTUS\data\sample\API и машинное обучение.pdf"

doc = pymupdf.open(pdf_path)

pages = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    page_data = {
        "source": pdf_path,
        "page": page_num + 1,
        "text": text
    }
    pages.append(page_data)


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
for page_num, page in enumerate(doc):
    text = clean_text(page.get_text())
    text = page.get_text()
    page_chunks = split_text(text)

    for chunk_num, chunk_text in enumerate(page_chunks):
        chunk_data = {
            "source": pdf_path,
            "page": page_num + 1,
            "chunk_number": chunk_num + 1,
            "text": chunk_text
        }
        all_chunks.append(chunk_data)

# %%

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# sample_text = all_chunks[0]["text"]

# embedding = model.encode(sample_text)

texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts)

# %%
query = "Определение переобучения модели в машинном обучении"
query_embedding = model.encode(query)
scores = cos_sim(query_embedding, embeddings)[0]
best_idx = scores.argmax().item()

# %%

print("Score:", scores[best_idx].item())
print("Page:", all_chunks[best_idx]["page"])
print("Chunk:", all_chunks[best_idx]["chunk_number"])
print(all_chunks[best_idx]["text"])
  
    
# %%
# for i, chunk in enumerate(all_chunks):
#     if "переобучение" in chunk["text"].lower():
#         print("Index:", i)
#         print("Page:", chunk["page"])
#         print(chunk["text"])
#         print("----------------")
# %%
for i, chunk in enumerate(all_chunks):
    if "переобучение" in chunk["text"].lower():
        print("Index:", i)
        print("Score:", scores[i].item())
# %%
top_indices = scores.argsort(descending=True)[:5]

for idx in top_indices:
    idx = idx.item()

    print("Score:", scores[idx].item())
    print("Page:", all_chunks[idx]["page"])
    print("Chunk:", all_chunks[idx]["chunk_number"])
    print(all_chunks[idx]["text"])
    print("----------------")
# %%

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# %%
pairs = []

for idx in top_indices:
    idx = idx.item()

    pairs.append([
        query,
        all_chunks[idx]["text"]
    ])