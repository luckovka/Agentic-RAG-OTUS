import pymupdf

pdf_path = "/Users/luckovka/ML/projects/Agentic-RAG-OTUS/data/sample/API и машинное обучение.pdf"

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

#text = clean_text(page.get_text())

def split_text(text, chunk_size=1000, overlap=150):
    text_chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        text_chunks.append(chunk)

        start = end - overlap

    return text_chunks

all_chunks = []

for page_num, page in enumerate(doc):
    text = clean_text(page.get_text())
    page_chunks = split_text(text)

    for chunk_num, chunk_text in enumerate(page_chunks):
        chunk_data = {
            "source": pdf_path,
            "page": page_num + 1,
            "chunk_number": chunk_num + 1,
            "text": chunk_text
        }
        all_chunks.append(chunk_data)

print(f"Total chunks created: {len(all_chunks)}")
print(f"Sample chunk: {all_chunks[15]}")
print(f"Sample chunk: {all_chunks[16]}")
print(f"Sample chunk: {all_chunks[17]}")
  
    