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

text = clean_text(page.get_text())

print(pages[2]["text"])
    