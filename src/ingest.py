import pymupdf

pdf_path = "/Users/luckovka/ML/projects/Agentic-RAG-OTUS/data/sample/API и машинное обучение.pdf"

doc = pymupdf.open(pdf_path)

for page_num, page in enumerate(doc):
    text = page.get_text()
    print(f"\n--- PAGE {page_num + 1} ---\n")
    print(text[:1000])
    