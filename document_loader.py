# document_loader.py
import os
from langchain_core.documents import Document  # <-- changed from langchain.schema
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents_with_lines():
    """
    Load all .txt files from DOCS_DIR, split into chunks,
    and attach metadata: source file name, start_line, end_line.
    Returns a list of LangChain Document objects.
    """
    docs = []
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Split lines into overlapping chunks
        for i in range(0, len(lines), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_lines = lines[i:i + CHUNK_SIZE]
            if not chunk_lines:
                continue
            start_line = i + 1  # lines are 1-indexed for citation
            end_line = i + len(chunk_lines)
            content = "".join(chunk_lines)

            metadata = {
                "source": filename,
                "start_line": start_line,
                "end_line": end_line
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
    return docs