# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "policy_docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # any sentence-transformers model

# LLM (Ollama)
LLM_MODEL = "qwen2.5-coder:7b"        # ensure it's pulled via ollama

# Retrieval settings
RETRIEVAL_K = 4                        # number of chunks to retrieve
CHUNK_SIZE = 10                         # lines per chunk
CHUNK_OVERLAP = 2                        # overlap lines between chunks