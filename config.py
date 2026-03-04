# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAGAN_DOCS_DIR = os.path.join(BASE_DIR, "gagan_bot", "policy_docs")
GAGAN_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

HARDIK_DOCS_DIR = os.path.join(BASE_DIR, "hardik_bot")
HARDIK_CSV_PATH = os.path.join(HARDIK_DOCS_DIR, "deepseek_csv_20260302_c82ca1.csv")
HARDIK_PERSIST_DIR = os.path.join(HARDIK_DOCS_DIR, "chrome_langchain_db")

# Shared LLM (both bots use the same model)
LLM_MODEL = "qwen2.5-coder:7b"
EMBEDDING_MODEL = "mxbai-embed-large"  # For Hardik's bot

# Retrieval settings
RETRIEVAL_K = 4

# ADD THESE LINES - Chunking settings for Gagan's bot
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2