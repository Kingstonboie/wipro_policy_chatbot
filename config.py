# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "policy_docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration - Auto-detect environment
if os.getenv('STREAMLIT_RUNTIME_ENV') == 'production' or os.getenv('CLOUD_DEPLOYMENT'):
    # Running on Streamlit Cloud
    LLM_MODEL = "HuggingFace"
    HF_MODEL = "microsoft/phi-2"  # You can change this to any model on HuggingFace
    # Free models that work well: "microsoft/phi-2", "google/flan-t5-large", "mistralai/Mistral-7B-Instruct-v0.1"
else:
    # Local development
    LLM_MODEL = "qwen2.5-coder:7b"
    HF_MODEL = None  # Not used locally

# Retrieval settings
RETRIEVAL_K = 4
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2