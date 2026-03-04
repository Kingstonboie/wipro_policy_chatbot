# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "policy_docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# More reliable Streamlit Cloud detection
IS_STREAMLIT_CLOUD = (
    os.path.exists('/mount/src') or 
    os.getenv('STREAMLIT_RUNTIME_ENV') == 'production' or
    os.getenv('STREAMLIT_SHARING') is not None or
    os.getenv('STREAMLIT_DEPLOYMENT_ID') is not None
)

# LLM Configuration - Auto-detect environment
if IS_STREAMLIT_CLOUD:
    # Running on Streamlit Cloud
    print("✅ Running on Streamlit Cloud - Using HuggingFace")
    LLM_MODEL = "HuggingFace"
    HF_MODEL = os.getenv('HF_MODEL', 'microsoft/phi-2')  # Allow override via secrets
else:
    # Local development
    print("💻 Running locally - Using Ollama")
    LLM_MODEL = "qwen2.5-coder:7b"
    HF_MODEL = None  # Not used locally

# Retrieval settings
RETRIEVAL_K = 4
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2

print(f"📋 Config: LLM_MODEL={LLM_MODEL}, RETRIEVAL_K={RETRIEVAL_K}")

# Export HF_MODEL for use in other files
if IS_STREAMLIT_CLOUD:
    HF_MODEL = os.getenv('HF_MODEL', 'microsoft/phi-2')