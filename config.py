# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gagan's Bot paths
GAGAN_DOCS_DIR = os.path.join(BASE_DIR, "gagan_bot", "policy_docs")
GAGAN_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Hardik's Bot paths  
HARDIK_DOCS_DIR = os.path.join(BASE_DIR, "hardik_bot")
HARDIK_CSV_PATH = os.path.join(HARDIK_DOCS_DIR, "deepseek_csv_20260302_c82ca1.csv")
HARDIK_PERSIST_DIR = os.path.join(HARDIK_DOCS_DIR, "chrome_langchain_db")

# Embedding models - DIFFERENT for each bot
GAGAN_EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # For Gagan's bot (HuggingFace)
HARDIK_EMBEDDING_MODEL = "mxbai-embed-large"    # For Hardik's bot (Ollama)

# AWS Configuration
USE_AWS = os.getenv('USE_AWS', 'false').lower() == 'true'
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')

# LLM Selection
if USE_AWS:
    LLM_MODEL = "bedrock"
    print(f"✅ Using AWS Bedrock with model: {BEDROCK_MODEL_ID}")
else:
    LLM_MODEL = "qwen2.5-coder:7b"
    print(f"💻 Using local Ollama with model: {LLM_MODEL}")

# Retrieval settings
RETRIEVAL_K = 4
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2