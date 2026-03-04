# hardik_bot/vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import sys

# Import config from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HARDIK_CSV_PATH, HARDIK_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_K

# Global variable to cache retriever
_retriever = None

def get_retriever():
    """Get or create the retriever (cached)"""
    global _retriever
    
    if _retriever is not None:
        return _retriever
    
    print("="*50)
    print("Starting Hardik's Bot Vector Store Setup")
    print("="*50)
    
    # Check if CSV exists
    if not os.path.exists(HARDIK_CSV_PATH):
        print(f"ERROR: CSV file not found at {HARDIK_CSV_PATH}")
        # Return empty retriever
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=HARDIK_PERSIST_DIR
        )
        _retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        return _retriever
    
    # Read CSV
    print(f"Reading CSV from: {HARDIK_CSV_PATH}")
    df = pd.read_csv(HARDIK_CSV_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Check if we need to add documents
    add_documents = not os.path.exists(HARDIK_PERSIST_DIR)
    
    if add_documents:
        print("Creating new vector store...")
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            # Combine title and content for search
            content = f"{row['title']} {row['content']}"
            
            document = Document(
                page_content=content,
                metadata={
                    "title": row['title'],
                    "category": row['category'],
                    "tags": row['tags']
                },
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
        
        print(f"Created {len(documents)} documents")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=HARDIK_PERSIST_DIR,
            ids=ids
        )
        print("Vector store created successfully!")
    else:
        print("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=HARDIK_PERSIST_DIR,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully!")
    
    _retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    print(f"Retriever created with k={RETRIEVAL_K}")
    print("="*50)
    
    return _retriever