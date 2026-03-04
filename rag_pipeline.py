# rag_pipeline.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import shutil
import requests
import json

from config import EMBEDDING_MODEL, LLM_MODEL, PERSIST_DIR, RETRIEVAL_K, HF_MODEL
from document_loader import load_documents_with_lines

print("="*50)
print("Starting RAG Pipeline Setup")
print("="*50)

# ---------------------------
# 1. Initialize embedding model
# ---------------------------
print("1. Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------------------
# 2. Load documents and create vector store
# ---------------------------
print("2. Loading documents...")
documents = load_documents_with_lines()
print(f"   Loaded {len(documents)} document chunks")

if len(documents) == 0:
    print("   ERROR: No documents loaded! Check policy_docs folder.")
    exit()

# Show unique sources
sources = set([doc.metadata['source'] for doc in documents])
print(f"   Documents found: {sources}")

print("3. Creating vector store...")
# Delete existing store if it exists to force rebuild
if os.path.exists(PERSIST_DIR):
    print("   Removing existing vector store...")
    shutil.rmtree(PERSIST_DIR)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
print("   Vector store created successfully!")

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
print(f"4. Retriever created with k={RETRIEVAL_K}")

# ---------------------------
# 3. Format retrieved documents
# ---------------------------
def format_docs(docs):
    if not docs:
        return ""
    
    formatted = []
    for doc in docs:
        source = doc.metadata["source"]
        start = doc.metadata["start_line"]
        end = doc.metadata["end_line"]
        formatted.append(
            f"[From {source} lines {start}-{end}]:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# ---------------------------
# 4. Define prompt template
# ---------------------------
prompt_template = """
You are an AI assistant for Wipro policies. Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have information on that."
Always cite the source document and line numbers in your answer using the format: from "DocumentName" Line X-Y, Line A-B.

Context:
{context}

Question: {question}

Answer (with citations):"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# ---------------------------
# 5. LLM Initialization - Choose based on environment
# ---------------------------
print(f"5. Initializing LLM with model: {LLM_MODEL}")

def call_huggingface_api(prompt):
    """Call HuggingFace's free inference API"""
    API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN', '')}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)
    except Exception as e:
        print(f"Error calling HuggingFace API: {e}")
        return f"Error: {str(e)}"

def call_ollama(prompt):
    """Call local Ollama instance"""
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=LLM_MODEL)
    return llm.invoke(prompt)

# Choose the appropriate LLM function based on environment
if LLM_MODEL == "HuggingFace":
    print(f"   Using HuggingFace model: {HF_MODEL}")
    llm_func = call_huggingface_api
else:
    print("   Using local Ollama")
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=LLM_MODEL)
    llm_func = llm.invoke

# ---------------------------
# 6. Set up message history for conversation memory
# ---------------------------
print("6. Setting up conversation memory...")

# Store for session histories
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def format_chat_history(messages):
    """Convert chat messages to string format"""
    formatted = []
    for msg in messages:
        if msg.type == "human":
            formatted.append(f"Human: {msg.content}")
        elif msg.type == "ai":
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

# ---------------------------
# 7. Custom chain that includes history
# ---------------------------
def rag_with_history(question, session_id):
    # Get chat history
    chat_history = get_session_history(session_id)
    history_str = format_chat_history(chat_history.messages)
    
    # Retrieve relevant docs
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    # Build the full prompt with history
    full_prompt = f"""You are an AI assistant for Wipro policies. Use the following context and chat history to answer the question.
If you cannot find the answer in the context, say "I don't have information on that."
Always cite the source document and line numbers in your answer using the format: from "DocumentName" Line X-Y, Line A-B.

Chat History:
{history_str}

Context:
{context}

Question: {question}

Answer (with citations):"""
    
    # Generate response using the appropriate LLM function
    if LLM_MODEL == "HuggingFace":
        response = call_huggingface_api(full_prompt)
        # Clean up response if needed (HuggingFace sometimes returns the prompt + answer)
        if full_prompt in response:
            response = response.replace(full_prompt, "").strip()
    else:
        response = llm.invoke(full_prompt)
    
    # Add to history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    return response

print("="*50)
print("RAG pipeline with memory ready!")
print("="*50)

# ---------------------------
# 8. Quick test
# ---------------------------
print("\nTesting retrieval for 'sick leave':")
test_results = vectorstore.similarity_search("sick leave", k=2)
print(f"Found {len(test_results)} results")
for i, doc in enumerate(test_results):
    print(f"  Result {i+1}: {doc.metadata['source']} (lines {doc.metadata['start_line']}-{doc.metadata['end_line']})")
    print(f"  Preview: {doc.page_content[:100].replace(chr(10), ' ')}...")
print("="*50)