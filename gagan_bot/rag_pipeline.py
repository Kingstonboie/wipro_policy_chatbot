# gagan_bot/rag_pipeline.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import shutil

# Import config from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GAGAN_DOCS_DIR, GAGAN_PERSIST_DIR, RETRIEVAL_K, LLM_MODEL

from gagan_bot.document_loader import load_documents_with_lines

print("="*50)
print("Starting Gagan's Bot RAG Pipeline")
print("="*50)

# ---------------------------
# 1. Initialize embedding model
# ---------------------------
print("1. Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------
# 2. Load documents and create vector store
# ---------------------------
print("2. Loading documents...")
documents = load_documents_with_lines()
print(f"   Loaded {len(documents)} document chunks")

if len(documents) == 0:
    print("   ERROR: No documents loaded! Check policy_docs folder.")
    # Don't exit, just continue with empty store

# Show unique sources
if documents:
    sources = set([doc.metadata['source'] for doc in documents])
    print(f"   Documents found: {sources}")

print("3. Creating vector store...")
# Delete existing store if it exists to force rebuild
if os.path.exists(GAGAN_PERSIST_DIR):
    print("   Removing existing vector store...")
    shutil.rmtree(GAGAN_PERSIST_DIR)

if documents:
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=GAGAN_PERSIST_DIR
    )
    print("   Vector store created successfully!")
else:
    # Create empty vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=GAGAN_PERSIST_DIR
    )
    print("   Empty vector store created!")

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
        source = doc.metadata.get("source", "Unknown")
        start = doc.metadata.get("start_line", 0)
        end = doc.metadata.get("end_line", 0)
        formatted.append(
            f"[From {source} lines {start}-{end}]:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# ---------------------------
# 4. Define prompt template with chat history
# ---------------------------
prompt_template = """
You are an AI assistant for Wipro policies. Use the following context and chat history to answer the question.
If you cannot find the answer in the context, say "I don't have information on that."
Always cite the source document and line numbers in your answer using the format: from "DocumentName" Line X-Y, Line A-B.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer (with citations):"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["chat_history", "context", "question"]
)

# ---------------------------
# 5. Initialize LLM (using shared model)
# ---------------------------
print("5. Initializing LLM...")
llm = OllamaLLM(model=LLM_MODEL)

# ---------------------------
# 6. Create a function that includes history in the context
# ---------------------------
print("6. Building RAG chain with memory...")

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

def rag_with_history(question, session_id):
    # Get chat history
    chat_history = get_session_history(session_id)
    history_str = format_chat_history(chat_history.messages)
    
    # Retrieve relevant docs
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    # Generate response
    response = llm.invoke(PROMPT.format(
        chat_history=history_str,
        context=context,
        question=question
    ))
    
    # Add to history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    return response

print("="*50)
print("Gagan's Bot RAG pipeline ready!")
print("="*50)