# rag_pipeline.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import shutil

from config import GAGAN_EMBEDDING_MODEL, LLM_MODEL, GAGAN_PERSIST_DIR, RETRIEVAL_K, USE_AWS, BEDROCK_MODEL_ID

# Import Bedrock only if using AWS
if USE_AWS:
    from bedrock_llm import BedrockLLM

from gagan_bot.document_loader import load_documents_with_lines

print("="*50)
print("Starting RAG Pipeline Setup")
print("="*50)

# ---------------------------
# 1. Initialize embedding model
# ---------------------------
print("1. Initializing embedding model...")
from config import GAGAN_EMBEDDING_MODEL
embeddings = HuggingFaceEmbeddings(model_name=GAGAN_EMBEDDING_MODEL)

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
if os.path.exists(GAGAN_PERSIST_DIR):  # Changed from PERSIST_DIR
    print("   Removing existing vector store...")
    shutil.rmtree(GAGAN_PERSIST_DIR)  # Changed from PERSIST_DIR

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=GAGAN_PERSIST_DIR  # Changed from PERSIST_DIR
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
# 5. Initialize LLM - Choose based on config
# ---------------------------
print("5. Initializing LLM...")

if USE_AWS:
    print(f"   Using AWS Bedrock with model: {BEDROCK_MODEL_ID}")
    llm = BedrockLLM()
else:
    print(f"   Using local Ollama with model: {LLM_MODEL}")
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
    
    # Generate response (works same for both Ollama and Bedrock)
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
print("RAG pipeline with memory ready!")
print("="*50)

# ---------------------------
# 7. Quick test
# ---------------------------
print("\nTesting retrieval for 'sick leave':")
test_results = vectorstore.similarity_search("sick leave", k=2)
print(f"Found {len(test_results)} results")
for i, doc in enumerate(test_results):
    print(f"  Result {i+1}: {doc.metadata['source']} (lines {doc.metadata['start_line']}-{doc.metadata['end_line']})")
    print(f"  Preview: {doc.page_content[:100].replace(chr(10), ' ')}...")
print("="*50)