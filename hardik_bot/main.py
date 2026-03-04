# hardik_bot/main.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import sys

# Import config from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL
from hardik_bot.vector import get_retriever

# Initialize model with shared LLM
model = OllamaLLM(model=LLM_MODEL)

template = """
You are an expert in AWS field.

Here are the relevant documents: {answers}

Here is the question to answer: {question}

Please provide a helpful answer based on the documents above.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

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

def ask_hardik_bot(question, session_id):
    """Main function to be called from app.py"""
    # Get chat history
    chat_history = get_session_history(session_id)
    history_str = format_chat_history(chat_history.messages)
    
    # Get retriever and search for relevant docs
    retriever = get_retriever()
    answers = retriever.invoke(question)
    
    # Format answers as text
    answers_text = "\n\n".join([doc.page_content for doc in answers])
    
    # Include history in the prompt
    enhanced_question = f"Chat History:\n{history_str}\n\nCurrent Question: {question}"
    
    # Generate response
    result = chain.invoke({"answers": answers_text, "question": enhanced_question})
    
    # Add to history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(result)
    
    return result