# app.py
import streamlit as st
from rag_pipeline import rag_with_history, get_session_history
import uuid

st.set_page_config(page_title="Wipro Policy Chatbot", page_icon="💼")
st.title("📄 Wipro Policy Chatbot (Prototype)")
st.caption("This chatbot uses **SAMPLE / FAKE** policy documents. Not actual Wipro data.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Create a unique session ID for this browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask about Wipro policies...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                # Use our custom function with memory
                response = rag_with_history(prompt, st.session_state.session_id)
            except Exception as e:
                response = f"Error: {str(e)}"
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})