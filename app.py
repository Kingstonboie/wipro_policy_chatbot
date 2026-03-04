# app.py
import streamlit as st
import uuid

# Import both bots
from gagan_bot.rag_pipeline import rag_with_history as gagan_rag
from gagan_bot.rag_pipeline import get_session_history as gagan_get_session
from hardik_bot.main import ask_hardik_bot

st.set_page_config(page_title="AWS & Wipro Policy Chatbot", page_icon="🤖")
st.title("Multi-Bot Chatbot Platform")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "bot_choice" not in st.session_state:
    st.session_state.bot_choice = None

# Bot selection screen
if st.session_state.bot_choice is None:
    st.markdown("## 👋 Welcome! Please select a bot:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Gagan's Bot")
        st.markdown("**Wipro Policy Expert**")
        st.markdown("- Answers questions about Wipro policies")
        st.markdown("- Uses sample policy documents")
        st.markdown("- Provides citations from source documents")
        if st.button("Select Gagan's Bot", use_container_width=True):
            st.session_state.bot_choice = "gagan"
            st.rerun()
    
    with col2:
        st.markdown("###  Hardik's Bot")
        st.markdown("**AWS Service Expert**")
        st.markdown("- Answers questions about AWS services")
        st.markdown("- Uses CSV database of AWS services")
        st.markdown("- Covers AWS-Wipro partnership info")
        if st.button("Select Hardik's Bot", use_container_width=True):
            st.session_state.bot_choice = "hardik"
            st.rerun()
    
    st.stop()

# Display current bot and option to switch
bot_name = "Gagan's Bot (Wipro Policies)" if st.session_state.bot_choice == "gagan" else "Hardik's Bot (AWS Services)"
st.sidebar.title(f"Current: {bot_name}")
if st.sidebar.button("🔄 Switch Bot"):
    st.session_state.bot_choice = None
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask your question...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response based on bot choice
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                if st.session_state.bot_choice == "gagan":
                    response = gagan_rag(prompt, st.session_state.session_id)
                else:
                    response = ask_hardik_bot(prompt, st.session_state.session_id)
            except Exception as e:
                response = f"Error: {str(e)}"
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})