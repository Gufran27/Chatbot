import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Portfolio Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("Portfolio Chatbot")
st.write("Ask questions about ASG Solutions")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.write(st.session_state.session_id)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_input
                    }
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = "Backend error. Please check FastAPI server."

            except requests.exceptions.ConnectionError:
                answer = "FastAPI backend is not running. Please start backend first."

            st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })