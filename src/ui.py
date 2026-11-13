import streamlit as st
from src.app import run_query

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🐙"
)

st.title("🐙 Agentic RAG")
st.caption("Справочник ML-инженера")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🐙" if message["role"] == "assistant" else None

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

query = st.chat_input("Задайте вопрос по ML")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Ктулху думает... 🐙"):
        result = run_query(query)

    answer = result["answer"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant", avatar="🐙"):
        st.markdown(answer)