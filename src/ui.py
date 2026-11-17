
import streamlit as st
import re
from src.app import run_query

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🐙"
)

# Состояние приложения
if "messages" not in st.session_state:
    st.session_state.messages = []

if "cthulhu_mode" not in st.session_state:
    st.session_state.cthulhu_mode = False


# Обработка секретных команд

import re

def process_chat_input():
    raw_query = st.session_state.get("chat_query", "")

    if not raw_query:
        return

    command = raw_query.strip()

    if re.match(r"^ктулху\b", command, flags=re.IGNORECASE):
        st.session_state.cthulhu_mode = True

        model_query = re.sub(
            r"^ктулху\b",
            "",
            command,
            count=1,
            flags=re.IGNORECASE
        ).lstrip(" ,:;!?—-")

        st.session_state.pending_query = model_query
        st.session_state.pending_raw_query = raw_query
        st.session_state.pending_secret_message = (
            "⚠️ ОБНАРУЖЕНА АНОМАЛИЯ В ЯЗЫКОВОЙ МОДЕЛИ\n\n"
            "Древний разум пробудился.\n\n"
            "Ваш запрос очень важен для нас. "
            "К сожалению, нам совершенно безразлично."
        )

    elif command.casefold() == "изыди":
        st.session_state.cthulhu_mode = False
        st.session_state.pending_query = ""
        st.session_state.pending_raw_query = raw_query
        st.session_state.pending_secret_message = (
            "🐙 Древнее зло изгнано. "
            "Справочник снова к вашим услугам!"
        )

    else:
        st.session_state.pending_query = raw_query
        st.session_state.pending_raw_query = raw_query
        st.session_state.pending_secret_message = None


# def handle_secret_command(query):
#     command = query.strip()

#     # Пробуждение
#     match = re.match(
#         r"^ктулху\b[\s,!:—-]*(.*)$",
#         command,
#         flags=re.IGNORECASE | re.DOTALL
#     )

#     if match and not st.session_state.cthulhu_mode:
#         st.session_state.cthulhu_mode = True

#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": (
#                 "⚠️ ОБНАРУЖЕНА АНОМАЛИЯ В ЯЗЫКОВОЙ МОДЕЛИ\n\n"
#                 "Древний разум пробудился.\n\n"
#                 "Ваш запрос очень важен для нас. "
#                 "К сожалению, нам совершенно безразлично."
#             )
#         })

#         # Сохраняем вопрос после обращения
#         question = match.group(1).strip()

#         if question:
#             st.session_state.pending_query = question

#         return True

    # Изгнание
    # if command.lower() == "изыди":
    #     st.session_state.cthulhu_mode = False

    #     st.session_state.messages.append({
    #         "role": "assistant",
    #         "content": (
    #             "🐙 Древнее зло изгнано. "
    #             "Справочник снова к вашим услугам!"
    #         )
    #     })

    #     return True

    # return False



# Оформление режима Ктулху

if st.session_state.cthulhu_mode:
    st.markdown("""
    <style>
    /* Основной фон */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #101C17 !important;
        color: #B1D2A8 !important;
    }

    /* Заголовки и обычный текст */
    .stApp h1,
    .stApp p {
        color: #B1D2A8;
    }

    /* Элементы списков */
    .stApp li,
    .stApp li p,
    .stApp li span {
        color: #B1D2A8 !important;
    }

    /* Сообщения чата */
    [data-testid="stChatMessage"] {
        background-color: #17291F;
        border: 1px solid #355444;
        border-radius: 12px;
    }

    /* Нижняя панель */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: #101C17 !important;
    }

    /* Само поле ввода */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        background-color: #17291F !important;
        border-color: #355444 !important;
    }

    /* Текст ввода и подсказка */
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #D4E5D0 !important;
        caret-color: #D4E5D0 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8FAF99 !important;
        opacity: 1;
    }

    /* Кнопка отправки */
    [data-testid="stChatInput"] button {
        background-color: #355444 !important;
        border-color: #355444 !important;
        color: #D4E5D0 !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #456D54 !important;
    }

    /* Верхняя панель */
    [data-testid="stHeader"] {
        background: #101C17 !important;
    }
    
    @font-face {
        font-family: "CthulhuRunes";
        src: url("/app/static/CthulhuRunes-Regular.woff2") format("woff2");
        font-weight: normal;
        font-style: normal;
    }
    
    .cthulhu-title {
        font-family: "CthulhuRunes", serif !important;
        font-size: 48px;
        color: #B1D2A8 !important;
        text-align: center;
        letter-spacing: 4px;
    }
    
    .cthulhu-abyss {
        font-family: "CthulhuRunes", serif !important;
        font-size: 24px;
        color: #789780 !important;
        text-align: center;
        letter-spacing: 3px;
        margin-top: 20px;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)



# Заголовок
if st.session_state.cthulhu_mode:
    st.markdown(
        '<h1 class="cthulhu-title">Ктулху пробудился</h1>',
        unsafe_allow_html=True
    )
    st.caption("Р'льехский справочник запретных знаний")
else:
    st.title("🐙 Agentic RAG")
    st.caption("Справочник ML-инженера")


# 6. Обработка ожидающего запроса

raw_query = st.session_state.pop("pending_raw_query", None)
model_query = st.session_state.pop("pending_query", None)
secret_message = st.session_state.pop(
    "pending_secret_message", None
)

if raw_query:
    st.session_state.messages.append({
        "role": "user",
        "content": raw_query
    })

    with st.chat_message("user"):
        st.markdown(raw_query)

    if secret_message:
        st.session_state.messages.append({
            "role": "assistant",
            "content": secret_message
        })

        with st.chat_message("assistant", avatar="🐙"):
            st.markdown(secret_message)

    if model_query and model_query.strip():
        with st.spinner(
            "Ктулху изучает смертного..."
            if st.session_state.cthulhu_mode
            else "Ктулху думает... 🐙"
        ):
            result = run_query(
                model_query,
                cthulhu_mode=st.session_state.cthulhu_mode
            )

        answer = result["answer"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant", avatar="🐙"):
            st.markdown(answer)

            # 7. Поле ввода

if st.session_state.cthulhu_mode:
    st.markdown(
        '<div class="cthulhu-abyss">бездна смотрит в тебя</div>',
        unsafe_allow_html=True
    )
                
st.chat_input(
    "Шепните древнему божеству..."
    if st.session_state.cthulhu_mode
    else "Задайте вопрос по ML",
    key="chat_query",
    on_submit=process_chat_input
)