import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "you are a good friend really kind and smart as professor. If your client asks questions, teach them simply."}
    ]
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "responded" not in st.session_state:
    st.session_state.responded = False

# UI 제목
st.title("OpenAI 챗봇")
st.write("챗봇을 시작합니다. 메시지를 입력하세요.")

# 사용자 입력 받기
user_input = st.text_input("사용자:", key="user_input")

# 종료 버튼
if st.button("종료"):
    st.write("챗봇 종료.")
    st.stop()

# 사용자 입력 변화 감지 → responded 상태 초기화
if user_input != st.session_state.last_input:
    st.session_state.responded = False
    st.session_state.last_input = user_input

# 대화 출력
st.markdown("---")
st.subheader("대화 내용")
for message in st.session_state.chat_history:
    if message["role"] != "system":
        st.write(f"{message['role']}: {message['content']}")

# 입력 처리
if user_input and not st.session_state.responded:
    if user_input.lower() in ["종료", "exit", "quit"]:
        st.write("챗봇 종료.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.chat_history,
            stream=True
        )
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.stop()

    assistant_reply = ""
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            assistant_reply += chunk_message

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.responded = True
    st.rerun()

# 전체 대화 내용 보기
if st.button("전체 대화 내용 보기"):
    st.write("대화 내용:")
    for chat in st.session_state.chat_history:
        st.write(f"{chat['role']}: {chat['content']}")
