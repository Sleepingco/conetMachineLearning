import streamlit as st
st.title("💬 Streamlit Chat 예시")
# 세션 상태에 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = []
# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# 입력창
user_input = st.chat_input("메시지를 입력하세요")
# 입력 처리
if user_input:
# 사용자의 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
# 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)
# AI 응답 예시
ai_response = f"당신이 보낸 메시지: '{user_input}'를 받았습니다."
st.session_state.messages.append({"role": "assistant", "content": ai_response})
# AI 응답 표시
with st.chat_message("assistant"):
    st.markdown(ai_response)