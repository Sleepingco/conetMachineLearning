import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import datetime
st.write("1번 과제 -")
age = tuple(np.arange(0,100))
option = st.selectbox(
    "나이를 선택하세요",
    age
)
if st.button("입력"):
    if option<20:
        st.write("청소년 입니다:", option)
    else:
        st.write("성인 입니다",option)

st.write("2번 과제 -")

weight_input = st.text_input("몸무게를 입력하세요", placeholder="kg단위")
height_input = st.text_input("키를 입력하세요", placeholder="cm단위")

if weight_input and height_input:
    weight = int(weight_input)
    height = int(height_input)
    height = height/100
else:
    st.warning("몸무게와 키를 모두 입력해주세요.")
if st.button("계산하기"):
    bmi = weight / (height ** 2)
    st.write(f"BMI: {bmi:.2f}")
   
st.write("3번 과제 -")
if 'answer' not in st.session_state:
    st.session_state.answer = np.random.randint(1, 101)
guessNumber = st.text_input("1~100 사이의 숫자를 맞춰보세요",value="0")
# st.write(f"정답은: {st.seㄴssion_state.answer}")
if guessNumber:
    guessNumber = int(guessNumber)
    answer = st.session_state.answer
    if guessNumber > answer:
        st.warning(f"숫자가 커요 {guessNumber}")
    elif guessNumber < answer:
        st.error(f"숫자가 작아요 {guessNumber}")
    else:
        st.write('정답입니다')
        st.balloons()

st.write("4번 과제 -")
# 초기화
if "todos" not in st.session_state:
    st.session_state.todos = []

# 새 할 일 입력
new_task = st.text_input("새 할 일 입력")

if st.button("추가하기") and new_task:
    st.session_state.todos.append({"task": new_task, "done": False})

# 할 일 목록 출력 및 인터랙션
st.subheader("현재 할 일 목록")
for i, todo in enumerate(st.session_state.todos):
    col1, col2, col3 = st.columns([0.05, 0.8, 0.15])

    # 체크박스로 완료 여부 토글
    done = col1.checkbox("", value=todo["done"], key=f"done_{i}")
    st.session_state.todos[i]["done"] = done

    # 텍스트 표시 (완료되었으면 취소선)
    if done:
        col2.markdown(f"~~{todo['task']}~~")
    else:
        col2.markdown(todo["task"])

    # 삭제 버튼
    if col3.button("dels", key=f"delete_{i}"):
        st.session_state.todos.pop(i)
        st.experimental_rerun()

