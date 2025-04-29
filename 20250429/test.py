import streamlit as st

st.title('안녕하세요')
st.write('hello')
st.markdown("""
<h2 style='color:blue;'>HTML & CSS 연습</h2>
<p style='background-color:green; padding:10px;'>Streamlit에서도 기본 스타일링 가능합니다.</p>
""", unsafe_allow_html=True)
st.markdown("""
<h2 style="color:navy;">Streamlit HTML 기본 태그 예시</h2>
<p>HTML에서 가장 많이 쓰이는 <strong>텍스트 태그</strong>와 <strong>리스트 태그</strong> 예시입니다.</p>
<ul>
 <li> Python</li>
 <li> Streamlit</li>
 <li> HTML & CSS</li>
</ul>
<p>자세한 내용은 <a href="https://streamlit.io" target="_blank">Streamlit 공식 사이트</a>를 참고하세요.</p>
""", unsafe_allow_html=True)
st.markdown("""
<h2 style="color:green;">프로필 정보</h2>
<table style="width:50%; border-collapse: collapse;" border="1">
 <tr>
 <th>항목</th>
 <th>내용</th>
 </tr>
 <tr>
 <td>이름</td>
 <td>홍길동</td>
 </tr>
 <tr>
 <td>직업</td>
 <td>데이터 분석가</td>
 </tr>
</table>
""", unsafe_allow_html=True)
st.markdown("""
<h2 style="color:darkblue; font-size:30px;"> Streamlit에서 HTML + CSS</h2>
<p style="color:gray; font-size:18px;">이 문장은 <span style="color:red;">빨간색</span> 단어와 함께 표시됩니다.</p>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd

# HTML과 CSS를 사용한 상단 배너
st.markdown(
    """
    <div style='background-color: #0e1117; padding: 20px; border-radius: 10px;'>
        <h1 style='text-align: center; color: white;'>Welcome to Streamlit</h1>
        <p style='text-align: center; color: lightgray;'>HTML과 CSS를 활용한 상단 배너</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 데이터프레임 생성
data = {
    "과목": ["Python", "HTML/CSS"],
    "점수": [95, 90]
}
df = pd.DataFrame(data)

# 표 스타일 설정
st.markdown(
    """
    <style>
    .custom-table {
        background-color: #0e1117;
        color: white;
        border-collapse: collapse;
        width: 100%;
    }
    .custom-table th {
        background-color: #28a745;
        color: white;
        padding: 8px;
    }
    .custom-table td {
        padding: 8px;
        border-top: 1px solid #222;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 표 출력
st.markdown('<table class="custom-table">'
            '<tr><th>과목</th><th>점수</th></tr>' +
            ''.join(f'<tr><td>{row["과목"]}</td><td>{row["점수"]}</td></tr>' for _, row in df.iterrows()) +
            '</table>', unsafe_allow_html=True)

st.markdown('# 제목1')
st.markdown('## 제목2 :cat:')
st.markdown('**굵은 텍스트** 와 *기울 텍스트*를 같이 써봅시다.')
st.markdown("[구글로 이동](https://google.com)")
st.markdown('### 순서없는 리스트')
st.markdown("""
    - 사과
    - 바나나
    - 체리            
""")

st.markdown('### :red[순서있는 리스트] ')
st.markdown("""
    1. 사과
    2. 바나나
    3. 체리            
""")

st.markdown("> 이건 인용문 입니다.")
st.markdown("### 코드 블록")
st.markdown("""
```
def say_hello():
    print("Hello Streamlit")
```
""")

st.code("""
def greet():
    print("Hi Streamlit!")
""")
st.markdown("`print()` 함수는 출력을 위한 Python 함수 입니다")

# 컬러 텍스트
st.markdown(":red[빨간 텍스트] :blue[파란 텍스트] :green[초록 텍스트]")
# 배경 강조
st.markdown(":blue-background[파란 배경 강조 텍스트]")
st.markdown(":orange-background[주황 배경 강조 텍스트]")
# 이모지
st.markdown("이모지 테스트: :rocket: :smile: :tulip: :fire:")
# 수식 (LaTeX)
st.markdown("수식 예시: $\\sqrt{x^2 + y^2}$")
# 줄바꿈 (공백 두 칸 후 줄바꿈)
st.markdown("이 문장은 줄 바꿈을 합니다. \n이 줄은 아래로 내려옵니다.")

st.markdown("### Soft Return 예시(줄 끝에 공백 2칸)")
st.markdown("이 줄은 줄 끝에 공백 두 칸이 있습니다.  \n다음줄로 내려옵니다.")
st.markdown("### Sofr Return 예시 (역슬래시 n)")
st.markdown("이 줄은 역슬래시 n 사옹\n줄바꿈이 적용됩니다.")

st.markdown("### Hard Return 예시(문단 바꿈)")
st.markdown("""
            문단1입니다
            
            문단 2 입니다. 완전히 다른문단으로 처리됩니다
""")

st.write("Hello, *world!* :sunglasses:")
"magic"
"hello, Streamlit"
123+456

st.write(1234)
st.write(
    pd.DataFrame(
        {
            "first column":[1,2,3,4],
            "second cloumn":[10,20,30,40],
        }
    )
)