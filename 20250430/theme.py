import streamlit as st
# 타이틀 설정
st.title("📊Streamlit 기본 웹 앱")
# 텍스트 입력
name = st.text_input("이름을 입력하세요:")
# 버튼 클릭 시 동작
if st.button("인사하기"):
st.write(f"안녕하세요, {name}님! 😊")
# 슬라이더로 숫자 조정
number = st.slider("숫자(나이)를 선택하세요:", 0, 100, 50)
st.write(f"선택한 숫자(나이): {number}")
# =============================
# [1] 페이지 타이틀 설정
# =============================
# st.title()은 페이지 최상단에 가장 큰 제목을 표시합니다.
st.title("🌟Streamlit UI 요소 데모")
# 이모지는 :emoji_code: 형태로 넣을 수 있으며,
# 제목 텍스트 내에서도 삽입 가능합니다.
st.title("고양이도 코딩을 합니다 :cat:")
# =============================
# [2] 헤더와 서브헤더
# =============================
# st.header()는 섹션을 구분할 때 유용합니다.
st.header("📌헤더 영역입니다")
# st.subheader()는 header보다 작은 제목입니다.
st.subheader("이곳은 서브헤더입니다")
# =============================
# [3] 캡션 표시
# =============================
# st.caption()은 보통 부연 설명이나 메타 정보에 사용됩니다.
st.caption("작성자: Streamlit 연습생 | 날짜: 2025-03-25")
# =============================
# [4] 코드 블록 출력
# =============================
# Python 코드를 문자열로 정의하고, 코드 하이라이팅을 함께 적용
st.caption("아래는 Python 코드 예시입니다.")
example_code = '''
def calculate_area(radius):
pi = 3.14159
return pi * (radius 2)
print(calculate_area(5))
'''
st.code(example_code, language='python')

# =============================
# [5] 일반 텍스트 출력
# =============================
st.text("이것은 일반 텍스트입니다. 마크다운도, 스타일도 적용되지
않습니다.")
# =============================
# [6] 마크다운(Markdown) 문법
# =============================
# Streamlit에서는 마크다운 문법을 지원합니다. 텍스트 강조나 리스트
등을 작성할 수 있습니다.
st.markdown("Streamlit은 마크다운 문법을 지원 하며, _기울임_,
굵게 , `코드`도 가능합니다.")
st.markdown("- 리스트 항목 1\n- 리스트 항목 2\n- 리스트 항목 3")
# 컬러 텍스트 예시
st.markdown("여기서는 :red[빨간색], :green[초록색], :blue[파란색]
텍스트를 보여줍니다.")
# 이모지 + 마크다운 같이 쓰기
st.markdown("✔️이것은 체크 표시 이모지입니다")
# =============================
# [7] 수학 수식 표현 (LaTeX)
# =============================
# 수식을 표현할 때는 st.latex() 또는 마크다운에서 :color[$...$] 문법을 사용할
수 있습니다.
# 라텍 문법은 TeX 수학 표기법을 따릅니다.
# 일반 LaTeX 수식 출력
st.latex(r"E = mc^2")
# 마크다운에 포함된 색상+수식
st.markdown("삼각함수 항등식: :blue[$\\sin^2 \\theta + \\cos^2 \\theta = 1$]")
# 어려운 수식 설명을 위한 확장 영역
with st.expander("📘 수식 설명 보기"):
st.markdown("""
위 식은 삼각함수의 항등식 으로, 단위원 상에서 각도 θ에 대한
`사인과 코사인의 제곱의 합은 항상 1`이라는 성질을 말합니다.
다양한 수학적 증명과 물리 공식의 기본 단위로 자주 등장합니다.
""")
# =============================
# [8] 하단 캡션 및 크레딧
# =============================
st.caption("ⓒ 2025 Streamlit 예제 데모 - powered by ChatGPT")

# ========================================
# [1] 페이지 제목
# ========================================
st.title('📊데이터프레임 튜토리얼')
# ========================================
# [2] DataFrame 생성
# ========================================
# 예시: 가상의 학생 시험 점수 데이터
df_scores = pd.DataFrame({
'이름': ['홍길동', '김철수', '이영희', '박민수'],
'수학': [85, 90, 78, 92],
'영어': [88, 75, 95, 80]
})
# ✅st.dataframe(): 스크롤 가능하고 정렬 가능한 인터랙티브 테이블
# use_container_width=True → 화면 너비에 맞춰 테이블 확장
st.subheader("📌데이터프레임 출력 (인터랙티브)")
st.dataframe(df_scores, use_container_width=True)
# ========================================
# [3] 정적 테이블 출력
# ========================================
# ✅st.table(): 정적인 테이블 (정렬/스크롤 없음)
st.subheader("📌테이블 출력 (정적)")
st.table(df_scores)
# ========================================
# [4] 지표(metric) 출력
# ========================================
# ✅st.metric(label, value, delta): KPI나 숫자 변화 추이를 보여줄 때 사용
st.subheader("📌단일 메트릭 예시")
st.metric(label="📈 주간 평균 점수", value="84점", delta="2점 증가")
st.metric(label="🔥 수학 최고 점수", value="92점", delta="+4점")
# ========================================
# [5] 컬럼(Column)을 이용한 메트릭 정렬
# ========================================
# ✅st.columns(n): 여러 개의 컬럼으로 화면 분할
st.subheader("📌과목별 점수 변화")
# 세 개의 컬럼 생성
col1, col2, col3 = st.columns(3)
# 각 컬럼에 개별 메트릭 출력
col1.metric(label="수학 평균", value="86.3", delta="+2.1")
col2.metric(label="영어 평균", value="84.5", delta="-1.0")
col3.metric(label="전체 평균", value="85.4", delta="+0.6")

import streamlit as st
import pandas as pd
from datetime import datetime as dt
import datetime
# ========================================
# [1] 텍스트 입력 (여행지)
# ========================================
title = st.text_input(
label='✈️가고 싶은 여행지가 있나요?',
placeholder='예: 몰디브, 도쿄, 아이슬란드...'
)
if title:
st.write(f'당신이 선택한 여행지: :violet[{title}]')
# ========================================
# [2] 숫자 입력 (나이)
# ========================================
age = st.number_input(
label='🎂나이를 입력해 주세요.',
min_value=10,
max_value=100,
value=30,
step=1
)
st.write('당신이 입력하신 나이는: ', age)
# ========================================
# [3] 라디오 버튼 (커피 취향)
# ========================================
coffee = st.radio(
'☕어떤 커피를 좋아하시나요?',
('아메리카노', '라떼', '선택 안 함')
)
if coffee == '아메리카노':
st.write('시원한 :blue[아메리카노] 준비 중...')
elif coffee == '라떼':
st.write('부드러운 :orange[라떼] 추천드려요!')
else:
st.write('나중에 원하실 때 말씀해주세요 :relieved:')
# ========================================
# [4] 셀렉트 박스 (영화 장르)
# ========================================
genre = st.selectbox(
'🎬 좋아하는 영화 장르는 무엇인가요?',
['액션', '코미디', '다큐멘터리', '선택 안 함'],
index=3
)
if genre != '선택 안 함':
st.write(f'당신은 :green[{genre}] 장르를 좋아하시는군요!')
# ========================================
# [5] 멀티셀렉트 (과일 선택)
# ========================================
fruits = st.multiselect(
'🍓 좋아하는 과일을 모두 골라주세요!',
['딸기', '수박', '복숭아', '포도'],
default=['딸기', '포도']
)
st.write(f'🍇선택하신 과일: :violet[{fruits}]')
# ========================================
# [6] 숫자 범위 슬라이더
# ========================================
price_range = st.slider(
'💰 원하는 가격대 (만원)',
min_value=0.0,
max_value=100.0,
value=(30.0, 70.0)
)
st.write(f'선택한 가격 범위: {price_range[0]}만원 ~ {price_range[1]}만원')
# ========================================
# [7] 날짜/시간 슬라이더
# ========================================
start_time = st.slider(
"📅약속 가능한 시간 선택",
min_value=dt(2025, 3, 25, 9, 0),
max_value=dt(2025, 3, 25, 18, 0),
value=dt(2025, 3, 25, 13, 0),
step=datetime.timedelta(minutes=30),
format="HH:mm"
)
st.write("선택하신 시간:", start_time.strftime('%H:%M'))
# ========================================
# [8] 체크박스 (동의)
# ========================================
agree = st.checkbox('✅ 개인정보 제공에 동의합니다')
if agree:
st.info('감사합니다! 데이터를 보여드릴게요.')
# ========================================
# [9] 버튼 클릭
# ========================================
if st.button('📌이 버튼을 눌러보세요'):
st.success('✅ 버튼이 눌렸습니다!')
# ========================================
# [10] 데이터프레임 & 다운로드
# ========================================
# 예시 데이터 생성
df_movies = pd.DataFrame({
'영화 제목': ['파묘', '웡카', '듄2', '소풍'],
'관객 수(만)': [115, 87, 65, 30]
})
st.subheader("🎞️박스오피스 데이터")
st.dataframe(df_movies)
st.download_button(
label='⬇️영화 랭킹 CSV 다운로드',
data=df_movies.to_csv(index=False),
file_name='movie_rankings.csv',
mime='text/csv'
)
# ========================================
# [11] 메트릭 요약 정보
# ========================================
st.subheader("📊요약 지표")
col1, col2, col3 = st.columns(3)
col1.metric(label="평균 관객 수", value="74.25만", delta="-3.5만")
col2.metric(label="최고 관객 수", value="115만", delta="+10만")
col3.metric(label="예상 다음 주 증가율", value="5%", delta="↑ 1.2%")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ================================
# ✅ 한글 폰트 설정
# ================================
plt.rcParams['font.family'] = "NanumGothic" # Windows, Linux, macOS 공통
사용 가능
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
# ====================================
# 📊 예시 1: 데이터프레임 + 바 차트
# ====================================
st.title("학생 건강 정보 시각화")
# 학생 건강 정보 예시 데이터프레임 생성
data = pd.DataFrame({
'이름': ['민수', '지우', '하늘'],
'신장(cm)': [172, 165, 158],
'체중(kg)': [68.4, 55.2, 49.8]
})
st.subheader("📋 학생 건강 데이터")
st.dataframe(data, use_container_width=True)
# ====================================
# 🔹 matplotlib을 이용한 바 차트
# ====================================
st.subheader("📈신장 그래프 (matplotlib)")
fig, ax = plt.subplots()
ax.bar(data['이름'], data['신장(cm)'], color='skyblue')
ax.set_ylabel("신장 (cm)")
ax.set_title("학생별 신장 비교")
st.pyplot(fig)
# ====================================
# 🔹 seaborn을 이용한 바 차트 (색상 테마)
# ====================================
st.subheader("📊신장 그래프 (seaborn)")
fig, ax = plt.subplots()
sns.barplot(x='이름', y='신장(cm)', data=data, palette='pastel', ax=ax)
ax.set_title("학생별 신장 (seaborn)")
st.pyplot(fig)

# ====================================
# 📊 예시 2: 누적 막대그래프 + 오차
# ====================================
st.subheader("👨‍👩‍👧‍👦 그룹별 점수와 성별 누적 그래프")
# 가상의 그룹별 점수
labels = ['A조', 'B조', 'C조', 'D조', 'E조']
men_means = [70, 65, 80, 60, 75]
women_means = [75, 70, 60, 80, 70]
men_std = [5, 3, 6, 4, 5]
women_std = [4, 6, 5, 3, 4]
width = 0.5 # 막대 너비
fig, ax = plt.subplots()
# 남자 점수 막대
ax.bar(labels, men_means, width, yerr=men_std, label='남학생',
color='lightblue')
# 여자 점수는 남자 위에 누적
ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
label='여학생', color='pink')
ax.set_ylabel('점수')
ax.set_title('조별 평균 점수 (남/여)')
ax.legend()
st.pyplot(fig)
# ====================================
# 📦예시 3: 바코드 시각화
# ====================================
st.subheader("📡바이너리 데이터 시각화 (바코드 스타일)")
# 임의의 바이너리 데이터 (0과 1)
code = np.random.choice([0, 1], size=100)
# 바 너비와 해상도
pixel_per_bar = 4
dpi = 100
# 바코드 크기 설정
fig = plt.figure(figsize=(len(code) * pixel_per_bar / dpi, 2), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1]) # 전체 영역에 그림
ax.set_axis_off() # 축 제거
# 흑백으로 출력
ax.imshow(code.reshape(1, -1), cmap='binary', aspect='auto',
interpolation='nearest')
st.pyplot(fig)