import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from huggingface_hub import InferenceClient

def survey_results():
    # 1. 제목과 소개
    st.title("👨‍💻 2024 개발자 설문 결과")
    st.image("data:image/png;base64,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")
    st.markdown("개발자들이 가장 많이 사용하는 프로그래밍 언어는 무엇일까요? 🤔")
    st.markdown("이번 설문을 통해 다양한 언어의 인기 순위를 확인해보겠습니다!")

    # 2. 출처 표기
    st.markdown("##### 📑 데이터 출처: [Stack Overflow Developer Survey](https://survey.stackoverflow.co/)")
    st.caption("Stack Overflow 공식 개발자 설문조사 기반 데이터입니다.")


    chart_type = st.radio("차트 유형 선택", ["개발자 언어 순위", "AI", "지리","DataBases"])

    if chart_type == "개발자 언어 순위":
        # 3. 데이터 준비
        data = {
            "Language": [
                "JS", "HTML/CSS", "PY", "SQL", "TS", "Bash/Shell", "Java", "C#", "C++", "C",
                "PHP", "PowerShell", "Go", "Rust", "Kotlin", "Lua", "Dart", "Assembly", "Ruby",
                "Swift", "R", "Visual Basic", "MATLAB", "VBA", "Groovy", "Scala", "Perl",
                "GDScript", "Objective-C", "Elixir", "Haskell", "Delphi", "MicroPython",
                "Lisp", "Clojure", "Julia", "Zig", "Fortran", "Solidity", "Ada", "Erlang",
                "F#", "Apex", "Prolog", "OCaml", "Cobol", "Crystal", "Nim", "Zephyr"
            ],
            "Percentage": [
                62.3, 52.9, 51, 51, 38.5, 33.9, 30.3, 27.1, 23, 20.3,
                18.2, 13.8, 13.5, 12.6, 9.4, 6.2, 6, 5.4, 5.2,
                4.7, 4.3, 4.2, 4, 3.7, 3.3, 2.6, 2.5,
                2.3, 2.1, 2.1, 2, 1.8, 1.6,
                1.5, 1.2, 1.1, 1.1, 1.1, 1.1, 0.9, 0.9,
                0.9, 0.8, 0.8, 0.8, 0.7, 0.4, 0.4, 0.3  
            ]
        }
        df = pd.DataFrame(data)
        # 순위 추가 (사용 비율 기준 내림차순 정렬 후 순위 매김)
        df = df.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1부터 시작
        # df.insert(0, "Rank", df.index)  # 0번째(맨 앞)에 Rank 컬럼 추가
        col1, col2 = st.columns(2)
        with col1:
            # 4. 데이터프레임 간단히 보여주기
            st.dataframe(df)
        with col2:
            # 5. 상위 10개 언어 분석
            st.markdown("### 🥇 가장 인기 있는 Top 10 언어는?")
            top10 = df.sort_values(by="Percentage", ascending=False).head(10)

            # 6. 차트 그리기
            fig, ax = plt.subplots()
            plt.rcParams['font.family'] ='Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] =False
            colors = plt.cm.tab20(np.linspace(0, 1, len(top10)))
            ax.barh(top10["Language"], top10["Percentage"], color=colors)

            ax.invert_yaxis()  # 높은 수치를 위로
            ax.set_xlabel("Rario (%)")
            ax.set_title("Top 10 Language")

            st.pyplot(fig)

        # 7. 마무리
        st.success("개발자들은 주로 JS, HTML/CSS, PYTHON 등을 많이 사용하는 것으로 나타났습니다! 🚀")
        st.info("여러분은 어떤 언어를 주로 사용하고 계신가요?")

        # 무작위 행 추출
        randomNum1 = np.random.randint(0, len(df))
        randomNum2 = np.random.randint(0, len(df))
        randomNum3 = np.random.randint(0, len(df))

        r1 = df.iloc[randomNum1]
        r2 = df.iloc[randomNum2]
        r3 = df.iloc[randomNum3]

        # metric 카드 3개로 출력
        st.markdown("### 🔍 오늘 배워볼 랜덤 언어 3가지")
        col1, col2, col3 = st.columns(3)

        col1.metric(label="1st", value=r1["Language"], help="Try learning this language!")
        col2.metric(label="2nd", value=r2["Language"], help="Try learning this language!")
        col3.metric(label="3rd", value=r3["Language"], help="Try learning this language!")
    elif chart_type == "AI":
        # 슬라이드 단계를 선택
        section = st.select_slider(
            "AI 트렌드를 탐색해보세요",
            options=["감정과 사용 현황", "선호도", "효과성과 윤리"],
            value="감정과 사용 현황"
        )

        # 1. 감정과 사용 현황
        if section == "감정과 사용 현황":
            st.title("🤖 감정과 사용 현황")
            st.markdown("사용자들이 AI 기술에 대해 느끼는 감정과 실제 사용 현황을 살펴봅니다.")
            st.write("전체 응답자의 76%가 올해 개발 프로세스에 AI 도구를 사용하고 있거나 사용할 계획이라고 답했는데, 이는 작년(70%)보다 증가한 수치입니다. 또한, 올해는 훨씬 더 많은 개발자들이 AI 도구를 사용하고 있습니다(62% vs. 44%).")

            fig1, ax1 = plt.subplots()
            ax1.pie([61.8, 13.8, 24.4], labels=["사용중", "계획중", "계획 없음"], autopct="%1.1f%%", startangle=140)
            ax1.axis('equal')
            st.pyplot(fig1)

        # 2. 개발 도구
        elif section == "선호도":
            st.title("🛠️ 선호도")
            st.markdown("AI 개발에 사용자들의 선호도를 분석합니다")
            st.write("전체 응답자의 72%가 개발용 AI 도구에 대해 호의적이거나 매우 호의적이라고 답했습니다. 이는 작년 호의도 77%보다 낮은 수치이며, 호의도 하락은 사용 결과가 만족스럽지 않았기 때문일 수 있습니다.")

            dev_tools = pd.DataFrame({
                "도구": ["Very favorable", "Favorable", "Indifferent", "Unsure","Unfavorable","Very unfavorable"],
                "사용 비율": [23.6,48.3, 18.7, 3,5.2,1.2]
            })
            fig2, ax2 = plt.subplots()
            ax2.barh(dev_tools["도구"], dev_tools["사용 비율"], color="skyblue")
            ax2.invert_yaxis()
            ax2.set_xlabel("사용 비율 (%)")
            st.pyplot(fig2)

        # 3. 효과성과 윤리
        elif section == "효과성과 윤리":
            st.title("⚖️ 효과성과 윤리")
            st.markdown("AI 도구가 당신의 직업에 위협이 될까요?")
            fig3, ax3 = plt.subplots()
            ax3.bar(["NO", "I'm not sure", "Yes"], [68.3,19.6,12.1], color=["green", "orange", "red"])
            ax3.set_ylabel("응답 비율 (%)")
            st.pyplot(fig3)

            st.info("전문 개발자의 70%는 AI를 자신의 직업에 대한 위협으로 인식하지 않습니다.")
    elif chart_type == "지리":

        # 상위 10개 국가 + 위도 경도 포함된 데이터프레임
        data = [
            {"country": "United States", "percent": 18.9, "lat": 37.0902, "lon": -95.7129},
            {"country": "Germany", "percent": 8.4, "lat": 51.1657, "lon": 10.4515},
            {"country": "India", "percent": 7.2, "lat": 20.5937, "lon": 78.9629},
            {"country": "United Kingdom", "percent": 5.5, "lat": 55.3781, "lon": -3.4360},
            {"country": "Ukraine", "percent": 4.6, "lat": 48.3794, "lon": 31.1656},
            {"country": "Canada", "percent": 3.6, "lat": 56.1304, "lon": -106.3468},
            {"country": "France", "percent": 3.6, "lat": 46.2276, "lon": 2.2137},
            {"country": "Poland", "percent": 2.6, "lat": 51.9194, "lon": 19.1451},
            {"country": "Netherlands", "percent": 2.5, "lat": 52.1326, "lon": 5.2913},
            {"country": "Brazil", "percent": 2.3, "lat": -14.2350, "lon": -51.9253}
        ]
        df = pd.DataFrame(data)
        df = df.sort_values(by="percent", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1부터 시작

        selected_countries = st.multiselect("국가 선택", df["country"].tolist(), default=df["country"].tolist())
        filtered_df = df[df["country"].isin(selected_countries)]

        # 지도 표시
        st.title("Top 10 국가 백분율 시각화")
        st.map(filtered_df.rename(columns={"lat": "latitude", "lon": "longitude"}))
        with st.expander("자세한 표를 보려면 클릭하세요."):
            st.dataframe(df[["country", "percent"]])
    elif chart_type == "DataBases":

        # 클라우드 플랫폼 사용 비율 데이터 정리
        cloud_data = {
            "Service": [
                "Amazon Web Services", "Microsoft Azure", "Google Cloud", "Cloudflare", "Firebase",
                "Vercel", "Digital Ocean", "Heroku", "Netlify", "VMware", "Hetzner", "Supabase",
                "Linode, now Akamai", "OVH", "Managed Hosting", "Oracle Cloud Infra.", "Render",
                "Fly.io", "OpenShift", "Databricks", "PythonAnywhere", "Vultr", "OpenStack",
                "Alibaba Cloud", "IBM Cloud Or Watson", "Scaleway", "Colocation"
            ],
            "Percentage": [
                48, 27.8, 25.1, 15.1, 13.9, 11.9, 11.7, 8.2, 7, 6.6, 5, 3.8, 3.1, 3, 3,
                2.9, 2.8, 2.6, 2.4, 2, 1.9, 1.7, 1.6, 1.2, 1.1, 0.9, 0.7
            ]
        }
        
        cloud_df = pd.DataFrame(cloud_data)
        top10 = cloud_df.sort_values(by="Percentage", ascending=False).head(10)
        # 파이 차트 출력
        st.title("클라우드 플랫폼 사용 비율 - 원형 차트")
        fig, ax = plt.subplots()
    
        ax.pie(top10["Percentage"], labels=top10["Service"], autopct="%1.1f%%", startangle=140)
        ax.axis('equal')  # 원형 유지
        st.pyplot(fig)

        # 6. 차트 그리기
        fig, ax = plt.subplots()
        plt.rcParams['font.family'] ='Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] =False
        colors = plt.cm.tab20(np.linspace(0, 1, len(cloud_df)))
        ax.barh(top10["Service"], top10["Percentage"], color=colors)
        ax.invert_yaxis()  # 높은 수치를 위로
        ax.set_xlabel("Rario (%)")
        ax.set_title("Top 10 Language")
        st.pyplot(fig)

        st.markdown("가장 인기있는 DB 3종")
        img1, img2, img3 = st.columns(3)
        with img1:
            st.header("AWS")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZSy14TJ_EjEdf9QDWYSwKhFUq7DrJ_8NAcw&s")
        with img2:
            st.header("Azure")
            st.image("https://blog.kakaocdn.net/dn/5BRkR/btrGjFoEAkL/C4eNzgUxGnVulyTnyBW7Xk/img.png")
        with img3:
            st.header("Google Cloud")
            st.image("data:image/png;base64,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")
    

def ai_recommendation():
    # --- 페이지 설정 ---
    st.set_page_config(page_title="언어를 추천 받아보세요", layout="centered")
    st.title("🧠 Hugging Face 언어를 추천기")

    # --- Hugging Face API 설정 ---
    API_TOKEN = ""
    client = InferenceClient(token=API_TOKEN)
    user_question = st.text_input("💬 개발 목표를 입력하세요", placeholder="ex) i want to make game server")
    if user_question:
        prompt = (
            f"The user said: '{user_question}'\n\n"
            "Please respond with a concise list:\n\n"
            "Languages:\n"
            "- Language 1: Reason\n"
            "- Language 2: Reason\n\n"
            "Database:\n"
            "- Database name: Reason\n\n"
            "Tools:\n"
            "- Tool name: Purpose"
        )




        response = client.text_generation(
            prompt=prompt,
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_new_tokens=80
        )
        # 결과 출력
        st.markdown("🧠 **AI 추천 결과**")
        st.write(response.strip())

# navigation 설정
nav = st.navigation([
    st.Page(survey_results),
    st.Page(ai_recommendation)
])
nav.run()
st.markdown("해당 사이트가 얼마나 마음에 드는지 평가해주세요.")
stars = st.feedback("stars")
# 저장 버튼
if st.button("제출하기"):
    # 이전 피드백 데이터 로드 (없으면 새로 생성)
    if os.path.exists("ratings.csv"):
        df = pd.read_csv("ratings.csv")
    else:
        df = pd.DataFrame(columns=["rating"])

    # 현재 점수 추가 후 저장
    df = df._append({"rating": stars}, ignore_index=True)
    df.to_csv("ratings.csv", index=False)

    st.success("별점이 저장되었습니다!")

# 평균 별점 출력
if os.path.exists("ratings.csv"):

    df = pd.read_csv("ratings.csv")
    avg = df["rating"].mean()
    count = len(df)
    st.markdown(f"### ⭐ 평균 별점: {avg:.2f}점 ({count}명 참여)")