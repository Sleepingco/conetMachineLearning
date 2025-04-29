import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 제목과 소개
st.title("👨‍💻 2024 개발자 설문 결과")
st.markdown("개발자들이 가장 많이 사용하는 프로그래밍 언어는 무엇일까요? 🤔")
st.markdown("이번 설문을 통해 다양한 언어의 인기 순위를 확인해보겠습니다!")

# 2. 출처 표기
st.markdown("##### 📑 데이터 출처: [Stack Overflow Developer Survey](https://survey.stackoverflow.co/)")
st.caption("Stack Overflow 공식 개발자 설문조사 기반 데이터입니다.")

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

chart_type = st.radio("차트 유형 선택", ["개발자 언어 순위", "AI", "DataBases"])
df = pd.DataFrame(data)
if chart_type == "개발자 언어 순위":
    # 순위 추가 (사용 비율 기준 내림차순 정렬 후 순위 매김)
    df = df.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1부터 시작
    # df.insert(0, "Rank", df.index)  # 0번째(맨 앞)에 Rank 컬럼 추가

    # 4. 데이터프레임 간단히 보여주기
    st.dataframe(df)

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


