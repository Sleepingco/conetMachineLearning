import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import datetime

# 주요 종목 리스트
stock_list = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '005380': '현대차',
    '068270': '셀트리온',
    '051910': 'LG화학',
    '005490': 'POSCO홀딩스',
    '028260': '삼성물산',
    '035420': 'NAVER'
}

# 날짜 입력
start_date = st.date_input("시작일", datetime.date(2025, 1, 1))
end_date = st.date_input("종료일", datetime.date.today())

# 다중 종목 선택
selected_codes = st.multiselect(
    label='조회할 종목을 선택하세요',
    options=list(stock_list.keys()),
    format_func=lambda x: f"{x} - {stock_list[x]}",
    default=['005930', '000660']
)

# 조건이 모두 입력된 경우
if selected_codes and start_date and end_date:
    try:
        df_close = pd.DataFrame()

        for code in selected_codes:
            df = fdr.DataReader(code, start=start_date, end=end_date)[['Close']]
            df.rename(columns={"Close": stock_list.get(code, code)}, inplace=True)
            df_close = pd.concat([df_close, df], axis=1)

        if df_close.empty:
            st.warning("❗ 해당 기간에 주가 데이터가 없습니다.")
        else:
            st.subheader("📈 종목별 종가 비교 차트")
            st.line_chart(df_close)

            st.write("📋 최근 5일 종가 데이터")
            st.dataframe(df_close.tail(5))

    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")