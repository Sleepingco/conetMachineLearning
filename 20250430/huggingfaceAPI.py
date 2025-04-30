

import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import requests

# --- 페이지 설정 ---
st.set_page_config(page_title="이미지 분류 with Hugging Face", layout="centered")
st.title("🧠 Hugging Face 이미지 분류기")

# --- Hugging Face API 설정 ---
API_TOKEN = ""
client = InferenceClient(token=API_TOKEN)

# --- 이미지 업로드 ---
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
    image_bytes = uploaded_file.read()

    with st.spinner("분류 중입니다..."):
        try:    
            # --- 이미지 분류 요청 ---
            results = client.image_classification(
                image_bytes,
                model="google/vit-base-patch16-224"
            )

            st.success("✅ 분류 완료!")
            st.write(results)

            # 가장 높은 확률 라벨 1개 추출 (사용하고 싶다면)
            main_label = results[0]["label"].split(",")[0].strip()

            # 상위 라벨 이름 추출 (앞의 단어만)
            top_labels = [r["label"].split(",")[0].strip() for r in results[:5]]
            label_list = ", ".join(top_labels)

            prompt = (
                "You are an AI trained to write short, poetic image captions.\n"
                "Given the following image classification results, write one natural English sentence "
                "that gently describes what might be seen in the photo.\n\n"
                f"Labels: {label_list}\n\n"
                "Caption:"
            )



            # 결과 출력
            response = client.text_generation(prompt=prompt, model="google/flan-t5-base", max_new_tokens=50)
            st.markdown("📘 **이미지 설명 생성 결과**")
            st.write(response)

        except Exception as e:
            st.error(f"⚠️ 오류 발생: {e}")
else:
    st.info("👈 왼쪽에서 이미지를 업로드해 주세요.")




# # Hugging Face API 설정
# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# API_TOKEN = "" # Hugging Face에서 발급받은 개인 API 토큰 입력
# headers = {
#     "Authorization": f"Bearer {API_TOKEN}"
# }
# # 요약 요청 함수
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# # st.set_page_config(page_title="영문텍스트 요약(API)")
# st.title("영문 텍스트 요약(HF API)")
# text = st.text_area(" 요약할 텍스트를 입력하세요", height=200, placeholder="긴 뉴스 기사나 문단을 입력해보세요...")
# if st.button("요약 시작"):
#     if len(text) < 30:
#         st.warning("최소 30자 이상 입력해 주세요.")
#     else:
#         with st.spinner("요약 중..."):
#             output = query({
#                 "inputs": text,
#                 "parameters": {"max_length": 100, "min_length": 30, "do_sample": False}
#             })
#     try:
#         summary = output[0]["summary_text"]
#         st.success("요약 완료")
#         st.markdown(f"**요약 결과:** {summary}")
#     except Exception as e:
#         st.error(f"요약 실패: {e}")