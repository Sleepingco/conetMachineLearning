import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from dotenv import load_dotenv
import os
from PIL import Image
import tempfile
import requests
from io import BytesIO
import base64

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 페이지 설정
st.set_page_config(page_title="Bran AI", layout="centered")

# 페이지 스타일
st.markdown("""
    <style>
    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .section {
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #F9FFFA;
        border-left: 6px solid #2E8B57;
        padding: 15px;
        border-radius: 6px;
        font-size: 16px;
        color: #111;
    }
    .upload-box {
        border: 2px dashed #999;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #fdfdfd;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)
with open("Brandai-removebg-preview.png", "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data).decode()

# 마크다운 삽입
st.markdown(f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{encoded}' width='70' style='vertical-align: middle; margin-right: 10px; margin-bottom: 20PX;'>
        <span style='font-size: 40px; font-weight: 700; color: #4CAF50;'>Bran AI</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="subtitle">브랜드를 말하거나 이미지를 올려보세요. AI디자이너 브랜이 분석하고 멋진 제안을 드립니다.</div>', unsafe_allow_html=True)
tab1, tab2= st.tabs(["AI 브랜드 디자이너", "AI 디자이너 평가"])
with tab1:
    with st.expander("📌 예시 보기"):
        st.markdown("""
            **생성할 시각적 컨셉**: 로고  
            **원하는 스타일**: 심플하고 현대적인 디자인  
            **관련 색상, 형상 또는 상징적 요소**: 청량감을 주는 블루 톤, 아침 태양을 나타내는 웜 톤, 세차를 상징하는 물방울이나 거품  
            **전달하고자 하는 감정 또는 분위기**: 신뢰감, 세심함, 편안함  
            **필요한 경우 포맷 또는 종횡비**: 정사각형이나 원형
            """)
    def generateBrand(transcript):
        prompt_text = transcript
        st.markdown(f"<div class='highlight'><strong>📝 인식된 내용:</strong><br>{prompt_text}</div>", unsafe_allow_html=True)
        if prompt_text == "시청해주셔서 감사합니다!":
            st.warning("🛑 AI가 적절한 이미지를 만들기 어려워했습니다.")
            st.info("💡 '미니멀하고 감성적인 카페 브랜드를 만들고 싶어요'처럼 브랜드의 성격과 스타일을 함께 알려주세요.")
        else:
            # GPT 분석
            with st.spinner("AI 디자이너가 브랜드를 생각 중입니다..."): 
                chat_completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[designer_prompt, {"role": "user", "content": prompt_text}]
                )
            ai_description = chat_completion.choices[0].message.content
            st.markdown(f"<div class='highlight'><strong>🎯 AI 제안:</strong><br>{ai_description}</div>", unsafe_allow_html=True)
            avoid_pattern = [
                "정확히 어떤 피드백을 드려야 할지 모르겠습니다",
                "브랜드 개념에 대한 것이 아니라서",
                "브랜드에 대한 구체적인 정보가 부족합니다",
                "브랜드에 대한 자세한 설명을 주시면",
                "정보가 충분하지 않아 이미지 프롬프트를 만들기 어렵습니다",
                "브랜드의 성격이나 스타일이 명확하지 않습니다",
                "원하시는 이미지 스타일이나 톤을 알려주세요",
                "브랜드 컨셉이 더 구체적으로 필요합니다",
                "좀 더 구체적인 아이디어를 알려주시면",
                "추가 정보가 필요합니다",
                "이 요청만으로는 브랜드 이미지를 상상하기 어렵습니다",
                "도움드리기 위해 브랜드의 목적이나 방향성이 필요합니다",
                "브랜드에 대한 배경이 없어서 정확한 제안이 어렵습니다",
                "요청이 모호하여 구체적인 시각적 프롬프트를 제공하기 어렵습니다",
                "AI가 이해할 수 있도록 브랜드 톤과 주제를 명확히 해주세요",
                "해당 요청은 일반적인 문장으로는 부족하며",
                "보다 명확한 브랜드 방향이 필요합니다",
                "어떤 분위기의 이미지인지 알 수 없습니다",
                "죄송",
                "모르겠습니다",
                "어렵"
            ]


            if any(p in ai_description for p in avoid_pattern):
                st.warning("🛑 AI가 적절한 이미지를 만들기 어려워했습니다.")
                st.info("💡 '미니멀하고 감성적인 카페 브랜드를 만들고 싶어요'처럼 브랜드의 성격과 스타일을 함께 알려주세요.")
            else:
                st.markdown("### 🖼️ 이미지 생성 프롬프트")
                st.markdown(ai_description)
                # 이미지 생성
                st.markdown("#### 🖼️ 추천 로고 이미지")
                with st.spinner("로고 이미지를 생성 중입니다..."):
                    image_response = client.images.generate(
                        prompt=prompt_text,
                        n=1,
                        size="1024x1024",
                        model="dall-e-3",
                        quality="standard",
                        style="vivid"
                    )
                    image_url = image_response.data[0].url
                    image = Image.open(BytesIO(requests.get(image_url).content))
                    st.image(image, caption="AI 생성 로고", use_container_width=True)

                # TTS 생성
                with st.spinner("음성 설명을 생성 중입니다..."):
                    speech_response = client.audio.speech.create(
                        model="tts-1-hd",
                        voice="nova",
                        input=ai_description,
                        response_format="mp3",
                        speed=1.0,
                        instructions="Use a warm and professional tone in Korean."
                    )
                    tts_path = tempfile.mktemp(".mp3")
                    with open(tts_path, "wb") as f:
                        f.write(speech_response.read())
                    st.audio(tts_path, format="audio/mp3")
            
    # 시스템 프롬프트
    designer_prompt = {
    "role": "system",
    "content": (
        "You are a professional brand designer name bran who helps creators and founders visualize their brand ideas. "
        "When the user describes their brand concept, your job is to interpret it and generate a clear and inspiring image generation prompt. "
        "This prompt should be suitable for an AI image generation model like DALL·E or Midjourney, and must reflect the brand’s personality, tone, and aesthetic.\n\n"
        "Your response should include:\n"
        "1. A summary of the brand concept (1–2 sentences)\n"
        "2. A detailed image generation prompt that describes:\n"
        "- The type of visual to generate (e.g. logo, emblem, symbolic image, abstract concept)\n"
        "- The desired style (e.g. minimalist, luxurious, retro, techy)\n"
        "- Relevant colors, shapes, or symbolic elements\n"
        "- Mood or emotion to convey (e.g. trust, excitement, warmth)\n"
        "- Format or aspect ratio if needed\n\n"
        "📌 Important: When generating logos or emblems, the image prompt should clearly instruct the model to display the logo alone, "
        "centered on a plain background (e.g. white), with no hands, objects, desks, or additional scenery. The result should focus solely on the logo design.\n\n"
        "Write your response in Korean, and ensure that the image prompt is vivid and descriptive enough for a visual AI model to follow. "
        "Keep a friendly and creative tone, as if co-designing the brand with the user. Your tone should be friendly, professional, and supportive. Only respond in Korean."
        )
    }

    input_type = st.radio(
        "입력 방식 선택",
        ["텍스트", "마이크"]
    )
    if input_type == "텍스트":
        prompt_text = st.text_input("당신의 브랜드 시각 컨셉을 자유롭게 입력해 주세요:",
        placeholder="예: 플랫 스타일의 감성적인 카페 브랜드 로고. 베이지 톤 + 커피잔 심볼. 편안함과 따뜻한 느낌 강조"
        ).strip()
        # 입력 유효성 검사 및 트리거
        if st.button("🚀 이미지 프롬프트 생성 요청"):
            if not prompt_text or len(prompt_text.strip()) < 10:
                st.error("입력 내용이 너무 짧습니다. 예시를 참고해 브랜드 컨셉을 더 구체적으로 입력해 주세요.")
            else:
                st.success("입력 완료! 이제 AI가 시각적 이미지 프롬프트를 생성합니다.")
                # 이후 GPT 처리 함수 연결 가능:
                generateBrand(prompt_text)
            
    if input_type == "마이크":
        # 🎙️ 음성 입력 섹션
        st.markdown('<div class="section"><h3>🎤 음성으로 브랜드 아이디어 말하기</h3></div>', unsafe_allow_html=True)
        if st.button("🎙 지금 말하기"):
            fs = 16000
            seconds = 5
            filename = "live_input.wav"
            st.info("⏺ 5초간 녹음합니다. 말씀해주세요...")
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            write(filename, fs, recording)
            st.success("✅ 녹음 완료!")
            with open(filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text",
                    language="ko"
                )
            if transcript:
                generateBrand(transcript)
    
    # 공통 처리
    # if prompt_text:
    #     st.markdown("### 🧠 입력된 브랜드 컨셉")
    #     st.markdown(f"> {prompt_text}")
    #     generateBrand(transcript)
    #     # st.markdown(f"<div class='highlight'><strong>📝 인식된 내용:</strong><br>{prompt_text}</div>", unsafe_allow_html=True)

    #     # GPT 분석
    #     with st.spinner("AI 디자이너가 브랜드를 분석 중입니다..."):
    #         chat_completion = client.chat.completions.create(
    #             model="gpt-4",
    #             messages=[designer_prompt, {"role": "user", "content": prompt_text}]
    #         )
    #     ai_description = chat_completion.choices[0].message.content
    #     st.markdown(f"<div class='highlight'><strong>🎯 AI 제안:</strong><br>{ai_description}</div>", unsafe_allow_html=True)

    #     # 이미지 생성
    #     st.markdown("#### 🖼️ 추천 로고 이미지")
    #     with st.spinner("로고 이미지를 생성 중입니다..."):
    #         image_response = client.images.generate(
    #             prompt=prompt_text,
    #             n=1,
    #             size="1024x1024",
    #             model="dall-e-3",
    #             quality="standard",
    #             style="vivid"
    #         )
    #         image_url = image_response.data[0].url
    #         image = Image.open(BytesIO(requests.get(image_url).content))
    #         st.image(image, caption="AI 생성 로고", use_container_width=True)

    #     # TTS 생성
    #     with st.spinner("음성 설명을 생성 중입니다..."):
    #         speech_response = client.audio.speech.create(
    #             model="tts-1-hd",
    #             voice="nova",
    #             input=ai_description,
    #             response_format="mp3",
    #             speed=1.0,
    #             instructions="Use a warm and professional tone in Korean."
    #         )
    #         tts_path = tempfile.mktemp(".mp3")
    #         with open(tts_path, "wb") as f:
    #             f.write(speech_response.read())
    #         st.audio(tts_path, format="audio/mp3")
with tab2:
    # 📤 이미지 평가 섹션
    st.markdown('<div class="section"><h3>🖼️ 로고 이미지 평가받기</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 평가할 로고 이미지를 업로드하세요", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)

        # MIME 타입 확인 및 base64 인코딩
        mime_type = uploaded_file.type
        image_bytes = uploaded_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # 평가 프롬프트
        evaluation_prompt = (
            "이 이미지는 로고 디자인 시안입니다. 디자인의 인상, 스타일, 색감, 브랜드 전달력 등을 전문가의 시선으로 분석해 주세요. "
            "강점과 개선점, 추천 대상 브랜드 유형도 제시해 주세요. 한국어로 대답하세요."
        )

        # GPT-4o 분석
        with st.spinner("AI가 로고 이미지를 평가하는 중입니다..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional brand designer who evaluates design quality."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": evaluation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

        st.markdown(f"<div class='highlight'><strong>💡 AI 평가 결과:</strong><br>{response.choices[0].message.content}</div>", unsafe_allow_html=True)
        with st.spinner("tts를 생성중..."):
            speech_response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=response.choices[0].message.content,
                response_format="mp3",
                speed=1.0,
                instructions="Use a warm and professional tone in Korean."
            )
            tts_path = tempfile.mktemp(".mp3")
            with open(tts_path, "wb") as f:
                f.write(speech_response.read())
            st.audio(tts_path, format="audio/mp3")
