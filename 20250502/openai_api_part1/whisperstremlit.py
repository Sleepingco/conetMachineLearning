from openai import OpenAI
import os
from dotenv import load_dotenv
from playsound import playsound # 외부 스피커 재생
from pathlib import Path
import pygame
import time
import sounddevice as sd
from scipy.io.wavfile import write
import streamlit as st
# client.audio.transcriptions.create(
#     file=..., # 필수: 오디오 파일 객체
#     model="whisper-1", # 필수: 모델 이름 (현재는 whisper-1 고정)
#     prompt=..., # 선택: 힌트 문장 제공 (맥락 강화)
#     response_format=..., # 선택: 응답 형식 ("json", "text", "srt", "verbose_json", "vtt")
#     temperature=..., # 선택: 창의성/랜덤성 조절 (0.0~1.0)
#     language=..., # 선택: 입력 음성의 언어 (예: "en", "ko", "es")
#     timestamp_granularities=... # 선택: 세그먼트 정보 단위 설정 (베타)
# )

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
st.title("한국어 to whisper자막 to gpt 영어 번역")
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if st.button("지금 부터 5초간 녹음하기"):
    # OpenAI API 클라이언트
    
    # 기본 설정
    fs = 16000 # 샘플레이트 (16kHz)
    seconds = 5 # 녹음 시간 (초)
    filename = "live_input.wav"
    # 1. 녹음 시작
    st.error(" 지금부터 5초간 녹음합니다...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    st.success("녹음 완료")
    # 2. Whisper로 한국어 텍스트 변환
    with open(filename, "rb") as audio_file:
        st.session_state.transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="text",
        language="ko"
    )
    st.markdown("### 인식된 자막(한국어)")
    st.write(st.session_state)
# 언어 기준 리스트 (국가가 아닌)
languages = [
    "Afrikaans",
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Mandarin",
    "Norwegian",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swedish",
    "Thai",
    "Turkish",
    "Urdu",
    "Vietnamese"
]

# 사용자 선택
selected_language = st.selectbox("TTS로 변환할 언어를 선택하세요:", languages)

st.success(f"선택한 언어: {selected_language}")

# GPT 프롬프트용 안내 메시지 생성 예시
gpt_instruction = f"Translate the following Korean sentence into natural {selected_language}."
st.write("GPT에 전달할 시스템 메시지:")
st.code(gpt_instruction, language='text')
if st.button("번역 시작"):
    if not st.session_state.transcript:
        st.warning("먼저 음성을 녹음하고 텍스트를 생성하세요.")
    else:
        # 3. GPT로 영어 번역
        messages = [
            {"role": "system", "content": gpt_instruction},
            {"role": "user", "content": st.session_state.transcript}
        ]
        translation_response = client.chat.completions.create(
            model="gpt-4o-mini", # 또는 "gpt-3.5-turbo"
            messages=messages,
            temperature=0.3
        )
        english_translation = translation_response.choices[0].message.content
        st.markdown("### [자동 번역 - 영어]")
        st.write(english_translation)

        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=english_translation,
            # instructions="Speak in a cheerful and positive tone."
        )
        audio_data = response.content
        # 파일 경로 설정
        output_path = "engResponse.mp3"
        speech_file_path = Path.cwd() / "engResponse.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print(f" file save : {output_path}")
        try:
            pygame.mixer.music.unload()
        except:
            pass # 첫 실행 시에는 unload 대상이 없음
        print(f"오디오 파일 저장 완료: {speech_file_path}")
        # 오디오 재생
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(str(speech_file_path))
        pygame.mixer.music.play()
        # 끝날 때까지 대기
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
        # 명시적 업로드
        pygame.mixer.music.unload()
        print("재생 완료 및 파일 해제")