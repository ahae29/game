import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 모델 로드
model = load_model('my_model.h5')  # Teachable Machine에서 저장한 모델 파일 경로
model.save('my_model.h5')

# 클래스 이름
class_names = ['가위', '바위']  # 클래스 이름을 실제 클래스에 맞게 수정하세요.

# Streamlit 애플리케이션 제목
st.title("웹캠 동영상 분류기")


# 웹캠 영상 입력
video_input = st.camera_input("웹캠을 실행하세요")

