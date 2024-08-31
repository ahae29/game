import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 모델 로드
model = load_model('my_model.h5')  # Teachable Machine에서 저장한 모델 파일 경로

# 클래스 이름
class_names = ['가위', '바위']  # 클래스 이름을 실제 클래스에 맞게 수정하세요.

# Streamlit 애플리케이션 제목
st.title("웹캠 동영상 분류기")

# 웹캠 영상 입력
video_input = st.camera_input("웹캠을 실행하세요")

if video_input is not None:
    # 이미지 열기
    img = Image.open(video_input)
    img = img.resize((224, 224))  # 모델 입력 크기에 맞게 조정
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    # 예측
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # 결과 출력
    st.image(img, caption='촬영한 이미지', use_column_width=True)
    st.write(f"예측 클래스: {class_names[class_index]}")
    st.write(f"신뢰도: {confidence:.2f}")

else:
    st.write("웹캠을 시작하려면 사진을 찍어주세요.")
