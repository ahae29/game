import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 모델 로드
model = load_model('my_model.h5')  # Teachable Machine에서 저장한 모델 파일 경로

# 클래스 이름
class_names = ['Class1', 'Class2', 'Class3']  # 클래스 이름을 실제 클래스에 맞게 수정하세요.

# Streamlit 애플리케이션 제목
st.title("웹캠 동영상 분류기")

# 웹캠 영상 출력
run = st.checkbox('웹캠 실행')

if run:
    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("웹캠을 열 수 없습니다.")
            break

        # 이미지 전처리
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        img = Image.fromarray(img)  # OpenCV 이미지를 PIL 이미지로 변환
        img = img.resize((224, 224))  # 모델 입력 크기에 맞게 조정
        img_array = np.array(img) / 255.0  # 정규화
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

        # 예측
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]

        # 결과 출력
        stframe.image(frame, channels='BGR')
        st.write(f"예측 클래스: {class_names[class_index]}")
        st.write(f"신뢰도: {confidence:.2f}")

    cap.release()
else:
    st.write("웹캠을 시작하려면 체크박스를 선택하세요.")
