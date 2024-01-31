import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

model_path = r"C:\Users\rahul\Downloads\landmarks_data9"
model = load_model(model_path)


def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks:
        data.extend([landmark.x, landmark.y])
    data = np.array([data])
    return data


cap = cv2.VideoCapture(0)

label=['UP','DOWN','LEFT','RIGHT','FLIP','LAND']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            preprocessed_landmarks = preprocess_landmarks(landmarks)
             
            prediction = model.predict(preprocessed_landmarks)
            class_id = np.argmax(prediction)
            cv2.putText(image, f'Class: {label[class_id-1]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
