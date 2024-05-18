import cv2
import mediapipe as mp
import numpy as np
import pickle

from build_the_model import build_mlp_model
from prepare_data import load_data

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the model weights and label map
with open('model_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

# Load data to get label_map
X, y, label_map = load_data()  # Ensure label_map is loaded again
input_shape = (X.shape[1],)
num_classes = len(np.unique(y))

# Load the trained model
model = build_mlp_model(input_shape, num_classes)
model.set_weights(loaded_weights)


def predict_sign(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)

            landmarks = np.array(landmarks).flatten().reshape(1, -1)
            predictions = model.predict(landmarks)
            sign_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            return sign_idx, confidence
    return None, None


cap = cv2.VideoCapture(0)
inverse_label_map = {v: k for k, v in label_map.items()}  # Create inverse label map

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sign_idx, confidence = predict_sign(frame, model)
    if sign_idx is not None:
        sign = inverse_label_map[sign_idx]
        cv2.putText(frame, f'{sign} ({confidence * 100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
