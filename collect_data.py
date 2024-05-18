import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Function to collect data for a given label.
def collect_data(label, num_samples, save_dir='data'):
    cap = cv2.VideoCapture(0)
    collected_samples = 0
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    while collected_samples < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks.
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)

                # Save landmarks to file.
                landmarks = np.array(landmarks).flatten()
                np.save(os.path.join(label_dir, f'{label}_{collected_samples}.npy'), landmarks)
                collected_samples += 1

        cv2.imshow('Collecting Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# List of labels and the number of samples for each label.
labels = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
num_samples = 100

# Collect data for each label.
for label in labels:
    print(f'Collecting data for label: {label}')
    collect_data(label, num_samples)
