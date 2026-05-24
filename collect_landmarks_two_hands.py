import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- CONFIGURATION ---
# This is the name of the dataset file we will create
DATASET_PATH = "landmark_dataset_two_hands.csv"

# This is the complete list of all signs we will collect
SIGNS_TO_COLLECT = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'NONE'
]
NUM_IMAGES_PER_SIGN = 100 # Number of samples to collect for each sign

# --- MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
# IMPORTANT: We are telling MediaPipe to look for up to 2 hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- SCRIPT ---
# Create the CSV file and write the header for 42 landmarks (21 for left hand + 21 for right hand)
header = ['class']
for i in range(42):
    header += [f'x{i}', f'y{i}', f'z{i}']

with open(DATASET_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

cap = cv2.VideoCapture(0)

# Loop through each sign in our list
for sign_index, sign in enumerate(SIGNS_TO_COLLECT):
    # --- User Prompt ---
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        prompt_text = f"GET READY FOR SIGN: {sign}. Press 'S' to start."
        cv2.putText(frame, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    print(f"--- Starting collection for sign: {sign} ---")
    
    image_count = 0
    while image_count < NUM_IMAGES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Save the data when the 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if results.multi_hand_landmarks:
                all_landmarks = []
                handedness_list = [res.classification[0].label for res in results.multi_handedness]
                
                # Process left hand first if present, otherwise fill with zeros
                if 'Left' in handedness_list:
                    hand_landmarks = results.multi_hand_landmarks[handedness_list.index('Left')]
                    wrist = hand_landmarks.landmark[0]
                    for lm in hand_landmarks.landmark:
                        all_landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                else:
                    all_landmarks.extend([0.0] * 63) # 21 landmarks * 3 coords = 63 zeros

                # Process right hand second if present, otherwise fill with zeros
                if 'Right' in handedness_list:
                    hand_landmarks = results.multi_hand_landmarks[handedness_list.index('Right')]
                    wrist = hand_landmarks.landmark[0]
                    for lm in hand_landmarks.landmark:
                        all_landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                else:
                    all_landmarks.extend([0.0] * 63)

                # Save the data to the CSV file
                row = [sign_index] + all_landmarks
                with open(DATASET_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                image_count += 1
                print(f"Collected sample {image_count} for sign {sign}")

        # Draw landmarks on the frame for visualization
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        count_text = f"Collected: {image_count} / {NUM_IMAGES_PER_SIGN}"
        cv2.putText(frame, count_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

cap.release()
cv2.destroyAllWindows()