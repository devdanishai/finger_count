#dani code 21/dec/2024
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the finger tips (indices of landmarks)
finger_tips = [4, 8, 12, 16, 20]

# MediaPipe Hands configuration
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the image for a mirror view and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image for hand landmarks
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw the hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmark coordinates for the hand
                    landmarks = hand_landmarks.landmark

                    # List to store whether each finger is raised
                    fingers_up = []

                    # Thumb logic
                    if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)

                    # Other fingers logic
                    for tip in finger_tips[1:]:
                        # Apply a threshold to detect if the finger is raised
                        if landmarks[tip].y < landmarks[tip - 2].y - 0.05:  # Adjust threshold as needed
                            fingers_up.append(1)
                        else:
                            fingers_up.append(0)

                    # Count the number of fingers up
                    total_fingers = fingers_up.count(1)

                    # Optional: Print debug information
                    print(f"Hand {i+1}: Total fingers detected = {total_fingers}, Fingers up = {fingers_up}")

                    # Set text position for left or right hand
                    text_position = (10, 50 + i * 50)  # Adjust Y position for each hand

                    # Display the count with red color and bold
                    cv2.putText(frame, f'Fingers: {total_fingers}', text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Red color (BGR), bold

            # Display the result
            cv2.imshow("Finger Counting", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()