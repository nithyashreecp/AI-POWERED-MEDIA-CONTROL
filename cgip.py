import cv2
import mediapipe as mp
import numpy as np
import pyautogui  # For media controls
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe for hands and face detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

paused = False  # Track if media is currently paused

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(img_rgb)
    hand_results = hands.process(img_rgb)

    # Face detection to control play/pause
    if face_results.detections:
        if paused:
            pyautogui.press('playpause')
            paused = False
        cv2.putText(img, "Playing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        if not paused:
            pyautogui.press('playpause')
            paused = True
        cv2.putText(img, "Paused - Face Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hand gesture volume control
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert landmark positions to pixel coordinates
            h, w, _ = img.shape
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw circles on thumb and index finger tips
            cv2.circle(img, (thumb_tip_x, thumb_tip_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_tip_x, index_tip_y), 10, (0, 0, 255), cv2.FILLED)

            # Draw a line between thumb and index finger tips
            cv2.line(img, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 255, 0), 3)

            # Calculate the distance between thumb and index finger tips
            distance = np.linalg.norm([thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y])

            # Normalize the distance to the volume range
            vol = np.interp(distance, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display the volume level
            vol_percentage = np.interp(distance, [30, 200], [0, 100])
            cv2.putText(img, f'Volume: {int(vol_percentage)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Face Detection and Hand Volume Control", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
