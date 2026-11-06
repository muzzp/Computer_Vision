"""
Webcam face detector using OpenCV.

Reports whether the camera is currently looking at a human face.
q or ESC = quitpython face_detector.py [--camera 0]
Use venv called 'face'
"""

import argparse
import sys
from collections import deque

import cv2

# Number of recent frames used to smooth the result (reduces flicker)
HISTORY_LEN = 7
# Fraction of recent frames that must contain a face to report "FACE"
THRESHOLD = 0.6


def load_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        sys.exit(f"Could not load Haar cascade from: {path}")
    return cascade


def detect_faces(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting
    return cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,      # higher = fewer false positives
        minSize=(80, 80),    # ignore tiny detections
    )


def draw_status(frame, face_present: bool, count: int):
    h, w = frame.shape[:2]
    colour = (0, 200, 0) if face_present else (0, 0, 220)
    label = f"HUMAN FACE DETECTED ({count})" if face_present else "NO FACE DETECTED"

    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    cv2.circle(frame, (24, 22), 10, colour, -1)
    cv2.putText(frame, label, (46, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, colour, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Webcam face detector")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. "
                 "Check it's connected and not in use by another app.")

    cascade = load_cascade()
    history = deque(maxlen=HISTORY_LEN)

    print("Running. Press 'q' or ESC in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror view
        faces = detect_faces(cascade, frame)

        history.append(len(faces) > 0)
        face_present = sum(history) / len(history) >= THRESHOLD

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        draw_status(frame, face_present, len(faces))
        cv2.imshow("Face Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
