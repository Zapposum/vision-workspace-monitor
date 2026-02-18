import cv2
from ultralytics import YOLO
import os
import time

MODEL_PATH = "runs/detect/train13/weights/best.engine"
SAVE_DIR = "detections"
CONF = 0.4

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Detector started...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    results = model(frame, conf=CONF)
    boxes = results[0].boxes
    annotated = results[0].plot()

    if boxes is not None and len(boxes) > 0:
        timestamp = int(time.time() * 1000)
        filename = f"{SAVE_DIR}/det_{timestamp}.jpg"
        cv2.imwrite(filename, annotated)

    cv2.imshow("Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
