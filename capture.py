import cv2
import os

SAVE_DIR = "dataset/images/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

counter = 657

print("Нажмите 's' чтобы сохранить кадр")
print("Нажмите ESC чтобы выйти")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка камеры")
        break

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = f"{SAVE_DIR}/img_{counter}.jpg"
        cv2.imwrite(filename, frame)
        print("Сохранено:", filename)
        counter += 1

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
