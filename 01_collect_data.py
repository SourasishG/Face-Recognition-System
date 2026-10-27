import cv2
import os

# Task 1 & 2: Collect face dataset and organize by person identity
person_name = input("Enter person's name: ").strip().lower()
save_folder = os.path.join("dataset", person_name)
os.makedirs(save_folder, exist_ok=True)

# Count existing images to avoid overwriting (allows adding glasses photos)
existing_count = len(os.listdir(save_folder))
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
count = 0
pad = 20  # Padding for glasses and full face boundary

print(f"[INFO] Collecting 30 photos for '{person_name}'. Press 'q' to quit.")

while count < 30:
    ret, frame = cap.read()
    if not ret:
        break

    # Natural mirror selfie view
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face with padding
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)

        face_crop = frame[y1:y2, x1:x2]
        img_name = f"face_{existing_count + count}.jpg"
        cv2.imwrite(os.path.join(save_folder, img_name), face_crop)
        count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        break

    cv2.putText(frame, f"Captured: {count}/30", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Register Face", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[SUCCESS] Saved 30 photos for '{person_name}'. Total now: {existing_count + count}")
