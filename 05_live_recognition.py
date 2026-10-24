import cv2
import pickle
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# Task 10: Live Face Recognition System using Webcam
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

print("[INFO] Loading FaceNet deep learning model...")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load registered face database
with open("models/face_model.pkl", "rb") as f:
    saved_data = pickle.load(f)

database = saved_data["centroids"]
threshold = 0.65  # Cosine similarity threshold for identifying registered users

# Load face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face and resize to 160x160
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        resized = cv2.resize(face_crop, (160, 160))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [-1, 1]
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        tensor = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)

        # Find best matching registered person
        best_name = "Unknown"
        best_score = -1.0

        for name, centroid in database.items():
            similarity = float(np.dot(emb, centroid))
            if similarity > best_score:
                best_score = similarity
                best_name = name

        # Check if similarity meets threshold
        if best_score >= threshold:
            label = f"{best_name} ({best_score:.2f})"
            box_color = (0, 255, 0)  # Green for recognized person
        else:
            label = f"Unknown ({best_score:.2f})"
            box_color = (0, 0, 255)  # Red for unknown visitor

        # Draw bounding box and name tag
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
