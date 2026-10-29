import cv2
import pickle
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

# Task 10: Live Face Recognition System using Webcam
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
with open("models/face_model.pkl", "rb") as f:
    database = pickle.load(f)["database"]

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        y1 = max(0, y - 20)
        y2 = min(frame.shape[0], y + h + 20)
        x1 = max(0, x - 20)
        x2 = min(frame.shape[1], x + w + 20)

        face_crop = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor((rgb.astype(np.float32) - 127.5) / 128.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)

        best_name = "Unknown"
        best_score = -1.0
        for name, user_embs in database.items():
            score = float(np.max(np.dot(user_embs, emb)))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.65:
            label = f"{best_name} ({int(best_score * 100)}%)"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({int(best_score * 100)}%)"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Live Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
