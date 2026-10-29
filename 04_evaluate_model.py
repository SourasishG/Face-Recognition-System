import os
import cv2
import pickle
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from facenet_pytorch import InceptionResnetV1

# Task 8 & 9: Test Model with Known/Unknown Persons and Calculate Performance Metrics
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
with open("models/face_model.pkl", "rb") as f:
    database = pickle.load(f)["database"]

test_dir = os.path.join("dataset_split", "test")
y_true = []
y_pred = []

for person in os.listdir(test_dir):
    p_dir = os.path.join(test_dir, person)
    if not os.path.isdir(p_dir):
        continue
    for filename in os.listdir(p_dir):
        img = cv2.imread(os.path.join(p_dir, filename))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor((rgb.astype(np.float32) - 127.5) / 128.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)

        best_name = "unknown"
        best_score = -1.0
        for name, user_embs in database.items():
            score = float(np.max(np.dot(user_embs, emb)))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.65:
            y_pred.append(best_name)
        else:
            y_pred.append("unknown")
        y_true.append(person)

print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
print(classification_report(y_true, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
